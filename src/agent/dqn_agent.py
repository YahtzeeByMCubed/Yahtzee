"""DQNAgent — epsilon-greedy action selection, prioritized experience
replay, learning step, and weight persistence.

This module owns everything that touches the network at training time
EXCEPT the network architecture itself (that lives in src/agent/model.py).

Design doc §3.5.2 (masking), §3.2 (PER), §3.5.1 (architecture), §3.4
(reward strategy), §2.2 (interface).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.agent.model import DQN


# -- masking ---------------------------------------------------------------

def select_legal_action(raw_q_values: np.ndarray, legal_moves_mask: np.ndarray) -> int:
    """Add the legal-moves mask to raw Q-values and argmax.

    Per design doc §3.5.2:
      - legal_moves_mask is shape (45,), 0.0 for legal, -np.inf for illegal.
      - Adding it to Q-values leaves legal Q-values unchanged and pushes
        illegal ones to -inf, so argmax cannot pick an illegal move.
    """
    return int(np.argmax(raw_q_values + legal_moves_mask))


# -- experience replay -----------------------------------------------------

@dataclass
class Transition:
    state: np.ndarray         # (24,)
    action: int               # 0..44
    reward: float             # 0.0 most of the time; terminal_score/100 at end
    next_state: np.ndarray    # (24,)
    done: bool


class PrioritizedReplayBuffer:
    """Sum-tree-backed PER (design doc §3.2).

    With strictly sparse rewards, the rare end-of-game transitions carry the
    only learning signal — uniform sampling drowns them out, so PER is not
    optional here.

    Sampling is proportional to priority^alpha; importance-sampling weights
    of (N * P(i))^-beta are returned and applied to the loss to correct the
    bias introduced by non-uniform sampling.

    Priorities stored in the tree are already alpha-scaled — i.e. the leaf
    value is (|td_err| + eps)^alpha. Sampling proportional to that leaf is
    therefore sampling proportional to priority^alpha.
    """

    _PRIORITY_EPS = 1e-6  # keeps zero-TD transitions from getting probability 0

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        # Binary heap layout: parents at i, children at 2i+1 and 2i+2.
        # Leaves occupy [capacity-1, 2*capacity-1); internal sums above.
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list[Transition | None] = [None] * capacity
        self._write = 0
        self._size = 0
        self.max_priority = 1.0

    def push(self, transition: Transition) -> None:
        tree_idx = self._write + self.capacity - 1
        self.data[self._write] = transition
        self._update_tree(tree_idx, self.max_priority)
        self._write = (self._write + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        """Return (transitions, tree_indices, is_weights).

        tree_indices are absolute positions in self.tree (not data indices).
        Pass them straight back to update_priorities after computing TD errors.
        """
        total = self.tree[0]
        segment = total / batch_size

        transitions: list[Transition] = []
        tree_indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            tree_idx = self._retrieve(s)
            data_idx = tree_idx - (self.capacity - 1)
            transitions.append(self.data[data_idx])
            tree_indices[i] = tree_idx
            priorities[i] = self.tree[tree_idx]

        sampling_probs = priorities / total
        is_weights = (self._size * sampling_probs) ** (-beta)
        is_weights /= is_weights.max()

        return transitions, tree_indices, is_weights.astype(np.float32)

    def update_priorities(self, indices, td_errors) -> None:
        for tree_idx, td_err in zip(indices, td_errors):
            priority = (abs(float(td_err)) + self._PRIORITY_EPS) ** self.alpha
            self._update_tree(int(tree_idx), priority)
            if priority > self.max_priority:
                self.max_priority = priority

    def __len__(self) -> int:
        return self._size

    # -- sum-tree internals --------------------------------------------

    def _update_tree(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def _retrieve(self, s: float) -> int:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right


# -- agent -----------------------------------------------------------------

class DQNAgent:
    """Wraps online + target networks, replay buffer, and epsilon schedule.

    Public API (design doc §2.4):
        get_action(state, mask) -> int               # < 10ms inference
        update_memory(s, a, r, s_next, done) -> None
        learn() -> float | None                      # returns loss or None if buffer too small
        save_weights(path: str) -> None              # writes a .pt file
        load_weights(path: str) -> None
    """

    def __init__(self,
                 buffer_capacity: int = 200_000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 target_sync_steps: int = 1_000,
                 eps_start: float = 1.0,
                 eps_end: float = 0.05,
                 eps_decay_steps: int = 200_000,
                 device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.online = DQN().to(self.device)
        self.target = DQN().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(buffer_capacity)

        self.batch_size = batch_size
        self.gamma = gamma
        self.target_sync_steps = target_sync_steps
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.global_step = 0

    # -- inference ------------------------------------------------------

    def get_action(self,
                   state: np.ndarray,
                   legal_mask: np.ndarray,
                   greedy: bool = False) -> int:
        """Pick an action. `greedy=True` forces epsilon=0 (used by evaluate)."""
        if not greedy and np.random.random() < self.epsilon(self.global_step):
            legal_indices = np.flatnonzero(legal_mask == 0.0)
            return int(np.random.choice(legal_indices))

        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s).squeeze(0).cpu().numpy()
        return select_legal_action(q, legal_mask)

    def epsilon(self, step: int) -> float:
        if step >= self.eps_decay_steps:
            return self.eps_end
        frac = step / self.eps_decay_steps
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    # -- training -------------------------------------------------------

    def update_memory(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(Transition(
            state=np.asarray(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
        ))

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        transitions, tree_indices, is_weights = self.buffer.sample(self.batch_size)

        states = torch.as_tensor(
            np.stack([t.state for t in transitions]), dtype=torch.float32, device=self.device,
        )
        next_states = torch.as_tensor(
            np.stack([t.next_state for t in transitions]), dtype=torch.float32, device=self.device,
        )
        actions = torch.as_tensor(
            [t.action for t in transitions], dtype=torch.int64, device=self.device,
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [t.reward for t in transitions], dtype=torch.float32, device=self.device,
        ).unsqueeze(1)
        dones = torch.as_tensor(
            [float(t.done) for t in transitions], dtype=torch.float32, device=self.device,
        ).unsqueeze(1)
        is_w = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Double DQN: argmax under online, evaluate under target. Vanilla
        # DQN biases targets up by maxing under the same network it trains.
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            target_q = self.target(next_states).gather(1, next_actions)
            y = rewards + (1.0 - dones) * self.gamma * target_q

        predicted_q = self.online(states).gather(1, actions)
        td_errors = y - predicted_q
        loss = (is_w * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.update_priorities(
            tree_indices,
            td_errors.detach().squeeze(1).cpu().numpy(),
        )

        self.global_step += 1
        if self.global_step % self.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    # -- persistence ----------------------------------------------------

    def save_weights(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online.state_dict(), out)

    def load_weights(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.online.load_state_dict(state_dict)
        self.target.load_state_dict(state_dict)


# -- training loop entry point --------------------------------------------

def train(num_episodes: int = 100_000,
          save_path: str = "models/dqn.pt",
          eval_interval: int = 1_000) -> None:
    """Outer training loop — drives the brain-in-a-vat.

    High-level shape:
        env = YahtzeeStateMachine()                      # sim fakes by default
        agent = DQNAgent()
        for ep in range(num_episodes):
            s = env.reset(); done = False
            while not done:
                mask   = env.get_legal_mask()
                a      = agent.get_action(s, mask)
                s_next, r, done = env.step(a)
                agent.update_memory(s, a, r, s_next, done)
                agent.learn()
                s = s_next
            if ep % eval_interval == 0:
                avg = evaluate(agent, num_games=100)
                print(f"ep={ep} eval_avg={avg:.1f}")
        agent.save_weights(save_path)

    Performance target (design doc §2.2): consistent average score >= 190
    over a 100-game evaluation period.
    """
    from src.engine.yahtzee_env import YahtzeeStateMachine

    env = YahtzeeStateMachine()
    agent = DQNAgent()

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            mask = env.get_legal_mask()
            action = agent.get_action(state, mask)
            next_state, reward, done = env.step(action)
            agent.update_memory(state, action, reward, next_state, done)
            agent.learn()
            state = next_state

        if ep % eval_interval == 0:
            avg = evaluate(agent, num_games=100)
            print(f"ep={ep} eval_avg={avg:.1f}")

    agent.save_weights(save_path)


def evaluate(agent: "DQNAgent", num_games: int = 100) -> float:
    """Greedy evaluation — epsilon=0, returns mean total score."""
    from src.engine.yahtzee_env import REWARD_NORMALIZER, YahtzeeStateMachine

    env = YahtzeeStateMachine()
    score_sum = 0.0
    for _ in range(num_games):
        state = env.reset()
        done = False
        terminal_reward = 0.0
        while not done:
            mask = env.get_legal_mask()
            action = agent.get_action(state, mask, greedy=True)
            state, reward, done = env.step(action)
            if done:
                terminal_reward = reward
        score_sum += terminal_reward * REWARD_NORMALIZER
    return score_sum / num_games
