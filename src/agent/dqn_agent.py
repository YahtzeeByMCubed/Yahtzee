"""DQNAgent — epsilon-greedy action selection, prioritized experience
replay, learning step, and weight persistence.

This module owns everything that touches the network at training time
EXCEPT the network architecture itself (that lives in src/agent/model.py).

Design doc §3.5.2 (masking), §3.2 (PER), §3.5.1 (architecture), §3.4
(reward strategy), §2.2 (interface).

python -c "from src.agent.dqn_agent import train; \
           train(num_episodes=100_000, \
                 load_path='models/dqn.pt', \
                 save_path='models/dqn_v2.pt')"
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.agent.model import DQN, STATE_DIM, ACTION_DIM


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
    state: np.ndarray            # (24,)
    action: int                  # 0..44
    reward: float                # 0.0 most of the time; terminal_score/100 at end
    next_state: np.ndarray       # (24,)
    done: bool
    # Legal-action mask for next_state (45,), 0.0 legal / -inf illegal.
    # Used to mask the Double-DQN bootstrap argmax so illegal actions in
    # s_next can't poison the target — the failure mode that caused the
    # 100k-episode run to diverge into millions-magnitude loss.
    next_legal_mask: np.ndarray


class PrioritizedReplayBuffer:
    """Sum-tree-backed PER (design doc §3.2).

    With strictly sparse rewards, the rare end-of-game transitions carry the
    only learning signal — uniform sampling drowns them out, so PER is not
    optional here.

    Sampling is proportional to priority^alpha; importance-sampling weights
    of (N * P(i))^-beta are returned and applied to the loss to correct the
    bias introduced by non-uniform sampling.

    Tree layout: a flat array of size 2*capacity - 1. Index 0 is the root,
    leaves live at indices [capacity-1, 2*capacity-2]. Leaf i corresponds
    to data slot i. Each internal node holds the sum of its two children's
    priorities, so tree[0] is the total priority of the buffer.
    """

    _PRIORITY_FLOOR = 1e-6  # added to |td_error| before exponentiation

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data: List[Optional[Transition]] = [None] * self.capacity
        self.write = 0
        self.size = 0
        # Priorities stored in the tree are already raised to alpha.
        # max_priority is also kept in that same "stored" form.
        self.max_priority = 1.0

    # -- internal sum-tree primitives ----------------------------------

    def _propagate(self, leaf_idx: int, change: float) -> None:
        """Bubble a delta from a leaf up to the root."""
        parent = (leaf_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def _set_priority(self, leaf_idx: int, new_priority: float) -> None:
        change = new_priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = new_priority
        self._propagate(leaf_idx, change)

    def _retrieve(self, value: float) -> int:
        """Walk down from root to find the leaf whose cumulative priority
        contains `value`. Returns the tree index of the chosen leaf."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

    # -- public API ----------------------------------------------------

    def push(self, transition: Transition) -> None:
        """Append a transition with maximum stored priority so it is
        guaranteed to be sampled at least once before its priority is
        refined by an actual TD-error update."""
        leaf_idx = self.capacity - 1 + self.write
        self.data[self.write] = transition
        self._set_priority(leaf_idx, self.max_priority)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Stratified sampling across `batch_size` segments of total priority.

        Returns:
            transitions: list of `batch_size` transitions
            indices:     np.ndarray of tree indices (for update_priorities)
            is_weights:  np.ndarray of shape (batch_size,), max-normalized
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        total = float(self.tree[0])
        segment = total / batch_size

        transitions: List[Transition] = []
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            leaf_idx = self._retrieve(value)
            data_idx = leaf_idx - (self.capacity - 1)
            # Edge case: if priorities are zero at a leaf for any reason,
            # fall back to a valid populated slot.
            if self.data[data_idx] is None:
                data_idx = data_idx % self.size
                leaf_idx = self.capacity - 1 + data_idx
            transitions.append(self.data[data_idx])
            indices[i] = leaf_idx
            priorities[i] = self.tree[leaf_idx]

        sampling_probs = priorities / max(total, 1e-12)
        is_weights = (self.size * sampling_probs) ** (-beta)
        is_weights = is_weights / max(is_weights.max(), 1e-12)

        return transitions, indices, is_weights.astype(np.float32)

    def update_priorities(self, indices, td_errors) -> None:
        """priority_i = (|td_error_i| + epsilon) ** alpha. Bumps
        max_priority so freshly-pushed transitions stay competitive."""
        indices = np.asarray(indices)
        td_errors = np.asarray(td_errors, dtype=np.float64)
        for leaf_idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self._PRIORITY_FLOOR) ** self.alpha
            self._set_priority(int(leaf_idx), priority)
            if priority > self.max_priority:
                self.max_priority = priority

    def __len__(self) -> int:
        return self.size


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
                 lr: float = 5e-5,
                 target_sync_steps: int = 1_000,
                 grad_clip_norm: float = 10.0,
                 eps_start: float = 1.0,
                 eps_end: float = 0.05,
                 eps_decay_steps: int = 200_000,
                 device: str | torch.device = "cpu",
                 learn_every: int = 1,
                 beta_start: float = 0.4,
                 beta_end: float = 1.0,
                 beta_anneal_steps: int = 1_000_000) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_sync_steps = target_sync_steps
        self.grad_clip_norm = grad_clip_norm
        self.learn_every = max(1, int(learn_every))
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = max(1, int(beta_anneal_steps))
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.online = DQN().to(self.device)
        self.target = DQN().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(buffer_capacity)
        self.global_step = 0

        # Most recent raw Q-values from get_action / get_q_values — read
        # by the GUI's display_q_chart per design doc §2.3. Populated as
        # a numpy array of shape (45,); None until first inference.
        self.last_q_values: Optional[np.ndarray] = None

    # -- inference ------------------------------------------------------

    def get_action(self, state: np.ndarray, legal_mask: np.ndarray) -> int:
        eps = self.epsilon(self.global_step)

        # Always run a forward so the GUI's confidence chart has fresh
        # Q-values even when ε-greedy explores. This is microseconds on
        # the 24→45 MLP — cheaper than branching twice.
        q_values = self.get_q_values(state)

        if np.random.random() < eps:
            legal_indices = np.flatnonzero(legal_mask == 0.0)
            return int(np.random.choice(legal_indices))

        return select_legal_action(q_values, legal_mask)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return the raw 45-D Q-value vector for `state`. Used by the GUI
        for confidence display (design doc §2.3) and internally by
        get_action. Side-effect: caches into self.last_q_values."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            q_values = self.online(state_tensor).cpu().numpy()
        self.last_q_values = q_values
        return q_values

    def epsilon(self, step: int) -> float:
        frac = min(step / max(self.eps_decay_steps, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def beta(self, step: int) -> float:
        """PER importance-sampling exponent. Anneals linearly from
        beta_start (less aggressive bias correction, OK while the policy
        is bad) to beta_end (full bias correction, important late in
        training). Standard schedule from Schaul et al."""
        frac = min(step / self.beta_anneal_steps, 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    # -- training -------------------------------------------------------

    def update_memory(self, state, action, reward, next_state, done, next_legal_mask=None) -> None:
        """Spec signature is (s, a, r, s', done); next_legal_mask is an
        optional 6th arg. If omitted, falls back to an all-legal mask —
        callers that want correct Double-DQN bootstrap targets MUST pass
        the real mask, since otherwise the bootstrap argmax can pick an
        illegal next-state action and recreate the divergence bug."""
        if next_legal_mask is None:
            next_legal_mask = np.zeros(ACTION_DIM, dtype=np.float32)
        self.buffer.push(Transition(
            state, int(action), float(reward),
            next_state, bool(done), next_legal_mask,
        ))

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        # Skip the heavy compute on (learn_every - 1) of every learn_every
        # calls. Per cProfile, learn() is ~85% of training time, dominated
        # by Adam, network forward/backward, and the PER sum-tree walks.
        # Standard DQN recipe (Atari paper) is learn-every-4. global_step
        # still increments so target sync and ε decay remain in env-step
        # units regardless of learn_every.
        self.global_step += 1
        if self.global_step % self.learn_every != 0:
            return None

        beta = self.beta(self.global_step)
        batch, indices, is_weights = self.buffer.sample(self.batch_size, beta=beta)

        states = torch.as_tensor(
            np.stack([t.state for t in batch]),
            dtype=torch.float32, device=self.device,
        )
        actions = torch.as_tensor(
            np.array([t.action for t in batch], dtype=np.int64),
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            np.array([t.reward for t in batch], dtype=np.float32),
            device=self.device,
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.stack([t.next_state for t in batch]),
            dtype=torch.float32, device=self.device,
        )
        dones = torch.as_tensor(
            np.array([t.done for t in batch], dtype=np.float32),
            device=self.device,
        ).unsqueeze(1)
        next_masks = torch.as_tensor(
            np.stack([t.next_legal_mask for t in batch]),
            dtype=torch.float32, device=self.device,
        )
        is_w = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Double-DQN target: argmax under online (masked to legal next-state
        # actions), evaluate under target. Without the mask, the argmax can
        # pick an illegal action whose Q-value the network has hallucinated
        # — the bootstrap then propagates that hallucination back, target
        # values blow up, and Q-values diverge into the millions. For
        # terminal transitions the mask is all -inf, but (1 - done) zeros
        # that branch out of y, so the all-illegal mask is harmless there.
        with torch.no_grad():
            next_q_online = self.online(next_states) + next_masks
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            y = rewards + (1.0 - dones) * self.gamma * next_q

        predicted = self.online(states).gather(1, actions)
        td_errors = y - predicted
        # Huber (smooth-L1) is bounded-gradient at large errors, which —
        # together with grad clipping — keeps PER from positive-feedback
        # divergence when an outlier transition gets sampled.
        elementwise_loss = F.smooth_l1_loss(predicted, y, reduction="none")
        loss = (is_w * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self.buffer.update_priorities(
            indices,
            td_errors.detach().abs().squeeze(1).cpu().numpy(),
        )

        if self.global_step % self.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    # -- persistence ----------------------------------------------------

    def save_weights(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(self.online.state_dict(), path)

    def load_weights(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.online.load_state_dict(state_dict)
        self.target.load_state_dict(state_dict)


# -- environment bootstrap -------------------------------------------------

def _ensure_env_importable() -> None:
    """Put the repo root on sys.path so the top-level `environment` package
    resolves regardless of how this module is invoked. Mirrors the pattern
    used by scripts/smoke_run_env_agent.py."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# -- training loop entry point --------------------------------------------

def train(num_episodes: int = 100_000,
          save_path: str = "models/dqn.pt",
          eval_interval: int = 1_000,
          device: str | torch.device | None = None,
          load_path: Optional[str] = None,
          eps_start: Optional[float] = None,
          learn_every: int = 1,
          checkpoint_interval: int = 10_000,
          gamma: float = 0.99,
          beta_anneal_steps: int = 1_000_000) -> None:
    """Outer training loop — drives the brain-in-a-vat.

    Performance target (design doc §2.2): consistent average score >= 190
    over a 100-game evaluation period.
    """
    _ensure_env_importable()
    from environment.yahtzee_env import YahtzeeEnv

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device={device}", flush=True)

    # On resume, default ε_start to 0.1 instead of 1.0 so we don't waste
    # the first 200k steps re-exploring with a network that already knows
    # something. Caller can override either way via the kwarg.
    if eps_start is None:
        eps_start = 0.1 if load_path is not None else 1.0

    env = YahtzeeEnv()
    agent = DQNAgent(
        device=device,
        eps_start=eps_start,
        learn_every=learn_every,
        gamma=gamma,
        beta_anneal_steps=beta_anneal_steps,
    )
    if load_path is not None:
        agent.load_weights(load_path)
        print(f"Resumed weights from {load_path} (eps_start={eps_start})", flush=True)

    t0 = time.time()
    for ep in range(num_episodes):
        state = env.reset()
        mask = env.get_legal_mask()
        done = False
        ep_loss = 0.0
        loss_steps = 0

        while not done:
            action = agent.get_action(state, mask)
            next_state, reward, done = env.step(action)
            next_mask = env.get_legal_mask()
            agent.update_memory(state, action, reward, next_state, done, next_mask)
            loss = agent.learn()
            if loss is not None:
                ep_loss += loss
                loss_steps += 1
            state = next_state
            mask = next_mask

        # Periodic checkpoint so a 30-hour run doesn't lose everything if
        # something crashes or you Ctrl-C. Writes to <save_path>.ckpt.
        if checkpoint_interval > 0 and ep > 0 and ep % checkpoint_interval == 0:
            ckpt_path = save_path + ".ckpt"
            agent.save_weights(ckpt_path)

        if eval_interval > 0 and ep > 0 and ep % eval_interval == 0:
            avg = evaluate(agent, num_games=100)
            avg_loss = ep_loss / max(loss_steps, 1)
            eps_now = agent.epsilon(agent.global_step)
            elapsed = time.time() - t0
            eta = elapsed * (num_episodes - ep) / ep
            beta_now = agent.beta(agent.global_step)
            print(
                f"ep={ep} eval_avg={avg:.1f} "
                f"loss={avg_loss:.4f} eps={eps_now:.3f} beta={beta_now:.3f} "
                f"buffer={len(agent.buffer)} "
                f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                flush=True,
            )

    agent.save_weights(save_path)
    print(f"Saved final weights to {save_path}")


def evaluate(agent: "DQNAgent", num_games: int = 100) -> float:
    """Greedy evaluation — epsilon=0, returns mean total score."""
    _ensure_env_importable()
    from environment.yahtzee_env import YahtzeeEnv

    totals: List[int] = []
    for game_idx in range(num_games):
        env = YahtzeeEnv()
        state = env.reset()
        done = False

        while not done:
            mask = env.get_legal_mask()
            with torch.no_grad():
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32, device=agent.device,
                )
                q_values = agent.online(state_tensor).cpu().numpy()
            action = select_legal_action(q_values, mask)
            state, _, done = env.step(action)

        totals.append(env.scorecard.total())

    return float(np.mean(totals))


if __name__ == "__main__":
    train()
