"""DQNAgent — epsilon-greedy action selection, prioritized experience
replay, learning step, and weight persistence.

This module owns everything that touches the network at training time
EXCEPT the network architecture itself (that lives in src/agent/model.py).

Design doc §3.5.2 (masking), §3.2 (PER), §3.5.1 (architecture), §3.4
(reward strategy), §2.2 (interface).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# from src.agent.model import DQN, STATE_DIM, ACTION_DIM


# -- masking ---------------------------------------------------------------

def select_legal_action(raw_q_values: np.ndarray, legal_moves_mask: np.ndarray) -> int:
    """Add the legal-moves mask to raw Q-values and argmax.

    Per design doc §3.5.2:
      - legal_moves_mask is shape (45,), 0.0 for legal, -np.inf for illegal.
      - Adding it to Q-values leaves legal Q-values unchanged and pushes
        illegal ones to -inf, so argmax cannot pick an illegal move.
    """
    # TODO: return int(np.argmax(raw_q_values + legal_moves_mask)).
    raise NotImplementedError


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
    """

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        # TODO: allocate sum-tree storage of size `capacity`. Track:
        #   - max_priority seen (used as the priority for fresh transitions).
        #   - alpha (priority exponent).
        raise NotImplementedError

    def push(self, transition: Transition) -> None:
        # TODO: append at max_priority so new memories are guaranteed to
        # be sampled at least once before their priority is updated.
        raise NotImplementedError

    def sample(self, batch_size: int, beta: float = 0.4):
        # TODO: return (transitions, indices, is_weights).
        # Sampling is proportional to priority^alpha; IS-weights are
        # (N * P(i))^-beta, normalized by max weight in the batch.
        raise NotImplementedError

    def update_priorities(self, indices, td_errors) -> None:
        # TODO: priority_i = (|td_error_i| + epsilon) ^ alpha. Also bump
        # self.max_priority so fresh transitions stay competitive.
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO
        raise NotImplementedError


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
        # TODO:
        #   - Build self.online = DQN().to(device).
        #   - Build self.target = DQN().to(device); copy online weights and
        #     freeze gradients (it's only updated by hard sync).
        #   - Build self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr).
        #   - Build self.buffer = PrioritizedReplayBuffer(buffer_capacity).
        #   - Track self.global_step = 0 for the epsilon schedule and target sync.
        raise NotImplementedError

    # -- inference ------------------------------------------------------

    def get_action(self, state: np.ndarray, legal_mask: np.ndarray) -> int:
        # TODO:
        #   1. With probability epsilon(self.global_step), pick UNIFORMLY
        #      from {i : legal_mask[i] == 0.0}. Never explore illegal moves —
        #      it wastes the exploration budget.
        #   2. Otherwise, run online.forward(state) under torch.no_grad(),
        #      pull a numpy array, and return select_legal_action(q, mask).
        # Latency target: < 10ms on CPU (design doc §2.2).
        raise NotImplementedError

    def epsilon(self, step: int) -> float:
        # TODO: linear decay from eps_start at step 0 to eps_end at
        # eps_decay_steps; clamp at eps_end thereafter.
        raise NotImplementedError

    # -- training -------------------------------------------------------

    def update_memory(self, state, action, reward, next_state, done) -> None:
        # TODO: self.buffer.push(Transition(state, action, reward, next_state, done)).
        raise NotImplementedError

    def learn(self) -> float | None:
        # TODO:
        #   1. If len(self.buffer) < self.batch_size, return None (warmup).
        #   2. Sample (batch, indices, is_weights) from self.buffer.
        #   3. Compute Double-DQN targets:
        #        next_actions = argmax(online(next_state), dim=1)
        #        target_q     = target(next_state).gather(1, next_actions)
        #        y            = reward + (1 - done) * gamma * target_q
        #      The "argmax under online, evaluate under target" pattern is
        #      what makes this Double DQN — vanilla DQN biases targets up.
        #   4. predicted_q = online(state).gather(1, action)
        #   5. td_errors = y - predicted_q
        #   6. loss = (is_weights * td_errors.pow(2)).mean()
        #   7. optimizer.zero_grad(); loss.backward(); optimizer.step()
        #   8. self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        #   9. If self.global_step % target_sync_steps == 0, hard-copy weights:
        #        self.target.load_state_dict(self.online.state_dict())
        #   10. self.global_step += 1; return loss.item().
        raise NotImplementedError

    # -- persistence ----------------------------------------------------

    def save_weights(self, path: str) -> None:
        # TODO: torch.save(self.online.state_dict(), path). Make sure parent
        # dir exists; write to .pt extension.
        raise NotImplementedError

    def load_weights(self, path: str) -> None:
        # TODO: state_dict = torch.load(path, map_location=self.device);
        # online.load_state_dict; target.load_state_dict.
        raise NotImplementedError


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
    # TODO: implement the loop above.
    raise NotImplementedError


def evaluate(agent: "DQNAgent", num_games: int = 100) -> float:
    """Greedy evaluation — epsilon=0, returns mean total score."""
    # TODO: run num_games full games with epsilon clamped to 0; return the
    # mean of scorecard.total() across games. Don't push transitions into
    # the buffer during eval.
    raise NotImplementedError
