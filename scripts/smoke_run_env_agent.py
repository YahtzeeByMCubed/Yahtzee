import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.yahtzee_env import YahtzeeEnv
from agent.model import DQN
from agent.action_selection import select_epsilon_greedy_action


def run_one_game(seed=42, epsilon=1.0):
    env = YahtzeeEnv(seed=seed)
    model = DQN()

    state = env.reset()
    done = False

    step_count = 0
    total_reward = 0.0

    rng = np.random.default_rng(seed)

    while not done:
        legal_mask = env.get_legal_mask()

        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = model(state_tensor)

        action = select_epsilon_greedy_action(
            q_values=q_values,
            legal_mask=legal_mask,
            epsilon=epsilon,
            rng=rng,
        )

        state, reward, done = env.step(action)

        step_count += 1
        total_reward += reward

        print(
            f"Step {step_count:02d} | "
            f"Turn {env.turn:02d} | "
            f"Roll {env.current_roll} | "
            f"Action {action:02d} | "
            f"Reward {reward:.2f} | "
            f"Done {done}"
        )

    print()
    print("Game finished.")
    print(f"Final score: {env.scorecard.total()}")
    print(f"Final reward: {total_reward:.2f}")
    print(f"Steps taken: {step_count}")


if __name__ == "__main__":
    run_one_game(seed=42, epsilon=1.0)