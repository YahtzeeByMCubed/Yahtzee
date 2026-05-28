import os
import sys
import unittest

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.engine.yahtzee_env import YahtzeeEnv
from src.engine.constants import STATE_DIM, NUM_ACTIONS
from src.agent.model import DQN
from src.agent.action_selection import select_legal_action, select_epsilon_greedy_action


class TestEnvAgentIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.env = YahtzeeEnv(seed=42)
        self.model = DQN()

    def test_dqn_can_choose_legal_action_from_environment_state(self):
        state = self.env.reset()
        legal_mask = self.env.get_legal_mask()

        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state_tensor)

        action = select_legal_action(q_values, legal_mask)

        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, NUM_ACTIONS)
        self.assertEqual(legal_mask[action], 0.0)

    def test_dqn_selected_action_can_be_stepped_in_environment(self):
        state = self.env.reset()
        legal_mask = self.env.get_legal_mask()

        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state_tensor)

        action = select_legal_action(q_values, legal_mask)

        next_state, reward, done = self.env.step(action)

        self.assertIsInstance(next_state, np.ndarray)
        self.assertEqual(next_state.shape, (STATE_DIM,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_untrained_dqn_can_complete_one_full_game_with_masking(self):
        state = self.env.reset()

        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            legal_mask = self.env.get_legal_mask()

            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state_tensor)

            action = select_epsilon_greedy_action(
                q_values=q_values,
                legal_mask=legal_mask,
                epsilon=0.5,
                rng=np.random.default_rng(123 + step_count),
            )

            self.assertEqual(legal_mask[action], 0.0)

            state, reward, done = self.env.step(action)

            self.assertTrue(np.all(np.isfinite(state)))
            self.assertTrue(np.isfinite(reward))

            total_reward += reward
            step_count += 1

            # Maximum possible actions:
            # 13 turns × 3 actions per turn = 39
            self.assertLessEqual(step_count, 39)

        self.assertTrue(done)
        self.assertTrue(self.env.done)
        self.assertGreaterEqual(total_reward, 0.0)

    def test_untrained_dqn_can_complete_many_games_without_illegal_actions(self):
        num_games = 50

        for game_idx in range(num_games):
            env = YahtzeeEnv(seed=game_idx)
            state = env.reset()

            done = False
            step_count = 0
            total_reward = 0.0

            rng = np.random.default_rng(game_idx)

            while not done:
                legal_mask = env.get_legal_mask()

                state_tensor = torch.tensor(state, dtype=torch.float32)
                q_values = self.model(state_tensor)

                action = select_epsilon_greedy_action(
                    q_values=q_values,
                    legal_mask=legal_mask,
                    epsilon=1.0,
                    rng=rng,
                )

                self.assertEqual(
                    legal_mask[action],
                    0.0,
                    msg=f"Illegal action selected in game {game_idx}: {action}",
                )

                state, reward, done = env.step(action)

                self.assertEqual(state.shape, (STATE_DIM,))
                self.assertTrue(np.all(np.isfinite(state)))
                self.assertTrue(np.isfinite(reward))

                total_reward += reward
                step_count += 1

                self.assertLessEqual(step_count, 39)

            self.assertTrue(done)
            self.assertGreaterEqual(total_reward, 0.0)

    def test_terminal_state_after_full_game_is_zero_vector(self):
        state = self.env.reset()

        done = False

        while not done:
            legal_mask = self.env.get_legal_mask()

            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state_tensor)

            action = select_epsilon_greedy_action(
                q_values=q_values,
                legal_mask=legal_mask,
                epsilon=1.0,
                rng=np.random.default_rng(42),
            )

            state, reward, done = self.env.step(action)

        np.testing.assert_array_equal(
            state,
            np.zeros(STATE_DIM, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)