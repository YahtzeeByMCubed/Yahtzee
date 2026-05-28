import os
import sys
import unittest

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.action_selection import (
    select_legal_action,
    select_epsilon_greedy_action,
    NoLegalActionsError,
)

from src.engine.constants import NUM_ACTIONS


class TestActionSelection(unittest.TestCase):
    def make_all_legal_mask(self):
        return np.zeros(NUM_ACTIONS, dtype=np.float32)

    def make_all_illegal_mask(self):
        return np.full(NUM_ACTIONS, -np.inf, dtype=np.float32)

    # ============================================================
    # select_legal_action()
    # ============================================================

    def test_select_legal_action_chooses_highest_legal_q_value(self):
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        q_values[10] = 5.0
        q_values[20] = 15.0
        q_values[30] = 10.0

        legal_mask = self.make_all_legal_mask()

        action = select_legal_action(q_values, legal_mask)

        self.assertEqual(action, 20)

    def test_select_legal_action_ignores_illegal_high_q_values(self):
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # This is the highest raw Q-value, but it will be illegal.
        q_values[7] = 999.0

        # This is the best legal action.
        q_values[40] = 10.0

        legal_mask = self.make_all_legal_mask()
        legal_mask[7] = -np.inf

        action = select_legal_action(q_values, legal_mask)

        self.assertEqual(action, 40)

    def test_select_legal_action_raises_error_if_no_actions_are_legal(self):
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        legal_mask = self.make_all_illegal_mask()

        with self.assertRaises(NoLegalActionsError):
            select_legal_action(q_values, legal_mask)

    def test_select_legal_action_accepts_torch_tensor_q_values(self):
        q_values = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        q_values[12] = 50.0

        legal_mask = self.make_all_legal_mask()

        action = select_legal_action(q_values, legal_mask)

        self.assertEqual(action, 12)

    def test_select_legal_action_accepts_torch_tensor_mask(self):
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        q_values[33] = 8.0

        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)

        action = select_legal_action(q_values, legal_mask)

        self.assertEqual(action, 33)

    def test_select_legal_action_rejects_wrong_q_value_shape(self):
        q_values = np.zeros(NUM_ACTIONS - 1, dtype=np.float32)
        legal_mask = self.make_all_legal_mask()

        with self.assertRaises(ValueError):
            select_legal_action(q_values, legal_mask)

    def test_select_legal_action_rejects_wrong_mask_shape(self):
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        legal_mask = np.zeros(NUM_ACTIONS - 1, dtype=np.float32)

        with self.assertRaises(ValueError):
            select_legal_action(q_values, legal_mask)

    def test_select_legal_action_returns_first_max_when_tied(self):
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        q_values[5] = 10.0
        q_values[12] = 10.0

        legal_mask = self.make_all_legal_mask()

        action = select_legal_action(q_values, legal_mask)

        # np.argmax returns the first maximum.
        self.assertEqual(action, 5)

    # ============================================================
    # select_epsilon_greedy_action(), epsilon = 0
    # ============================================================

    def test_epsilon_zero_always_chooses_greedy_legal_action(self):
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        q_values[3] = 5.0
        q_values[25] = 20.0
        q_values[44] = 999.0

        legal_mask = self.make_all_legal_mask()

        # Make the true highest Q-value illegal.
        legal_mask[44] = -np.inf

        rng = np.random.default_rng(123)

        for _ in range(50):
            action = select_epsilon_greedy_action(
                q_values=q_values,
                legal_mask=legal_mask,
                epsilon=0.0,
                rng=rng,
            )

            self.assertEqual(action, 25)

    # ============================================================
    # select_epsilon_greedy_action(), epsilon = 1
    # ============================================================

    def test_epsilon_one_chooses_only_from_legal_actions(self):
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # Give an illegal action a huge value to make sure random selection
        # is based on the legal mask, not raw Q-values.
        q_values[44] = 999.0

        legal_mask = self.make_all_illegal_mask()

        legal_actions = {2, 7, 40}
        for action_idx in legal_actions:
            legal_mask[action_idx] = 0.0

        rng = np.random.default_rng(123)

        for _ in range(100):
            action = select_epsilon_greedy_action(
                q_values=q_values,
                legal_mask=legal_mask,
                epsilon=1.0,
                rng=rng,
            )

            self.assertIn(action, legal_actions)

    def test_epsilon_one_can_return_multiple_legal_actions_over_many_trials(self):
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)

        legal_mask = self.make_all_illegal_mask()
        legal_actions = {1, 5, 9, 13}

        for action_idx in legal_actions:
            legal_mask[action_idx] = 0.0

        rng = np.random.default_rng(42)

        selected = set()

        for _ in range(100):
            action = select_epsilon_greedy_action(
                q_values=q_values,
                legal_mask=legal_mask,
                epsilon=1.0,
                rng=rng,
            )
            selected.add(action)

        self.assertTrue(selected.issubset(legal_actions))
        self.assertGreater(len(selected), 1)

    # ============================================================
    # Epsilon validation
    # ============================================================

    def test_epsilon_greedy_rejects_negative_epsilon(self):
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        legal_mask = self.make_all_legal_mask()

        with self.assertRaises(ValueError):
            select_epsilon_greedy_action(q_values, legal_mask, epsilon=-0.1)

    def test_epsilon_greedy_rejects_epsilon_above_one(self):
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        legal_mask = self.make_all_legal_mask()

        with self.assertRaises(ValueError):
            select_epsilon_greedy_action(q_values, legal_mask, epsilon=1.1)

    def test_epsilon_greedy_raises_error_if_no_actions_are_legal(self):
        q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        legal_mask = self.make_all_illegal_mask()

        with self.assertRaises(NoLegalActionsError):
            select_epsilon_greedy_action(q_values, legal_mask, epsilon=1.0)

    # ============================================================
    # Integrated DQN-style example
    # ============================================================

    def test_action_selection_matches_masked_argmax_logic(self):
        q_values = np.arange(NUM_ACTIONS, dtype=np.float32)
        legal_mask = self.make_all_legal_mask()

        # Highest raw Q-value is action 44, but block actions 40-44.
        legal_mask[40:45] = -np.inf

        action = select_legal_action(q_values, legal_mask)

        self.assertEqual(action, 39)

    def test_only_one_legal_action_returns_that_action(self):
        q_values = np.random.default_rng(1).normal(size=NUM_ACTIONS).astype(np.float32)

        legal_mask = self.make_all_illegal_mask()
        legal_mask[18] = 0.0

        greedy_action = select_legal_action(q_values, legal_mask)
        random_action = select_epsilon_greedy_action(
            q_values,
            legal_mask,
            epsilon=1.0,
            rng=np.random.default_rng(5),
        )

        self.assertEqual(greedy_action, 18)
        self.assertEqual(random_action, 18)


if __name__ == "__main__":
    unittest.main(verbosity=2)