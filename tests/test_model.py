import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.model import DQN
from environment.constants import STATE_DIM, NUM_ACTIONS


class TestDQN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = DQN()

    # ============================================================
    # Single-state input
    # ============================================================

    def test_dqn_accepts_single_state_shape_24(self):
        state = torch.zeros(STATE_DIM, dtype=torch.float32)

        q_values = self.model(state)

        self.assertIsInstance(q_values, torch.Tensor)

    def test_dqn_outputs_45_q_values_for_single_state(self):
        state = torch.zeros(STATE_DIM, dtype=torch.float32)

        q_values = self.model(state)

        self.assertEqual(q_values.shape, (NUM_ACTIONS,))

    def test_dqn_single_output_contains_finite_values(self):
        state = torch.ones(STATE_DIM, dtype=torch.float32)

        q_values = self.model(state)

        self.assertTrue(torch.all(torch.isfinite(q_values)))

    # ============================================================
    # Batch input
    # ============================================================

    def test_dqn_accepts_batch_input_shape_8_by_24(self):
        batch = torch.zeros((8, STATE_DIM), dtype=torch.float32)

        q_values = self.model(batch)

        self.assertIsInstance(q_values, torch.Tensor)

    def test_dqn_outputs_batch_shape_8_by_45(self):
        batch = torch.zeros((8, STATE_DIM), dtype=torch.float32)

        q_values = self.model(batch)

        self.assertEqual(q_values.shape, (8, NUM_ACTIONS))

    def test_dqn_batch_output_contains_finite_values(self):
        batch = torch.randn((8, STATE_DIM), dtype=torch.float32)

        q_values = self.model(batch)

        self.assertTrue(torch.all(torch.isfinite(q_values)))

    # ============================================================
    # Invalid shapes
    # ============================================================

    def test_dqn_rejects_single_state_with_wrong_length(self):
        invalid_state = torch.zeros(STATE_DIM - 1, dtype=torch.float32)

        with self.assertRaises(ValueError):
            self.model(invalid_state)

    def test_dqn_rejects_batch_with_wrong_state_dimension(self):
        invalid_batch = torch.zeros((8, STATE_DIM - 1), dtype=torch.float32)

        with self.assertRaises(ValueError):
            self.model(invalid_batch)

    def test_dqn_rejects_three_dimensional_input(self):
        invalid_input = torch.zeros((2, 8, STATE_DIM), dtype=torch.float32)

        with self.assertRaises(ValueError):
            self.model(invalid_input)

    def test_dqn_rejects_scalar_input(self):
        invalid_input = torch.tensor(1.0)

        with self.assertRaises(ValueError):
            self.model(invalid_input)

    # ============================================================
    # Model properties
    # ============================================================

    def test_dqn_stores_state_and_action_dimensions(self):
        self.assertEqual(self.model.state_dim, STATE_DIM)
        self.assertEqual(self.model.action_dim, NUM_ACTIONS)

    def test_dqn_can_be_created_with_custom_dimensions(self):
        model = DQN(state_dim=10, action_dim=4)

        state = torch.zeros(10, dtype=torch.float32)
        q_values = model(state)

        self.assertEqual(q_values.shape, (4,))
        self.assertEqual(model.state_dim, 10)
        self.assertEqual(model.action_dim, 4)

    def test_dqn_has_trainable_parameters(self):
        params = list(self.model.parameters())

        self.assertGreater(len(params), 0)

        trainable_params = [p for p in params if p.requires_grad]
        self.assertGreater(len(trainable_params), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)