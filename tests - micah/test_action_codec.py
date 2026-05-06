import os
import sys
import unittest

# Allows this test to run directly from VS Code or PowerShell.
# It adds AI_Assignment/ to Python's import path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.action_codec import (
    ActionCodec,
    ActionCodecError,
    InvalidActionError,
)

from environment.constants import (
    NUM_ACTIONS,
    NUM_HOLD_ACTIONS,
    SCORE_OFFSET,
    NUM_CATEGORIES,
    Category,
)


class TestActionCodec(unittest.TestCase):
    # ============================================================
    # validate_action()
    # ============================================================

    def test_validate_action_accepts_all_valid_actions(self):
        for action_idx in range(NUM_ACTIONS):
            with self.subTest(action_idx=action_idx):
                ActionCodec.validate_action(action_idx)

    def test_validate_action_rejects_negative_action(self):
        with self.assertRaises(InvalidActionError):
            ActionCodec.validate_action(-1)

    def test_validate_action_rejects_action_equal_to_num_actions(self):
        with self.assertRaises(InvalidActionError):
            ActionCodec.validate_action(NUM_ACTIONS)

    def test_validate_action_rejects_action_above_valid_range(self):
        invalid_actions = [45, 46, 100, 999]

        for action_idx in invalid_actions:
            with self.subTest(action_idx=action_idx):
                with self.assertRaises(InvalidActionError):
                    ActionCodec.validate_action(action_idx)

    def test_validate_action_rejects_non_integer_actions(self):
        invalid_actions = [
            0.0,
            1.5,
            "0",
            "32",
            None,
            [1],
            {"action": 1},
        ]

        for action_idx in invalid_actions:
            with self.subTest(action_idx=action_idx):
                with self.assertRaises(InvalidActionError):
                    ActionCodec.validate_action(action_idx)

    # ============================================================
    # is_hold()
    # ============================================================

    def test_is_hold_returns_true_for_actions_0_to_31(self):
        for action_idx in range(0, NUM_HOLD_ACTIONS):
            with self.subTest(action_idx=action_idx):
                self.assertTrue(ActionCodec.is_hold(action_idx))

    def test_is_hold_returns_false_for_score_actions_32_to_44(self):
        for action_idx in range(SCORE_OFFSET, NUM_ACTIONS):
            with self.subTest(action_idx=action_idx):
                self.assertFalse(ActionCodec.is_hold(action_idx))

    def test_is_hold_rejects_invalid_actions(self):
        invalid_actions = [-1, NUM_ACTIONS, "5", 5.0, None]

        for action_idx in invalid_actions:
            with self.subTest(action_idx=action_idx):
                with self.assertRaises(InvalidActionError):
                    ActionCodec.is_hold(action_idx)

    # ============================================================
    # is_score()
    # ============================================================

    def test_is_score_returns_false_for_hold_actions_0_to_31(self):
        for action_idx in range(0, NUM_HOLD_ACTIONS):
            with self.subTest(action_idx=action_idx):
                self.assertFalse(ActionCodec.is_score(action_idx))

    def test_is_score_returns_true_for_actions_32_to_44(self):
        for action_idx in range(SCORE_OFFSET, NUM_ACTIONS):
            with self.subTest(action_idx=action_idx):
                self.assertTrue(ActionCodec.is_score(action_idx))

    def test_is_score_rejects_invalid_actions(self):
        invalid_actions = [-1, NUM_ACTIONS, "32", 32.0, None]

        for action_idx in invalid_actions:
            with self.subTest(action_idx=action_idx):
                with self.assertRaises(InvalidActionError):
                    ActionCodec.is_score(action_idx)

    # ============================================================
    # Hold/score range separation
    # ============================================================

    def test_hold_and_score_actions_do_not_overlap(self):
        for action_idx in range(NUM_ACTIONS):
            with self.subTest(action_idx=action_idx):
                is_hold = ActionCodec.is_hold(action_idx)
                is_score = ActionCodec.is_score(action_idx)

                # Every valid action should be exactly one of these.
                self.assertNotEqual(is_hold, is_score)

    def test_boundary_actions_are_classified_correctly(self):
        self.assertTrue(ActionCodec.is_hold(0))
        self.assertTrue(ActionCodec.is_hold(31))

        self.assertFalse(ActionCodec.is_hold(32))
        self.assertFalse(ActionCodec.is_hold(44))

        self.assertFalse(ActionCodec.is_score(0))
        self.assertFalse(ActionCodec.is_score(31))

        self.assertTrue(ActionCodec.is_score(32))
        self.assertTrue(ActionCodec.is_score(44))

    # ============================================================
    # decode_hold_action()
    # ============================================================

    def test_decode_hold_action_zero_rerolls_all_dice(self):
        mask = ActionCodec.decode_hold_action(0)

        self.assertEqual(mask, [False, False, False, False, False])

    def test_decode_hold_action_31_keeps_all_dice(self):
        mask = ActionCodec.decode_hold_action(31)

        self.assertEqual(mask, [True, True, True, True, True])

    def test_decode_hold_action_5_matches_expected_binary_pattern(self):
        # 5 in binary is 00101.
        # Reading from least significant bit to most significant bit:
        # die 0 = True
        # die 1 = False
        # die 2 = True
        # die 3 = False
        # die 4 = False
        mask = ActionCodec.decode_hold_action(5)

        self.assertEqual(mask, [True, False, True, False, False])

    def test_decode_hold_action_10_matches_expected_binary_pattern(self):
        # 10 in binary is 01010.
        mask = ActionCodec.decode_hold_action(10)

        self.assertEqual(mask, [False, True, False, True, False])

    def test_decode_hold_action_16_keeps_only_fifth_die(self):
        # 16 in binary is 10000.
        mask = ActionCodec.decode_hold_action(16)

        self.assertEqual(mask, [False, False, False, False, True])

    def test_decode_hold_action_returns_five_booleans_for_every_hold_action(self):
        for action_idx in range(NUM_HOLD_ACTIONS):
            with self.subTest(action_idx=action_idx):
                mask = ActionCodec.decode_hold_action(action_idx)

                self.assertEqual(len(mask), 5)

                for value in mask:
                    self.assertIsInstance(value, bool)

    def test_decode_hold_action_round_trip_binary_encoding(self):
        """
        This checks that every decoded keep mask can be converted back
        into the original action index.

        Example:
            action 5 -> [True, False, True, False, False]
            reconstructed = 1 + 4 = 5
        """

        for action_idx in range(NUM_HOLD_ACTIONS):
            with self.subTest(action_idx=action_idx):
                mask = ActionCodec.decode_hold_action(action_idx)

                reconstructed = 0
                for bit_position, keep in enumerate(mask):
                    if keep:
                        reconstructed += 2 ** bit_position

                self.assertEqual(reconstructed, action_idx)

    def test_decode_hold_action_rejects_score_actions(self):
        score_actions = list(range(SCORE_OFFSET, NUM_ACTIONS))

        for action_idx in score_actions:
            with self.subTest(action_idx=action_idx):
                with self.assertRaises(InvalidActionError):
                    ActionCodec.decode_hold_action(action_idx)

    def test_decode_hold_action_rejects_invalid_actions(self):
        invalid_actions = [-1, NUM_ACTIONS, "5", 5.0, None]

        for action_idx in invalid_actions:
            with self.subTest(action_idx=action_idx):
                with self.assertRaises(InvalidActionError):
                    ActionCodec.decode_hold_action(action_idx)

    # ============================================================
    # decode_score_action()
    # ============================================================

    def test_decode_score_action_32_is_category_0_ones(self):
        category_idx = ActionCodec.decode_score_action(32)

        self.assertEqual(category_idx, 0)
        self.assertEqual(category_idx, Category.ONES)

    def test_decode_score_action_40_is_category_8_full_house(self):
        category_idx = ActionCodec.decode_score_action(40)

        self.assertEqual(category_idx, 8)
        self.assertEqual(category_idx, Category.FULL_HOUSE)

    def test_decode_score_action_44_is_category_12_chance(self):
        category_idx = ActionCodec.decode_score_action(44)

        self.assertEqual(category_idx, 12)
        self.assertEqual(category_idx, Category.CHANCE)

    def test_decode_score_action_maps_all_score_actions_to_categories_0_to_12(self):
        for action_idx in range(SCORE_OFFSET, NUM_ACTIONS):
            with self.subTest(action_idx=action_idx):
                category_idx = ActionCodec.decode_score_action(action_idx)
                expected_category_idx = action_idx - SCORE_OFFSET

                self.assertEqual(category_idx, expected_category_idx)
                self.assertGreaterEqual(category_idx, 0)
                self.assertLess(category_idx, NUM_CATEGORIES)

    def test_decode_score_action_round_trip_with_score_offset(self):
        """
        This checks the core SCORE_OFFSET idea:

            action_idx = SCORE_OFFSET + category_idx
            category_idx = action_idx - SCORE_OFFSET
        """

        for category_idx in range(NUM_CATEGORIES):
            with self.subTest(category_idx=category_idx):
                action_idx = SCORE_OFFSET + category_idx
                decoded_category = ActionCodec.decode_score_action(action_idx)

                self.assertEqual(decoded_category, category_idx)

    def test_decode_score_action_rejects_hold_actions(self):
        hold_actions = list(range(0, NUM_HOLD_ACTIONS))

        for action_idx in hold_actions:
            with self.subTest(action_idx=action_idx):
                with self.assertRaises(InvalidActionError):
                    ActionCodec.decode_score_action(action_idx)

    def test_decode_score_action_rejects_invalid_actions(self):
        invalid_actions = [-1, NUM_ACTIONS, "32", 32.0, None]

        for action_idx in invalid_actions:
            with self.subTest(action_idx=action_idx):
                with self.assertRaises(InvalidActionError):
                    ActionCodec.decode_score_action(action_idx)

    # ============================================================
    # Error hierarchy
    # ============================================================

    def test_invalid_action_error_is_action_codec_error(self):
        self.assertTrue(issubclass(InvalidActionError, ActionCodecError))


if __name__ == "__main__":
    unittest.main(verbosity=2)