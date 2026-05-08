import os
import sys
import unittest

import numpy as np

# Allows this test to run directly from VS Code or PowerShell.
# It adds AI_Assignment/ to Python's import path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.dice_manager import (
    DiceManager,
    DiceManagerError,
    InvalidKeepMaskError,
)

from environment.constants import (
    NUM_DICE,
    NUM_DIE_FACES,
)


class FakeRNG:
    """
    Small fake RNG for deterministic reroll tests.

    DiceManager expects an object with:
        integers(low, high, size)
    """

    def __init__(self, values):
        self.values = np.array(values, dtype=np.int32)

    def integers(self, low, high, size):
        if size != len(self.values):
            raise ValueError(
                f"FakeRNG expected size {len(self.values)}, got {size}."
            )

        return self.values.copy()


class TestDiceManager(unittest.TestCase):
    def setUp(self):
        self.dm = DiceManager(seed=42)

    # ============================================================
    # Initial State
    # ============================================================

    def test_initial_dice_start_as_zeros(self):
        dice = self.dm.get_dice()

        self.assertIsInstance(dice, np.ndarray)
        self.assertEqual(dice.shape, (NUM_DICE,))
        self.assertEqual(dice.dtype, np.int32)
        np.testing.assert_array_equal(dice, np.zeros(NUM_DICE, dtype=np.int32))

    # ============================================================
    # roll_all()
    # ============================================================

    def test_roll_all_returns_five_dice(self):
        dice = self.dm.roll_all()

        self.assertIsInstance(dice, np.ndarray)
        self.assertEqual(dice.shape, (NUM_DICE,))
        self.assertEqual(dice.dtype, np.int32)

    def test_roll_all_values_are_between_1_and_6(self):
        dice = self.dm.roll_all()

        self.assertTrue(np.all(dice >= 1))
        self.assertTrue(np.all(dice <= NUM_DIE_FACES))

    def test_roll_all_stores_dice_in_sorted_order(self):
        dice = self.dm.roll_all()

        self.assertTrue(np.all(dice[:-1] <= dice[1:]))

    def test_roll_all_updates_internal_dice_state(self):
        rolled = self.dm.roll_all()
        stored = self.dm.get_dice()

        np.testing.assert_array_equal(rolled, stored)

    def test_same_seed_produces_same_first_roll(self):
        dm1 = DiceManager(seed=123)
        dm2 = DiceManager(seed=123)

        roll1 = dm1.roll_all()
        roll2 = dm2.roll_all()

        np.testing.assert_array_equal(roll1, roll2)

    def test_different_seeds_can_produce_different_rolls(self):
        dm1 = DiceManager(seed=1)
        dm2 = DiceManager(seed=2)

        roll1 = dm1.roll_all()
        roll2 = dm2.roll_all()

        # This should normally be true. It is technically possible but very unlikely
        # for two different seeds to produce the exact same first sorted roll.
        self.assertFalse(np.array_equal(roll1, roll2))

    # ============================================================
    # reroll()
    # ============================================================

    def test_reroll_keep_all_dice_does_not_change_dice(self):
        self.dm.set_dice_for_testing([1, 2, 3, 4, 5])
        before = self.dm.get_dice()

        after = self.dm.reroll([True, True, True, True, True])

        np.testing.assert_array_equal(after, before)
        np.testing.assert_array_equal(self.dm.get_dice(), before)

    def test_reroll_keep_no_dice_replaces_all_dice(self):
        self.dm.set_dice_for_testing([1, 1, 1, 1, 1])

        # Force the rerolled dice to be known.
        self.dm.rng = FakeRNG([2, 3, 4, 5, 6])

        after = self.dm.reroll([False, False, False, False, False])

        expected = np.array([2, 3, 4, 5, 6], dtype=np.int32)
        np.testing.assert_array_equal(after, expected)
        np.testing.assert_array_equal(self.dm.get_dice(), expected)

    def test_reroll_keeps_true_positions_and_rerolls_false_positions(self):
        self.dm.set_dice_for_testing([1, 2, 3, 4, 5])

        # Current sorted dice are [1, 2, 3, 4, 5]
        # Keep first and last dice: 1 and 5
        # Reroll middle three dice into 6, 6, 6
        self.dm.rng = FakeRNG([6, 6, 6])

        after = self.dm.reroll([True, False, False, False, True])

        expected = np.array([1, 5, 6, 6, 6], dtype=np.int32)
        np.testing.assert_array_equal(after, expected)

    def test_reroll_result_is_sorted(self):
        self.dm.set_dice_for_testing([1, 2, 3, 4, 5])

        # Keep 5, reroll others into unsorted values.
        # DiceManager should sort the final result.
        self.dm.rng = FakeRNG([4, 1, 6, 2])

        after = self.dm.reroll([False, False, False, False, True])

        expected = np.array([1, 2, 4, 5, 6], dtype=np.int32)
        np.testing.assert_array_equal(after, expected)
        self.assertTrue(np.all(after[:-1] <= after[1:]))

    def test_reroll_updates_internal_dice_state(self):
        self.dm.set_dice_for_testing([1, 2, 3, 4, 5])
        self.dm.rng = FakeRNG([6, 6, 6])

        after = self.dm.reroll([True, True, False, False, False])
        stored = self.dm.get_dice()

        expected = np.array([1, 2, 6, 6, 6], dtype=np.int32)

        np.testing.assert_array_equal(after, expected)
        np.testing.assert_array_equal(stored, expected)

    # ============================================================
    # Invalid keep masks
    # ============================================================

    def test_reroll_rejects_keep_mask_that_is_too_short(self):
        self.dm.set_dice_for_testing([1, 2, 3, 4, 5])

        with self.assertRaises(InvalidKeepMaskError):
            self.dm.reroll([True, False, True, False])

    def test_reroll_rejects_keep_mask_that_is_too_long(self):
        self.dm.set_dice_for_testing([1, 2, 3, 4, 5])

        with self.assertRaises(InvalidKeepMaskError):
            self.dm.reroll([True, False, True, False, True, False])

    def test_reroll_rejects_non_boolean_keep_mask_values(self):
        self.dm.set_dice_for_testing([1, 2, 3, 4, 5])

        invalid_masks = [
            [1, 0, 1, 0, 1],
            ["True", False, True, False, True],
            [True, False, None, False, True],
            [True, False, 0.0, False, True],
        ]

        for mask in invalid_masks:
            with self.subTest(mask=mask):
                with self.assertRaises(InvalidKeepMaskError):
                    self.dm.reroll(mask)

    # ============================================================
    # set_dice_for_testing()
    # ============================================================

    def test_set_dice_for_testing_sets_sorted_dice(self):
        self.dm.set_dice_for_testing([6, 1, 3, 2, 3])

        expected = np.array([1, 2, 3, 3, 6], dtype=np.int32)
        np.testing.assert_array_equal(self.dm.get_dice(), expected)

    def test_set_dice_for_testing_rejects_too_few_dice(self):
        with self.assertRaises(ValueError):
            self.dm.set_dice_for_testing([1, 2, 3, 4])

    def test_set_dice_for_testing_rejects_too_many_dice(self):
        with self.assertRaises(ValueError):
            self.dm.set_dice_for_testing([1, 2, 3, 4, 5, 6])

    def test_set_dice_for_testing_rejects_non_integer_dice(self):
        invalid_dice_sets = [
            [1, 2, 3, 4, 5.0],
            [1, 2, 3, 4, "5"],
            [1, 2, 3, 4, None],
        ]

        for dice in invalid_dice_sets:
            with self.subTest(dice=dice):
                with self.assertRaises(ValueError):
                    self.dm.set_dice_for_testing(dice)

    def test_set_dice_for_testing_rejects_dice_below_1(self):
        with self.assertRaises(ValueError):
            self.dm.set_dice_for_testing([0, 1, 2, 3, 4])

    def test_set_dice_for_testing_rejects_dice_above_6(self):
        with self.assertRaises(ValueError):
            self.dm.set_dice_for_testing([1, 2, 3, 4, 7])

    # ============================================================
    # face_counts()
    # ============================================================

    def test_face_counts_returns_six_values(self):
        self.dm.set_dice_for_testing([1, 3, 3, 4, 6])

        counts = self.dm.face_counts()

        self.assertIsInstance(counts, np.ndarray)
        self.assertEqual(counts.shape, (NUM_DIE_FACES,))
        self.assertEqual(counts.dtype, np.float32)

    def test_face_counts_correctly_counts_each_face(self):
        self.dm.set_dice_for_testing([1, 3, 3, 4, 6])

        counts = self.dm.face_counts()

        expected = np.array([1, 0, 2, 1, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(counts, expected)

    def test_face_counts_for_all_same_dice(self):
        self.dm.set_dice_for_testing([6, 6, 6, 6, 6])

        counts = self.dm.face_counts()

        expected = np.array([0, 0, 0, 0, 0, 5], dtype=np.float32)
        np.testing.assert_array_equal(counts, expected)

    # ============================================================
    # get_dice()
    # ============================================================

    def test_get_dice_returns_copy_not_reference(self):
        self.dm.set_dice_for_testing([1, 2, 3, 4, 5])

        dice_copy = self.dm.get_dice()
        dice_copy[0] = 99

        # Internal dice should not change.
        expected = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        np.testing.assert_array_equal(self.dm.get_dice(), expected)

    # ============================================================
    # __str__()
    # ============================================================

    def test_string_output_contains_class_name_and_dice(self):
        self.dm.set_dice_for_testing([1, 2, 3, 4, 5])

        output = str(self.dm)

        self.assertIn("DiceManager", output)
        self.assertIn("dice=", output)
        self.assertIn("[1, 2, 3, 4, 5]", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)