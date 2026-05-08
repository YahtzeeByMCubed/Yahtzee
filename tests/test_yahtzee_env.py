import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.yahtzee_env import (
    YahtzeeEnv,
    IllegalActionError,
    GameOverError,
    sparse_terminal_reward,
)

from environment.constants import (
    STATE_DIM,
    NUM_ACTIONS,
    NUM_HOLD_ACTIONS,
    SCORE_OFFSET,
    MAX_ROLLS_PER_TURN,
    NUM_CATEGORIES,
    Category,
)


class TestYahtzeeEnv(unittest.TestCase):
    def setUp(self):
        self.env = YahtzeeEnv(seed=42)
        self.env.reset()

    # ============================================================
    # reset()
    # ============================================================

    def test_reset_returns_24_d_state_vector(self):
        state = self.env.reset()

        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(state.dtype, np.float32)

    def test_reset_starts_at_turn_1_roll_1(self):
        self.env.turn = 7
        self.env.current_roll = 3
        self.env.done = True

        self.env.reset()

        self.assertEqual(self.env.turn, 1)
        self.assertEqual(self.env.current_roll, 1)
        self.assertFalse(self.env.done)

    def test_reset_rolls_five_valid_dice(self):
        self.env.reset()
        dice = self.env.get_dice()

        self.assertEqual(dice.shape, (5,))
        self.assertTrue(np.all(dice >= 1))
        self.assertTrue(np.all(dice <= 6))

    # ============================================================
    # State Vector
    # ============================================================

    def test_state_vector_has_correct_face_counts(self):
        self.env.set_dice_for_testing([1, 3, 3, 4, 6])

        state = self.env.construct_state_vector()

        expected_counts = np.array([1, 0, 2, 1, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(state[0:6], expected_counts)

    def test_state_vector_has_roll_1_one_hot_after_reset(self):
        state = self.env.reset()

        expected_roll = np.array([1, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(state[6:9], expected_roll)

    def test_state_vector_has_roll_2_one_hot_after_one_hold_action(self):
        self.env.step(31)

        state = self.env.construct_state_vector()

        expected_roll = np.array([0, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(state[6:9], expected_roll)

    def test_state_vector_has_roll_3_one_hot_after_two_hold_actions(self):
        self.env.step(31)
        self.env.step(31)

        state = self.env.construct_state_vector()

        expected_roll = np.array([0, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(state[6:9], expected_roll)

    def test_state_vector_open_categories_start_all_open(self):
        state = self.env.reset()

        expected_open = np.ones(NUM_CATEGORIES, dtype=np.float32)
        np.testing.assert_array_equal(state[9:22], expected_open)

    def test_state_vector_updates_filled_category_to_zero(self):
        # Score Ones.
        self.env.set_dice_for_testing([1, 1, 1, 2, 3])
        self.env.step(SCORE_OFFSET + Category.ONES)

        state = self.env.construct_state_vector()

        self.assertEqual(state[9 + Category.ONES], 0.0)

    # ============================================================
    # Legal Mask
    # ============================================================

    def test_legal_mask_has_shape_45(self):
        mask = self.env.get_legal_mask()

        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, (NUM_ACTIONS,))
        self.assertEqual(mask.dtype, np.float32)

    def test_hold_actions_are_legal_on_roll_1(self):
        self.env.current_roll = 1

        mask = self.env.get_legal_mask()

        self.assertTrue(np.all(mask[0:NUM_HOLD_ACTIONS] == 0.0))

    def test_hold_actions_are_legal_on_roll_2(self):
        self.env.current_roll = 2

        mask = self.env.get_legal_mask()

        self.assertTrue(np.all(mask[0:NUM_HOLD_ACTIONS] == 0.0))

    def test_hold_actions_are_illegal_on_roll_3(self):
        self.env.current_roll = 3

        mask = self.env.get_legal_mask()

        self.assertTrue(np.all(np.isneginf(mask[0:NUM_HOLD_ACTIONS])))

    def test_open_score_categories_are_legal(self):
        mask = self.env.get_legal_mask()

        score_mask = mask[SCORE_OFFSET:NUM_ACTIONS]
        self.assertTrue(np.all(score_mask == 0.0))

    def test_filled_score_category_becomes_illegal(self):
        self.env.set_dice_for_testing([1, 1, 1, 2, 3])

        # Score Ones.
        self.env.step(SCORE_OFFSET + Category.ONES)

        mask = self.env.get_legal_mask()

        ones_action = SCORE_OFFSET + Category.ONES
        self.assertTrue(np.isneginf(mask[ones_action]))

    def test_other_open_categories_remain_legal_after_one_category_filled(self):
        self.env.set_dice_for_testing([1, 1, 1, 2, 3])

        # Score Ones.
        self.env.step(SCORE_OFFSET + Category.ONES)

        mask = self.env.get_legal_mask()

        twos_action = SCORE_OFFSET + Category.TWOS
        chance_action = SCORE_OFFSET + Category.CHANCE

        self.assertEqual(mask[twos_action], 0.0)
        self.assertEqual(mask[chance_action], 0.0)

    def test_all_actions_are_illegal_when_done(self):
        self.env.done = True

        mask = self.env.get_legal_mask()

        self.assertTrue(np.all(np.isneginf(mask)))

    # ============================================================
    # Hold Action Step Behaviour
    # ============================================================

    def test_hold_action_increments_current_roll(self):
        self.assertEqual(self.env.current_roll, 1)

        # Action 31 means keep all dice.
        next_state, reward, done = self.env.step(31)

        self.assertEqual(self.env.current_roll, 2)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(next_state.shape, (STATE_DIM,))

    def test_two_hold_actions_move_to_roll_3(self):
        self.env.step(31)
        self.env.step(31)

        self.assertEqual(self.env.current_roll, 3)

    def test_hold_action_is_rejected_on_roll_3(self):
        self.env.current_roll = 3

        with self.assertRaises(IllegalActionError):
            self.env.step(31)

    def test_invalid_action_is_rejected(self):
        with self.assertRaises(IllegalActionError):
            self.env.step(NUM_ACTIONS)

    # ============================================================
    # Score Action Step Behaviour
    # ============================================================

    def test_score_action_advances_to_next_turn(self):
        self.env.set_dice_for_testing([1, 1, 1, 2, 3])

        next_state, reward, done = self.env.step(SCORE_OFFSET + Category.ONES)

        self.assertEqual(self.env.turn, 2)
        self.assertEqual(self.env.current_roll, 1)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(next_state.shape, (STATE_DIM,))

    def test_score_action_locks_category(self):
        self.env.set_dice_for_testing([1, 1, 1, 2, 3])

        self.env.step(SCORE_OFFSET + Category.ONES)

        self.assertFalse(self.env.scorecard.is_category_open(Category.ONES))
        self.assertEqual(self.env.scorecard.category_score(Category.ONES), 3)

    def test_scoring_filled_category_is_rejected(self):
        self.env.set_dice_for_testing([1, 1, 1, 2, 3])
        self.env.step(SCORE_OFFSET + Category.ONES)

        with self.assertRaises(IllegalActionError):
            self.env.step(SCORE_OFFSET + Category.ONES)

    # ============================================================
    # Sparse Terminal Reward
    # ============================================================

    def test_sparse_terminal_reward_function(self):
        self.assertEqual(sparse_terminal_reward(0), 0.0)
        self.assertEqual(sparse_terminal_reward(250), 2.5)
        self.assertEqual(sparse_terminal_reward(375), 3.75)

    def test_non_terminal_score_action_gives_zero_reward(self):
        self.env.set_dice_for_testing([1, 1, 1, 2, 3])

        _, reward, done = self.env.step(SCORE_OFFSET + Category.ONES)

        self.assertEqual(reward, 0.0)
        self.assertFalse(done)

    def test_terminal_reward_only_given_when_scorecard_is_full(self):
        """
        Fill 12 categories manually, then use step() to score the final category.

        This checks:
        - reward is 0 before terminal state
        - final score action gives total_score / 100
        - done becomes True
        - terminal state is zeros
        """

        scoring_plan = [
            (Category.ONES, [1, 1, 1, 1, 1]),              # 5
            (Category.TWOS, [2, 2, 2, 2, 2]),              # 10
            (Category.THREES, [3, 3, 3, 3, 3]),            # 15
            (Category.FOURS, [4, 4, 4, 4, 4]),             # 20
            (Category.FIVES, [5, 5, 5, 5, 5]),             # 25
            (Category.SIXES, [6, 6, 6, 6, 6]),             # 30
            (Category.THREE_OF_A_KIND, [2, 2, 2, 4, 5]),   # 15
            (Category.FOUR_OF_A_KIND, [6, 6, 6, 6, 1]),    # 25
            (Category.FULL_HOUSE, [3, 3, 5, 5, 5]),        # 25
            (Category.SMALL_STRAIGHT, [1, 2, 3, 4, 6]),    # 30
            (Category.LARGE_STRAIGHT, [2, 3, 4, 5, 6]),    # 40
            (Category.YAHTZEE, [4, 4, 4, 4, 4]),           # 50
        ]

        # Manually fill 12 categories without using env.step(),
        # so the final category can be tested through step().
        for category, dice in scoring_plan:
            self.env.scorecard.commit(category, dice)

        self.assertFalse(self.env.scorecard.is_full())
        self.assertFalse(self.env.done)

        # Final remaining category is Chance.
        self.env.set_dice_for_testing([1, 2, 3, 4, 5])

        terminal_state, reward, done = self.env.step(SCORE_OFFSET + Category.CHANCE)

        expected_total = self.env.scorecard.total()
        expected_reward = expected_total / 100.0

        self.assertTrue(done)
        self.assertTrue(self.env.done)
        self.assertEqual(reward, expected_reward)
        self.assertGreater(reward, 0.0)
        np.testing.assert_array_equal(
            terminal_state,
            np.zeros(STATE_DIM, dtype=np.float32),
        )

    def test_step_after_game_done_raises_game_over_error(self):
        self.env.done = True

        with self.assertRaises(GameOverError):
            self.env.step(SCORE_OFFSET + Category.CHANCE)

    # ============================================================
    # Full episode smoke test
    # ============================================================

    def test_environment_can_finish_full_game_by_scoring_categories(self):
        env = YahtzeeEnv(seed=123)
        env.reset()

        categories = [
            Category.ONES,
            Category.TWOS,
            Category.THREES,
            Category.FOURS,
            Category.FIVES,
            Category.SIXES,
            Category.THREE_OF_A_KIND,
            Category.FOUR_OF_A_KIND,
            Category.FULL_HOUSE,
            Category.SMALL_STRAIGHT,
            Category.LARGE_STRAIGHT,
            Category.YAHTZEE,
            Category.CHANCE,
        ]

        done = False
        final_reward = 0.0

        for category in categories:
            _, reward, done = env.step(SCORE_OFFSET + category)
            final_reward = reward

            if category != Category.CHANCE:
                self.assertFalse(done)
                self.assertEqual(reward, 0.0)

        self.assertTrue(done)
        self.assertTrue(env.done)
        self.assertEqual(final_reward, env.scorecard.total() / 100.0)

    # ============================================================
    # String Output
    # ============================================================

    def test_string_output_contains_environment_status(self):
        output = str(self.env)

        self.assertIn("YahtzeeEnv", output)
        self.assertIn("turn=", output)
        self.assertIn("current_roll=", output)
        self.assertIn("done=", output)
        self.assertIn("dice=", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)