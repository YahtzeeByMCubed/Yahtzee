import unittest

from environment.scorecard_manager import (
    ScorecardManager,
    ScorecardError,
    CategoryAlreadyFilledError,
    InvalidCategoryError,
    InvalidDiceError,
)

from environment.constants import (
    NUM_CATEGORIES,
    UPPER_BONUS_VALUE,
    UPPER_BONUS_THRESHOLD,
    YAHTZEE_SCORE_VALUE,
    YAHTZEE_BONUS_VALUE,
    Category,
)


class TestScorecardManager(unittest.TestCase):
    def setUp(self):
        self.sm = ScorecardManager()

    # ============================================================
    # Initial State
    # ============================================================

    def test_initial_scorecard_starts_empty(self):
        self.assertEqual(len(self.sm.scores), NUM_CATEGORIES)
        self.assertEqual(self.sm.scores, [None] * NUM_CATEGORIES)

        self.assertEqual(self.sm.get_open(), [1] * NUM_CATEGORIES)
        self.assertFalse(self.sm.is_full())

        self.assertEqual(self.sm.upper_section_total, 0)
        self.assertEqual(self.sm.upper_section_bonus, 0)
        self.assertEqual(self.sm.lower_section_total, 0)
        self.assertEqual(self.sm.grand_total, 0)
        self.assertEqual(self.sm.total(), 0)

        self.assertEqual(self.sm.upper_prog(), 0.0)

        # This assumes you add the alias:
        # def has_bonus(self): return self.has_yahtzee_bonus()
        self.assertEqual(self.sm.has_yahtzee_bonus(), 0)

    # ============================================================
    # Category State / Locking
    # ============================================================

    def test_category_is_open_before_commit_and_locked_after_commit(self):
        self.assertTrue(self.sm.is_category_open(Category.ONES))

        score = self.sm.commit(Category.ONES, [1, 1, 1, 2, 3])

        self.assertEqual(score, 3)
        self.assertEqual(self.sm.category_score(Category.ONES), 3)
        self.assertFalse(self.sm.is_category_open(Category.ONES))
        self.assertEqual(self.sm.get_open()[Category.ONES], 0)

    def test_reusing_category_raises_error(self):
        self.sm.commit(Category.ONES, [1, 1, 1, 2, 3])

        with self.assertRaises(CategoryAlreadyFilledError):
            self.sm.commit(Category.ONES, [1, 1, 1, 1, 1])

    def test_calculate_score_does_not_modify_scorecard(self):
        score = self.sm.calculate_score(Category.THREES, [3, 3, 3, 4, 5])

        self.assertEqual(score, 9)
        self.assertEqual(self.sm.scores, [None] * NUM_CATEGORIES)
        self.assertEqual(self.sm.get_open(), [1] * NUM_CATEGORIES)

    # ============================================================
    # Upper Section Scoring
    # ============================================================

    def test_upper_section_scores_correctly(self):
        self.assertEqual(self.sm.calculate_score(Category.ONES, [1, 1, 1, 2, 3]), 3)
        self.assertEqual(self.sm.calculate_score(Category.TWOS, [2, 2, 1, 3, 4]), 4)
        self.assertEqual(self.sm.calculate_score(Category.THREES, [3, 3, 3, 4, 5]), 9)
        self.assertEqual(self.sm.calculate_score(Category.FOURS, [4, 4, 4, 4, 1]), 16)
        self.assertEqual(self.sm.calculate_score(Category.FIVES, [5, 5, 2, 3, 4]), 10)
        self.assertEqual(self.sm.calculate_score(Category.SIXES, [6, 6, 6, 1, 2]), 18)

    def test_upper_bonus_applies_at_63_or_more(self):
        scoring_plan = [
            (Category.ONES, [1, 1, 1, 1, 1]),          # 5
            (Category.TWOS, [2, 2, 2, 2, 2]),          # 10
            (Category.THREES, [3, 3, 3, 3, 3]),        # 15
            (Category.FOURS, [4, 4, 4, 4, 4]),         # 20
            (Category.FIVES, [5, 5, 5, 5, 5]),         # 25
            (Category.SIXES, [6, 6, 6, 2, 2]),         # 18
        ]

        for category, dice in scoring_plan:
            self.sm.commit(category, dice)

        # Upper subtotal = 5 + 10 + 15 + 20 + 25 + 18 = 93
        self.assertEqual(self.sm.upper_section_total, 93)
        self.assertEqual(self.sm.upper_section_bonus, UPPER_BONUS_VALUE)
        self.assertEqual(self.sm.upper_section_total_with_bonus, 93 + UPPER_BONUS_VALUE)
        self.assertEqual(self.sm.upper_prog(), 1.0)

    def test_upper_bonus_does_not_apply_below_63(self):
        self.sm.commit(Category.ONES, [1, 1, 1, 1, 1])      # 5
        self.sm.commit(Category.TWOS, [2, 2, 2, 2, 2])      # 10
        self.sm.commit(Category.THREES, [3, 3, 3, 3, 3])    # 15

        self.assertEqual(self.sm.upper_section_total, 30)
        self.assertEqual(self.sm.upper_section_bonus, 0)
        self.assertAlmostEqual(self.sm.upper_prog(), 30 / UPPER_BONUS_THRESHOLD)

    # ============================================================
    # Lower Section Scoring
    # ============================================================

    def test_three_of_a_kind_scoring(self):
        self.assertEqual(
            self.sm.calculate_score(Category.THREE_OF_A_KIND, [2, 2, 2, 5, 6]),
            17,
        )

        self.assertEqual(
            self.sm.calculate_score(Category.THREE_OF_A_KIND, [2, 2, 3, 5, 6]),
            0,
        )

    def test_four_of_a_kind_scoring(self):
        self.assertEqual(
            self.sm.calculate_score(Category.FOUR_OF_A_KIND, [6, 6, 6, 6, 2]),
            26,
        )

        self.assertEqual(
            self.sm.calculate_score(Category.FOUR_OF_A_KIND, [6, 6, 6, 2, 2]),
            0,
        )

    def test_full_house_scoring(self):
        self.assertEqual(
            self.sm.calculate_score(Category.FULL_HOUSE, [3, 3, 5, 5, 5]),
            25,
        )

        self.assertEqual(
            self.sm.calculate_score(Category.FULL_HOUSE, [3, 3, 3, 3, 5]),
            0,
        )

    def test_small_straight_scoring(self):
        self.assertEqual(
            self.sm.calculate_score(Category.SMALL_STRAIGHT, [1, 2, 3, 4, 6]),
            30,
        )

        self.assertEqual(
            self.sm.calculate_score(Category.SMALL_STRAIGHT, [1, 2, 2, 5, 6]),
            0,
        )

    def test_large_straight_scoring(self):
        self.assertEqual(
            self.sm.calculate_score(Category.LARGE_STRAIGHT, [1, 2, 3, 4, 5]),
            40,
        )

        self.assertEqual(
            self.sm.calculate_score(Category.LARGE_STRAIGHT, [2, 3, 4, 5, 6]),
            40,
        )

        self.assertEqual(
            self.sm.calculate_score(Category.LARGE_STRAIGHT, [1, 2, 3, 4, 6]),
            0,
        )

    def test_yahtzee_scoring(self):
        self.assertEqual(
            self.sm.calculate_score(Category.YAHTZEE, [4, 4, 4, 4, 4]),
            YAHTZEE_SCORE_VALUE,
        )

        self.assertEqual(
            self.sm.calculate_score(Category.YAHTZEE, [4, 4, 4, 4, 5]),
            0,
        )

    def test_chance_scoring(self):
        self.assertEqual(
            self.sm.calculate_score(Category.CHANCE, [1, 2, 3, 4, 5]),
            15,
        )

    # ============================================================
    # Dice Normalisation
    # ============================================================

    def test_dice_are_sorted_before_scoring(self):
        # Unsorted dice should still count as a large straight.
        self.assertEqual(
            self.sm.calculate_score(Category.LARGE_STRAIGHT, [5, 3, 1, 4, 2]),
            40,
        )

        # Unsorted dice should still count as a small straight.
        self.assertEqual(
            self.sm.calculate_score(Category.SMALL_STRAIGHT, [6, 4, 3, 2, 1]),
            30,
        )

    # ============================================================
    # Yahtzee Bonus
    # ============================================================

    def test_has_bonus_after_scoring_yahtzee_as_50(self):
        self.sm.commit(Category.YAHTZEE, [4, 4, 4, 4, 4])

        self.assertEqual(self.sm.category_score(Category.YAHTZEE), 50)
        self.assertEqual(self.sm.has_yahtzee_bonus(), 1)

    def test_extra_yahtzee_bonus_added_when_eligible(self):
        self.sm.commit(Category.YAHTZEE, [4, 4, 4, 4, 4])
        self.sm.commit(Category.FOURS, [4, 4, 4, 4, 4])

        self.assertEqual(self.sm.yahtzee_bonus, YAHTZEE_BONUS_VALUE)
        self.assertEqual(self.sm.category_score(Category.FOURS), 20)

        # Upper = 20
        # Lower = Yahtzee 50 + Yahtzee bonus 100
        # Grand total = 170
        self.assertEqual(self.sm.upper_section_total, 20)
        self.assertEqual(self.sm.lower_section_total, 150)
        self.assertEqual(self.sm.grand_total, 170)

    def test_no_yahtzee_bonus_if_yahtzee_category_not_already_scored_as_50(self):
        self.sm.commit(Category.CHANCE, [6, 6, 6, 6, 6])

        self.assertEqual(self.sm.category_score(Category.CHANCE), 30)
        self.assertEqual(self.sm.yahtzee_bonus, 0)
        self.assertEqual(self.sm.has_yahtzee_bonus(), 0)

    # ============================================================
    # Full Scorecard / Total
    # ============================================================

    def test_full_scorecard_and_grand_total(self):
        scoring_plan = [
            (Category.ONES, [1, 1, 1, 1, 1]),                 # 5
            (Category.TWOS, [2, 2, 2, 2, 2]),                 # 10
            (Category.THREES, [3, 3, 3, 3, 3]),               # 15
            (Category.FOURS, [4, 4, 4, 4, 4]),                # 20
            (Category.FIVES, [5, 5, 5, 5, 5]),                # 25
            (Category.SIXES, [6, 6, 6, 6, 6]),                # 30

            (Category.THREE_OF_A_KIND, [2, 2, 2, 4, 5]),      # 15
            (Category.FOUR_OF_A_KIND, [6, 6, 6, 6, 1]),       # 25
            (Category.FULL_HOUSE, [3, 3, 5, 5, 5]),           # 25
            (Category.SMALL_STRAIGHT, [1, 2, 3, 4, 6]),       # 30
            (Category.LARGE_STRAIGHT, [2, 3, 4, 5, 6]),       # 40
            (Category.YAHTZEE, [4, 4, 4, 4, 4]),              # 50
            (Category.CHANCE, [1, 2, 3, 4, 5]),               # 15
        ]

        for category, dice in scoring_plan:
            self.sm.commit(category, dice)

        self.assertTrue(self.sm.is_full())
        self.assertEqual(self.sm.get_open(), [0] * NUM_CATEGORIES)

        # Upper = 5 + 10 + 15 + 20 + 25 + 30 = 105
        # Upper bonus = 35
        # Lower = 15 + 25 + 25 + 30 + 40 + 50 + 15 = 200
        # Grand total = 105 + 35 + 200 = 340
        self.assertEqual(self.sm.upper_section_total, 105)
        self.assertEqual(self.sm.upper_section_bonus, 35)
        self.assertEqual(self.sm.lower_section_total, 200)
        self.assertEqual(self.sm.total(), 340)

    # ============================================================
    # Invalid Inputs
    # ============================================================

    def test_invalid_category_raises_error(self):
        with self.assertRaises(InvalidCategoryError):
            self.sm.calculate_score(-1, [1, 2, 3, 4, 5])

        with self.assertRaises(InvalidCategoryError):
            self.sm.calculate_score(13, [1, 2, 3, 4, 5])

        with self.assertRaises(InvalidCategoryError):
            self.sm.calculate_score("chance", [1, 2, 3, 4, 5])

    def test_invalid_dice_length_raises_error(self):
        with self.assertRaises(InvalidDiceError):
            self.sm.calculate_score(Category.CHANCE, [1, 2, 3, 4])

        with self.assertRaises(InvalidDiceError):
            self.sm.calculate_score(Category.CHANCE, [1, 2, 3, 4, 5, 6])

    def test_invalid_dice_values_raise_error(self):
        with self.assertRaises(InvalidDiceError):
            self.sm.calculate_score(Category.CHANCE, [0, 1, 2, 3, 4])

        with self.assertRaises(InvalidDiceError):
            self.sm.calculate_score(Category.CHANCE, [1, 2, 3, 4, 7])

        with self.assertRaises(InvalidDiceError):
            self.sm.calculate_score(Category.CHANCE, [1, 2, 3, 4, 5.0])

    # ============================================================
    # Display / Dictionary Output
    # ============================================================

    def test_as_dict_contains_expected_summary_fields(self):
        self.sm.commit(Category.ONES, [1, 1, 1, 2, 3])
        data = self.sm.as_dict()

        self.assertEqual(data["Ones"], 3)
        self.assertIn("Upper Section Total", data)
        self.assertIn("Upper Section Bonus", data)
        self.assertIn("Upper Section Total With Bonus", data)
        self.assertIn("Yahtzee Bonus", data)
        self.assertIn("Lower Section Total", data)
        self.assertIn("Grand Total", data)


if __name__ == "__main__":
    unittest.main()