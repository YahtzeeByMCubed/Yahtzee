"""
Scorecard manager for Yahtzee.

This class is responsible for:
- storing all 13 category scores
- checking which categories are still open
- calculating category scores from dice
- committing scores into the scorecard
- calculating upper bonus
- calculating Yahtzee bonus
- calculating the final game total

It does NOT:
- roll dice
- choose actions
- train the DQN
- build the full state vector
"""

from __future__ import annotations

from numbers import Integral
from typing import Iterable, List, Optional

from .constants import (
    NUM_CATEGORIES,
    NUM_DICE,
    NUM_DIE_FACES,
    UPPER_BONUS_THRESHOLD,
    UPPER_BONUS_VALUE,
    YAHTZEE_SCORE_VALUE,
    YAHTZEE_BONUS_VALUE,
    FULL_HOUSE_SCORE_VALUE,
    SMALL_STRAIGHT_SCORE_VALUE,
    LARGE_STRAIGHT_SCORE_VALUE,
    Category,
    CATEGORY_NAMES,
    UPPER_CATEGORIES,
    LOWER_CATEGORIES,
)


class ScorecardError(Exception):
    """Base error for scorecard-related issues."""


class CategoryAlreadyFilledError(ScorecardError):
    """Raised when trying to score into a category that is already filled."""


class InvalidCategoryError(ScorecardError):
    """Raised when a category index is outside the valid range 0-12."""


class InvalidDiceError(ScorecardError):
    """Raised when dice input is invalid."""


class ScorecardManager:
    """
    Object-oriented Yahtzee scorecard.

    Internal representation:
        self.scores = [None] * 13

    Where:
        None = category has not been used yet
        int  = category has been scored and locked

    Category mapping:
        0  Ones
        1  Twos
        2  Threes
        3  Fours
        4  Fives
        5  Sixes
        6  Three of a Kind
        7  Four of a Kind
        8  Full House
        9  Small Straight
        10 Large Straight
        11 Yahtzee
        12 Chance
    """

    def __init__(self) -> None:
        self.scores: List[Optional[int]] = [None] * NUM_CATEGORIES
        self.yahtzee_bonus: int = 0

    # ============================================================
    # Validation / Normalisation
    # ============================================================

    def _validate_category(self, category_idx: int) -> None:
        """Ensure the category index is valid."""
        if not isinstance(category_idx, int):
            raise InvalidCategoryError(
                f"Category index must be an int, got {type(category_idx).__name__}."
            )

        if category_idx < 0 or category_idx >= NUM_CATEGORIES:
            raise InvalidCategoryError(
                f"Invalid category index {category_idx}. "
                f"Expected value between 0 and {NUM_CATEGORIES - 1}."
            )

    def _normalise_dice(self, dice: Iterable[int]) -> List[int]:
        """
        Validate and sort dice into ascending order.

        This is important because the environment should treat:
            [6, 1, 3, 3, 2]
        and:
            [1, 2, 3, 3, 6]
        as the same dice state.
        """

        dice_list = list(dice)

        if len(dice_list) != NUM_DICE:
            raise InvalidDiceError(
                f"Expected exactly {NUM_DICE} dice, got {len(dice_list)}."
            )

        for value in dice_list:
            if not isinstance(value, Integral):
                raise InvalidDiceError(
                    f"Dice values must be integers, got {type(value).__name__}."
                )

            if value < 1 or value > NUM_DIE_FACES:
                raise InvalidDiceError(
                    f"Invalid die value {value}. Expected values from 1 to {NUM_DIE_FACES}."
                )

        return sorted(int(value) for value in dice_list)

    # ============================================================
    # Category State
    # ============================================================

    def is_category_open(self, category_idx: int) -> bool:
        """Return True if the category has not been scored yet."""
        self._validate_category(category_idx)
        return self.scores[category_idx] is None

    def get_open(self) -> List[int]:
        """
        Return a 13-element list representing open categories.

        Returns:
            1 = category is open
            0 = category is locked / already scored
        """
        return [1 if score is None else 0 for score in self.scores]

    def is_full(self) -> bool:
        """Return True when all 13 categories have been scored."""
        return all(score is not None for score in self.scores)

    # ============================================================
    # Score Calculation
    # ============================================================

    def calculate_score(self, category_idx: int, dice: Iterable[int]) -> int:
        """
        Calculate the score for a category without modifying the scorecard.

        This function is pure game logic:
        - It sorts the dice
        - Counts face occurrences
        - Applies the Yahtzee scoring rule for the selected category
        """

        self._validate_category(category_idx)
        dice_sorted = self._normalise_dice(dice)

        counts = [dice_sorted.count(face) for face in range(1, NUM_DIE_FACES + 1)]
        total = sum(dice_sorted)
        unique = set(dice_sorted)

        category = Category(category_idx)

        # ----------------------------
        # Upper Section
        # ----------------------------
        if category == Category.ONES:
            return dice_sorted.count(1) * 1

        if category == Category.TWOS:
            return dice_sorted.count(2) * 2

        if category == Category.THREES:
            return dice_sorted.count(3) * 3

        if category == Category.FOURS:
            return dice_sorted.count(4) * 4

        if category == Category.FIVES:
            return dice_sorted.count(5) * 5

        if category == Category.SIXES:
            return dice_sorted.count(6) * 6

        # ----------------------------
        # Lower Section
        # ----------------------------
        if category == Category.THREE_OF_A_KIND:
            return total if max(counts) >= 3 else 0

        if category == Category.FOUR_OF_A_KIND:
            return total if max(counts) >= 4 else 0

        if category == Category.FULL_HOUSE:
            non_zero_counts = sorted(c for c in counts if c > 0)
            return FULL_HOUSE_SCORE_VALUE if non_zero_counts == [2, 3] else 0

        if category == Category.SMALL_STRAIGHT:
            has_small_straight = (
                {1, 2, 3, 4}.issubset(unique)
                or {2, 3, 4, 5}.issubset(unique)
                or {3, 4, 5, 6}.issubset(unique)
            )
            return SMALL_STRAIGHT_SCORE_VALUE if has_small_straight else 0

        if category == Category.LARGE_STRAIGHT:
            has_large_straight = (
                dice_sorted == [1, 2, 3, 4, 5]
                or dice_sorted == [2, 3, 4, 5, 6]
            )
            return LARGE_STRAIGHT_SCORE_VALUE if has_large_straight else 0

        if category == Category.YAHTZEE:
            return YAHTZEE_SCORE_VALUE if max(counts) == 5 else 0

        if category == Category.CHANCE:
            return total

        raise InvalidCategoryError(f"Unhandled category index: {category_idx}")

    def commit(self, category_idx: int, dice: Iterable[int]) -> int:
        """
        Score the dice into the selected category and lock that category.

        Returns:
            The score placed into that category.

        Raises:
            CategoryAlreadyFilledError if the category has already been used.
        """

        self._validate_category(category_idx)

        if not self.is_category_open(category_idx):
            category_name = CATEGORY_NAMES[category_idx]
            raise CategoryAlreadyFilledError(
                f"Cannot score {category_name}. Category is already filled."
            )

        dice_sorted = self._normalise_dice(dice)
        score = self.calculate_score(category_idx, dice_sorted)

        # Add Yahtzee bonus if eligible.
        #
        # Simplified project rule:
        # If the player has already scored 50 in Yahtzee and rolls another Yahtzee,
        # they receive +100 bonus when committing a later category.
        if self._is_extra_yahtzee(dice_sorted, category_idx):
            self.yahtzee_bonus += YAHTZEE_BONUS_VALUE

        self.scores[category_idx] = score
        return score

    def _is_extra_yahtzee(self, dice: Iterable[int], category_idx: int) -> bool:
        """
        Return True if this roll qualifies for an additional Yahtzee bonus.

        This occurs when:
        - the current dice are all the same
        - the Yahtzee category has already been scored as 50
        - the player is now scoring a different open category
        """

        dice_sorted = self._normalise_dice(dice)

        is_yahtzee_roll = len(set(dice_sorted)) == 1
        yahtzee_already_scored = self.scores[Category.YAHTZEE] == YAHTZEE_SCORE_VALUE
        scoring_different_category = category_idx != Category.YAHTZEE

        return is_yahtzee_roll and yahtzee_already_scored and scoring_different_category

    # ============================================================
    # Upper Section
    # ============================================================

    @property
    def upper_section_total(self) -> int:
        """Return upper section total before the 35-point bonus."""
        total = 0

        for category in UPPER_CATEGORIES:
            score = self.scores[category]
            if score is not None:
                total += score

        return total

    @property
    def upper_section_bonus(self) -> int:
        """Return 35 if upper section total is at least 63, otherwise 0."""
        return (
            UPPER_BONUS_VALUE
            if self.upper_section_total >= UPPER_BONUS_THRESHOLD
            else 0
        )

    @property
    def upper_section_total_with_bonus(self) -> int:
        """Return upper section total including the upper bonus."""
        return self.upper_section_total + self.upper_section_bonus

    def upper_prog(self) -> float:
        """
        Return normalized upper section progress.

        This is used in the 24-D state vector.

        Example:
            upper total = 31.5 -> 0.5
            upper total = 63   -> 1.0
            upper total = 93   -> 1.0
        """
        return min(self.upper_section_total / UPPER_BONUS_THRESHOLD, 1.0)

    # ============================================================
    # Lower Section
    # ============================================================

    @property
    def lower_section_total(self) -> int:
        """Return lower section total including Yahtzee bonuses."""
        total = 0

        for category in LOWER_CATEGORIES:
            score = self.scores[category]
            if score is not None:
                total += score

        total += self.yahtzee_bonus
        return total

    # ============================================================
    # Total / Bonus Helpers
    # ============================================================

    @property
    def grand_total(self) -> int:
        """Return total Yahtzee score including all bonuses."""
        return self.upper_section_total_with_bonus + self.lower_section_total

    def total(self) -> int:
        """Alias used by the environment."""
        return self.grand_total

    def has_yahtzee_bonus(self) -> int:
        """
        Return 1 if the scorecard is eligible for future Yahtzee bonuses.

        This is used as the final feature in the 24-D state vector.
        """
        return 1 if self.scores[Category.YAHTZEE] == YAHTZEE_SCORE_VALUE else 0

    # ============================================================
    # Display / Debugging
    # ============================================================

    def category_score(self, category_idx: int) -> Optional[int]:
        """Return the stored score for a category."""
        self._validate_category(category_idx)
        return self.scores[category_idx]

    def as_dict(self) -> dict:
        """Return the scorecard as a readable dictionary."""
        data = {}

        for idx, name in enumerate(CATEGORY_NAMES):
            data[name] = self.scores[idx]

        data["Upper Section Total"] = self.upper_section_total
        data["Upper Section Bonus"] = self.upper_section_bonus
        data["Upper Section Total With Bonus"] = self.upper_section_total_with_bonus
        data["Yahtzee Bonus"] = self.yahtzee_bonus
        data["Lower Section Total"] = self.lower_section_total
        data["Grand Total"] = self.grand_total

        return data

    def __str__(self) -> str:
        """Return a readable scorecard printout."""

        lines = []

        lines.append("Upper Section:")
        for category in UPPER_CATEGORIES:
            idx = int(category)
            score = self.scores[idx]
            lines.append(f"  {CATEGORY_NAMES[idx]}: {score}")

        lines.append(f"  Upper Total: {self.upper_section_total}")
        lines.append(f"  Upper Bonus: {self.upper_section_bonus}")
        lines.append(
            f"  Upper Total With Bonus: {self.upper_section_total_with_bonus}"
        )

        lines.append("")
        lines.append("Lower Section:")
        for category in LOWER_CATEGORIES:
            idx = int(category)
            score = self.scores[idx]
            lines.append(f"  {CATEGORY_NAMES[idx]}: {score}")

        lines.append(f"  Yahtzee Bonus: {self.yahtzee_bonus}")
        lines.append(f"  Lower Total: {self.lower_section_total}")

        lines.append("")
        lines.append(f"Grand Total: {self.grand_total}")

        return "\n".join(lines)