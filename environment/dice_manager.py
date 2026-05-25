"""
Dice manager for the Yahtzee environment.

Responsible for:
- rolling all dice
- rerolling only unheld dice
- keeping dice sorted in ascending order
- producing face counts for the 24-D state vector
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional

import numpy as np

from .constants import NUM_DICE, NUM_DIE_FACES, MAX_ROLLS_PER_TURN


class DiceManagerError(Exception):
    """Base error for dice manager issues."""


class InvalidKeepMaskError(DiceManagerError):
    """Raised when the keep mask is invalid."""


class DiceManager:
    """
    Handles dice state for Yahtzee.

    Dice are always stored in ascending order.

    Example:
        [6, 1, 3, 2, 3] becomes [1, 2, 3, 3, 6]
    """
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.dice = np.zeros(NUM_DICE, dtype=np.int32)

        # GUI-compatible fields
        self.roll_count = 0
        self.keep_mask = [False] * NUM_DICE

    def roll_all(self) -> np.ndarray:
        """
        Roll all 5 dice and store them sorted.

        Returns:
            np.ndarray of shape (5,)
        """
        self.dice = np.sort(
            self.rng.integers(1, NUM_DIE_FACES + 1, size=NUM_DICE)
        ).astype(np.int32)

        self.roll_count = 1
        self.keep_mask = [False] * NUM_DICE

        return self.dice.copy()

    def reroll(self, keep_mask: Optional[Iterable[bool]] = None) -> np.ndarray:
        """
        Reroll dice where keep_mask is False.

        This supports both:
            env.reroll(keep_mask)
        and:
            GUI button calling dice_manager.reroll()

        Args:
            keep_mask:
                True  = keep this die
                False = reroll this die
        """

        if keep_mask is None:
            mask = list(self.keep_mask)
        else:
            mask = list(keep_mask)

        if len(mask) != NUM_DICE:
            raise InvalidKeepMaskError(
                f"Expected keep mask of length {NUM_DICE}, got {len(mask)}."
            )

        for value in mask:
            if not isinstance(value, bool):
                raise InvalidKeepMaskError(
                    f"Keep mask values must be bool, got {type(value).__name__}."
                )

        # Do nothing if already at roll 3.
        # The environment legal mask should prevent this,
        # but this keeps the GUI safe too.
        if self.roll_count >= MAX_ROLLS_PER_TURN:
            return self.dice.copy()

        kept_dice = self.dice[np.array(mask, dtype=bool)]
        num_to_reroll = NUM_DICE - len(kept_dice)

        new_dice = self.rng.integers(
            1,
            NUM_DIE_FACES + 1,
            size=num_to_reroll,
        )

        final_dice = np.sort(
            np.concatenate([kept_dice, new_dice])
        ).astype(np.int32)

        self.dice = final_dice
        self.roll_count += 1

        # Rebuild keep_mask after sorting, so the GUI still highlights kept dice.
        self.keep_mask = self._build_keep_mask_after_sort(
            kept_dice=kept_dice,
            final_dice=final_dice,
        )

        return self.dice.copy()

    def set_dice_for_testing(self, dice: Iterable[int]) -> None:
        """
        Manually set dice for deterministic tests.

        This should mainly be used in unit tests.
        """

        dice_list = list(dice)

        if len(dice_list) != NUM_DICE:
            raise ValueError(f"Expected {NUM_DICE} dice, got {len(dice_list)}.")

        for value in dice_list:
            if not isinstance(value, int):
                raise ValueError("Dice values must be integers.")

            if value < 1 or value > NUM_DIE_FACES:
                raise ValueError(
                    f"Dice values must be between 1 and {NUM_DIE_FACES}."
                )

        self.dice = np.sort(np.array(dice_list, dtype=np.int32))
        self.keep_mask = [False] * NUM_DICE

    def face_counts(self) -> np.ndarray:
        """
        Return count of each dice face.

        Example:
            dice = [1, 3, 3, 4, 6]
            returns [1, 0, 2, 1, 0, 1]

        Used as the first 6 values in the 24-D state vector.
        """

        return np.array(
            [np.sum(self.dice == face) for face in range(1, NUM_DIE_FACES + 1)],
            dtype=np.float32,
        )

    def get_dice(self) -> np.ndarray:
        """Return a copy of the current dice."""
        return self.dice.copy()

    def __str__(self) -> str:
        return f"DiceManager(dice={self.dice.tolist()})"
    
    def roll(self) -> np.ndarray:
        """
        GUI-compatible alias for roll_all().
        """
        return self.roll_all()
    
    def toggle_keep(self, index: int) -> None:
        """
        GUI-compatible method.

        Toggles whether a die is kept between rolls.
        """

        if not isinstance(index, int):
            raise TypeError(f"Dice index must be int, got {type(index).__name__}.")

        if index < 0 or index >= NUM_DICE:
            raise IndexError(f"Dice index {index} is out of range.")

        if self.roll_count >= MAX_ROLLS_PER_TURN:
            return

        self.keep_mask[index] = not self.keep_mask[index]


    def get_kept_dice(self) -> List[int]:
        """
        Return dice currently marked as kept.

        Used by the GUI kept dice display.
        """

        return [
            int(value)
            for value, keep in zip(self.dice, self.keep_mask)
            if keep
        ]


    def _build_keep_mask_after_sort(self, kept_dice, final_dice) -> List[bool]:
        """
        Rebuild keep_mask after sorting the final dice.

        This lets the GUI keep highlighting the dice that were held.
        """

        remaining = Counter(int(v) for v in kept_dice)
        new_mask = []

        for value in final_dice:
            value = int(value)

            if remaining[value] > 0:
                new_mask.append(True)
                remaining[value] -= 1
            else:
                new_mask.append(False)

        return new_mask