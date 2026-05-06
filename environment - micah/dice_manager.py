"""
Dice manager for the Yahtzee environment.

Responsible for:
- rolling all dice
- rerolling only unheld dice
- keeping dice sorted in ascending order
- producing face counts for the 24-D state vector
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from .constants import NUM_DICE, NUM_DIE_FACES


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

    def roll_all(self) -> np.ndarray:
        """
        Roll all 5 dice and store them sorted.

        Returns:
            np.ndarray of shape (5,)
        """
        self.dice = np.sort(
            self.rng.integers(1, NUM_DIE_FACES + 1, size=NUM_DICE)
        ).astype(np.int32)

        return self.dice.copy()

    def reroll(self, keep_mask: Iterable[bool]) -> np.ndarray:
        """
        Reroll dice where keep_mask is False.

        Args:
            keep_mask:
                Iterable of 5 booleans.
                True  = keep this die
                False = reroll this die

        Returns:
            Sorted np.ndarray of shape (5,)
        """

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

        kept_dice = self.dice[np.array(mask, dtype=bool)]
        num_to_reroll = NUM_DICE - len(kept_dice)

        new_dice = self.rng.integers(
            1,
            NUM_DIE_FACES + 1,
            size=num_to_reroll,
        )

        self.dice = np.sort(
            np.concatenate([kept_dice, new_dice])
        ).astype(np.int32)

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