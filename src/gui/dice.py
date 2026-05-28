"""Stateful Dice manager for Yahtzee.

This class manages the current dice values, the roll count (1-3), and the 
keep mask. Because the environment always sorts dice in ascending order, 
the position of the dice (0-4) corresponds semantically to their rank, 
allowing the AI's 5-bit hold action to reliably map to the correct dice.
"""

from __future__ import annotations

import numpy as np

DICE_PER_ROLL = 5
DIE_FACES = 6


class Dice:
    """Stores the values of the dice and manages roll states."""
    
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.values = np.zeros(DICE_PER_ROLL, dtype=int)
        self.keep_mask = [False] * DICE_PER_ROLL
        self.roll_count = 0

    def reset(self) -> None:
        """Resets the dice state for a new turn."""
        self.values = np.zeros(DICE_PER_ROLL, dtype=int)
        self.keep_mask = [False] * DICE_PER_ROLL
        self.roll_count = 0

    def roll(self) -> np.ndarray:
        """First roll of a turn. Rolls all 5 dice, sorts them, increments roll count."""
        self.values = self.rng.integers(1, DIE_FACES + 1, size=DICE_PER_ROLL)
        self.values.sort()
        self.keep_mask = [False] * DICE_PER_ROLL
        self.roll_count = 1
        return self.values.copy()

    def reroll(self, keep_mask: list[bool] | None = None) -> np.ndarray:
        """Rerolls unkept dice, sorts them, increments roll count.
        
        Args:
            keep_mask: Optional list of 5 booleans. If provided, overrides 
                       the internal keep_mask. True means keep, False means reroll.
        """
        if self.roll_count == 0:
            return self.roll()
            
        if self.roll_count >= 3:
            raise ValueError("Max rolls (3) reached for this turn.")
        
        if keep_mask is not None:
            if len(keep_mask) != DICE_PER_ROLL:
                raise ValueError(f"keep_mask must have exactly {DICE_PER_ROLL} elements")
            self.keep_mask = keep_mask
            
        kept_values = [self.values[i] for i in range(DICE_PER_ROLL) if self.keep_mask[i]]
        num_to_roll = DICE_PER_ROLL - len(kept_values)
        
        if num_to_roll > 0:
            new_rolls = self.rng.integers(1, DIE_FACES + 1, size=num_to_roll).tolist()
            self.values = np.array(kept_values + new_rolls, dtype=int)
            self.values.sort()
            
        # Reset the mask for the new roll
        self.keep_mask = [False] * DICE_PER_ROLL
        self.roll_count += 1
        
        return self.values.copy()

    def face_counts(self) -> np.ndarray:
        """Returns shape (DIE_FACES,) int array — count of dice showing each face 1..6.
        Used directly by the state vector [0:6] slice.
        """
        counts = np.zeros(DIE_FACES, dtype=int)
        for val in self.values:
            if 1 <= val <= DIE_FACES:
                counts[val - 1] += 1
        return counts

    def toggle_keep(self, index: int) -> None:
        """Toggles the keep status of a die at the given index (GUI helper)."""
        if 0 <= index < DICE_PER_ROLL:
            self.keep_mask[index] = not self.keep_mask[index]
            
    def get_kept_dice(self) -> list[int]:
        """Returns a list of the values currently marked to be kept (GUI helper)."""
        return [int(self.values[i]) for i in range(DICE_PER_ROLL) if self.keep_mask[i]]
