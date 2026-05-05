"""Dice utilities — rolling and face counting.

The canonical 32-element keep/reroll mapping for hold actions
(indices 0..31) lives in src/engine/yahtzee_env.py
(decode_hold_action) — that's the bridge between the integer action
index and the 5-element keep/reroll mask.
"""

from __future__ import annotations

import numpy as np

DICE_PER_ROLL = 5
DIE_FACES = 6


def roll(rng: np.random.Generator, n: int = DICE_PER_ROLL) -> np.ndarray:
    # TODO: return shape (n,) int array of values uniformly drawn from
    # [1, DIE_FACES].
    raise NotImplementedError


def reroll(rng: np.random.Generator, current: np.ndarray, keep_mask: list[bool]) -> np.ndarray:
    # TODO: take the current 5-die array and a 5-element keep mask, return
    # a new 5-die array where indices with keep_mask[i] == True are
    # preserved and the rest are freshly rolled.
    raise NotImplementedError


def face_counts(dice: np.ndarray) -> np.ndarray:
    # TODO: return shape (DIE_FACES,) int array — count of dice showing
    # each face value 1..6. Used directly by the state vector [0:6] slice.
    raise NotImplementedError
