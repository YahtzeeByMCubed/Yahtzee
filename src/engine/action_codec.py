"""
Action codec for the Yahtzee DQN environment.

The DQN has 45 discrete actions:

Actions 0-31:
    Hold/reroll combinations for 5 dice.

Actions 32-44:
    Score categories 0-12.
"""

from __future__ import annotations

from typing import List

from .constants import (
    NUM_ACTIONS,
    NUM_HOLD_ACTIONS,
    SCORE_OFFSET,
    NUM_CATEGORIES,
)


class ActionCodecError(Exception):
    """Base error for action codec issues."""


class InvalidActionError(ActionCodecError):
    """Raised when an action index is invalid."""


class ActionCodec:
    """Utility class for decoding DQN action indices."""

    @staticmethod
    def validate_action(action_idx: int) -> None:
        """Ensure action index is between 0 and 44."""
        if not isinstance(action_idx, int):
            raise InvalidActionError(
                f"Action index must be int, got {type(action_idx).__name__}."
            )

        if action_idx < 0 or action_idx >= NUM_ACTIONS:
            raise InvalidActionError(
                f"Invalid action index {action_idx}. Expected 0 to {NUM_ACTIONS - 1}."
            )

    @staticmethod
    def is_hold(action_idx: int) -> bool:
        """
        Return True if action is a hold/reroll action.

        Hold actions are 0-31.
        """
        ActionCodec.validate_action(action_idx)
        return 0 <= action_idx < NUM_HOLD_ACTIONS

    @staticmethod
    def is_score(action_idx: int) -> bool:
        """
        Return True if action is a score action.

        Score actions are 32-44.
        """
        ActionCodec.validate_action(action_idx)
        return SCORE_OFFSET <= action_idx < NUM_ACTIONS

    @staticmethod
    def decode_hold_action(action_idx: int) -> List[bool]:
        """
        Decode a hold action into a 5-value keep mask.

        Example:
            action 0:
                binary 00000
                [False, False, False, False, False]
                reroll all dice

            action 31:
                binary 11111
                [True, True, True, True, True]
                keep all dice

            action 5:
                binary 00101
                [True, False, True, False, False]
        """

        if not ActionCodec.is_hold(action_idx):
            raise InvalidActionError(
                f"Action {action_idx} is not a hold action. Expected 0-31."
            )

        return [bool((action_idx >> i) & 1) for i in range(5)]

    @staticmethod
    def decode_score_action(action_idx: int) -> int:
        """
        Decode score action into category index.

        Example:
            action 32 -> category 0, Ones
            action 40 -> category 8, Full House
            action 44 -> category 12, Chance
        """

        if not ActionCodec.is_score(action_idx):
            raise InvalidActionError(
                f"Action {action_idx} is not a score action. Expected 32-44."
            )

        category_idx = action_idx - SCORE_OFFSET

        if category_idx < 0 or category_idx >= NUM_CATEGORIES:
            raise InvalidActionError(
                f"Decoded invalid category index {category_idx}."
            )

        return category_idx