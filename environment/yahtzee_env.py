"""
Yahtzee environment orchestrator for the DQN agent.

This class connects:
- DiceManager
- ScorecardManager
- ActionCodec

It provides the DQN-facing interface:
- reset()
- step(action_idx)
- get_legal_mask()
- construct_state_vector()

The environment uses a sparse reward strategy:
- reward = 0.0 during the game
- reward = final_score / 100.0 when the scorecard is complete
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .action_codec import ActionCodec, InvalidActionError
from .constants import (
    STATE_DIM,
    NUM_ACTIONS,
    NUM_HOLD_ACTIONS,
    SCORE_OFFSET,
    MAX_ROLLS_PER_TURN,
    TURNS_PER_GAME,
)
from .dice_manager import DiceManager
from .scorecard_manager import ScorecardManager


class YahtzeeEnvError(Exception):
    """Base error for Yahtzee environment issues."""


class IllegalActionError(YahtzeeEnvError):
    """Raised when the agent tries to take an illegal action."""


class GameOverError(YahtzeeEnvError):
    """Raised when step() is called after the game is already complete."""


def sparse_terminal_reward(total_score: int) -> float:
    """
    Convert final Yahtzee score into normalized sparse reward.

    Example:
        total_score = 250
        reward = 2.5
    """
    return total_score / 100.0


class YahtzeeEnv:
    """
    Main Yahtzee environment.

    Responsibilities:
        1. Manage turn number
        2. Manage current roll number
        3. Hold/reroll dice
        4. Score categories
        5. Produce legal action mask
        6. Produce 24-D state vector
        7. Return sparse terminal reward
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

        self.dice_manager = DiceManager(seed=seed)
        self.scorecard = ScorecardManager()

        self.current_roll = 1
        self.turn = 1
        self.done = False

    # ============================================================
    # Core DQN Interface
    # ============================================================

    def reset(self) -> np.ndarray:
        """
        Reset the environment to a new game.

        Returns:
            Initial 24-D state vector.
        """

        self.dice_manager = DiceManager(seed=self.seed)
        self.scorecard = ScorecardManager()

        self.current_roll = 1
        self.turn = 1
        self.done = False

        self.dice_manager.roll_all()

        return self.construct_state_vector()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one DQN action.

        Args:
            action_idx:
                Integer from 0 to 44.

                0-31:
                    Hold/reroll actions.

                32-44:
                    Score category actions.

        Returns:
            next_state, reward, done
        """

        if self.done:
            raise GameOverError("Cannot call step() because the game is already complete.")

        try:
            ActionCodec.validate_action(action_idx)
        except InvalidActionError as exc:
            raise IllegalActionError(str(exc)) from exc

        legal_mask = self.get_legal_mask()

        if np.isneginf(legal_mask[action_idx]):
            raise IllegalActionError(
                f"Action {action_idx} is illegal at turn {self.turn}, roll {self.current_roll}."
            )

        if ActionCodec.is_hold(action_idx):
            return self._step_hold(action_idx)

        if ActionCodec.is_score(action_idx):
            return self._step_score(action_idx)

        raise IllegalActionError(f"Unhandled action index: {action_idx}")

    # ============================================================
    # Action Handling
    # ============================================================

    def _step_hold(self, action_idx: int) -> Tuple[np.ndarray, float, bool]:
        """
        Handle actions 0-31.

        These actions keep some dice and reroll the rest.
        """

        keep_mask = ActionCodec.decode_hold_action(action_idx)

        self.dice_manager.reroll(keep_mask)
        self.current_roll = self.dice_manager.roll_count

        reward = 0.0
        done = False

        return self.construct_state_vector(), reward, done

    def _step_score(self, action_idx: int) -> Tuple[np.ndarray, float, bool]:
        """
        Handle actions 32-44.

        These actions commit the current dice to a scorecard category,
        end the current turn, and either:
        - start the next turn
        - or finish the game
        """

        category_idx = ActionCodec.decode_score_action(action_idx)
        current_dice = self.dice_manager.get_dice()

        self.scorecard.commit(category_idx, current_dice)

        if self.scorecard.is_full():
            self.done = True
            reward = sparse_terminal_reward(self.scorecard.total())

            terminal_state = np.zeros(STATE_DIM, dtype=np.float32)
            return terminal_state, reward, True

        self.turn += 1
        self.current_roll = 1

        if self.turn > TURNS_PER_GAME:
            raise YahtzeeEnvError(
                f"Turn counter exceeded {TURNS_PER_GAME}. "
                "This should not happen if the scorecard has 13 categories."
            )

        self.dice_manager.roll_all()
        self.current_roll = self.dice_manager.roll_count

        reward = 0.0
        done = False

        return self.construct_state_vector(), reward, done

    # ============================================================
    # Legal Action Mask
    # ============================================================

    def get_legal_mask(self) -> np.ndarray:
        """
        Return a 45-D legal action mask.

        Values:
            0.0     = legal action
            -np.inf = illegal action

        Rules:
            - Hold actions 0-31 are legal on roll 1 and roll 2.
            - Hold actions are illegal on roll 3.
            - Score actions 32-44 are legal only if the category is open.
            - Filled score categories are illegal.
        """

        mask = np.full(NUM_ACTIONS, -np.inf, dtype=np.float32)

        if self.done:
            return mask

        # Hold/reroll actions are legal before roll 3.
        if self.current_roll < MAX_ROLLS_PER_TURN:
            mask[0:NUM_HOLD_ACTIONS] = 0.0

        # Score actions are legal if the scorecard category is open.
        open_categories = self.scorecard.get_open()

        for category_idx, is_open in enumerate(open_categories):
            if is_open == 1:
                action_idx = SCORE_OFFSET + category_idx
                mask[action_idx] = 0.0

        return mask

    # ============================================================
    # State Vector
    # ============================================================

    def construct_state_vector(self) -> np.ndarray:
        """
        Build the 24-D state vector used by the DQN.

        Layout:
            [0:6]   dice face counts for faces 1-6
            [6:9]   current roll one-hot encoding
            [9:22]  open scorecard categories
            [22]    upper section progress
            [23]    Yahtzee bonus eligibility
        """

        face_counts = self.dice_manager.face_counts()

        roll_one_hot = np.zeros(MAX_ROLLS_PER_TURN, dtype=np.float32)
        roll_one_hot[self.current_roll - 1] = 1.0

        open_categories = np.array(self.scorecard.get_open(), dtype=np.float32)

        upper_progress = np.array(
            [self.scorecard.upper_prog()],
            dtype=np.float32,
        )

        yahtzee_bonus_flag = np.array(
            [self.scorecard.has_yahtzee_bonus()],
            dtype=np.float32,
        )

        state = np.concatenate(
            [
                face_counts,
                roll_one_hot,
                open_categories,
                upper_progress,
                yahtzee_bonus_flag,
            ]
        ).astype(np.float32)

        if state.shape != (STATE_DIM,):
            raise YahtzeeEnvError(
                f"State vector has shape {state.shape}, expected ({STATE_DIM},)."
            )

        return state

    # ============================================================
    # Testing / Debug Helpers
    # ============================================================

    def set_dice_for_testing(self, dice) -> None:
        """
        Manually set dice for deterministic environment tests.

        This should not be used by the DQN during training.
        """
        self.dice_manager.set_dice_for_testing(dice)

    def get_dice(self) -> np.ndarray:
        """Return a copy of the current dice."""
        return self.dice_manager.get_dice()

    def __str__(self) -> str:
        return (
            f"YahtzeeEnv("
            f"turn={self.turn}, "
            f"current_roll={self.current_roll}, "
            f"done={self.done}, "
            f"dice={self.dice_manager.get_dice().tolist()}"
            f")"
        )