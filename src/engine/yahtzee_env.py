"""Core state machine — the Logic Engine's main orchestrator.

Per design doc §2.1 and §3.5.3, the YahtzeeStateMachine class:
  - Drives the 13-turn game cycle (each turn = up to 3 rolls).
  - Validates every move against the rules (rule integrity = 100%).
  - Builds the 24-D state vector consumed by the DQN.
  - Computes the strict sparse reward (0 every step, terminal_score/100 on
    turn 13).

The constructor accepts a vision node and a robot client behind abstract
interfaces (see src/perception, src/robotics) so the simulator can inject
in-memory fakes during training (the "brain-in-a-vat" mode) and the demo
can swap in real hardware without touching this file.

This module also owns the action-space encoding (45 discrete actions)
because action decoding is fundamentally a property of the environment,
not the agent.
"""

from __future__ import annotations

import numpy as np

# from src.engine.scorecard import ScorecardManager
# from src.engine.dice import roll, face_counts
# from src.perception.vision_node import VisionNode
# from src.robotics.execution_wrapper import RobotClient

# -- dimensions ------------------------------------------------------------

STATE_DIM = 24
NUM_ACTIONS = 45
NUM_HOLD_ACTIONS = 32         # indices 0..31
NUM_SCORE_ACTIONS = 13        # indices 32..44
SCORE_OFFSET = 32             # action_idx -= SCORE_OFFSET to get category 0..12

MAX_ROLLS_PER_TURN = 3
TURNS_PER_GAME = 13
DICE_PER_ROLL = 5

REWARD_NORMALIZER = 100.0     # design doc §3.4 — terminal_score / 100


# -- action-space helpers --------------------------------------------------

def is_hold(action_idx: int) -> bool:
    # TODO: action_idx in [0, NUM_HOLD_ACTIONS)
    raise NotImplementedError


def is_score(action_idx: int) -> bool:
    # TODO: action_idx in [SCORE_OFFSET, NUM_ACTIONS)
    raise NotImplementedError


def decode_hold_action(action_idx: int) -> list[bool]:
    # TODO: return 5-element list, True == keep this die index. Bit i of
    # action_idx (i in [0, 4]) maps to dice slot i. Bit set = keep, bit
    # unset = reroll. So action_idx=0 rerolls all five; action_idx=31
    # keeps all five (only useful before roll 3).
    raise NotImplementedError


def decode_score_action(action_idx: int) -> int:
    # TODO: return ScorecardManager category index 0..12.
    raise NotImplementedError


# -- reward ----------------------------------------------------------------

def sparse_terminal_reward(total_score: int) -> float:
    # TODO: per design doc §3.4, return total_score / REWARD_NORMALIZER.
    # Used only when done=True; intermediate steps must return 0.0.
    raise NotImplementedError


# -- environment -----------------------------------------------------------

class YahtzeeStateMachine:
    """Orchestrates one Yahtzee game from reset to terminal reward.

    Public API (matches design doc §2.4):
        reset() -> np.ndarray                         # initial 24-D state
        step(action_idx: int) -> (next_state, reward, done)
        get_legal_mask() -> np.ndarray                # 45-D, 0.0 or -inf
    """

    def __init__(self, vision_node=None, robot_client=None, seed: int | None = None) -> None:
        # TODO: store injected dependencies; default to in-memory simulator
        # fakes when None (brain-in-a-vat path). The fakes should live in
        # src/perception/vision_node.py (SimVisionNode) and
        # src/robotics/execution_wrapper.py (SimRobotClient).
        # TODO: instantiate ScorecardManager.
        # TODO: initialize roll counter (1..MAX_ROLLS_PER_TURN) and current
        # dice array (size DICE_PER_ROLL).
        # TODO: initialize an np.random.Generator from `seed` so training
        # runs are reproducible.
        raise NotImplementedError

    # -- public lifecycle ------------------------------------------------

    def reset(self) -> np.ndarray:
        # TODO: clear scorecard, reset roll counter to 1, roll all 5 dice
        # via the robot/vision (or sim fakes), return construct_state_vector().
        raise NotImplementedError

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool]:
        # TODO:
        # 1. Validate action_idx against get_legal_mask() (defensive — the
        #    DQN-side mask should already enforce this, but trust nothing).
        # 2. Dispatch to robot_client.execute_physical_move(action_idx) and
        #    raise on robot_status.success == False.
        # 3. Update internal phase via _update_game_phase(action_idx):
        #      hold (0..31): reroll non-kept dice, increment roll count.
        #      score (32..44): commit category, reset roll count to 1, roll
        #          all 5 dice for the next turn (unless the game is over).
        # 4. Compute next_state via construct_state_vector().
        # 5. done = scorecard.is_full().
        # 6. reward = sparse_terminal_reward(scorecard.total()) if done else 0.0.
        raise NotImplementedError

    # -- state vector construction --------------------------------------

    def construct_state_vector(self) -> np.ndarray:
        # TODO: assemble the 24-D vector per design doc §3.1:
        #   [0:6]   — count of each face value 1..6 currently on the table.
        #   [6:9]   — one-hot of current roll (Roll 1, Roll 2, or Roll 3).
        #   [9:22]  — 13 binary flags: 1 if scorecard category still open.
        #   [22]    — upper-section progress, normalized: upper_score / 63.
        #   [23]    — Yahtzee-bonus eligibility flag (0/1).
        # Sanity-check (§4.2): if vision returns != 5 dice, raise and let
        # the GUI surface the error rather than fabricating a state.
        raise NotImplementedError

    def get_legal_mask(self) -> np.ndarray:
        # TODO: return a 45-D float array where legal action indices are 0.0
        # and illegal indices are -np.inf. Rules:
        #   - On roll MAX_ROLLS_PER_TURN, all hold actions (0..31) are illegal —
        #     the agent MUST score.
        #   - Score categories whose cell is already filled are illegal.
        #   - On rolls 1 and 2, both holding and scoring are legal.
        raise NotImplementedError

    # -- internal helpers -----------------------------------------------

    def _update_game_phase(self, action_idx: int) -> None:
        # TODO: state transitions for hold vs. score actions.
        raise NotImplementedError
