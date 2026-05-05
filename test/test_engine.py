"""Tests for the Logic Engine.

All tests currently skip — they describe the contracts the engine must
satisfy as it's filled in. Remove the @pytest.mark.skip decorator on each
test as the corresponding behavior is implemented.
"""

from __future__ import annotations

import pytest

# from src.engine.yahtzee_env import (
#     YahtzeeStateMachine, STATE_DIM, NUM_ACTIONS, NUM_HOLD_ACTIONS,
#     SCORE_OFFSET, MAX_ROLLS_PER_TURN,
# )
# from src.engine.scorecard import ScorecardManager, NUM_CATEGORIES


@pytest.mark.skip(reason="YahtzeeStateMachine not yet implemented.")
def test_reset_returns_24d_state():
    # env = YahtzeeStateMachine()
    # state = env.reset()
    # assert state.shape == (STATE_DIM,)
    pass


@pytest.mark.skip(reason="YahtzeeStateMachine not yet implemented.")
def test_intermediate_reward_is_zero():
    # Design doc §3.4 — reward MUST be 0 on every intermediate step.
    pass


@pytest.mark.skip(reason="YahtzeeStateMachine not yet implemented.")
def test_terminal_reward_normalized_by_100():
    # Design doc §3.4 — terminal reward = total_score / 100.
    pass


@pytest.mark.skip(reason="YahtzeeStateMachine not yet implemented.")
def test_legal_mask_blocks_filled_categories():
    # Once a category is locked, its action index (32 + cat) MUST be -inf.
    pass


@pytest.mark.skip(reason="YahtzeeStateMachine not yet implemented.")
def test_legal_mask_blocks_holds_on_roll_3():
    # On the 3rd roll, all hold actions (0..31) MUST be -inf — the agent
    # must commit to a category.
    pass


@pytest.mark.skip(reason="ScorecardManager not yet implemented.")
def test_upper_bonus_awarded_at_threshold():
    # 35 pts when upper subtotal >= 63.
    pass


@pytest.mark.skip(reason="ScorecardManager not yet implemented.")
def test_yahtzee_bonus_awards_100_per_extra():
    # First Yahtzee scores 50 in the Yahtzee cell; subsequent Yahtzees
    # score 100 bonus each (and the joker rules apply for the category
    # selection — check the design doc and the standard Yahtzee rules).
    pass
