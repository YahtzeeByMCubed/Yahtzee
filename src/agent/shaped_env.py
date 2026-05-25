"""Reward-shaping wrapper around the top-level YahtzeeEnv.

Design doc §3.4 mandates strict sparse rewards: 0 every step, only the
terminal step yields total_score/100. This is a deliberate "no greedy
trap" choice but makes credit assignment hard and plateaus the agent
around ~150 average score on this network and reasonable compute.

This wrapper is a §3.4 ablation. It replaces the per-step reward with
the **delta in scorecard.total()** caused by that step:

    shaped_reward = (total_after - total_before) / 100

Properties:
- Hold actions never change `scorecard.total()`, so they still return 0.
  The agent still has to learn dice→category strategy from the same
  signal as before for hold decisions.
- Score actions return `score_added / 100`, immediate feedback per commit.
- Bonuses (upper +35, Yahtzee +100) appear as reward spikes the moment
  their thresholds are crossed — *exactly* the credit assignment the
  sparse-reward agent struggled to learn.
- Total return over a full game equals the original sparse final reward
  (`final_total / 100`), so cross-comparison against sparse-trained
  models is well-defined.

Does NOT modify the top-level `environment/` package — composes with it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from environment.yahtzee_env import YahtzeeEnv


class ShapedYahtzeeEnv:
    """Drop-in replacement for YahtzeeEnv with per-commit reward shaping.

    Same public surface as YahtzeeEnv (reset, step, get_legal_mask,
    scorecard, dice_manager, done) so train() doesn't need to know which
    env it's holding.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.env = YahtzeeEnv(seed=seed)

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool]:
        prev_total = self.env.scorecard.total()
        state, _, done = self.env.step(action_idx)
        shaped_reward = (self.env.scorecard.total() - prev_total) / 100.0
        return state, shaped_reward, done

    def get_legal_mask(self) -> np.ndarray:
        return self.env.get_legal_mask()

    @property
    def scorecard(self):
        return self.env.scorecard

    @property
    def dice_manager(self):
        return self.env.dice_manager

    @property
    def done(self) -> bool:
        return self.env.done

    @property
    def turn(self) -> int:
        return self.env.turn

    @property
    def current_roll(self) -> int:
        return self.env.current_roll
