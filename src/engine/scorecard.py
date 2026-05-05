"""ScorecardManager — tracks the 13 Yahtzee categories, the upper-section
bonus, and Yahtzee-bonus eligibility.

Canonical category indices used everywhere in this codebase:
    0..5   Upper:  Ones, Twos, Threes, Fours, Fives, Sixes
    6      Three of a Kind
    7      Four of a Kind
    8      Full House                  (25 pts)
    9      Small Straight              (30 pts)
    10     Large Straight              (40 pts)
    11     Yahtzee                     (50 pts, +100 bonus per extra)
    12     Chance
"""

from __future__ import annotations

NUM_CATEGORIES = 13
UPPER_BONUS_THRESHOLD = 63
UPPER_BONUS_VALUE = 35
YAHTZEE_BONUS_VALUE = 100


class ScorecardManager:
    def __init__(self) -> None:
        # TODO: per-category cells (None = open, int = locked score).
        # TODO: yahtzee_bonus_eligible flag, set True once the first Yahtzee
        # has been scored in the Yahtzee cell.
        raise NotImplementedError

    # -- queries used by the state vector -------------------------------

    def get_open(self) -> list[int]:
        # TODO: return 13-element list, 1 if open else 0. Order = canonical
        # index above.
        raise NotImplementedError

    def upper_prog(self) -> float:
        # TODO: sum(upper cells filled) / UPPER_BONUS_THRESHOLD, clamped to <= 1.0.
        raise NotImplementedError

    def has_bonus(self) -> int:
        # TODO: 1 if eligible for the +100 Yahtzee joker bonus, else 0.
        raise NotImplementedError

    def is_full(self) -> bool:
        # TODO: True iff all 13 categories are locked.
        raise NotImplementedError

    def total(self) -> int:
        # TODO: sum of all locked scores
        #     + UPPER_BONUS_VALUE if upper subtotal >= UPPER_BONUS_THRESHOLD
        #     + YAHTZEE_BONUS_VALUE * (extra Yahtzees rolled).
        raise NotImplementedError

    # -- mutation -------------------------------------------------------

    def commit(self, category_idx: int, dice: list[int]) -> int:
        # TODO: scoring rules per category. Raises ValueError if the
        # category is already locked. Returns the points awarded — note
        # that this return value is for GUI display only; reward signal
        # remains strictly sparse and is computed from total() at game end.
        raise NotImplementedError
