"""
Shared constants for the Yahtzee DQN environment.

The environment uses:
- 13 Yahtzee score categories
- 32 hold/reroll actions
- 13 score actions
- 45 total actions
- 24-dimensional state vector
"""

from enum import IntEnum


# ============================================================
# DQN / Environment Dimensions
# ============================================================

STATE_DIM = 24

NUM_DICE = 5
NUM_DIE_FACES = 6

NUM_CATEGORIES = 13

NUM_HOLD_ACTIONS = 32          # Actions 0-31
NUM_SCORE_ACTIONS = 13         # Actions 32-44
NUM_ACTIONS = 45               # 32 hold actions + 13 score actions

SCORE_OFFSET = 32              # Score action index = 32 + category index


# ============================================================
# Game Rules
# ============================================================

MAX_ROLLS_PER_TURN = 3
TURNS_PER_GAME = 13

UPPER_BONUS_THRESHOLD = 63
UPPER_BONUS_VALUE = 35

YAHTZEE_SCORE_VALUE = 50
YAHTZEE_BONUS_VALUE = 100

FULL_HOUSE_SCORE_VALUE = 25
SMALL_STRAIGHT_SCORE_VALUE = 30
LARGE_STRAIGHT_SCORE_VALUE = 40


# ============================================================
# Category Indices
# ============================================================

class Category(IntEnum):
    """
    Canonical Yahtzee category indices.

    These indices are used everywhere:
    - scorecard.scores[category]
    - score action decoding
    - legal action masking
    """

    ONES = 0
    TWOS = 1
    THREES = 2
    FOURS = 3
    FIVES = 4
    SIXES = 5

    THREE_OF_A_KIND = 6
    FOUR_OF_A_KIND = 7
    FULL_HOUSE = 8
    SMALL_STRAIGHT = 9
    LARGE_STRAIGHT = 10
    YAHTZEE = 11
    CHANCE = 12


CATEGORY_NAMES = [
    "Ones",
    "Twos",
    "Threes",
    "Fours",
    "Fives",
    "Sixes",
    "Three of a Kind",
    "Four of a Kind",
    "Full House",
    "Small Straight",
    "Large Straight",
    "Yahtzee",
    "Chance",
]


UPPER_CATEGORIES = (
    Category.ONES,
    Category.TWOS,
    Category.THREES,
    Category.FOURS,
    Category.FIVES,
    Category.SIXES,
)

LOWER_CATEGORIES = (
    Category.THREE_OF_A_KIND,
    Category.FOUR_OF_A_KIND,
    Category.FULL_HOUSE,
    Category.SMALL_STRAIGHT,
    Category.LARGE_STRAIGHT,
    Category.YAHTZEE,
    Category.CHANCE,
)