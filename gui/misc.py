import random
import numpy as np

from environment.scorecard_manager import ScorecardManager


def generate_random_yahtzee_game():
    card = ScorecardManager()

    card.scores = [
        random.randint(0, 5) * 1,                  # Ones
        random.randint(0, 5) * 2,                  # Twos
        random.randint(0, 5) * 3,                  # Threes
        random.randint(0, 5) * 4,                  # Fours
        random.randint(0, 5) * 5,                  # Fives
        random.randint(0, 5) * 6,                  # Sixes
        random.choice([0, random.randint(5, 30)]), # Three of a kind
        random.choice([0, random.randint(5, 30)]), # Four of a kind
        random.choice([0, 25]),                    # Full house
        random.choice([0, 30]),                    # Small straight
        random.choice([0, 40]),                    # Large straight
        random.choice([0, 50]),                    # Yahtzee
        random.randint(5, 30),                     # Chance
    ]

    card.yahtzee_bonus = random.choice([0, 100])
    return card


def generate_random_q():
    raw_values = np.random.rand(45)

    max_idx = np.argmax(raw_values)

    if max_idx < 32:
        target_idx = random.randint(32, 44)
        raw_values[max_idx], raw_values[target_idx] = raw_values[target_idx], raw_values[max_idx]

    probabilities = raw_values / np.sum(raw_values)
    probabilities = np.round(probabilities, 4)

    diff = 1.0 - np.sum(probabilities)
    current_max_idx = np.argmax(probabilities)
    probabilities[current_max_idx] += diff
    probabilities = np.round(probabilities, 4)

    legal_mask = np.zeros(45, dtype=np.float32)

    for i in range(45):
        if i != current_max_idx and random.random() < 0.2:
            legal_mask[i] = -np.inf

    return probabilities, legal_mask