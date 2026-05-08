import random
from yahtzee_class import YahtzeeScoreGame

def generate_random_yahtzee_game():
    game = YahtzeeScoreGame()
    game.ones = random.randint(0, 5) * 1
    game.twos = random.randint(0, 5) * 2
    game.threes = random.randint(0, 5) * 3
    game.fours = random.randint(0, 5) * 4
    game.fives = random.randint(0, 5) * 5
    game.sixes = random.randint(0, 5) * 6
    game.three_of_a_kind = random.choice([0, random.randint(5, 30)])
    game.four_of_a_kind = random.choice([0, random.randint(5, 30)])
    game.full_house = random.choice([0, 25])
    game.small_straight = random.choice([0, 30])
    game.large_straight = random.choice([0, 40])
    game.yahtzee = random.choice([0, 50])
    game.chance = random.randint(5, 30)
    game.yahtzee_bonus = random.choice([0, 100])
    return game

def generate_random_q():
    import numpy as np
    
    # Generate 45 random numbers
    raw_values = np.random.rand(45)
    
    # Ensure they are unique (highly likely with random.rand, but we can enforce)
    # by adding a tiny unique noise if needed. rand() is usually sufficient.
    
    # We want the max value to be in the range [32, 44]
    max_idx = np.argmax(raw_values)
    
    if max_idx < 32:
        # Swap the max value with a random index in the target range
        target_idx = random.randint(32, 44)
        raw_values[max_idx], raw_values[target_idx] = raw_values[target_idx], raw_values[max_idx]
        
    # Normalize so they sum to 1
    probabilities = raw_values / np.sum(raw_values)
    
    # Round to 4 decimal places
    probabilities = np.round(probabilities, 4)
    
    # The rounding might slightly break the sum = 1.0 constraint.
    # Adjust the max value to absorb the rounding error difference
    diff = 1.0 - np.sum(probabilities)
    current_max_idx = np.argmax(probabilities)
    probabilities[current_max_idx] += diff
    probabilities = np.round(probabilities, 4) # clean up floating point artifacts from addition
    
    # Generate a mock legal mask
    legal_mask = np.zeros(45, dtype=np.float32)
    # Randomly make some actions illegal (-np.inf)
    for i in range(45):
        if i != current_max_idx and random.random() < 0.2: # Keep max_idx legal
            legal_mask[i] = -np.inf
            
    return probabilities, legal_mask
    