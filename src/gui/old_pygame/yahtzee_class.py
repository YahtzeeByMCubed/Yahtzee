class YahtzeeScoreGame:
    def __init__(self):
        # Upper Section
        self.ones = None
        self.twos = None
        self.threes = None
        self.fours = None
        self.fives = None
        self.sixes = None

        # Lower Section
        self.three_of_a_kind = None
        self.four_of_a_kind = None
        self.full_house = None
        self.small_straight = None
        self.large_straight = None
        self.yahtzee = None
        self.chance = None

        # Bonus
        self.yahtzee_bonus = 0

    @property
    def upper_section_total(self):
        """Calculates the total score of the upper section before bonus."""
        scores = [self.ones, self.twos, self.threes, self.fours, self.fives, self.sixes]
        return sum(s for s in scores if s is not None)

    @property
    def upper_section_bonus(self):
        """Returns 35 if the upper section total is 63 or more, else 0."""
        return 35 if self.upper_section_total >= 63 else 0

    @property
    def upper_section_total_with_bonus(self):
        """Total of upper section including the bonus."""
        return self.upper_section_total + self.upper_section_bonus

    @property
    def lower_section_total(self):
        """Calculates the total score of the lower section."""
        scores = [
            self.three_of_a_kind,
            self.four_of_a_kind,
            self.full_house,
            self.small_straight,
            self.large_straight,
            self.yahtzee,
            self.chance,
            self.yahtzee_bonus
        ]
        return sum(s for s in scores if s is not None)

    @property
    def grand_total(self):
        """Calculates the grand total of both sections."""
        return self.upper_section_total_with_bonus + self.lower_section_total

    def is_complete(self):
        """Checks if all categories on the score game have been filled."""
        categories = [
            self.ones, self.twos, self.threes, self.fours, self.fives, self.sixes,
            self.three_of_a_kind, self.four_of_a_kind, self.full_house,
            self.small_straight, self.large_straight, self.yahtzee, self.chance
        ]
        return all(c is not None for c in categories)

    def __str__(self):
        """Returns a string representation of the score game."""
        return (
            f"Upper Section:\n"
            f"  Ones: {self.ones}\n"
            f"  Twos: {self.twos}\n"
            f"  Threes: {self.threes}\n"
            f"  Fours: {self.fours}\n"
            f"  Fives: {self.fives}\n"
            f"  Sixes: {self.sixes}\n"
            f"  Upper Total: {self.upper_section_total}\n"
            f"  Upper Bonus: {self.upper_section_bonus}\n"
            f"  Upper Total (with Bonus): {self.upper_section_total_with_bonus}\n"
            f"Lower Section:\n"
            f"  3 of a Kind: {self.three_of_a_kind}\n"
            f"  4 of a Kind: {self.four_of_a_kind}\n"
            f"  Full House: {self.full_house}\n"
            f"  Small Straight: {self.small_straight}\n"
            f"  Large Straight: {self.large_straight}\n"
            f"  Yahtzee: {self.yahtzee}\n"
            f"  Chance: {self.chance}\n"
            f"  Yahtzee Bonus: {self.yahtzee_bonus}\n"
            f"  Lower Total: {self.lower_section_total}\n"
            f"Grand Total: {self.grand_total}\n"
        )
