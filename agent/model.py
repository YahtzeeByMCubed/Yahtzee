"""
DQN model for the Yahtzee agent.

Input:
    24-dimensional Yahtzee state vector

Output:
    45 Q-values, one for each possible action

Action layout:
    0-31  = hold/reroll actions
    32-44 = score category actions
"""

from __future__ import annotations

import torch
import torch.nn as nn

from environment.constants import STATE_DIM, NUM_ACTIONS


class DQN(nn.Module):
    """
    Standard feed-forward Deep Q-Network.

    Architecture:
        24 -> 128 -> 128 -> 64 -> 45
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = NUM_ACTIONS) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state:
                Either shape (24,) or (batch_size, 24)

        Returns:
            Q-values with shape:
                (45,) if input was (24,)
                (batch_size, 45) if input was (batch_size, 24)
        """

        if state.dim() == 1:
            if state.shape[0] != self.state_dim:
                raise ValueError(
                    f"Expected state shape ({self.state_dim},), got {tuple(state.shape)}."
                )

            return self.network(state)

        if state.dim() == 2:
            if state.shape[1] != self.state_dim:
                raise ValueError(
                    f"Expected batch shape (batch_size, {self.state_dim}), "
                    f"got {tuple(state.shape)}."
                )

            return self.network(state)

        raise ValueError(
            f"Expected state tensor with 1 or 2 dimensions, got {state.dim()} dimensions."
        )