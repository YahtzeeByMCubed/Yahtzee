"""The PyTorch DQN architecture — design doc §3.5.1.

Standard MLP: 24-D state -> 128 -> 128 -> 64 -> 45 Q-values.
ReLU activations on hidden layers; no activation on the output (raw Q-values).
"""

from __future__ import annotations

import torch
import torch.nn as nn

STATE_DIM = 24
ACTION_DIM = 45


class DQN(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
