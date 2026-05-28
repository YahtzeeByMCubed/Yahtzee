"""
Legal action selection for the Yahtzee DQN.

The DQN always outputs 45 Q-values, but not every action is legal.
The environment provides a 45-D legal mask:

    0.0     = legal action
    -inf    = illegal action

The selected action is:

    argmax(q_values + legal_mask)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.engine.constants import NUM_ACTIONS


class ActionSelectionError(Exception):
    """Base error for action selection issues."""


class NoLegalActionsError(ActionSelectionError):
    """Raised when the legal mask contains no legal actions."""


def _to_numpy_1d(values, name: str) -> np.ndarray:
    """
    Convert a tensor/list/array into a 1-D NumPy array.
    """

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    arr = np.asarray(values, dtype=np.float32)

    if arr.shape != (NUM_ACTIONS,):
        raise ValueError(
            f"{name} must have shape ({NUM_ACTIONS},), got {arr.shape}."
        )

    return arr


def select_legal_action(q_values, legal_mask) -> int:
    """
    Select the highest-valued legal action.

    Args:
        q_values:
            45 raw Q-values from the DQN.

        legal_mask:
            45 values from the environment.
            Legal actions are 0.0.
            Illegal actions are -inf.

    Returns:
        Best legal action index from 0 to 44.
    """

    q = _to_numpy_1d(q_values, "q_values")
    mask = _to_numpy_1d(legal_mask, "legal_mask")

    legal_indices = np.flatnonzero(mask == 0.0)

    if len(legal_indices) == 0:
        raise NoLegalActionsError("No legal actions are available.")

    masked_q_values = q + mask

    return int(np.argmax(masked_q_values))


def select_epsilon_greedy_action(
    q_values,
    legal_mask,
    epsilon: float,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Select an action using epsilon-greedy exploration.

    Behaviour:
        - With probability epsilon, choose a random legal action.
        - Otherwise, choose the best legal action.

    Args:
        q_values:
            45 raw Q-values from the DQN.

        legal_mask:
            45-D legal action mask.

        epsilon:
            Exploration probability between 0.0 and 1.0.

        rng:
            Optional NumPy random generator for deterministic tests.
    """

    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError(f"epsilon must be between 0.0 and 1.0, got {epsilon}.")

    q = _to_numpy_1d(q_values, "q_values")
    mask = _to_numpy_1d(legal_mask, "legal_mask")

    legal_indices = np.flatnonzero(mask == 0.0)

    if len(legal_indices) == 0:
        raise NoLegalActionsError("No legal actions are available.")

    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < epsilon:
        return int(rng.choice(legal_indices))

    return select_legal_action(q, mask)