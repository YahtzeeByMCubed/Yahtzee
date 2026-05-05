"""Tests for the DQN agent.

All tests currently skip — they describe the contracts the agent must
satisfy as it's filled in. Remove the @pytest.mark.skip decorator on
each test as the corresponding behavior is implemented.
"""

from __future__ import annotations

import pytest

# import numpy as np
# import torch
# from src.agent.model import DQN, STATE_DIM, ACTION_DIM
# from src.agent.dqn_agent import DQNAgent, select_legal_action


@pytest.mark.skip(reason="DQN not yet implemented.")
def test_dqn_output_shape():
    # net = DQN()
    # out = net(torch.zeros(1, STATE_DIM))
    # assert out.shape == (1, ACTION_DIM)
    pass


@pytest.mark.skip(reason="DQN not yet implemented.")
def test_inference_under_10ms():
    # Design doc §2.2 — inference budget is 10ms on the deployment host.
    # Warm up first (first forward pass JITs CUDA kernels / allocates
    # workspace), then measure 100 calls and assert mean < 10ms.
    pass


@pytest.mark.skip(reason="Action masking not yet implemented.")
def test_argmax_skips_illegal_actions():
    # q = np.zeros(45); q[5] = 100.0          # would be greedy
    # mask = np.zeros(45); mask[5] = -np.inf  # ...but it's illegal
    # q[7] = 50.0                              # next-best legal
    # assert select_legal_action(q, mask) == 7
    pass


@pytest.mark.skip(reason="DQNAgent not yet implemented.")
def test_agent_never_returns_illegal_action():
    # Property test: across many random states, agent.get_action(state, mask)
    # returns an index where mask[idx] == 0.0 (i.e. legal). This is the
    # "0% illegal moves" guarantee from design doc §2.2.
    pass


@pytest.mark.skip(reason="PrioritizedReplayBuffer not yet implemented.")
def test_per_oversamples_high_td_error_transitions():
    # Push a mix of low- and high-TD-error transitions, draw many samples,
    # and assert the high-TD ones appear disproportionately often.
    pass


@pytest.mark.skip(reason="DQNAgent not yet implemented.")
def test_save_and_load_round_trip(tmp_path):
    # Save weights, instantiate a new agent, load weights, and assert
    # state dicts are equal.
    pass
