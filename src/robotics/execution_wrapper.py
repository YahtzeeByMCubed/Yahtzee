"""Robot execution wrapper — STUBBED for the brain-in-a-vat phase.

This module is the bridge between an action_idx (0..44) chosen by the DQN
and the physical UR3e motion that realizes it. In the real system this
class is an `rclpy.action.ActionClient` that talks to the
PlayYahtzee.action server (see src/robotics/action_server.py).

Until that layer is built, the simulator path injects a SimRobotClient
that simply rolls the dice in software and returns success.

Both real and sim implementations satisfy the RobotClient Protocol below
so YahtzeeStateMachine can stay agnostic to which one it's holding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RobotResult:
    success: bool
    final_poses: list = field(default_factory=list)  # poses of kept dice in base frame
    error_message: str = ""


class RobotClient(Protocol):
    """Interface contract — both real and sim clients implement this."""

    def execute_physical_move(self, action_index: int) -> RobotResult:
        ...

    def cancel_goal(self) -> None:
        # E-Stop hook (design doc §6).
        ...


# -- real implementation: STUB --------------------------------------------

class Ur3eRobotClient:
    """Real ROS 2 implementation. STUB — to be filled in for hardware bring-up.

    Decode logic (design doc §3.5.4):
        action_idx < 32:  hold combination chosen
            - decode_hold_action(idx) -> 5-bool keep mask
            - move kept dice to the safe zone outside the camera FOV
            - sweep remaining dice into the cup, re-roll
        action_idx >= 32: score category chosen
            - return all 5 dice to the start area
            - block until "Confirm Human Move" is pressed on the GUI

    Safety (design doc §6):
        - The UR3e force-torque limit is configured to trip a protective
          stop above 15 N. We do not catch the protective stop here; we
          surface it as RobotResult(success=False, error_message=...).
        - cancel_goal() must propagate to the underlying action client so
          the software E-Stop is responsive.
    """

    def __init__(self) -> None:
        # TODO:
        #   - rclpy.init().
        #   - Create a Node and an ActionClient<PlayYahtzee>.
        #   - Wait for the server to be ready (with a timeout).
        raise NotImplementedError

    def execute_physical_move(self, action_index: int) -> RobotResult:
        # TODO:
        #   - Build PlayYahtzee.Goal(action_index=action_index).
        #   - send_goal_async, await result.
        #   - Translate ROS Result -> RobotResult dataclass.
        raise NotImplementedError

    def cancel_goal(self) -> None:
        # TODO: cancel the in-flight goal handle (if any). Idempotent.
        raise NotImplementedError


# -- simulator fake: STUB -------------------------------------------------

class SimRobotClient:
    """In-memory fake used during brain-in-a-vat training.

    Performs the dice manipulation in software:
        - hold action: keep the bits set in decode_hold_action(idx),
          reroll the rest using the env's RNG.
        - score action: just signal success — the env's _update_game_phase
          does the actual scorecard mutation.
    Returns RobotResult(success=True) unconditionally; there's no physics.

    TODO:
        - Hold a reference (or callable) to the env's RNG and dice array.
        - Implement execute_physical_move per the rules above.
        - cancel_goal() is a no-op.
    """

    def __init__(self, dice_ref, rng) -> None:
        raise NotImplementedError

    def execute_physical_move(self, action_index: int) -> RobotResult:
        raise NotImplementedError

    def cancel_goal(self) -> None:
        raise NotImplementedError
