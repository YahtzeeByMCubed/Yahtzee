"""Perception layer — STUBBED for the brain-in-a-vat phase.

In the real system this module:
    - Reads RGB-D frames from an Intel RealSense D435if (pyrealsense2).
    - Runs a YOLOv8 model trained on a touching-dice dataset (ultralytics).
    - Applies a static hand-eye transform to map detections from image
      space into the UR3e base frame.
    - Enforces the 5-dice sanity check (design doc §4.2) and surfaces an
      error to the dashboard if the detected count != 5.

Until that layer is built, the simulator path injects a SimVisionNode
that returns the dice values stored on the YahtzeeStateMachine — i.e. the
"vision" reads ground truth straight from the simulator.

Both real and sim implementations satisfy the VisionNode Protocol below
so YahtzeeStateMachine can stay agnostic to which one it's holding.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VisionSanityError(RuntimeError):
    """Raised when the camera cannot see exactly 5 dice clearly.

    Per design doc §4.2, the State Machine refuses to construct a state
    vector when this fires; the dashboard surfaces the error to the human.
    """


class VisionNode(Protocol):
    """Interface contract for any vision provider (real or simulated)."""

    def get_dice_faces(self) -> np.ndarray:
        # Return shape (5,) int array of face values 1..6.
        # MUST raise VisionSanityError if exactly 5 dice are not visible.
        ...

    def get_dice_poses(self) -> list:
        # Return 5 poses in the UR3e base frame. Used only by the robot
        # wrapper, not by the DQN.
        ...


# -- real implementation: STUB --------------------------------------------

class RealSenseYoloVisionNode:
    """Real perception node. STUB — to be filled in for hardware bring-up.

    TODO:
        - Open the RealSense pipeline at startup (RGB + depth, aligned).
        - Load YOLOv8 weights from models/yolov8_dice.pt.
        - On get_dice_faces():
            1. Pull aligned RGB frame.
            2. Run YOLO inference, filter by confidence threshold.
            3. If detections != 5, raise VisionSanityError.
            4. Sort detections left-to-right (or by depth) for stable
               die-index assignment across frames.
            5. Return the 5 face-class IDs.
        - On get_dice_poses():
            1. Project each detection's center pixel into the depth frame.
            2. Apply the static hand-eye transform (loaded once at init).
            3. Return 5 poses in the UR3e base frame.
    """

    def __init__(self, weights_path: str = "models/yolov8_dice.pt") -> None:
        raise NotImplementedError

    def get_dice_faces(self) -> np.ndarray:
        raise NotImplementedError

    def get_dice_poses(self) -> list:
        raise NotImplementedError


# -- simulator fake: STUB -------------------------------------------------

class SimVisionNode:
    """In-memory fake used during brain-in-a-vat training.

    Reads dice values straight off a reference to the simulator's internal
    state — there's no image processing, no noise, no occlusion. This is
    deliberate: training in sim is about learning Yahtzee strategy, not
    learning vision robustness.

    TODO:
        - Hold a reference (or callable) that returns the simulator's
          current 5-die array.
        - get_dice_faces() returns that array unchanged.
        - get_dice_poses() returns dummy poses or raises NotImplementedError
          (the SimRobotClient ignores poses anyway).
    """

    def __init__(self, dice_source) -> None:
        # `dice_source` is a callable () -> np.ndarray of shape (5,).
        raise NotImplementedError

    def get_dice_faces(self) -> np.ndarray:
        raise NotImplementedError

    def get_dice_poses(self) -> list:
        raise NotImplementedError
