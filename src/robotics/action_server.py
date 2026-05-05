"""ROS 2 PlayYahtzee.action server — STUBBED for later hardware bring-up.

The action server runs on the robot host. It owns trajectory generation
and gripper control; the rest of the system never touches MoveIt2 directly.

Action interface (design doc §5.2):
    Goal:
        int32 action_index            # 0..44, decoded server-side
    Feedback:
        string current_op             # e.g. "Picking die 2 of 3"
        int32  dice_moved             # cumulative
    Result:
        bool success
        geometry_msgs/Pose[] final_poses
        string error_message

This file is intentionally a thin Python stub. The real server will
likely live inside a sibling ROS 2 package built with colcon — i.e.
`yahtzee_robot/` containing `package.xml`, `setup.py`, an `action/`
directory with `PlayYahtzee.action`, and a launch file. When that happens,
the contents of this file should move into that package's
`yahtzee_robot/play_yahtzee_server.py` and this stub can be deleted.

TODO when bringing up the hardware layer:
    - Decide whether action_server.py stays in this Python package or
      moves into a colcon-built ROS package. The latter is the ROS 2 norm.
    - If it stays here, add a `setup.py` console_script entry-point so the
      server can be run as `ros2 run yahtzee play_yahtzee_server`.
"""

from __future__ import annotations


class PlayYahtzeeActionServer:
    """rclpy ActionServer wrapping the UR3e MoveIt2 stack. STUB."""

    def __init__(self) -> None:
        # TODO:
        #   - rclpy.init(); create a Node "play_yahtzee_action_server".
        #   - Create an ActionServer for PlayYahtzee, with execute_callback
        #     and cancel_callback wired up.
        #   - Initialize MoveIt2 commander for trajectory planning.
        #   - Subscribe to the F/T sensor topic; trigger a protective stop
        #     and abort the current goal if force exceeds 15 N (§6).
        raise NotImplementedError

    def execute_callback(self, goal_handle):
        # TODO: dispatch on goal_handle.request.action_index per
        # Ur3eRobotClient docstring. Publish feedback at >= 2 Hz with the
        # current operation string and dice-moved count. Return Result on
        # completion or on cancel/abort.
        raise NotImplementedError

    def cancel_callback(self, goal_handle):
        # TODO: accept the cancellation, halt the current trajectory, and
        # return rclpy.action.CancelResponse.ACCEPT. The E-Stop button on
        # the GUI calls cancel_goal_async() on the client side and ends up
        # here.
        raise NotImplementedError
