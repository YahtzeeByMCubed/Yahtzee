# Yahtzee
This project integrates a Deep Q-Network (DQN) with a UR3e robotic manipulator and YOLOv8-based vision to play Yahtzee autonomously against a human opponent. This document outlines the complete architecture where the AI agent acts as the primary decision-maker, facilitated by a ROS 2 (Humble) Action Server and a robust logic-filtered state machine.

The system is designed to achieve a target average score of 210+ through high-fidelity simulation training. By utilizing a strictly sparse reward strategy, the AI learns holistic, forward-looking strategies over the entirety of a game rather than chasing immediate, greedy payouts. Utilizing a physical "remove-from-play" dice management strategy via the UR3e gripper, the system physically executes the AI's intent, while an Intel RealSense D435if camera provides RGB-D perception interpreted by a CNN model.

## Repository layout

```
src/
  engine/        Logic Engine — Yahtzee rules, scorecard, 24-D state vector
    yahtzee_env.py   YahtzeeStateMachine + 45-action space + sparse reward
    scorecard.py     ScorecardManager (13 categories, upper bonus, Yahtzee bonus)
    dice.py          Roll / reroll utilities
  agent/         AI Agent — Deep Q-Network and decision-making
    model.py         DQN nn.Module (24 -> 128 -> 128 -> 64 -> 45)
    dqn_agent.py     DQNAgent: epsilon-greedy + PER + learn step + masking
  gui/           PyQt6 dashboard (passive observer of game state)
    gui.py           Dashboard, ScorecardView, QChartView
  perception/    YOLOv8 + RealSense — STUB (interface only)
  robotics/      ROS 2 PlayYahtzee action client/server — STUB (interface only)

test/            pytest suite (currently all skipped — contracts only)
models/          DQN weight checkpoints (.pt) and YOLO weights
scripts/         setup.sh, train.sh
main.py          mode dispatcher (train | demo)
```

The brain-in-a-vat (engine + agent + GUI) is being built first. The
`perception/` and `robotics/` packages contain only Protocol-based
interfaces and stub implementations; the simulator path uses in-memory
fakes so training never depends on hardware.

## Getting started (Linux)

```bash
git clone <repo>
cd Yahtzee
scripts/setup.sh                         # creates .venv with python3.10
source .venv/bin/activate
pytest                                   # runs the (currently-skipped) test suite
```

ROS 2 (Humble) and `pyrealsense2` are required only for the hardware
demo and are intentionally excluded from the default install. See
`scripts/setup.sh` for the optional install commands.
