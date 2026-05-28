# Yahtzee
This project integrates a Deep Q-Network (DQN) with a UR3e robotic manipulator and YOLOv8-based vision to play Yahtzee autonomously against a human opponent. This document outlines the complete architecture where the AI agent acts as the primary decision-maker, facilitated by a ROS 2 (Humble) Action Server and a robust logic-filtered state machine.

The system is designed to achieve a target average score of 210+ through high-fidelity simulation training. By utilizing a strictly sparse reward strategy, the AI learns holistic, forward-looking strategies over the entirety of a game rather than chasing immediate, greedy payouts. Utilizing a physical "remove-from-play" dice management strategy via the UR3e gripper, the system physically executes the AI's intent, while an Intel RealSense D435if camera provides RGB-D perception interpreted by a CNN model.

## Group information
- **Micah**: Programmed the environment
- **Marcus**: Programmed the GUI
- **Matt**: Trained the AI agent

## Repository layout

```
src/
  engine/        Logic Engine — Yahtzee rules, scorecard, 24-D state vector
    yahtzee_env.py        YahtzeeEnv: 45-action space, sparse terminal reward
    scorecard_manager.py  13 categories, upper bonus, Yahtzee bonus
    dice_manager.py       Roll / reroll utilities
    action_codec.py       Encode/decode the 45-action space
    constants.py          Shared dimensions and category indices
  agent/         AI Agent — Deep Q-Network and decision-making
    model.py              DQN nn.Module (24 -> 128 -> 128 -> 64 -> 45)
    dqn_agent.py          DQNAgent: epsilon-greedy + PER + learn step + masking
    action_selection.py   Legal-mask + epsilon-greedy action selection
    shaped_env.py         §3.4 ablation: per-commit reward shaping wrapper
  gui/           PyQt6 dashboard (passive observer of game state)
    gui.py                Dashboard, ScoreQChart, YahtzeeBoard, DicePanel
    dice.py, misc.py      Dice widget + scorecard helpers
    assets/               yahtzee_card.png and fonts
  perception/    YOLOv8 + RealSense — STUB (interface only)
  robotics/      ROS 2 PlayYahtzee action client/server — STUB (interface only)

tests/           pytest suite covering the env, agent, and action codec
models/          DQN weight checkpoints (.pt) and YOLO weights
scripts/         setup.sh (env install), smoke_run_env_agent.py
main.py          mode dispatcher (train | demo)
```

The brain-in-a-vat (engine + agent + GUI) runs without hardware. The
`perception/` and `robotics/` packages contain only Protocol-based
interfaces and stub implementations; the simulator path uses in-memory
fakes so training never depends on a robot or camera.

## Getting started (Linux)

### Quick setup (recommended)

```bash
git clone https://github.com/YahtzeeByMCubed/Yahtzee.git
cd Yahtzee
scripts/setup.sh                  # creates .venv, installs brain-in-a-vat deps
source .venv/bin/activate
pytest                            # verify install
```

### Manual setup

If you'd rather not run the script:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### Optional hardware extras

The brain-in-a-vat install above is enough to train, evaluate, and demo
the agent in the simulator. The perception and robotics layers need extra deps:

```bash
# YOLOv8 + Intel RealSense (Linux x86_64, Python 3.10 only)
pip install -e ".[vision]"

# ROS 2 Humble — required only for the robot action server.
# Cannot be installed via pip; use apt:
sudo apt install ros-humble-desktop
```

### Requirements

- Python 3.10 (the project pins `>=3.10,<3.11` because `pyrealsense2`
  only ships wheels for 3.10).
- A virtualenv-capable Python install (`python3.10-venv` on Debian/Ubuntu).
- (Optional) NVIDIA GPU + CUDA-enabled PyTorch for faster training; CPU works fine for demo and short runs.

## Run commands

The program is run using the `main.py` script, which provides a CLI with two primary modes: `train` and `demo`.

### Demo Mode
Launch the GUI dashboard to manually play or watch the AI agent.

```bash
python main.py demo [OPTIONS]
```

**Options:**
* `--weights <path>`: Path to `.pt` file to load trained agent weights.
* `--autoplay`: Have the agent play games on a loop. (Requires `--weights`).
* `--speed-ms <int>`: Delay between autoplay actions in milliseconds (default: 500).
* `--device <cpu|cuda>`: Set the compute device.
* `--seed <int>`: Random seed for the environment (default: 42).

**Example:**
```bash
python main.py demo --weights models/model3.pt --autoplay --speed-ms 500
```
Models available: `model1.pt`, `model2.pt.ckpt`, `model3.pt`

### Training Mode
Run the DQN training loop against the simulated environment.

```bash
python main.py train [OPTIONS]
```

**Options:**
* `--num-episodes <int>`: Number of training episodes (default: 100,000).
* `--save-path <path>`: Path to save the final model (default: `models/dqn.pt`).
* `--resume <path>`: Path to a `.pt` file to resume training from.
* `--eval-interval <int>`: Evaluation interval (default: 1,000).
* `--checkpoint-interval <int>`: Save checkpoint every N episodes (default: 10,000, 0 to disable).
* `--learn-every <int>`: Run gradient step every N env steps (default: 1).
* `--gamma <float>`: Bootstrap discount factor (default: 0.99).
* `--beta-anneal-steps <int>`: Env-steps for PER beta linear annealing (default: 1,000,000).
* `--shaped-reward`: Use shaped rewards instead of strict sparse rewards.
* `--eps-start <float>`: Override initial epsilon (0.0-1.0).
* `--device <cpu|cuda>`: Set the compute device (auto-detected if omitted).

**Example:**
```bash
python main.py train --num-episodes 100000 --save-path models/dqn.pt
```