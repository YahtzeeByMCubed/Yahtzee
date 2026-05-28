"""Entry point — toggles between Training Mode and Demo Mode.

Training Mode:  runs the DQN training loop against the simulated environment
                (brain-in-a-vat — no robot, no camera).
Demo Mode:      launches the GUI dashboard. Optional --weights loads a
                trained agent for live Q-value display per design doc §2.3.

Examples:
    python main.py train --num-episodes 100000 --save-path models/dqn.pt
    python main.py train --resume models/dqn.pt --num-episodes 200000
    python main.py demo --weights models/dqn.pt
"""

from __future__ import annotations

import argparse
import os
import sys


def _bootstrap_path() -> None:
    """Put the repo root on sys.path so the top-level packages import
    cleanly regardless of how main.py is invoked."""
    repo_root = os.path.abspath(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def cmd_train(args: argparse.Namespace) -> None:
    _bootstrap_path()
    from src.agent.dqn_agent import train

    train(
        num_episodes=args.num_episodes,
        save_path=args.save_path,
        eval_interval=args.eval_interval,
        device=args.device,
        load_path=args.resume,
        eps_start=args.eps_start,
        learn_every=args.learn_every,
        checkpoint_interval=args.checkpoint_interval,
        gamma=args.gamma,
        beta_anneal_steps=args.beta_anneal_steps,
        shaped_reward=args.shaped_reward,
    )


def cmd_demo(args: argparse.Namespace) -> None:
    _bootstrap_path()
    import numpy as np
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    from src.engine.yahtzee_env import YahtzeeEnv
    from src.gui.gui import Dashboard

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QWidget { color: black; }
        QLabel { color: black; }
        QPushButton { color: black; }
    """)

    env = YahtzeeEnv(seed=args.seed)
    state = env.reset()
    window = Dashboard(env=env)

    if args.weights:
        from src.agent.dqn_agent import DQNAgent, select_legal_action
        agent = DQNAgent(device=args.device or "cpu")
        agent.load_weights(args.weights)
        print(f"Loaded weights from {args.weights}")
    else:
        agent = None

    def render(s):
        if agent is not None:
            q = agent.get_q_values(s)
        else:
            q = np.zeros(45, dtype=np.float32)
        window.refresh_ui(
            state_vector=s,
            q_values=q,
            legal_mask=env.get_legal_mask(),
        )

    render(state)
    window.show()

    if args.autoplay:
        if agent is None:
            print("--autoplay needs --weights; falling back to manual mode.", file=sys.stderr)
        else:
            # Disable manual interaction so clicks don't race the timer.
            window.board.on_score_clicked = None

            state_box = {"state": state}

            def step_once():
                s = state_box["state"]
                if env.done:
                    print(f"Game over — final {env.scorecard.total()}. Resetting.")
                    s = env.reset()
                    state_box["state"] = s
                    render(s)
                    return
                mask = env.get_legal_mask()
                q = agent.get_q_values(s)
                action = select_legal_action(q, mask)
                next_s, _, _ = env.step(action)
                state_box["state"] = next_s
                render(next_s)

            timer = QTimer()
            timer.timeout.connect(step_once)
            timer.start(args.speed_ms)
            print(f"Autoplay running at {args.speed_ms}ms/action. Close window to exit.")

    sys.exit(app.exec())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yahtzee",
        description="DQN-driven Yahtzee agent: train against the simulator or demo via GUI.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Run DQN training against the simulator.")
    p_train.add_argument("--num-episodes", type=int, default=100_000)
    p_train.add_argument("--save-path", type=str, default="models/dqn.pt")
    p_train.add_argument("--eval-interval", type=int, default=1_000)
    p_train.add_argument("--device", type=str, default=None,
                         help="cpu | cuda. Auto-detected if omitted.")
    p_train.add_argument("--resume", type=str, default=None,
                         help="Path to a .pt to resume from. ε_start defaults to 0.1 on resume.")
    p_train.add_argument("--eps-start", type=float, default=None,
                         help="Override initial epsilon (0.0–1.0).")
    p_train.add_argument("--checkpoint-interval", type=int, default=10_000,
                         help="Write <save-path>.ckpt every N episodes (default 10000). "
                              "Set 0 to disable.")
    p_train.add_argument("--learn-every", type=int, default=1,
                         help="Run a gradient step every N env steps. "
                              "1 = original DQN behaviour. 4 is the Atari-paper "
                              "default and gives ~3.5x speedup with similar "
                              "sample efficiency.")
    p_train.add_argument("--gamma", type=float, default=0.99,
                         help="Bootstrap discount. 0.997 propagates the sparse "
                              "terminal reward through more of the episode.")
    p_train.add_argument("--beta-anneal-steps", type=int, default=1_000_000,
                         help="Env-steps over which PER β linearly anneals "
                              "0.4 → 1.0. β resets to 0.4 on resume.")
    p_train.add_argument("--shaped-reward", action="store_true",
                         help="§3.4 ablation: per-commit reward = score_added/100 "
                              "instead of strict sparse. Total return per game "
                              "still equals final_score/100, so eval scores are "
                              "directly comparable to sparse-trained models.")
    p_train.set_defaults(func=cmd_train)

    p_demo = sub.add_parser("demo", help="Launch the GUI dashboard.")
    p_demo.add_argument("--weights", type=str, default=None,
                        help="Optional .pt path to load trained agent weights.")
    p_demo.add_argument("--device", type=str, default=None)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--autoplay", action="store_true",
                        help="Have the agent play games on a loop. Requires --weights.")
    p_demo.add_argument("--speed-ms", type=int, default=500,
                        help="Delay between autoplay actions (ms). Default 500.")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
