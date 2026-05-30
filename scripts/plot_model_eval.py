"""Compare trained models by playing N greedy games each in the simulator.

Unlike plot_training_comparison.py (which needs training_*.log files), this
script works straight from the saved weights in models/. Every model plays the
*same* seeded dice sequence, so the comparison is paired and reproducible.

    python scripts/plot_model_eval.py                 # all models, 200 games
    python scripts/plot_model_eval.py --games 500
    python scripts/plot_model_eval.py --models models/model3.pt models/model3_v2.pt

Writes model_eval_comparison.png to the repo root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def discover_models(models_dir: Path) -> list[Path]:
    """Return one weight file per distinct model, preferring .pt over .pt.ckpt.

    models/ holds both model3.pt and model3.pt.ckpt for some runs; we want the
    final .pt and fall back to the .ckpt only when no .pt exists (e.g. model2)."""
    by_name: dict[str, Path] = {}
    for path in sorted(models_dir.glob("*.pt*")):
        name = path.name[: -len(".ckpt")] if path.name.endswith(".ckpt") else path.name
        name = name[: -len(".pt")] if name.endswith(".pt") else name
        # Prefer a real .pt over a .ckpt for the same model name.
        if name in by_name and by_name[name].suffix == ".pt":
            continue
        by_name[name] = path
    return [by_name[k] for k in sorted(by_name)]


def play_games(weights: Path, num_games: int, device: str) -> np.ndarray:
    """Load weights and play `num_games` greedy games on seeds 0..N-1."""
    from src.agent.dqn_agent import DQNAgent
    from src.agent.action_selection import select_legal_action
    from src.engine.yahtzee_env import YahtzeeEnv

    agent = DQNAgent(device=device)
    agent.load_weights(str(weights))

    totals = np.empty(num_games, dtype=np.int32)
    for game_idx in range(num_games):
        env = YahtzeeEnv(seed=game_idx)
        state = env.reset()
        done = False
        while not done:
            mask = env.get_legal_mask()
            with torch.no_grad():
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32, device=agent.device
                )
                q_values = agent.online(state_tensor).cpu().numpy()
            action = select_legal_action(q_values, mask)
            state, _, done = env.step(action)
        totals[game_idx] = env.scorecard.total()
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=200,
                        help="Games to play per model (default 200).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--models", nargs="*", type=Path, default=None,
                        help="Explicit weight files. Default: auto-discover models/.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "model_eval_comparison.png")
    args = parser.parse_args()

    models = args.models or discover_models(REPO_ROOT / "models")
    if not models:
        raise SystemExit("No model weight files found in models/.")

    import matplotlib.pyplot as plt

    labels, results = [], []
    for path in models:
        print(f"Evaluating {path.name} over {args.games} games...", flush=True)
        totals = play_games(path, args.games, args.device)
        labels.append(path.name.replace(".pt.ckpt", "").replace(".pt", ""))
        results.append(totals)
        print(f"  mean={totals.mean():.1f}  median={np.median(totals):.0f}  "
              f"min={totals.min()}  max={totals.max()}", flush=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    positions = range(1, len(results) + 1)

    bp = ax.boxplot(results, positions=positions, widths=0.55, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black", "markersize": 7},
                    medianprops={"color": "#444", "linewidth": 1.5})

    # Jittered raw scores behind each box so the spread is visible.
    rng = np.random.default_rng(0)
    for pos, totals in zip(positions, results):
        x = pos + (rng.random(len(totals)) - 0.5) * 0.25
        ax.scatter(x, totals, s=8, alpha=0.25, color="#1f77b4", zorder=1)

    # Annotate each box with its mean.
    for pos, totals in zip(positions, results):
        ax.annotate(f"μ={totals.mean():.1f}", xy=(pos, totals.mean()),
                    xytext=(pos + 0.30, totals.mean()), va="center",
                    fontsize=9, color="black")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model")
    ax.set_ylabel("Final game score")
    ax.set_title(f"Yahtzee DQN — eval score distribution ({args.games} paired games each)")
    ax.axhline(190, color="green", linestyle="--", alpha=0.5,
               label="design-doc target (190)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    fig.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
