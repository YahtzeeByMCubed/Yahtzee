"""Compare eval_avg curves across all training_*.log files in the repo root."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
LINE_RE = re.compile(r"ep=(\d+)\s+eval_avg=([-\d.]+)")


def parse_log(path: Path) -> tuple[list[int], list[float]]:
    eps, scores = [], []
    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if m:
            eps.append(int(m.group(1)))
            scores.append(float(m.group(2)))
    return eps, scores


def main() -> None:
    log_files = sorted(REPO_ROOT.glob("training*.log"))
    if not log_files:
        raise SystemExit("No training*.log files found in repo root.")

    plt.figure(figsize=(11, 6))
    for path in log_files:
        eps, scores = parse_log(path)
        if not eps:
            continue
        label = path.stem.replace("training_", "").replace("training", "baseline")
        plt.plot(eps, scores, label=f"{label}  (final={scores[-1]:.1f})", linewidth=1.5)

    plt.xlabel("Episode")
    plt.ylabel("Eval avg score")
    plt.title("Yahtzee DQN training: eval_avg across runs")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = REPO_ROOT / "training_comparison.png"
    plt.savefig(out, dpi=130)
    print(f"Wrote {out}")
    plt.show()


if __name__ == "__main__":
    main()
