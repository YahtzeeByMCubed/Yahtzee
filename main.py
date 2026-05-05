"""Entry point — toggle between Training Mode and Demo Mode.

Training Mode:  runs the DQN training loop against the simulated environment
                (brain-in-a-vat — no robot, no camera).
Demo Mode:      launches the GUI and (optionally) wires in the live perception
                + robot subsystems for an end-to-end game.

TODO:
- Wire argparse with subcommands `train` and `demo`.
- In `demo`, support `--simulated-robot` and `--simulated-vision` flags so the
  brain-in-a-vat can be exercised through the GUI without the physical stack.
- In `train`, expose hyperparams (num_episodes, save_path, eval_interval) as CLI
  flags that forward to src.agent.trainer.train().
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Mode dispatcher not yet implemented.")


if __name__ == "__main__":
    main()
