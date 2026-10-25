"""``fraud-ae`` command line: one sub-command per project phase.

Every sub-command is a thin wrapper over a pure function in the corresponding module so that
tests exercise the functions directly and the CLI stays trivial.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config, load_config

PROJECT_ROOT = Path.cwd()


def _add_config(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/base.toml", help="TOML experiment config")


def _add_run(p: argparse.ArgumentParser, required: bool = True) -> None:
    p.add_argument("--run", required=required, help="run directory, e.g. runs/ae_base")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-ae",
        description="Fraud detection with a deep autoencoder (ZTKIK Project 12).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Phase 1: preprocess the dataset into data/processed")
    _add_config(p)

    p = sub.add_parser("train", help="Phase 2: train the autoencoder on legitimate rows")
    _add_config(p)
    _add_run(p)

    p = sub.add_parser("threshold", help="Phase 3: choose the MSE threshold on val-legit")
    _add_run(p)
    p.add_argument("--strategy", choices=["quantile", "youden", "max_f1"])
    p.add_argument("--target-fpr", type=float)

    p = sub.add_parser("evaluate", help="Phase 4: metrics and figures on the test set")
    _add_run(p)

    p = sub.add_parser("baselines", help="Phase 5: PCA and Isolation Forest baselines")
    _add_config(p)

    p = sub.add_parser("all", help="Phases 1-4 in sequence")
    _add_config(p)
    _add_run(p)
    return parser


def cmd_prepare(cfg: Config) -> int:
    from .data import prepare

    prepare(cfg.data, PROJECT_ROOT)
    return 0


def cmd_train(cfg: Config, run: Path) -> int:
    from .data import load_splits
    from .train import train

    splits = load_splits(PROJECT_ROOT / cfg.data.processed_dir)
    train(cfg, splits.X_train, splits.X_val[splits.y_val == 0], run)
    return 0


def cmd_threshold(run: Path, strategy: str | None, target_fpr: float | None) -> int:
    raise NotImplementedError("Phase 3: wire scoring.calibrate/score + threshold.choose_threshold")


def cmd_evaluate(run: Path) -> int:
    raise NotImplementedError("Phase 4: wire scoring + evaluate.evaluate + plots")


def cmd_baselines(cfg: Config) -> int:
    from .baselines import run_baselines
    from .data import load_splits

    run_baselines(cfg, load_splits(PROJECT_ROOT / cfg.data.processed_dir), PROJECT_ROOT / "runs")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            return cmd_prepare(load_config(args.config))
        if args.command == "train":
            return cmd_train(load_config(args.config), Path(args.run))
        if args.command == "threshold":
            return cmd_threshold(Path(args.run), args.strategy, args.target_fpr)
        if args.command == "evaluate":
            return cmd_evaluate(Path(args.run))
        if args.command == "baselines":
            return cmd_baselines(load_config(args.config))
        if args.command == "all":
            cfg = load_config(args.config)
            run = Path(args.run)
            for step in (
                lambda: cmd_prepare(cfg),
                lambda: cmd_train(cfg, run),
                lambda: cmd_threshold(run, None, None),
                lambda: cmd_evaluate(run),
            ):
                rc = step()
                if rc:
                    return rc
            return 0
    except NotImplementedError as exc:
        print(f"fraud-ae {args.command}: not implemented yet ({exc})", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
