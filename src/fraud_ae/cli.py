"""``fraud-ae`` command line: one sub-command per project phase.

Every sub-command is a thin wrapper over a pure function in the corresponding module so that
tests exercise the functions directly and the CLI stays trivial.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from . import __version__
from .config import Config, ConfigError, load_config, validate
from .data import DataError

SYNTHETIC_CSV = "data/raw/synthetic.csv"
SYNTHETIC_SUFFIX = "_synthetic"


def project_root() -> Path:
    """Paths in configs are relative to the current working directory."""
    return Path.cwd()


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
    p.add_argument(
        "--synthetic",
        action="store_true",
        help=f"use a tiny generated dataset ({SYNTHETIC_CSV}, outputs under "
        f"<processed_dir>{SYNTHETIC_SUFFIX}) to check the pipeline without the real CSV",
    )
    p.add_argument("--figures-dir", help="where EDA figures go (default: report/figures)")

    p = sub.add_parser("train", help="Phase 2: train the autoencoder on legitimate rows")
    _add_config(p)
    _add_run(p)
    p.add_argument(
        "--synthetic",
        action="store_true",
        help=f"train on <processed_dir>{SYNTHETIC_SUFFIX} written by `prepare --synthetic`",
    )
    p.add_argument("--epochs", type=int, help="override train.epochs (smoke tests)")

    p = sub.add_parser("threshold", help="Phase 3: score val/test and choose the threshold")
    _add_run(p)
    p.add_argument("--strategy", choices=["quantile", "youden", "max_f1"], help="override")
    p.add_argument("--target-fpr", type=float, help="override threshold.target_fpr")
    p.add_argument(
        "--method", choices=["mse", "standardized", "mahalanobis"], help="override score.method"
    )
    p.add_argument(
        "--synthetic", action="store_true", help=f"read <processed_dir>{SYNTHETIC_SUFFIX}"
    )

    p = sub.add_parser("evaluate", help="Phase 4: metrics and figures on the test set")
    _add_run(p)

    p = sub.add_parser("baselines", help="Phase 5: PCA and Isolation Forest baselines")
    _add_config(p)

    p = sub.add_parser("all", help="Phases 1-4 in sequence")
    _add_config(p)
    _add_run(p)
    return parser


def cmd_prepare(cfg: Config, synthetic: bool = False, figures_dir: str | None = None) -> int:
    from .data import prepare, split_table_markdown

    root = project_root()
    dcfg = cfg.data
    if synthetic:
        from .synthetic import write_synthetic_csv

        write_synthetic_csv(root / SYNTHETIC_CSV)
        dcfg = dataclasses.replace(
            dcfg,
            zip_path="",
            raw_csv=SYNTHETIC_CSV,
            processed_dir=dcfg.processed_dir.rstrip("/") + SYNTHETIC_SUFFIX,
            expected_sha256="",
        )
        figs = Path(figures_dir) if figures_dir else root / dcfg.processed_dir / "figures"
    else:
        figs = Path(figures_dir) if figures_dir else root / "report" / "figures"

    _, report = prepare(dcfg, root, figures_dir=figs)
    rows, cb = report["rows"], report["class_balance"]
    print(
        f"raw rows: {rows['raw']:,}  duplicates dropped: {rows['duplicates_dropped']:,}  "
        f"rows: {rows['after_dedupe']:,}  frauds: {cb['fraud']:,} ({cb['fraud_rate']:.3%})"
    )
    print(f"sha256({dcfg.raw_csv}) = {report['source']['sha256']}")
    print(f"features ({len(report['features'])}): {', '.join(report['features'])}")
    print(split_table_markdown(report))
    print(f"artifacts: {root / dcfg.processed_dir}   figures: {figs}")
    return 0


def _processed_dir(cfg: Config, synthetic: bool) -> Path:
    d = cfg.data.processed_dir
    return project_root() / (d.rstrip("/") + SYNTHETIC_SUFFIX if synthetic else d)


def cmd_train(cfg: Config, run: Path, synthetic: bool = False, epochs: int | None = None) -> int:
    from .data import load_splits
    from .train import load_summary, train

    if epochs is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, epochs=epochs))
    splits = load_splits(_processed_dir(cfg, synthetic))
    train(cfg, splits.X_train, splits.X_val[splits.y_val == 0], run)
    s = load_summary(run)
    print(
        f"trained on {s['n_train']:,} legit rows ({s['input_dim']} features, "
        f"{s['n_parameters']:,} parameters), seed {s['seed']}"
    )
    print(
        f"epochs: {s['epochs_run']}/{s['max_epochs']}"
        f"{' (early stop)' if s['stopped_early'] else ''}  best epoch: {s['best_epoch']}  "
        f"best val-legit MSE: {s['best_val_legit_mse']:.5f}  "
        f"final train MSE: {s['final_train_mse']:.5f}  {s['wall_seconds']:.1f}s"
    )
    print(f"run directory: {run}")
    return 0


def cmd_threshold(
    run: Path,
    strategy: str | None = None,
    target_fpr: float | None = None,
    method: str | None = None,
    synthetic: bool = False,
) -> int:
    """Score val and test with the run's model, pick the threshold on val, save artifacts.

    The config comes from the run's own ``config.toml`` snapshot so that scoring always matches
    the model that was trained; CLI flags override single fields.
    """
    from .data import load_splits
    from .plots import score_histogram
    from .scoring import calibrate, save_calibration, save_scores, score
    from .threshold import choose_threshold, save_threshold
    from .train import CONFIG_FILE, load_model

    cfg = load_config(run / CONFIG_FILE)
    if method:
        cfg = dataclasses.replace(cfg, score=dataclasses.replace(cfg.score, method=method))
    th = cfg.threshold
    if strategy:
        th = dataclasses.replace(th, strategy=strategy)
    if target_fpr is not None:
        th = dataclasses.replace(th, target_fpr=target_fpr)
    cfg = dataclasses.replace(cfg, threshold=th)
    validate(cfg)

    splits = load_splits(_processed_dir(cfg, synthetic))
    model = load_model(run)
    calib = calibrate(model, splits.X_val[splits.y_val == 0])
    save_calibration(calib, run)
    scores_val = score(model, splits.X_val, cfg.score, calib)
    scores_test = score(model, splits.X_test, cfg.score, calib)
    save_scores(run, "val", scores_val, splits.y_val)
    save_scores(run, "test", scores_test, splits.y_test)

    result = choose_threshold(scores_val, splits.y_val, cfg.threshold, cfg.score.method)
    save_threshold(result, run)
    score_histogram(
        scores_val,
        splits.y_val,
        result.value,
        run / "figures" / "score_histogram_val.png",
        title=f"Validation scores ({cfg.score.method}) – {result.strategy} threshold",
    )
    rec = "n/a" if result.val_recall is None else f"{result.val_recall:.1%}"
    print(
        f"score method: {cfg.score.method}  strategy: {result.strategy}"
        f"{f' (target FPR {result.target_fpr:.2%})' if result.strategy == 'quantile' else ''}"
    )
    print(
        f"threshold: {result.value:.6g}  val FPR: {result.val_fpr:.3%} "
        f"({result.n_val_legit:,} legit)  val recall: {rec} ({result.n_val_fraud} frauds)"
    )
    print(f"artifacts: {run}/{{scores_val.npz, scores_test.npz, threshold.json, figures/}}")
    return 0


def cmd_evaluate(run: Path) -> int:
    raise NotImplementedError("Phase 4: wire scoring + evaluate.evaluate + plots")


def cmd_baselines(cfg: Config) -> int:
    from .baselines import run_baselines
    from .data import load_splits

    root = project_root()
    run_baselines(cfg, load_splits(root / cfg.data.processed_dir), root / "runs")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            return cmd_prepare(load_config(args.config), args.synthetic, args.figures_dir)
        if args.command == "train":
            return cmd_train(load_config(args.config), Path(args.run), args.synthetic, args.epochs)
        if args.command == "threshold":
            return cmd_threshold(
                Path(args.run), args.strategy, args.target_fpr, args.method, args.synthetic
            )
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
                lambda: cmd_threshold(run),
                lambda: cmd_evaluate(run),
            ):
                rc = step()
                if rc:
                    return rc
            return 0
    except NotImplementedError as exc:
        print(f"fraud-ae {args.command}: not implemented yet ({exc})", file=sys.stderr)
        return 2
    except (ConfigError, DataError, FileNotFoundError, ValueError) as exc:
        print(f"fraud-ae {args.command}: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
