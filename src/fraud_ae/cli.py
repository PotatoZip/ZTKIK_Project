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
DATA_SEED_SUFFIX = "_dseed"


def project_root() -> Path:
    """Paths in configs are relative to the current working directory."""
    return Path.cwd()


def _add_config(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/base.toml", help="TOML experiment config")


def _add_run(p: argparse.ArgumentParser, required: bool = True) -> None:
    p.add_argument("--run", required=required, help="run directory, e.g. runs/ae_base")


def _add_data_seed(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data-seed",
        type=int,
        help=f"override data.seed and use <processed_dir>{DATA_SEED_SUFFIX}<N> (repeated legit "
        "splits, Roadmap v2); threshold/evaluate follow the run's config snapshot",
    )


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
    p.add_argument(
        "--figures-dir",
        help="where EDA figures go (default: report/figures; <processed_dir>/figures with "
        "--synthetic or --data-seed)",
    )
    _add_data_seed(p)

    p = sub.add_parser("train", help="Phase 2: train the autoencoder on legitimate rows")
    _add_config(p)
    _add_run(p)
    _add_data_seed(p)
    p.add_argument(
        "--synthetic",
        action="store_true",
        help=f"train on <processed_dir>{SYNTHETIC_SUFFIX} written by `prepare --synthetic`",
    )
    p.add_argument("--epochs", type=int, help="override train.epochs (smoke tests)")
    p.add_argument("--seed", type=int, help="override train.seed (seed-variance ablation)")

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
    p.add_argument(
        "--synthetic", action="store_true", help=f"read <processed_dir>{SYNTHETIC_SUFFIX}"
    )

    p = sub.add_parser("baselines", help="Phase 5: PCA and Isolation Forest baselines")
    _add_config(p)
    p.add_argument(
        "--synthetic", action="store_true", help=f"read <processed_dir>{SYNTHETIC_SUFFIX}"
    )
    p.add_argument(
        "--suffix",
        help="run-name suffix, runs/baseline_<name><suffix> (default: '_temporal' for a temporal "
        "split, else none); use e.g. '_pooled' so a second protocol never overwrites runs",
    )
    _add_data_seed(p)

    p = sub.add_parser("compare", help="Phase 5: comparison table + figures from run directories")
    p.add_argument(
        "--runs",
        nargs="*",
        help="run directories (default: every runs/* with metrics.json, synthetic ones excluded)",
    )
    p.add_argument("--out", default="results/comparison.md", help="Markdown output")
    p.add_argument("--figures-dir", default="report/figures", help="where the figures go")
    p.add_argument(
        "--figure-prefix", default="", help="prefix for the figure file names (e.g. v2_protocol_)"
    )

    p = sub.add_parser("report", help="Phase 6: render report/report.md from the run artifacts")
    p.add_argument("--template", default="report/report_template.md")
    p.add_argument("--out", default="report/report.md")

    p = sub.add_parser(
        "diagnose", help="Phase 7: profile the missed test frauds of a run (+ ensemble bounds)"
    )
    _add_run(p)
    p.add_argument(
        "--against",
        nargs="*",
        default=[],
        help="other run directories scored on the same test split (overlap, fusion bounds)",
    )
    p.add_argument("--fpr", type=float, help="oracle FPR for the bounds (default: eval.target_fpr)")
    p.add_argument(
        "--synthetic", action="store_true", help=f"read <processed_dir>{SYNTHETIC_SUFFIX}"
    )

    p = sub.add_parser("all", help="Phases 1-4 in sequence")
    _add_config(p)
    _add_run(p)
    p.add_argument("--synthetic", action="store_true", help="run all phases on the synthetic set")
    _add_data_seed(p)
    return parser


def _with_data_seed(cfg: Config, data_seed: int | None) -> Config:
    """``--data-seed N``: a re-drawn legitimate split lives in ``<processed_dir>_dseed<N>``. The
    override changes the config, so ``train`` snapshots the *effective* config (ADR 17) and
    ``threshold`` / ``evaluate`` follow it without a flag (ADR 13)."""
    if data_seed is None:
        return cfg
    d = cfg.data
    d = dataclasses.replace(
        d,
        seed=data_seed,
        processed_dir=d.processed_dir.rstrip("/") + f"{DATA_SEED_SUFFIX}{data_seed}",
    )
    return dataclasses.replace(cfg, data=d)


def cmd_prepare(
    cfg: Config,
    synthetic: bool = False,
    figures_dir: str | None = None,
    data_seed: int | None = None,
) -> int:
    from .data import prepare, split_table_markdown

    root = project_root()
    cfg = _with_data_seed(cfg, data_seed)
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
    elif data_seed is not None:  # same frame, re-drawn split: keep report/figures/eda_* untouched
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


def cmd_train(
    cfg: Config,
    run: Path,
    synthetic: bool = False,
    epochs: int | None = None,
    seed: int | None = None,
    data_seed: int | None = None,
) -> int:
    from .data import load_splits
    from .train import load_summary, train

    cfg = _with_data_seed(cfg, data_seed)
    overrides = {k: v for k, v in (("epochs", epochs), ("seed", seed)) if v is not None}
    if overrides:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **overrides))
    splits = load_splits(_processed_dir(cfg, synthetic))
    if splits.X_train.shape[1] != cfg.data.n_features:
        raise DataError(
            f"{_processed_dir(cfg, synthetic)} has {splits.X_train.shape[1]} features but the "
            f"config expects {cfg.data.n_features} – run `fraud-ae prepare` with this config"
        )
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


def cmd_evaluate(run: Path, synthetic: bool = False) -> int:
    """Phase 4: read the test scores and the val-chosen threshold, compute ``metrics.json`` and
    the figures. This is the first (and only) place where test labels are read."""
    from .data import load_splits
    from .evaluate import evaluate, per_feature_summary, save_metrics
    from .plots import (
        per_feature_residuals,
        recall_vs_fpr,
        roc_pr_curves,
        score_histogram,
        threshold_sweep,
    )
    from .scoring import load_scores, residuals
    from .threshold import load_threshold
    from .train import CONFIG_FILE, load_model

    cfg = load_config(run / CONFIG_FILE)
    th = load_threshold(run)
    scores, y = load_scores(run, "test")
    metrics = evaluate(scores, y, th.value, cfg.eval, seed=cfg.train.seed)
    metrics["score_method"] = th.score_method
    metrics["threshold_strategy"] = th.strategy

    figs = run / "figures"
    roc_pr_curves(scores, y, figs / "roc_pr_test.png")
    recall_vs_fpr(
        scores,
        y,
        (cfg.eval.target_fpr, cfg.eval.target_recall),
        figs / "recall_vs_fpr_test.png",
        threshold=th.value,
    )
    threshold_sweep(scores, y, th.value, figs / "threshold_sweep_test.png")
    score_histogram(
        scores,
        y,
        th.value,
        figs / "score_histogram_test.png",
        title=f"Test scores ({th.score_method}) – threshold chosen on validation",
    )
    splits = load_splits(_processed_dir(cfg, synthetic))
    R = residuals(load_model(run), splits.X_test)
    metrics["per_feature"] = per_feature_summary(R, splits.y_test, splits.feature_names)
    per_feature_residuals(
        R, splits.y_test, splits.feature_names, figs / "per_feature_residuals_test.png"
    )
    save_metrics(metrics, run)

    at, c, b = metrics["at_threshold"], metrics["criterion"], metrics["bootstrap"]
    print(
        f"test set: {metrics['n_legit']:,} legit / {metrics['n_fraud']} fraud  "
        f"score: {th.score_method}  threshold: {th.value:.6g} ({th.strategy})"
    )
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}  PR-AUC: {metrics['pr_auc']:.4f}")
    print(
        "recall @ FPR:  "
        + "  ".join(f"{g['fpr_target']:.1%} → {g['recall']:.1%}" for g in metrics["recall_at_fpr"])
    )
    print(
        f"at threshold: TP {at['tp']}  FP {at['fp']}  TN {at['tn']:,}  FN {at['fn']}  "
        f"recall {at['recall']:.1%} [{at['recall_ci_wilson'][0]:.1%}, "
        f"{at['recall_ci_wilson'][1]:.1%}]  FPR {at['fpr']:.3%} "
        f"[{at['fpr_ci_wilson'][0]:.3%}, {at['fpr_ci_wilson'][1]:.3%}]  "
        f"precision {at['precision']:.1%}  F1 {at['f1']:.3f}"
    )
    br, bf = b["recall_at_threshold"], b["recall_at_target_fpr"]
    print(
        f"bootstrap ({b['samples']}): recall@threshold [{br['ci_lo']:.1%}, {br['ci_hi']:.1%}]  "
        f"recall@FPR {c['target_fpr']:.0%} [{bf['ci_lo']:.1%}, {bf['ci_hi']:.1%}]"
    )
    verdict = "MET" if metrics["meets_target"] else "NOT met"
    print(
        f"criterion recall ≥ {c['target_recall']:.0%} and FPR < {c['target_fpr']:.0%}: {verdict}  "
        f"(need TP ≥ {c['tp_required']}, FP ≤ {c['fp_allowed']})"
    )
    print(f"artifacts: {run}/metrics.json, {figs}/*_test.png")
    return 0


def cmd_baselines(
    cfg: Config,
    synthetic: bool = False,
    suffix: str | None = None,
    data_seed: int | None = None,
) -> int:
    """Phase 5: PCA(k) and Isolation Forest on the same splits, same threshold rule and
    evaluator as the AE; one run directory each under ``runs/``."""
    from .baselines import run_baselines
    from .data import load_splits

    cfg = _with_data_seed(cfg, data_seed)
    splits = load_splits(_processed_dir(cfg, synthetic))
    # run names carry the protocol so that a second split never overwrites the first one's runs
    if suffix is None:
        suffix = "_temporal" if cfg.data.split == "temporal" else ""
    suffix += (f"{DATA_SEED_SUFFIX}{data_seed}" if data_seed is not None else "") + (
        SYNTHETIC_SUFFIX if synthetic else ""
    )
    results = run_baselines(cfg, splits, project_root() / "runs", suffix=suffix)
    for name, m in results.items():
        at, b = m["at_threshold"], m["baseline"]
        extra = (
            f"explained var {b['explained_variance_ratio']:.1%}"
            if b["kind"] == "pca"
            else f"{b['n_estimators']} trees"
        )
        print(
            f"{name}: {b['model']} ({extra}, {b['wall_seconds']:.1f}s)  ROC-AUC {m['roc_auc']:.4f}"
            f"  test recall {at['recall']:.1%} [{at['recall_ci_wilson'][0]:.1%}, "
            f"{at['recall_ci_wilson'][1]:.1%}]  FPR {at['fpr']:.3%}  "
            f"meets_target: {m['meets_target']}"
        )
    return 0


def cmd_compare(
    runs: list[str] | None, out: str, figures_dir: str | None, figure_prefix: str = ""
) -> int:
    from .baselines import write_comparison
    from .evaluate import METRICS_FILE

    root = project_root()
    if runs:
        dirs = [Path(r) for r in runs]
    else:
        dirs = sorted(
            d
            for d in (root / "runs").glob("*")
            if (d / METRICS_FILE).exists() and SYNTHETIC_SUFFIX not in d.name
        )
    rows = write_comparison(
        dirs, Path(out), Path(figures_dir) if figures_dir else None, figure_prefix=figure_prefix
    )
    print(f"{len(rows)} runs → {out}" + (f" (figures in {figures_dir})" if figures_dir else ""))
    for r in rows:
        lo, hi = r["recall_ci"]
        print(
            f"  {r['run']:<28} {r['model']:<28} test recall {r['recall']:.1%} "
            f"[{lo:.1%}, {hi:.1%}] at FPR {r['fpr']:.3%}  {'✅' if r['meets_target'] else '❌'}"
        )
    return 0


def cmd_report(template: str, out: str) -> int:
    from .report import ReportError, render_report

    try:
        path = render_report(project_root(), template, out)
    except ReportError as exc:
        raise ValueError(f"unresolved placeholders:\n{exc}") from exc
    print(f"report written: {path}")
    return 0


def cmd_diagnose(
    run: Path, against: list[str], fpr: float | None = None, synthetic: bool = False
) -> int:
    """Phase 7: profile the frauds ``run`` misses on the test split, compare the missed set with
    other runs on the same split and write ``diagnostics.json`` + ``figures/missed_frauds.png``."""
    from .diagnose import FIGURE_FILE, diagnose, save_diagnostics
    from .plots import missed_frauds
    from .train import CONFIG_FILE

    cfg = load_config(run / CONFIG_FILE)
    summary, arrays = diagnose(
        run, [Path(a) for a in against], _processed_dir(cfg, synthetic), project_root(), fpr
    )
    save_diagnostics(summary, run)
    missed_frauds(arrays, run / FIGURE_FILE)

    g, tf = summary["groups"], summary["top_features"]
    lo, hi = summary["recall_ci_wilson"]
    print(
        f"{summary['run']}: {summary['n_missed']} of {summary['n_fraud']} test frauds missed at "
        f"threshold {summary['threshold']:.4g} (recall {summary['recall']:.1%} [{lo:.1%}, "
        f"{hi:.1%}], FPR {summary['fpr']:.3%})"
    )
    for name in ("legit", "detected", "missed"):
        r = g[name]
        print(
            f"  {name:<9} n={r['n']:<7,} score median {r['score_median']:.3g}  legit-percentile "
            f"median {r['legit_percentile_median']:.1%}  amount median {r['amount_median']:.2f}  "
            f"night share {r['night_share']:.1%}"
        )
    print(
        "  top features (detected vs legit): "
        + ", ".join(
            f"{f} {m:+.2f}/{d:+.2f}/{lm:+.2f}"
            for f, lm, d, m in zip(
                tf["features"],
                tf["legit_mean"],
                tf["detected_mean"],
                tf["missed_mean"],
                strict=True,
            )
        )
        + "  (missed / detected / legit mean)"
    )
    for o in summary["against"]:
        print(
            f"  vs {o['run']}: catches {o['caught_of_missed_own']} of the missed at its own "
            f"threshold, {o['caught_of_missed_oracle']} at oracle FPR {summary['fpr_target']:.0%}"
        )
    if summary["fusion"]:
        fu = summary["fusion"]
        parts = [
            f"{label} {fu[key]['recall']:.1%} at FPR {fu[key]['fpr']:.2%}"
            for label, key in (
                ("union@own", "union_own"),
                ("union@oracle", "union_oracle"),
                ("rank fusion", "rank_fusion"),
            )
        ]
        print(f"  bounds over {len(fu['runs'])} runs: " + ", ".join(parts))
    print(f"artifacts: {run}/diagnostics.json, {run / FIGURE_FILE}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            return cmd_prepare(
                load_config(args.config), args.synthetic, args.figures_dir, args.data_seed
            )
        if args.command == "train":
            return cmd_train(
                load_config(args.config),
                Path(args.run),
                args.synthetic,
                args.epochs,
                args.seed,
                args.data_seed,
            )
        if args.command == "threshold":
            return cmd_threshold(
                Path(args.run), args.strategy, args.target_fpr, args.method, args.synthetic
            )
        if args.command == "evaluate":
            return cmd_evaluate(Path(args.run), args.synthetic)
        if args.command == "baselines":
            return cmd_baselines(
                load_config(args.config), args.synthetic, args.suffix, args.data_seed
            )
        if args.command == "compare":
            return cmd_compare(args.runs, args.out, args.figures_dir, args.figure_prefix)
        if args.command == "report":
            return cmd_report(args.template, args.out)
        if args.command == "diagnose":
            return cmd_diagnose(Path(args.run), args.against, args.fpr, args.synthetic)
        if args.command == "all":
            cfg = load_config(args.config)
            run = Path(args.run)
            syn, ds = args.synthetic, args.data_seed
            for step in (
                lambda: cmd_prepare(cfg, syn, data_seed=ds),
                lambda: cmd_train(cfg, run, syn, data_seed=ds),
                lambda: cmd_threshold(run, synthetic=syn),
                lambda: cmd_evaluate(run, syn),
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
