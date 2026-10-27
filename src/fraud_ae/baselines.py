"""Phase 5 – classical baselines on the same splits, scored by the same evaluator, plus the
comparison table across all runs (ARCHITECTURE.md §3).

* PCA reconstruction error with ``k`` components (a linear autoencoder, same MSE units).
* Isolation Forest fitted on train-legit; score = ``-score_samples`` so higher = more anomalous.

Each baseline writes a run directory that follows the AE artifact contract minus ``model.pt`` /
``history.csv`` (``config.toml``, ``train_summary.json``, ``scores_{val,test}.npz``,
``threshold.json``, ``metrics.json``, ``figures/``), so :func:`write_comparison` treats AE runs
and baselines identically. The threshold is the same val-legit quantile as for the AE.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .config import Config, load_config
from .data import Splits
from .evaluate import METRICS_FILE, evaluate, load_metrics, save_metrics
from .scoring import save_scores
from .threshold import choose_threshold, load_threshold, save_threshold
from .train import SUMMARY_FILE, snapshot_config

PCA_COMPONENTS: tuple[int, ...] = (5, 10, 15)
IFOREST_ESTIMATORS = 200


# --------------------------------------------------------------------------- scores


def fit_pca(X_train: np.ndarray, n_components: int) -> PCA:
    X_train = np.asarray(X_train, dtype=np.float64)
    if not 0 < n_components <= X_train.shape[1]:
        raise ValueError("n_components must be in [1, n_features]")
    return PCA(n_components=n_components, svd_solver="full").fit(X_train)


def pca_reconstruction_error(pca: PCA, X: np.ndarray) -> np.ndarray:
    """Per-sample mean squared reconstruction error after projecting onto the fitted PCs."""
    X = np.asarray(X, dtype=np.float64)
    Xr = pca.inverse_transform(pca.transform(X))
    return ((X - Xr) ** 2).mean(axis=1)


def pca_scores(X_train: np.ndarray, X: np.ndarray, n_components: int) -> np.ndarray:
    """Per-sample squared reconstruction error after projecting onto ``n_components`` PCs."""
    return pca_reconstruction_error(fit_pca(X_train, n_components), X)


def fit_iforest(
    X_train: np.ndarray, seed: int, n_estimators: int = IFOREST_ESTIMATORS
) -> IsolationForest:
    return IsolationForest(
        n_estimators=n_estimators, max_samples="auto", random_state=seed, n_jobs=1
    ).fit(np.asarray(X_train, dtype=np.float64))


def iforest_anomaly(forest: IsolationForest, X: np.ndarray) -> np.ndarray:
    return -forest.score_samples(np.asarray(X, dtype=np.float64))


def iforest_scores(X_train: np.ndarray, X: np.ndarray, seed: int) -> np.ndarray:
    """Negated ``score_samples`` so that higher = more anomalous."""
    return iforest_anomaly(fit_iforest(X_train, seed), X)


# --------------------------------------------------------------------------- run directories


def _write_run(
    cfg: Config,
    splits: Splits,
    run_dir: Path,
    name: str,
    scores_val: np.ndarray,
    scores_test: np.ndarray,
    summary: dict[str, Any],
) -> dict[str, Any]:
    from .plots import recall_vs_fpr, roc_pr_curves, score_histogram

    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(cfg, run_dir)
    save_scores(run_dir, "val", scores_val, splits.y_val)
    save_scores(run_dir, "test", scores_test, splits.y_test)
    th = choose_threshold(scores_val, splits.y_val, cfg.threshold, score_method=name)
    save_threshold(th, run_dir)
    metrics = evaluate(scores_test, splits.y_test, th.value, cfg.eval, seed=cfg.train.seed)
    metrics["score_method"] = name
    metrics["threshold_strategy"] = th.strategy
    metrics["baseline"] = summary
    save_metrics(metrics, run_dir)
    (run_dir / SUMMARY_FILE).write_text(
        json.dumps(
            {"n_train": int(len(splits.X_train)), "seed": cfg.train.seed, **summary}, indent=2
        )
    )
    figs = run_dir / "figures"
    score_histogram(
        scores_val,
        splits.y_val,
        th.value,
        figs / "score_histogram_val.png",
        title=f"Validation scores ({name}) – {th.strategy} threshold",
    )
    roc_pr_curves(scores_test, splits.y_test, figs / "roc_pr_test.png")
    recall_vs_fpr(
        scores_test,
        splits.y_test,
        (cfg.eval.target_fpr, cfg.eval.target_recall),
        figs / "recall_vs_fpr_test.png",
        threshold=th.value,
    )
    return metrics


def run_baselines(
    cfg: Config,
    splits: Splits,
    runs_root: Path,
    suffix: str = "",
    pca_components: tuple[int, ...] = PCA_COMPONENTS,
) -> dict[str, dict[str, Any]]:
    """Fit PCA(k) for each ``k`` and an Isolation Forest on ``X_train``; write
    ``runs_root/baseline_<name><suffix>/`` for each and return ``{run_name: metrics}``."""
    runs_root = Path(runs_root)
    out: dict[str, dict[str, Any]] = {}
    d = splits.X_train.shape[1]
    for k in pca_components:
        if k > d:
            continue
        t0 = time.perf_counter()
        pca = fit_pca(splits.X_train, k)
        name = f"baseline_pca_k{k}{suffix}"
        summary = {
            "model": f"PCA k={k}",
            "kind": "pca",
            "n_components": int(k),
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            "wall_seconds": time.perf_counter() - t0,
        }
        out[name] = _write_run(
            cfg,
            splits,
            runs_root / name,
            f"pca_k{k}",
            pca_reconstruction_error(pca, splits.X_val),
            pca_reconstruction_error(pca, splits.X_test),
            summary,
        )
    t0 = time.perf_counter()
    forest = fit_iforest(splits.X_train, cfg.train.seed)
    name = f"baseline_iforest{suffix}"
    summary = {
        "model": f"IsolationForest ({IFOREST_ESTIMATORS} trees)",
        "kind": "iforest",
        "n_estimators": IFOREST_ESTIMATORS,
        "max_samples": int(forest.max_samples_),
        "wall_seconds": time.perf_counter() - t0,
    }
    out[name] = _write_run(
        cfg,
        splits,
        runs_root / name,
        "iforest",
        iforest_anomaly(forest, splits.X_val),
        iforest_anomaly(forest, splits.X_test),
        summary,
    )
    return out


# --------------------------------------------------------------------------- comparison

_SEED_SUFFIX = re.compile(r"_seed\d+$")


def _describe(run: Path) -> dict[str, Any]:
    """One flat record per run directory (AE or baseline) for the comparison table."""
    m = load_metrics(run)
    summary = json.loads((run / SUMMARY_FILE).read_text()) if (run / SUMMARY_FILE).exists() else {}
    cfg = load_config(run / "config.toml") if (run / "config.toml").exists() else None
    th = load_threshold(run)
    if "model" in summary:
        model = summary["model"]
    elif cfg is not None:
        dims = "-".join(
            str(x)
            for x in (summary.get("input_dim", "?"), *cfg.model.hidden_dims, cfg.model.latent_dim)
        )
        model = f"AE {dims}" + (" +BN" if cfg.model.batch_norm else "")
    else:
        model = "?"
    at = m["at_threshold"]
    r1 = next(
        (
            g
            for g in m["recall_at_fpr"]
            if abs(g["fpr_target"] - m["criterion"]["target_fpr"]) < 1e-12
        ),
        None,
    )
    return {
        "run": run.name,
        "family": _SEED_SUFFIX.sub("", run.name),
        "model": model,
        "split": cfg.data.split if cfg is not None else "?",
        "features": summary.get("input_dim", "?"),
        "score": m.get("score_method", th.score_method),
        "seed": summary.get("seed", "?"),
        "val_mse": summary.get("best_val_legit_mse"),
        "roc_auc": m["roc_auc"],
        "pr_auc": m["pr_auc"],
        "val_recall": th.val_recall,
        "val_fpr": th.val_fpr,
        "n_fraud": m["n_fraud"],
        "fpr": at["fpr"],
        "recall": at["recall"],
        "recall_ci": at["recall_ci_wilson"],
        "precision": at["precision"],
        "recall_at_target": r1["recall"] if r1 else float("nan"),
        "recall_at_target_ci": r1["recall_ci_wilson"] if r1 else [float("nan")] * 2,
        "meets_target": m["meets_target"],
        "seconds": summary.get("wall_seconds"),
    }


def comparison_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = [_describe(Path(r)) for r in run_dirs if (Path(r) / METRICS_FILE).exists()]
    return sorted(rows, key=lambda r: (-r["recall"], r["run"]))


def seed_families(
    rows: list[dict[str, Any]], min_size: int = 2
) -> dict[str, dict[str, list[float]]]:
    """``{family: {"val": [...], "test": [...]}}`` for families with ≥ ``min_size`` seeds."""
    fam: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        fam.setdefault(r["family"], {"val": [], "test": [], "seeds": []})
        fam[r["family"]]["val"].append(float(r["val_recall"]))
        fam[r["family"]]["test"].append(float(r["recall"]))
        fam[r["family"]]["seeds"].append(r["seed"])
    return {k: v for k, v in fam.items() if len(v["test"]) >= min_size}


def comparison_markdown(rows: list[dict[str, Any]], figures: dict[str, str] | None = None) -> str:
    pct = lambda x: "n/a" if x is None or x != x else f"{x:.1%}"  # noqa: E731
    lines = [
        "# Comparison – autoencoder vs baselines and ablations",
        "",
        "Generated by `fraud-ae compare` from `runs/*/metrics.json`. Thresholds are the "
        "0.9 % val-legit quantile of each run's own scores; test numbers use the untouched test "
        "split; intervals are Wilson 95 %. *Recall @ 1 % FPR* uses an oracle threshold on the "
        "test legit scores (ranking quality). Criterion: recall ≥ 85 % and FPR < 1 %.",
        "",
        "| Run | Model | Split | Score | Seed | ROC-AUC | PR-AUC | Val recall | Test FPR | "
        "Test recall [95 %] | Recall @ 1 % FPR [95 %] | Precision | Meets |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|:---:|",
    ]
    for r in rows:
        lo, hi = r["recall_ci"]
        lo1, hi1 = r["recall_at_target_ci"]
        lines.append(
            f"| `{r['run']}` | {r['model']} | {r['split']} | {r['score']} | {r['seed']} | "
            f"{r['roc_auc']:.4f} | {r['pr_auc']:.3f} | {pct(r['val_recall'])} | {r['fpr']:.3%} | "
            f"**{pct(r['recall'])}** [{pct(lo)}, {pct(hi)}] | {pct(r['recall_at_target'])} "
            f"[{pct(lo1)}, {pct(hi1)}] | {pct(r['precision'])} | "
            f"{'✅' if r['meets_target'] else '❌'} |"
        )
    fams = seed_families(rows)
    if fams:
        lines += [
            "",
            "## Seed variance",
            "",
            "| Family | Seeds | Val recall mean ± std | Test recall mean ± std | Test min – max |",
            "|---|---:|---:|---:|---|",
        ]
        for name, v in fams.items():
            va, te = np.asarray(v["val"]), np.asarray(v["test"])
            lines.append(
                f"| `{name}` | {len(te)} | {va.mean():.1%} ± {va.std(ddof=1):.1%} | "
                f"{te.mean():.1%} ± {te.std(ddof=1):.1%} | {te.min():.1%} – {te.max():.1%} |"
            )
    if figures:
        lines += ["", "## Figures", ""] + [f"* {k}: `{v}`" for k, v in figures.items()]
    return "\n".join(lines) + "\n"


def write_comparison(
    run_dirs: list[Path], out_md: Path, figures_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Build the comparison table (+ bar and seed-variance figures) from run directories."""
    from .plots import baseline_comparison, seed_variance

    rows = comparison_rows(run_dirs)
    if not rows:
        raise FileNotFoundError("no run directory with metrics.json among the given paths")
    figures: dict[str, str] = {}
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures["recall comparison"] = str(
            baseline_comparison(rows, figures_dir / "comparison_recall.png")
        )
        fams = seed_families(rows, min_size=3)
        if fams:
            figures["seed variance"] = str(seed_variance(fams, figures_dir / "seed_variance.png"))
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(comparison_markdown(rows, figures))
    return rows
