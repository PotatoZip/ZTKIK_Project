"""All figures (matplotlib, headless ``Agg``). Each function saves a PNG and returns its path.

Figures referenced by the report (ARCHITECTURE.md §3.6): training curves, score histogram with
threshold, ROC, PR, recall-vs-FPR with the (1 %, 85 %) target marker, threshold sweep,
per-feature residuals (explainability), seed variance, baseline comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Two-class categorical palette, fixed order: legit = blue, fraud = orange.
COLOR_LEGIT = "#2a78d6"
COLOR_FRAUD = "#eb6834"
COLOR_MISSED = "#8a5bc4"  # third class in the Phase 7 diagnosis: frauds below the threshold
COLOR_TEXT = "#52514e"
COLOR_GRID = "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLOR_GRID,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
        "legend.frameon": False,
        "font.size": 9,
    }
)


def _save(fig: plt.Figure, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- Phase 1: EDA


def eda_class_ratio(df: pd.DataFrame, out: Path) -> Path:
    """Horizontal bars with counts and shares; log x-axis so the fraud bar is visible."""
    counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.barh(["legit", "fraud"], counts.to_numpy(), color=[COLOR_LEGIT, COLOR_FRAUD], height=0.55)
    ax.set_xscale("log")
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    for i, c in enumerate(counts.to_numpy()):
        ax.annotate(
            f"{c:,}  ({c / total:.3%})",
            (c, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=COLOR_TEXT,
        )
    ax.set_xlim(right=counts.max() * 30)
    ax.set_xlabel("transactions (log scale)")
    ax.set_title(f"Class ratio – {total:,} transactions after de-duplication")
    ax.invert_yaxis()
    return _save(fig, out)


def eda_amount_distribution(df: pd.DataFrame, out: Path) -> Path:
    """Density of log1p(Amount) per class – the two populations overlap heavily."""
    legit = np.log1p(df.loc[df["Class"] == 0, "Amount"].to_numpy())
    fraud = np.log1p(df.loc[df["Class"] == 1, "Amount"].to_numpy())
    bins = np.linspace(0, max(legit.max(), fraud.max()) + 0.1, 50)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(
        legit,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color=COLOR_LEGIT,
        label=f"legit (median {np.expm1(np.median(legit)):.2f})",
    )
    ax.hist(
        fraud,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        color=COLOR_FRAUD,
        label=f"fraud (median {np.expm1(np.median(fraud)):.2f})",
    )
    ax.set_xlabel("log1p(Amount)")
    ax.set_ylabel("density")
    ax.set_title("Amount distribution by class (model input: log1p(Amount))")
    ax.legend(loc="upper right")
    return _save(fig, out)


def eda_fraud_rate_by_hour(df: pd.DataFrame, out: Path) -> Path:
    """Two stacked panels sharing the hour axis: traffic volume and fraud rate (one axis each,
    no dual-axis chart)."""
    from .data import hour_of_day  # avoid import cycle at module load

    hours = np.floor(hour_of_day(df["Time"])).astype(int)
    counts = np.bincount(hours, minlength=24)
    frauds = np.bincount(hours, weights=df["Class"].to_numpy(), minlength=24)
    rate = np.divide(frauds, counts, out=np.zeros(24), where=counts > 0)
    overall = df["Class"].mean()
    x = np.arange(24)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]}
    )
    ax1.bar(x, counts, color=COLOR_LEGIT, width=0.8)
    ax1.set_ylabel("transactions")
    ax1.set_title("Traffic and fraud rate by hour of day (hour = Time mod 24 h)")

    ax2.bar(x, rate * 100, color=COLOR_FRAUD, width=0.8)
    ax2.axhline(overall * 100, color=COLOR_TEXT, linewidth=1, linestyle="--")
    ax2.annotate(
        f"overall {overall:.3%}",
        (23.4, overall * 100),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        color=COLOR_TEXT,
    )
    ax2.set_ylabel("fraud rate [%]")
    ax2.set_xlabel("hour of day")
    ax2.set_xticks(x)
    return _save(fig, out)


def eda_figures(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """All Phase 1 figures; returns ``{name: path}``."""
    out_dir = Path(out_dir)
    return {
        "class_ratio": eda_class_ratio(df, out_dir / "eda_class_ratio.png"),
        "amount_distribution": eda_amount_distribution(df, out_dir / "eda_amount_distribution.png"),
        "fraud_rate_by_hour": eda_fraud_rate_by_hour(df, out_dir / "eda_fraud_rate_by_hour.png"),
    }


# --------------------------------------------------------------------------- later phases


def training_curves(history_csv: Path, out: Path) -> Path:
    """Train vs val-legit MSE per epoch (log y), best epoch marked; LR steps in a lower panel."""
    h = pd.read_csv(history_csv)
    best = int(h["val_legit_mse"].idxmin())
    fig, (ax, ax_lr) = plt.subplots(
        2, 1, figsize=(6.5, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(h["epoch"], h["train_mse"], color=COLOR_LEGIT, linewidth=2, label="train MSE")
    ax.plot(h["epoch"], h["val_legit_mse"], color=COLOR_FRAUD, linewidth=2, label="val-legit MSE")
    ax.scatter(
        [h.loc[best, "epoch"]], [h.loc[best, "val_legit_mse"]], s=40, zorder=3, color=COLOR_FRAUD
    )
    ax.annotate(
        f"best epoch {int(h.loc[best, 'epoch'])}: {h.loc[best, 'val_legit_mse']:.4f}",
        (h.loc[best, "epoch"], h.loc[best, "val_legit_mse"]),
        xytext=(6, 8),
        textcoords="offset points",
        color=COLOR_TEXT,
    )
    ax.set_yscale("log")
    ax.set_ylabel("MSE (standardized units)")
    ax.set_title(f"Training curves – {len(h)} epochs, early stopping on val-legit MSE")
    ax.legend(loc="upper right")
    ax_lr.step(h["epoch"], h["lr"], where="post", color=COLOR_TEXT, linewidth=1.5)
    ax_lr.set_yscale("log")
    ax_lr.set_ylabel("lr")
    ax_lr.set_xlabel("epoch")
    return _save(fig, out)


def score_histogram(
    scores: np.ndarray, y: np.ndarray, threshold: float, out: Path, title: str = ""
) -> Path:
    """Density of log10(score) per class with the threshold and the resulting legit FPR."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    log_s = np.log10(np.maximum(scores, 1e-12))
    legit, fraud = log_s[y == 0], log_s[y == 1]
    bins = np.linspace(log_s.min(), log_s.max() + 1e-9, 60)
    fpr = float((scores[y == 0] > threshold).mean()) if (y == 0).any() else float("nan")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(
        legit,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color=COLOR_LEGIT,
        label=f"legit (n={len(legit):,})",
    )
    if fraud.size:
        rec = float((scores[y == 1] > threshold).mean())
        ax.hist(
            fraud,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=COLOR_FRAUD,
            label=f"fraud (n={len(fraud):,}, recall {rec:.1%})",
        )
    ax.axvline(
        np.log10(max(threshold, 1e-12)),
        color=COLOR_TEXT,
        linestyle="--",
        linewidth=1.2,
        label=f"threshold {threshold:.3g} (legit FPR {fpr:.2%})",
    )
    ax.set_xlabel("log10(anomaly score)")
    ax.set_ylabel("density")
    ax.set_title(title or "Anomaly-score distribution and threshold")
    ax.legend(loc="upper right")
    return _save(fig, out)


def roc_pr_curves(scores: np.ndarray, y: np.ndarray, out: Path) -> Path:
    """ROC (FPR on a log axis, since only FPR < 1 % matters) and precision-recall curves."""
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    scores, y = np.asarray(scores, dtype=float), np.asarray(y)
    fpr, tpr, _ = roc_curve(y, scores)
    prec, rec, _ = precision_recall_curve(y, scores)
    auc, ap = roc_auc_score(y, scores), average_precision_score(y, scores)
    n_legit = int((y == 0).sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot(np.maximum(fpr, 0.5 / n_legit), tpr, color=COLOR_FRAUD, linewidth=2)
    ax1.set_xscale("log")
    ax1.set_xlim(0.5 / n_legit, 1)
    ax1.set_ylim(0, 1.02)
    ax1.axvline(0.01, color=COLOR_TEXT, linestyle="--", linewidth=1)
    ax1.annotate(
        "FPR 1 %", (0.01, 0.03), xytext=(4, 0), textcoords="offset points", color=COLOR_TEXT
    )
    ax1.set_xlabel("false-positive rate (log)")
    ax1.set_ylabel("recall (true-positive rate)")
    ax1.set_title(f"ROC – AUC {auc:.4f}")
    ax1.grid(axis="x")

    ax2.plot(rec, prec, color=COLOR_FRAUD, linewidth=2)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_title(f"Precision–recall – AP {ap:.4f}")
    fig.tight_layout()
    return _save(fig, out)


def recall_vs_fpr(
    scores: np.ndarray,
    y: np.ndarray,
    target: tuple[float, float],
    out: Path,
    threshold: float | None = None,
) -> Path:
    """Recall as a function of FPR (log axis) with the (target FPR, target recall) marker; the
    shaded box is the region where the criterion holds. ``threshold`` marks the Phase 3
    operating point."""
    from sklearn.metrics import roc_curve

    scores, y = np.asarray(scores, dtype=float), np.asarray(y)
    target_fpr, target_recall = target
    fpr, tpr, _ = roc_curve(y, scores)
    n_legit = int((y == 0).sum())
    x_min = 0.5 / n_legit

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axvspan(x_min, target_fpr, ymin=target_recall, ymax=1.0, color=COLOR_LEGIT, alpha=0.10, lw=0)
    ax.step(
        np.maximum(fpr, x_min),
        tpr,
        where="post",
        color=COLOR_FRAUD,
        linewidth=2,
        label="recall vs FPR (test)",
    )
    ax.axvline(target_fpr, color=COLOR_TEXT, linestyle="--", linewidth=1)
    ax.axhline(target_recall, color=COLOR_TEXT, linestyle="--", linewidth=1)
    ax.scatter(
        [target_fpr],
        [target_recall],
        marker="x",
        s=60,
        color=COLOR_TEXT,
        zorder=4,
        label=f"target: recall {target_recall:.0%} @ FPR {target_fpr:.0%}",
    )
    if threshold is not None:
        legit, fraud = scores[y == 0], scores[y == 1]
        op_fpr = float((legit > threshold).mean())
        op_rec = float((fraud > threshold).mean()) if fraud.size else float("nan")
        ax.scatter(
            [max(op_fpr, x_min)],
            [op_rec],
            s=50,
            color=COLOR_LEGIT,
            zorder=5,
            label=f"val-chosen threshold: recall {op_rec:.1%} @ FPR {op_fpr:.2%}",
        )
    ax.set_xscale("log")
    ax.set_xlim(x_min, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("false-positive rate on legitimate test rows (log)")
    ax.set_ylabel("recall on test frauds")
    ax.set_title("Recall vs FPR – criterion region shaded")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    return _save(fig, out)


def threshold_sweep(scores: np.ndarray, y: np.ndarray, threshold: float, out: Path) -> Path:
    """Recall, FPR and precision as functions of the threshold (log x), chosen threshold marked."""
    scores, y = np.asarray(scores, dtype=float), np.asarray(y)
    legit, fraud = scores[y == 0], scores[y == 1]
    lo = max(np.percentile(scores, 1), 1e-12)
    grid = np.logspace(np.log10(lo), np.log10(scores.max()), 300)
    legit_s, fraud_s = np.sort(legit), np.sort(fraud)
    fp = legit_s.size - np.searchsorted(legit_s, grid, side="right")
    tp = fraud_s.size - np.searchsorted(fraud_s, grid, side="right")
    fpr = fp / max(legit_s.size, 1)
    rec = tp / max(fraud_s.size, 1)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(grid), where=(tp + fp) > 0)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(grid, rec, color=COLOR_FRAUD, linewidth=2, label="recall")
    ax.plot(grid, fpr, color=COLOR_LEGIT, linewidth=2, label="FPR")
    ax.plot(grid, prec, color=COLOR_TEXT, linewidth=1.5, linestyle=":", label="precision")
    ax.axvline(
        threshold, color=COLOR_TEXT, linestyle="--", linewidth=1, label=f"threshold {threshold:.3g}"
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("threshold on the anomaly score (log)")
    ax.set_ylabel("rate")
    ax.set_title("Threshold sweep on the test set")
    ax.legend(loc="center right")
    return _save(fig, out)


def per_feature_residuals(
    residuals: np.ndarray, y: np.ndarray, feature_names: tuple[str, ...], out: Path
) -> Path:
    """Mean squared residual per input feature, legit vs fraud, sorted by the fraud/legit ratio
    (which features the autoencoder fails to reconstruct for frauds)."""
    R = np.asarray(residuals, dtype=float)
    y = np.asarray(y)
    legit = (R[y == 0] ** 2).mean(axis=0)
    fraud = (R[y == 1] ** 2).mean(axis=0) if (y == 1).any() else np.zeros(R.shape[1])
    order = np.argsort(fraud / np.maximum(legit, 1e-12))[::-1]
    names = [feature_names[i] for i in order]
    pos = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(6.5, 0.22 * len(names) + 1.4))
    ax.barh(pos - 0.2, legit[order], height=0.4, color=COLOR_LEGIT, label="legit")
    ax.barh(pos + 0.2, fraud[order], height=0.4, color=COLOR_FRAUD, label="fraud")
    ax.set_yticks(pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("mean squared residual (standardized units, log)")
    ax.set_title("Per-feature reconstruction error on the test set (sorted by fraud/legit ratio)")
    ax.legend(loc="lower right")
    return _save(fig, out)


def baseline_comparison(rows: list[dict], out: Path, target_recall: float = 0.85) -> Path:
    """Horizontal bars: test recall at the val-chosen threshold with Wilson 95 % whiskers, one
    row per run (``rows`` from ``baselines.comparison_rows``); the oracle recall at the target
    FPR is a hollow marker; the criterion is the dashed line."""
    rows = sorted(rows, key=lambda r: r["recall"])
    n = len(rows)
    y = np.arange(n)
    rec = np.array([r["recall"] for r in rows])
    lo = np.array([r["recall_ci"][0] for r in rows])
    hi = np.array([r["recall_ci"][1] for r in rows])
    oracle = np.array([r["recall_at_target"] for r in rows])
    colors = [
        COLOR_TEXT if r["model"].startswith(("PCA", "Isolation")) else COLOR_LEGIT for r in rows
    ]

    fig, ax = plt.subplots(figsize=(7.5, 0.32 * n + 1.6))
    ax.barh(y, rec, color=colors, height=0.62, alpha=0.9)
    ax.errorbar(rec, y, xerr=[rec - lo, hi - rec], fmt="none", ecolor=COLOR_TEXT, capsize=3, lw=1)
    ax.scatter(
        oracle,
        y,
        marker="o",
        facecolors="white",
        edgecolors=COLOR_FRAUD,
        s=28,
        zorder=4,
        label="recall @ 1 % FPR (oracle threshold)",
    )
    ax.axvline(
        target_recall,
        color=COLOR_FRAUD,
        linestyle="--",
        lw=1.2,
        label=f"target {target_recall:.0%}",
    )
    for yi, r in zip(y, rows, strict=True):
        ax.annotate(
            f"{r['recall']:.1%}  (FPR {r['fpr']:.2%})",
            (r["recall_ci"][1], yi),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=COLOR_TEXT,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['run']}  ·  {r['model']}" for r in rows], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("test recall at the val-chosen threshold (Wilson 95 %)")
    ax.set_title("Autoencoder (blue) vs baselines (grey) on the test split")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8)
    return _save(fig, out)


def seed_variance(families: dict[str, dict[str, list[float]]], out: Path) -> Path:
    """Box + points of val and test recall across seeds, one group per run family."""
    names = list(families)
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 3.5, 3.6))
    pos = np.arange(len(names))
    for off, key, color, label in (
        (-0.18, "val", COLOR_LEGIT, "val recall"),
        (0.18, "test", COLOR_FRAUD, "test recall"),
    ):
        data = [families[n][key] for n in names]
        keep = [i for i, d in enumerate(data) if len(d)]
        if not keep:
            continue
        data = [data[i] for i in keep]
        ax.boxplot(
            data,
            positions=pos[keep] + off,
            widths=0.3,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.25, "edgecolor": color},
            medianprops={"color": color, "lw": 2},
            whiskerprops={"color": color},
            capprops={"color": color},
            showfliers=False,
        )
        for p_, d in zip(pos[keep] + off, data, strict=True):
            ax.scatter(
                np.full(len(d), p_),
                d,
                color=color,
                s=18,
                zorder=3,
                label=label if p_ == pos[keep[0]] + off else None,
            )
    ax.axhline(0.85, color=COLOR_TEXT, linestyle="--", lw=1)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{n} ({len(families[n]['test'])} seeds)" for n in names], fontsize=8)
    ax.set_ylabel("recall at the val-chosen threshold")
    ax.set_title("Seed variance – validation vs test recall")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out)


# --------------------------------------------------------------------------- Phase 7: diagnosis


def missed_frauds(d, out: Path) -> Path:
    """Four panels on the test split: score densities (legit / detected / missed frauds) with the
    threshold, hour-of-day counts of detected vs missed frauds, amount distributions, and the
    mean raw value of the most separating features per group. ``d`` is a
    :class:`fraud_ae.diagnose.DiagnosisArrays`."""
    legit = d.y == 0
    groups = (
        ("legit", legit, COLOR_LEGIT),
        ("detected fraud", d.detected, COLOR_FRAUD),
        ("missed fraud", d.missed, COLOR_MISSED),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8))
    (ax_s, ax_h), (ax_a, ax_f) = axes

    log_s = np.log10(np.maximum(d.scores, 1e-12))
    bins = np.linspace(log_s.min(), log_s.max() + 1e-9, 60)
    for label, m, color in groups:
        if not m.any():
            continue
        kw = (
            {"histtype": "stepfilled", "alpha": 0.35}
            if label == "legit"
            else {"histtype": "step", "linewidth": 2}
        )
        ax_s.hist(
            log_s[m], bins=bins, density=True, color=color, label=f"{label} (n={m.sum():,})", **kw
        )
    ax_s.axvline(
        np.log10(max(d.threshold, 1e-12)), color=COLOR_TEXT, ls="--", lw=1.2, label="threshold"
    )
    ax_s.set_xlabel("log10(anomaly score)")
    ax_s.set_ylabel("density")
    ax_s.set_title("Scores: missed frauds sit inside the legit distribution")
    ax_s.legend(loc="upper right", fontsize=8)

    x = np.arange(24)
    for off, (label, m, color) in zip((-0.2, 0.2), groups[1:], strict=True):
        counts = np.bincount(np.floor(d.hours[m]).astype(int) % 24, minlength=24)
        ax_h.bar(x + off, counts, width=0.4, color=color, label=label)
    ax_h.set_xticks(x[::2])
    ax_h.set_xlabel("hour of day")
    ax_h.set_ylabel("frauds")
    ax_h.set_title("Hour of day")
    ax_h.legend(loc="upper right", fontsize=8)

    log_a = np.log10(1 + d.amount)
    bins_a = np.linspace(0, log_a.max() + 1e-9, 40)
    for label, m, color in groups:
        if not m.any():
            continue
        kw = (
            {"histtype": "stepfilled", "alpha": 0.35}
            if label == "legit"
            else {"histtype": "step", "linewidth": 2}
        )
        ax_a.hist(log_a[m], bins=bins_a, density=True, color=color, label=label, **kw)
    ax_a.set_xlabel("log10(1 + Amount)")
    ax_a.set_ylabel("density")
    ax_a.set_title("Amount")
    ax_a.legend(loc="upper right", fontsize=8)

    tf = d.top_features
    pos = np.arange(len(tf["features"]))
    for off, key, (label, _, color) in zip(
        (-0.27, 0.0, 0.27), ("legit_mean", "detected_mean", "missed_mean"), groups, strict=True
    ):
        ax_f.bar(pos + off, tf[key], width=0.26, color=color, label=label)
    ax_f.axhline(0, color=COLOR_TEXT, lw=0.8)
    ax_f.set_xticks(pos)
    ax_f.set_xticklabels(tf["features"])
    ax_f.set_ylabel("mean raw value")
    ax_f.set_title("Most separating features (detected vs legit)")
    ax_f.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------- Phase 9: few-label

# Fixed-order series palette for up to eight methods (identity never cycles).
SERIES_COLORS = (
    COLOR_LEGIT,
    COLOR_FRAUD,
    COLOR_MISSED,
    "#2f9e6e",
    "#c0398a",
    "#8a7a2a",
    "#3bb3c3",
    "#9a4b2f",
)


def few_label_curve(
    rows: list[dict], out: Path, reference: dict | None = None, target_recall: float = 0.85
) -> Path:
    """Out-of-fold recall (Wilson whiskers) vs labelled frauds per training fold, one line per
    method; ``rows`` are the ``few_label.json`` payloads with a ``run`` key. ``reference`` draws
    the legit-only AE as a horizontal line."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    budgets: set[float] = set()
    for i, r in enumerate(rows):
        pts = sorted(
            ((float(np.mean(b["labels_per_fold"])), b) for b in r["budgets"]), key=lambda t: t[0]
        )
        x = np.array([p[0] for p in pts])
        rec = np.array([p[1]["recall"] for p in pts])
        lo = np.array([p[1]["recall_ci_wilson"][0] for p in pts])
        hi = np.array([p[1]["recall_ci_wilson"][1] for p in pts])
        budgets.update(x.tolist())
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.errorbar(
            x,
            rec,
            yerr=[rec - lo, hi - rec],
            color=color,
            lw=2,
            marker="o",
            ms=5,
            capsize=2,
            label=r.get("model", r["run"]),
        )
    if reference:
        ax.axhline(
            reference["recall"],
            color=COLOR_TEXT,
            lw=1.2,
            ls=":",
            label=f"legit-only AE ({reference['recall']:.1%})",
        )
    ax.axhline(target_recall, color=COLOR_TEXT, lw=1, ls="--", label="target 85 %")
    ax.set_xscale("log")
    ticks = sorted(budgets)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("labelled frauds per training fold (log scale)")
    ax.set_ylabel("out-of-fold recall at FPR < 1 %")
    ax.set_ylim(0, 1.0)
    ax.set_title("What labels buy – recall vs label budget")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    return _save(fig, out)
