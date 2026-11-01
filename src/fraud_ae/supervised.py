"""Roadmap v2 Phase 9 – labelled frauds inside the model, evaluated out-of-fold.

Protocol (ARCHITECTURE.md, "Supervised protocol"): the frauds of the pooled test split are cut
into ``n_folds`` fixed folds (``folds.json``); the legit validation and test rows are cut into
the same number of folds. The fold-``k`` model is fitted on **all training legit rows** plus the
frauds of the other folds (optionally sub-sampled to a label budget), its threshold is the
``1 − target_fpr`` quantile of its scores on the legit validation rows, and it scores only the
fold-``k`` rows of validation and test. Scores are divided by the fold threshold, so the stored
scores of a run are out-of-fold and share the common threshold 1.0 (ADR 29 pattern), and the
usual ``evaluate`` yields the pooled out-of-fold recall over all frauds with its Wilson interval.
No fraud is ever scored by a model that saw it; the threshold never reads a fraud label.

Methods (``SupervisedConfig.methods``): ``logreg`` (class-weighted logistic regression), ``hgb``
(class-weighted ``HistGradientBoostingClassifier``), ``hgb_under`` (HGB on legit rows
under-sampled to ``undersample_ratio`` × frauds), ``mlp`` (PyTorch 31→20→10→1 trained on balanced
mini-batches), ``hybrid_logreg`` / ``hybrid_hgb`` (same classifiers on ``[x, AE residual, AE
score]`` from the legit-only AE in ``ae_run``), and the fine-tuned autoencoders ``neg_ae`` /
``sad_ae`` from :mod:`fraud_ae.semi`.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn

from .config import SEMI_METHODS, Config, SupervisedConfig
from .data import Splits
from .evaluate import evaluate, save_metrics, wilson_interval
from .model import Autoencoder
from .scoring import residuals, save_scores
from .threshold import ThresholdResult, fpr_at, quantile_threshold, save_threshold
from .train import SUMMARY_FILE, seed_everything, snapshot_config

FOLDS_FILE = "folds.json"


def needs_ae(method: str) -> bool:
    """Methods that use the legit-only autoencoder from ``supervised.ae_run``."""
    return method.startswith("hybrid") or method in SEMI_METHODS


FEW_LABEL_FILE = "few_label.json"
STRATEGY = "quantile_oof"  # recorded in threshold.json: per-fold val-legit quantiles, scores / thr
EPS = 1e-12


# --------------------------------------------------------------------------- folds


@dataclass(frozen=True)
class Folds:
    n_folds: int
    seed: int
    fraud_rows: np.ndarray  # raw (de-duplicated frame) row ids of the frauds, sorted
    fraud_fold: np.ndarray  # fold id per fraud, aligned with ``fraud_rows``
    val_fold: np.ndarray  # fold id per validation row
    test_fold: np.ndarray  # fold id per test row (frauds carry their ``fraud_fold``)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "seed": self.seed,
            "fraud_rows": self.fraud_rows.tolist(),
            "fraud_fold": self.fraud_fold.tolist(),
            "n_val": int(len(self.val_fold)),
            "n_test": int(len(self.test_fold)),
            "frauds_per_fold": np.bincount(self.fraud_fold, minlength=self.n_folds).tolist(),
        }


def make_folds(splits: Splits, n_folds: int, seed: int) -> Folds:
    """Balanced seeded folds. The fraud assignment is drawn first from a fresh generator, so it
    is identical for every legit split (data seed) of the pooled protocol: the frauds and their
    raw row ids are the same."""
    y_test = np.asarray(splits.y_test).astype(int)
    fraud_pos = np.flatnonzero(y_test == 1)
    if len(fraud_pos) < n_folds:
        raise ValueError(f"need at least {n_folds} frauds in the test split, got {len(fraud_pos)}")
    rng = np.random.default_rng(seed)
    fraud_fold = rng.permutation(np.arange(len(fraud_pos)) % n_folds)
    test_fold = rng.permutation(np.arange(len(y_test)) % n_folds)
    test_fold[fraud_pos] = fraud_fold
    val_fold = rng.permutation(np.arange(len(splits.y_val)) % n_folds)
    return Folds(
        n_folds=n_folds,
        seed=seed,
        fraud_rows=np.asarray(splits.idx_test)[fraud_pos],
        fraud_fold=fraud_fold,
        val_fold=val_fold,
        test_fold=test_fold,
    )


def save_folds(folds: Folds, run_dir: Path) -> Path:
    out = Path(run_dir) / FOLDS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(folds.to_dict(), indent=2))
    return out


def load_folds_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / FOLDS_FILE).read_text())


# --------------------------------------------------------------------------- features / models


def hybrid_features(ae: Autoencoder, X: np.ndarray) -> np.ndarray:
    """``[x, x − AE(x), mse(x)]`` – the legit-only autoencoder as a feature extractor."""
    R = residuals(ae, X)
    return np.hstack([np.asarray(X, dtype=np.float64), R, (R**2).mean(axis=1, keepdims=True)])


class _MLP(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 20), nn.ELU(), nn.Linear(20, 10), nn.ELU(), nn.Linear(10, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def fit_mlp(Xl: np.ndarray, Xf: np.ndarray, cfg: SupervisedConfig, seed: int) -> _MLP:
    """Small classifier with the AE encoder widths, trained on balanced mini-batches (480 legit +
    ``mlp_fraud_per_batch`` frauds drawn with replacement) for ``mlp_epochs`` passes over legit."""
    seed_everything(seed)
    Xl_t = torch.as_tensor(np.asarray(Xl), dtype=torch.float32)
    Xf_t = torch.as_tensor(np.asarray(Xf), dtype=torch.float32)
    model = _MLP(Xl_t.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    n_l, bf = len(Xl_t), cfg.mlp_fraud_per_batch
    bl = min(512 - bf, n_l)  # legit rows per batch (whole set when it is smaller)
    model.train()
    for _ in range(cfg.mlp_epochs):
        perm = torch.randperm(n_l, generator=gen)
        for start in range(0, n_l - bl + 1, bl):  # drop the last partial legit batch
            xl = Xl_t[perm[start : start + bl]]
            xf = Xf_t[torch.randint(len(Xf_t), (bf,), generator=gen)]
            x = torch.cat([xl, xf])
            y = torch.cat([torch.zeros(len(xl)), torch.ones(bf)])
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def mlp_probability(model: _MLP, X: np.ndarray) -> np.ndarray:
    xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    # sigmoid in float64 so that confident logits do not saturate to exactly 0 / 1 (ties)
    return torch.cat([torch.sigmoid(model(xb).double()) for xb in xt.split(8192)]).numpy()


def _sk_model(method: str, cfg: SupervisedConfig, seed: int):
    if method.endswith("logreg"):
        return LogisticRegression(C=cfg.logreg_c, class_weight="balanced", max_iter=2000)
    return HistGradientBoostingClassifier(
        max_iter=cfg.hgb_max_iter,
        learning_rate=cfg.hgb_learning_rate,
        max_leaf_nodes=cfg.hgb_max_leaf_nodes,
        class_weight=None if method == "hgb_under" else "balanced",
        early_stopping=False,
        random_state=seed,
    )


def fit_score(
    method: str,
    cfg: SupervisedConfig,
    Xl: np.ndarray,
    Xf: np.ndarray,
    X_eval: list[np.ndarray],
    seed: int,
    ae: Autoencoder | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Fit ``method`` on legit rows ``Xl`` + labelled frauds ``Xf`` and score each array in
    ``X_eval`` (higher = more fraudulent, strictly positive). Returns ``(scores, info)``."""
    if needs_ae(method) and ae is None:
        raise ValueError(f"{method} needs the legit-only autoencoder (supervised.ae_run)")
    if method in SEMI_METHODS:
        from .semi import finetune, semi_scores

        model, info = finetune(
            ae, cfg, Xl, Xf, seed, mode="margin" if method == "neg_ae" else "sad"
        )
        return [semi_scores(model, X, info) for X in X_eval], info
    if method.startswith("hybrid"):
        Xl, Xf = hybrid_features(ae, Xl), hybrid_features(ae, Xf)
        X_eval = [hybrid_features(ae, X) for X in X_eval]
    if method == "mlp":
        model = fit_mlp(Xl, Xf, cfg, seed)
        return [mlp_probability(model, X) for X in X_eval], {"n_features": int(Xl.shape[1])}
    if method == "hgb_under":
        rng = np.random.default_rng(seed)
        keep = rng.choice(
            len(Xl), size=min(len(Xl), cfg.undersample_ratio * len(Xf)), replace=False
        )
        Xl = np.asarray(Xl)[np.sort(keep)]
    if method not in ("logreg", "hgb", "hgb_under", "hybrid_logreg", "hybrid_hgb"):
        raise ValueError(f"unknown supervised method {method!r}")
    X = np.vstack([Xl, Xf]).astype(np.float64)
    y = np.r_[np.zeros(len(Xl), int), np.ones(len(Xf), int)]
    clf = _sk_model(method, cfg, seed).fit(X, y)
    return [clf.predict_proba(np.asarray(Xe, dtype=np.float64))[:, 1] for Xe in X_eval], {
        "n_features": int(X.shape[1]),
        "n_legit_fit": int(len(Xl)),
    }


# --------------------------------------------------------------------------- out-of-fold protocol


@dataclass(frozen=True)
class OofResult:
    scores_val: np.ndarray  # normalised (score / fold threshold); common threshold 1.0
    scores_test: np.ndarray
    thresholds: list[float]  # raw per-fold thresholds
    labels_per_fold: list[int]
    val_fpr_per_fold: list[float]
    wall_seconds: float
    info: dict[str, Any]


def run_oof(
    method: str,
    cfg: Config,
    splits: Splits,
    folds: Folds,
    budget: int | None,
    ae: Autoencoder | None = None,
) -> OofResult:
    """Fit one model per fold and assemble the out-of-fold, threshold-normalised scores."""
    sv, seed = cfg.supervised, cfg.train.seed
    y_test = np.asarray(splits.y_test).astype(int)
    fraud_pos = np.flatnonzero(y_test == 1)
    scores_val = np.full(len(splits.y_val), np.nan)
    scores_test = np.full(len(y_test), np.nan)
    thresholds, labels, val_fprs, info = [], [], [], {}
    t0 = time.perf_counter()
    for k in range(folds.n_folds):
        train_frauds = fraud_pos[folds.fraud_fold != k]
        if budget is not None and budget < len(train_frauds):
            rng = np.random.default_rng([seed, budget, k])
            train_frauds = np.sort(rng.choice(train_frauds, size=budget, replace=False))
        Xf = splits.X_test[train_frauds]
        val_k, test_k = folds.val_fold == k, folds.test_fold == k
        (s_val, s_test_k), info = fit_score(
            method, sv, splits.X_train, Xf, [splits.X_val, splits.X_test[test_k]], seed + k, ae
        )
        if not (np.isfinite(s_val).all() and np.isfinite(s_test_k).all()):
            raise ValueError(f"{method}: non-finite scores in fold {k} (training diverged?)")
        s_val_legit = s_val[np.asarray(splits.y_val) == 0]
        thr = max(quantile_threshold(s_val_legit, cfg.threshold.target_fpr), EPS)
        val_fpr = fpr_at(s_val_legit, thr)
        if not 0.5 * cfg.threshold.target_fpr <= val_fpr <= 1.5 * cfg.threshold.target_fpr:
            warnings.warn(
                f"{method} fold {k}: val FPR {val_fpr:.3%} far from the target "
                f"{cfg.threshold.target_fpr:.1%} (tied or saturated scores?)",
                stacklevel=2,
            )
        scores_val[val_k] = s_val[val_k] / thr
        scores_test[test_k] = s_test_k / thr
        thresholds.append(float(thr))
        labels.append(int(len(train_frauds)))
        val_fprs.append(val_fpr)
    assert not np.isnan(scores_val).any() and not np.isnan(scores_test).any()
    return OofResult(
        scores_val, scores_test, thresholds, labels, val_fprs, time.perf_counter() - t0, info
    )


def write_run(
    cfg: Config,
    splits: Splits,
    folds: Folds,
    run_dir: Path,
    method: str,
    result: OofResult,
    few_label: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    """Run-directory contract for an out-of-fold method (no model.pt: one model per fold)."""
    from .plots import recall_vs_fpr, roc_pr_curves, score_histogram

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(cfg, run_dir)
    save_folds(folds, run_dir)
    save_scores(run_dir, "val", result.scores_val, splits.y_val)
    save_scores(run_dir, "test", result.scores_test, splits.y_test)
    legit_val = result.scores_val[np.asarray(splits.y_val) == 0]
    th = ThresholdResult(
        strategy=STRATEGY,
        target_fpr=cfg.threshold.target_fpr,
        value=1.0,
        val_fpr=fpr_at(legit_val, 1.0),
        val_recall=None,
        score_method=method,
        n_val_legit=int(len(legit_val)),
        n_val_fraud=int((np.asarray(splits.y_val) == 1).sum()),
        segments={"per_fold": result.thresholds, "val_fpr_per_fold": result.val_fpr_per_fold},
    )
    save_threshold(th, run_dir)
    metrics = evaluate(result.scores_test, splits.y_test, 1.0, cfg.eval, seed=cfg.train.seed)
    metrics["score_method"] = method
    metrics["threshold_strategy"] = STRATEGY
    metrics["protocol"] = {
        "n_folds": folds.n_folds,
        "fold_seed": folds.seed,
        "labels_per_fold": result.labels_per_fold,
        "thresholds_per_fold": result.thresholds,
        "ae_run": cfg.supervised.ae_run if needs_ae(method) else None,
    }
    metrics["few_label"] = few_label
    save_metrics(metrics, run_dir)
    (run_dir / SUMMARY_FILE).write_text(
        json.dumps(
            {
                "model": label,
                "kind": "supervised",
                "method": method,
                "n_train": int(len(splits.X_train)),
                "labels_per_fold": result.labels_per_fold,
                "seed": cfg.train.seed,
                "wall_seconds": result.wall_seconds,
                "config": asdict(cfg.supervised),
                **{k: v for k, v in result.info.items() if isinstance(v, int | float | str)},
            },
            indent=2,
        )
    )
    (run_dir / FEW_LABEL_FILE).write_text(
        json.dumps({"method": method, "model": label, "budgets": few_label}, indent=2)
    )
    figs = run_dir / "figures"
    score_histogram(
        result.scores_val,
        splits.y_val,
        1.0,
        figs / "score_histogram_val.png",
        title=f"Validation scores ({method}, out-of-fold, normalised)",
    )
    roc_pr_curves(result.scores_test, splits.y_test, figs / "roc_pr_test.png")
    recall_vs_fpr(
        result.scores_test,
        splits.y_test,
        (cfg.eval.target_fpr, cfg.eval.target_recall),
        figs / "recall_vs_fpr_test.png",
        threshold=1.0,
    )
    return metrics


def few_label_row(
    budget: int | str, result: OofResult, splits: Splits, cfg: Config
) -> dict[str, Any]:
    y = np.asarray(splits.y_test).astype(int)
    alarm = result.scores_test > 1.0
    tp, fp = int((alarm & (y == 1)).sum()), int((alarm & (y == 0)).sum())
    n_f, n_l = int((y == 1).sum()), int((y == 0).sum())
    lo, hi = wilson_interval(tp, n_f)
    fpr = fp / n_l
    return {
        "budget": budget,
        "labels_per_fold": result.labels_per_fold,
        "tp": tp,
        "fp": fp,
        "recall": tp / n_f,
        "recall_ci_wilson": [lo, hi],
        "fpr": fpr,
        "meets_target": bool(tp / n_f >= cfg.eval.target_recall and fpr < cfg.eval.target_fpr),
        "wall_seconds": result.wall_seconds,
    }


LABELS = {
    "logreg": "LogReg (balanced)",
    "hgb": "HGB (balanced)",
    "hgb_under": "HGB (under-sampled legit)",
    "mlp": "MLP 31-20-10-1 (balanced batches)",
    "hybrid_logreg": "LogReg on [x, residual, AE score]",
    "hybrid_hgb": "HGB on [x, residual, AE score]",
    "neg_ae": "AE + margin loss on frauds",
    "sad_ae": "AE + Deep-SAD latent term",
}


def run_supervised(
    cfg: Config,
    splits: Splits,
    runs_root: Path,
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
    ae: Autoencoder | None,
    suffix: str = "",
    prefix: str = "sup_",
) -> dict[str, dict[str, Any]]:
    """Every method at every budget (plus all labels); one run directory per method holding the
    all-label contract and ``few_label.json``. Returns ``{run_name: metrics}``."""
    folds = make_folds(splits, cfg.supervised.n_folds, cfg.supervised.fold_seed)
    # labelled frauds available to the fold with the fewest: a budget must stay below it so
    # that every fold model sees exactly `budget` labels
    n_train_frauds = int(len(folds.fraud_fold) - np.bincount(folds.fraud_fold).max())
    out: dict[str, dict[str, Any]] = {}
    for method in methods:
        rows = []
        for b in sorted(budgets):
            if b >= n_train_frauds:
                continue  # covered by "all"
            rows.append(few_label_row(b, run_oof(method, cfg, splits, folds, b, ae), splits, cfg))
        full = run_oof(method, cfg, splits, folds, None, ae)
        rows.append(few_label_row("all", full, splits, cfg))
        name = f"{prefix}{method}{suffix}"
        out[name] = write_run(
            cfg, splits, folds, runs_root / name, method, full, rows, LABELS.get(method, method)
        )
    return out


# --------------------------------------------------------------------------- few-label table


def few_label_markdown(
    runs: list[Path], reference: dict[str, Any] | None = None
) -> tuple[str, list[dict]]:
    rows = []
    for r in runs:
        p = Path(r) / FEW_LABEL_FILE
        if p.exists():
            rows.append({"run": Path(r).name, **json.loads(p.read_text())})
    if not rows:
        raise FileNotFoundError("no run directory with few_label.json among the given paths")
    budgets = sorted({b["budget"] for r in rows for b in r["budgets"] if b["budget"] != "all"}) + [
        "all"
    ]
    pct = lambda x: f"{x:.1%}"  # noqa: E731
    lines = [
        "# Few-label curve – out-of-fold recall vs labelled frauds per training fold",
        "",
        "Pooled out-of-fold recall over all test frauds at the per-fold val-legit quantile "
        "threshold (FPR < 1 % by construction); Wilson 95 % intervals; `all` = every fraud "
        "outside the evaluated fold. Generated by `fraud-ae few-label` from "
        "`runs/*/few_label.json`.",
        "",
        "| Method | "
        + " | ".join(f"{b} labels" if b != "all" else "all labels" for b in budgets)
        + " |",
        "|---|" + "---|" * len(budgets),
    ]
    for r in rows:
        by = {b["budget"]: b for b in r["budgets"]}
        cells = []
        for b in budgets:
            if b in by:
                e = by[b]
                lo, hi = e["recall_ci_wilson"]
                flag = " ✅" if e["meets_target"] else ""
                cells.append(
                    f"**{pct(e['recall'])}** [{pct(lo)}, {pct(hi)}] @ {e['fpr']:.2%}{flag}"
                )
            else:
                cells.append("–")
        lines.append(f"| `{r['run']}` – {r['model']} | " + " | ".join(cells) + " |")
    if reference:
        lo, hi = reference["recall_ci_wilson"]
        lines += [
            "",
            f"Legit-only reference `{reference['run']}`: {pct(reference['recall'])} "
            f"[{pct(lo)}, {pct(hi)}] at FPR {reference['fpr']:.2%} (no labels).",
        ]
    return "\n".join(lines) + "\n", rows
