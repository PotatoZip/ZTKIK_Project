"""Roadmap v2 Phase 10 – robustness and deployment realism (ARCHITECTURE.md, "Robustness").

Three measurements on the Phase 9 winners (``hgb`` supervised, ``neg_ae`` AE-based) and the
legit-only AE:

* **Rolling-origin validation with a rolling threshold.** Models trained on the first 60 % of
  legitimate traffic (the Phase 5 temporal split; the supervised and fine-tuned models also see
  the frauds of that window) score the remaining 40 % in time order. The 40 % is cut into
  ``n_windows`` windows of equal legit count; window 0 is the calibration window, windows 1…
  are evaluated with (a) a *fixed* threshold – the 0.9 % legit quantile of window 0, the v1
  behaviour that drifted to 0.53 % FPR – and (b) a *rolling* threshold – the same quantile
  re-estimated on the previous window (legit rows only, i.e. assuming labels of the previous
  window are known; and label-free on all rows of the previous window). Target: FPR within
  ``band`` on every window.
* **Variance.** Model-seed and data-seed families of the winners from Phase 9 run directories.
* **Evasion probe.** Projected gradient descent on the inputs of the frauds the legit-only AE
  detects: for each L2 budget ε the attacker minimises the MSE score inside the ball ‖δ‖₂ ≤ ε;
  the share of detected frauds pushed below the threshold per budget, the median minimal budget,
  and the features that move. Budgets are in standardised feature units; the probe is an upper
  bound on attacker power because V1–V28 are PCA components that cannot be set directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from .config import Config, DataConfig
from .data import (
    FEATURES_FILE,
    SCALER_FILE,
    drop_duplicates,
    engineer_features,
    load_raw,
    load_report,
    load_splits,
)
from .evaluate import load_metrics, wilson_interval
from .model import Autoencoder
from .threshold import quantile_threshold
from .train import mse_per_sample

ROBUSTNESS_FILE = "robustness.json"
DEFAULT_BUDGETS = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
_SEED = re.compile(r"_seed(\d+)$")
_DSEED = re.compile(r"_dseed(\d+)$")


# --------------------------------------------------------------------------- temporal frame


@dataclass(frozen=True)
class TemporalFrame:
    """Rows of the temporal split in time order: the training legit rows, the frauds of the
    training window (dropped by the split, recovered from the raw CSV), and the evaluation rows
    (val + test, legit and fraud) with their ``Time``."""

    X_train_legit: np.ndarray
    X_train_fraud: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    time_eval: np.ndarray
    feature_names: tuple[str, ...]


def temporal_frame(processed_dir: Path, root: Path) -> TemporalFrame:
    processed_dir, root = Path(processed_dir), Path(root)
    report = load_report(processed_dir)
    if report.get("split_strategy") != "temporal":
        raise ValueError(f"{processed_dir} is not a temporal split (data.split = 'temporal')")
    dcfg = DataConfig(
        **{**report["config"], "legit_fractions": tuple(report["config"]["legit_fractions"])}
    )
    raw = load_raw(root / dcfg.raw_csv)
    df = drop_duplicates(raw)[0] if dcfg.drop_duplicates else raw.reset_index(drop=True)
    X_all = (
        joblib.load(processed_dir / SCALER_FILE)
        .transform(engineer_features(df, dcfg).to_numpy(dtype=np.float64))
        .astype(np.float32)
    )
    names = tuple(json.loads((processed_dir / FEATURES_FILE).read_text()))
    y_all = df["Class"].to_numpy(dtype=int)
    time_all = df["Time"].to_numpy(dtype=np.float64)
    splits = load_splits(processed_dir)
    t_train_end = report["time_windows"]["train"][1]
    train_fraud = np.flatnonzero((y_all == 1) & (time_all <= t_train_end))
    eval_idx = np.r_[splits.idx_val, splits.idx_test]
    eval_idx = eval_idx[np.lexsort((eval_idx, time_all[eval_idx]))]  # stable time order
    return TemporalFrame(
        X_train_legit=X_all[splits.idx_train],
        X_train_fraud=X_all[train_fraud],
        X_eval=X_all[eval_idx],
        y_eval=y_all[eval_idx],
        time_eval=time_all[eval_idx],
        feature_names=names,
    )


# --------------------------------------------------------------------------- rolling threshold


def window_ids(y: np.ndarray, n_windows: int) -> np.ndarray:
    """Window id per row (rows already in time order): windows hold equal numbers of legit rows;
    frauds join the window of the surrounding legit traffic."""
    y = np.asarray(y).astype(int)
    legit_pos = np.flatnonzero(y == 0)
    if n_windows < 2:
        raise ValueError("need at least 2 windows (one to calibrate, one to evaluate)")
    if len(legit_pos) < n_windows:
        raise ValueError("fewer legit rows than windows")
    ids = np.zeros(len(y), dtype=int)  # rows before the first legit row belong to window 0
    cuts = np.array_split(legit_pos, n_windows)
    starts = [c[0] for c in cuts] + [len(y)]
    for k in range(n_windows):
        ids[starts[k] : starts[k + 1]] = k
    return ids


def rolling_evaluation(
    scores: np.ndarray,
    y: np.ndarray,
    n_windows: int,
    target_fpr: float,
    band: tuple[float, float] = (0.008, 0.010),
) -> dict[str, Any]:
    """Per-window FPR / recall for a fixed threshold (calibrated on window 0) and for rolling
    thresholds re-estimated on the previous window (legit-only and label-free)."""
    scores, y = np.asarray(scores, dtype=np.float64), np.asarray(y).astype(int)
    ids = window_ids(y, n_windows)
    legit = y == 0

    def thr(mask: np.ndarray) -> float:
        return quantile_threshold(scores[mask], target_fpr)

    fixed = thr((ids == 0) & legit)
    strategies = {"fixed": [], "rolling_legit": [], "rolling_all": []}
    windows = []
    for k in range(1, n_windows):
        prev, cur = ids == k - 1, ids == k
        thresholds = {
            "fixed": fixed,
            "rolling_legit": thr(prev & legit),
            "rolling_all": thr(prev),
        }
        row: dict[str, Any] = {
            "window": k,
            "n_legit": int((cur & legit).sum()),
            "n_fraud": int((cur & ~legit).sum()),
        }
        for name, t in thresholds.items():
            fp = int((scores[cur & legit] > t).sum())
            tp = int((scores[cur & ~legit] > t).sum())
            row[name] = {
                "threshold": float(t),
                "fp": fp,
                "tp": tp,
                "fpr": fp / row["n_legit"],
                "recall": tp / row["n_fraud"] if row["n_fraud"] else float("nan"),
            }
            strategies[name].append(row[name])
        windows.append(row)
    summary = {}
    for name, rows in strategies.items():
        fprs = np.array([r["fpr"] for r in rows])
        tp, fp = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows)
        n_f = sum(w["n_fraud"] for w in windows)
        n_l = sum(w["n_legit"] for w in windows)
        lo, hi = wilson_interval(tp, n_f)
        summary[name] = {
            "fpr_min": float(fprs.min()),
            "fpr_max": float(fprs.max()),
            "fpr_mean": float(fprs.mean()),
            "windows_in_band": int(((fprs >= band[0]) & (fprs <= band[1])).sum()),
            "n_windows": len(rows),
            "all_in_band": bool(((fprs >= band[0]) & (fprs <= band[1])).all()),
            "pooled_fpr": fp / n_l,
            "pooled_recall": tp / n_f if n_f else float("nan"),
            "pooled_recall_ci_wilson": [lo, hi],
            "tp": tp,
            "fp": fp,
            "n_fraud": n_f,
            "n_legit": n_l,
        }
    return {
        "n_windows": n_windows,
        "target_fpr": target_fpr,
        "band": list(band),
        "windows": windows,
        "summary": summary,
    }


# --------------------------------------------------------------------------- variance


def variance_groups(
    runs_root: Path, family: str, suffix: str = ""
) -> dict[str, list[dict[str, Any]]]:
    """``{"model_seed": rows, "data_seed": rows}`` for ``runs/<family><suffix>``,
    ``<family>_seed<n><suffix>`` and ``<family>_dseed<n><suffix>`` (the base run belongs to both
    groups; ``suffix`` is ``_synthetic`` for the toy runs)."""
    runs_root = Path(runs_root)
    base = runs_root / f"{family}{suffix}"
    groups: dict[str, list[dict[str, Any]]] = {"model_seed": [], "data_seed": []}

    def row(run: Path, kind: str, n: int | str) -> dict[str, Any]:
        at = load_metrics(run)["at_threshold"]
        return {"run": run.name, kind: n, "recall": at["recall"], "fpr": at["fpr"], "tp": at["tp"]}

    if (base / "metrics.json").exists():
        groups["model_seed"].append(row(base, "seed", "base"))
        groups["data_seed"].append(row(base, "seed", "base"))
    for d in sorted(runs_root.glob(f"{family}_*{suffix}")):
        if not (d / "metrics.json").exists():
            continue
        stem = d.name[: -len(suffix)] if suffix else d.name
        if m := _SEED.search(stem):
            groups["model_seed"].append(row(d, "seed", int(m.group(1))))
        elif m := _DSEED.search(stem):
            groups["data_seed"].append(row(d, "seed", int(m.group(1))))
    return groups


def variance_summary(rows: list[dict[str, Any]], target_recall: float = 0.85) -> dict[str, Any]:
    rec = np.array([r["recall"] for r in rows], dtype=float)
    fpr = np.array([r["fpr"] for r in rows], dtype=float)
    return {
        "n": int(len(rows)),
        "recall_mean": float(rec.mean()) if len(rec) else float("nan"),
        "recall_std": float(rec.std(ddof=1)) if len(rec) > 1 else float("nan"),
        "recall_min": float(rec.min()) if len(rec) else float("nan"),
        "recall_max": float(rec.max()) if len(rec) else float("nan"),
        "fpr_min": float(fpr.min()) if len(fpr) else float("nan"),
        "fpr_max": float(fpr.max()) if len(fpr) else float("nan"),
        "target_recall": target_recall,
        "all_meet_recall": bool((rec >= target_recall).all()) if len(rec) else False,
        "runs": rows,
    }


# --------------------------------------------------------------------------- evasion probe


def evasion_probe(
    model: Autoencoder,
    X: np.ndarray,
    threshold: float,
    feature_names: tuple[str, ...],
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    steps: int = 200,
    lr: float = 0.05,
) -> dict[str, Any]:
    """Projected gradient descent on the MSE score inside the L2 ball of each budget.

    ``lr`` is the Adam step for the smallest budgets; it grows to ``eps / 50`` so that every
    budget's boundary is reachable within ``steps`` steps.
    """
    X0 = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    n = len(X0)
    if n == 0:
        raise ValueError("no detected frauds to perturb")
    model.eval()
    grad_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    minimal = np.full(n, np.inf)
    per_budget = []
    deltas_at: dict[float, np.ndarray] = {}
    x_norm = X0.norm(dim=1, keepdim=True)
    for eps in sorted(budgets):
        # Adam moves a coordinate by about lr per step; a fixed lr cannot reach the boundary of
        # a large ball in `steps` steps, so the step grows with the budget. Two starts: the
        # origin of the ball and the projected straight line towards the legit mean (x = 0).
        lr_eps = max(lr, eps / 50)
        shrink = -X0 * torch.clamp(eps / torch.clamp(x_norm, min=1e-12), max=1.0)
        best_score = torch.full((n,), float("inf"))
        best_delta = torch.zeros_like(X0)
        for start in (torch.zeros_like(X0), shrink):
            delta = start.clone().requires_grad_(True)
            opt = torch.optim.Adam([delta], lr=lr_eps)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                s = ((model(X0 + delta) - X0 - delta) ** 2).mean(dim=1)
                s.sum().backward()
                opt.step()
                with torch.no_grad():
                    norms = delta.norm(dim=1, keepdim=True)
                    delta.mul_(torch.clamp(eps / torch.clamp(norms, min=1e-12), max=1.0))
                    s_now = mse_per_sample(model, X0 + delta)
                    better = s_now < best_score
                    best_score = torch.where(better, s_now, best_score)
                    best_delta[better] = delta[better]
        evaded = (best_score <= threshold).numpy()
        minimal = np.where(evaded & np.isinf(minimal), eps, minimal)
        deltas_at[eps] = best_delta.numpy()
        per_budget.append(
            {
                "budget": float(eps),
                "evaded": int(evaded.sum()),
                "evaded_fraction": float(evaded.mean()),
                "score_median": float(best_score.median()),
            }
        )
    for p, flag in zip(model.parameters(), grad_flags, strict=True):
        p.requires_grad_(flag)

    def quantile(q: float) -> float | None:  # None when the quantile falls among never-evaded rows
        with np.errstate(invalid="ignore"):  # inf − inf inside the interpolation
            v = float(np.percentile(minimal, q))
        return v if np.isfinite(v) else None

    # feature movement at the smallest budget that evades at least half of the frauds
    half = next((b["budget"] for b in per_budget if b["evaded_fraction"] >= 0.5), None)
    movement = None
    if half is not None:
        d = deltas_at[half][minimal <= half]
        mean_abs = np.abs(d).mean(axis=0)
        mean_sq = (d**2).mean(axis=0)
        order = np.argsort(-mean_abs)
        movement = {
            "budget": half,
            "features": [feature_names[i] for i in order[:8]],
            "mean_abs_delta": mean_abs[order[:8]].tolist(),
            "share_of_l2": (mean_sq[order[:8]] / max(mean_sq.sum(), 1e-12)).tolist(),
        }
    return {
        "n_detected": int(n),
        "threshold": float(threshold),
        "steps": steps,
        "lr": lr,
        "per_budget": per_budget,
        "median_minimal_budget": quantile(50),
        "q25_minimal_budget": quantile(25),
        "q75_minimal_budget": quantile(75),
        "never_evaded": int(np.isinf(minimal).sum()),
        "fraud_l2_norm_median": float(np.median(x_norm.numpy())),
        "movement": movement,
    }


# --------------------------------------------------------------------------- markdown


def _pct(x: float, d: int = 1) -> str:
    return "n/a" if x is None or x != x else f"{x * 100:.{d}f} %"


def robustness_markdown(res: dict[str, Any]) -> str:
    lines = [
        "# Robustness and deployment realism (Roadmap v2, Phase 10)",
        "",
        "Generated by `fraud-ae robustness`. Rolling-origin validation on the temporal split "
        "(models trained on the first 60 % of legitimate traffic; supervised and fine-tuned "
        "models also see the frauds of that window), seed / data-seed variance of the Phase 9 "
        "winners, and the evasion probe on the legit-only autoencoder.",
        "",
        "## 1. Rolling threshold",
        "",
        f"The last 40 % of traffic (val + test of the temporal split) is cut into "
        f"{res['rolling']['n_windows']} windows of equal legit count in time order; window 0 "
        f"calibrates the fixed threshold, windows 1–{res['rolling']['n_windows'] - 1} are "
        f"evaluated. Target FPR band per window: [{_pct(res['rolling']['band'][0])}, "
        f"{_pct(res['rolling']['band'][1])}]; thresholds are the "
        f"{_pct(res['rolling']['target_fpr'])} quantile of the calibration rows.",
        "",
        "| Model | Threshold | Windows in band | FPR min – max (mean) | Pooled FPR | "
        "Pooled recall [95 %] |",
        "|---|---|---:|---|---:|---|",
    ]
    labels = {
        "fixed": "fixed (window 0, v1 behaviour)",
        "rolling_legit": "rolling, previous window legit rows",
        "rolling_all": "rolling, previous window all rows (label-free)",
    }
    for model, r in res["rolling"]["models"].items():
        for name, s in r["summary"].items():
            lo, hi = s["pooled_recall_ci_wilson"]
            lines.append(
                f"| `{model}` | {labels[name]} | {s['windows_in_band']} / {s['n_windows']} | "
                f"{_pct(s['fpr_min'], 2)} – {_pct(s['fpr_max'], 2)} ({_pct(s['fpr_mean'], 2)}) | "
                f"{_pct(s['pooled_fpr'], 2)} | {_pct(s['pooled_recall'])} [{_pct(lo)}, {_pct(hi)}] "
                f"({s['tp']} / {s['n_fraud']}) |"
            )
    if res["rolling"]["models"]:
        first = next(iter(res["rolling"]["models"].values()))
        lines += [
            "",
            "Per-window FPR (fixed / rolling-legit / rolling-all), frauds per window in "
            "parentheses:",
            "",
            "| Model | "
            + " | ".join(f"w{w['window']} ({w['n_fraud']})" for w in first["windows"])
            + " |",
            "|---|" + "---|" * len(first["windows"]),
        ]
        for model, r in res["rolling"]["models"].items():
            cells = [
                f"{_pct(w['fixed']['fpr'], 2)} / {_pct(w['rolling_legit']['fpr'], 2)} / "
                f"{_pct(w['rolling_all']['fpr'], 2)}"
                for w in r["windows"]
            ]
            lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 2. Variance of the winners",
        "",
        "| Family | Group | Runs | Recall mean ± std | Recall min – max | FPR min – max | "
        "All ≥ 85 % |",
        "|---|---|---:|---:|---|---|:---:|",
    ]
    for fam, groups in res["variance"].items():
        for kind, s in groups.items():
            if s["n"] < 2:  # a single run (the base alone) is not a variance measurement
                continue
            lines.append(
                f"| `{fam}` | {kind.replace('_', ' ')} | {s['n']} | {_pct(s['recall_mean'])} ± "
                f"{_pct(s['recall_std'])} | {_pct(s['recall_min'])} – {_pct(s['recall_max'])} | "
                f"{_pct(s['fpr_min'], 2)} – {_pct(s['fpr_max'], 2)} | "
                f"{'✅' if s['all_meet_recall'] else '❌'} |"
            )
    ev = res["evasion"]
    lines += [
        "",
        "## 3. Evasion probe (legit-only AE)",
        "",
        f"Projected gradient descent on the input of each of the {ev['n_detected']} detected test "
        f"frauds, minimising the MSE score inside an L2 ball (standardised feature units), "
        f"{ev['steps']} steps per budget. Median L2 norm of a detected fraud: "
        f"{ev['fraud_l2_norm_median']:.2f}. Median minimal budget to cross below the threshold "
        f"{ev['threshold']:.3g}: **"
        + (
            f"{ev['median_minimal_budget']:g}"
            if ev["median_minimal_budget"] is not None
            else "n/a (fewer than half of the frauds evade within the largest budget)"
        )
        + "**"
        + (
            f" (IQR {ev['q25_minimal_budget']:g} – {ev['q75_minimal_budget']:g})"
            if ev["q25_minimal_budget"] is not None and ev["q75_minimal_budget"] is not None
            else ""
        )
        + f"; never evaded within the largest budget: {ev['never_evaded']}.",
        "",
        "| L2 budget | Evaded | Share |",
        "|---:|---:|---:|",
    ]
    for b in ev["per_budget"]:
        lines.append(f"| {b['budget']:g} | {b['evaded']} | {_pct(b['evaded_fraction'])} |")
    if ev["movement"]:
        mv = ev["movement"]
        lines += [
            "",
            f"Features that move at budget {mv['budget']:g} (mean |δ| over the evaded frauds, "
            "share of the squared L2 budget): "
            + ", ".join(
                f"`{f}` {d:.2f} ({s:.0%})"
                for f, d, s in zip(
                    mv["features"], mv["mean_abs_delta"], mv["share_of_l2"], strict=True
                )
            )
            + ".",
        ]
    if res.get("figures"):
        lines += ["", "## Figures", ""] + [f"* {k}: `{v}`" for k, v in res["figures"].items()]
    return "\n".join(lines) + "\n"


def write_robustness(res: dict[str, Any], out_md: Path, out_json: Path | None = None) -> Path:
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(robustness_markdown(res))
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(res, indent=2, allow_nan=True))
    return out_md


def run_robustness(
    cfg: Config,
    root: Path,
    temporal_dir: Path,
    temporal_ae_run: Path,
    pooled_ae_run: Path,
    pooled_dir: Path,
    variance_families: tuple[str, ...],
    n_windows: int = 8,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    evasion_steps: int = 200,
    run_suffix: str = "",
) -> dict[str, Any]:
    """Compute all three measurements; returns the result dict (no files written)."""
    from .scoring import load_scores
    from .semi import finetune, semi_scores
    from .supervised import fit_score
    from .threshold import load_threshold
    from .train import load_model

    root = Path(root)
    frame = temporal_frame(temporal_dir, root)
    ae_t = load_model(temporal_ae_run)
    target = cfg.threshold.target_fpr
    models: dict[str, np.ndarray] = {}
    models["ae"] = mse_per_sample(ae_t, torch.as_tensor(frame.X_eval)).double().numpy()
    if len(frame.X_train_fraud):
        (s_hgb,), _ = fit_score(
            "hgb",
            cfg.supervised,
            frame.X_train_legit,
            frame.X_train_fraud,
            [frame.X_eval],
            cfg.train.seed,
        )
        models["hgb"] = s_hgb
        ft, info = finetune(
            ae_t, cfg.supervised, frame.X_train_legit, frame.X_train_fraud, cfg.train.seed
        )
        models["neg_ae"] = semi_scores(ft, frame.X_eval, info)
    band = (target - 0.001, cfg.eval.target_fpr)  # [0.8 %, 1.0 %] for the default targets
    rolling = {
        "n_windows": n_windows,
        "target_fpr": target,
        "band": list(band),
        "train_window_frauds": int(len(frame.X_train_fraud)),
        "n_eval_rows": int(len(frame.y_eval)),
        "n_eval_frauds": int(frame.y_eval.sum()),
        "models": {
            m: rolling_evaluation(s, frame.y_eval, n_windows, target, band)
            for m, s in models.items()
        },
    }
    variance = {
        fam: {
            k: variance_summary(v, cfg.eval.target_recall)
            for k, v in variance_groups(root / "runs", fam, run_suffix).items()
        }
        for fam in variance_families
    }
    ae_p = load_model(pooled_ae_run)
    scores, y = load_scores(pooled_ae_run, "test")
    thr = load_threshold(pooled_ae_run).value
    splits = load_splits(pooled_dir)
    if not np.array_equal(splits.y_test.astype(int), y.astype(int)):
        raise ValueError(f"{pooled_ae_run} was not scored on the test split in {pooled_dir}")
    detected = (y == 1) & (scores > thr)
    ev = evasion_probe(
        ae_p, splits.X_test[detected], thr, splits.feature_names, budgets, evasion_steps
    )
    return {"rolling": rolling, "variance": variance, "evasion": ev}
