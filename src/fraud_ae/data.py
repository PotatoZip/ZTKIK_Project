"""Phase 1 – data preprocessing for the Credit Card Fraud dataset.

Pipeline (see ARCHITECTURE.md §2)::

    dataset.zip → raw CSV (sha256 checked) → drop duplicates → feature engineering
    → seeded split (train: legit only) → StandardScaler fit on train → processed artifacts

Artifacts written to ``DataConfig.processed_dir``:
    splits.npz          X_train, X_val, y_val, X_test, y_test, idx_train, idx_val, idx_test
    scaler.joblib       fitted sklearn StandardScaler (train statistics only)
    feature_names.json  ordered list of model input columns
    data_report.json    row counts, duplicates dropped, class ratio, split sizes, sha256,
                        per-hour fraud rate, scaler statistics, config snapshot

``idx_*`` are row positions in the **de-duplicated** frame (``load_raw`` → ``drop_duplicates``),
so a row can always be traced back to the raw CSV via ``data_report.json``.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DataConfig

RAW_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
SECONDS_PER_DAY = 86_400
CSV_MEMBER = "creditcard.csv"

SPLITS_FILE = "splits.npz"
SCALER_FILE = "scaler.joblib"
FEATURES_FILE = "feature_names.json"
REPORT_FILE = "data_report.json"


class DataError(RuntimeError):
    """Raised for missing/corrupt input data or a checksum mismatch."""


@dataclass(frozen=True)
class Splits:
    """Scaled feature matrices and labels. ``y_train`` is implicitly all zeros."""

    X_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    feature_names: tuple[str, ...]


# --------------------------------------------------------------------------- input


def extract_dataset(zip_path: Path, raw_csv: Path) -> Path:
    """Unzip ``creditcard.csv`` into ``raw_csv`` if it is not already there."""
    raw_csv = Path(raw_csv)
    if raw_csv.exists():
        return raw_csv
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise DataError(
            f"neither {raw_csv} nor {zip_path} exists – download the Kaggle dataset "
            "(mlg-ulb/creditcardfraud) and place the zip at data.zip_path"
        )
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if Path(m).name == CSV_MEMBER]
        if not members:
            raise DataError(f"{zip_path} does not contain {CSV_MEMBER}")
        with zf.open(members[0]) as src, raw_csv.open("wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    return raw_csv


def sha256_of(path: Path) -> str:
    """Hex digest of a file, used to pin the dataset version in data_report.json."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_raw(raw_csv: Path) -> pd.DataFrame:
    """Read the CSV and assert the expected 31 columns."""
    df = pd.read_csv(raw_csv)
    if list(df.columns) != RAW_COLUMNS:
        raise DataError(
            f"{raw_csv}: unexpected columns {list(df.columns)[:5]}…, expected {RAW_COLUMNS[:5]}…"
        )
    if df.isna().any().any():
        raise DataError(f"{raw_csv}: contains missing values")
    if not set(df["Class"].unique()) <= {0, 1}:
        raise DataError(f"{raw_csv}: Class must be 0/1")
    df["Class"] = df["Class"].astype(int)
    return df


# --------------------------------------------------------------------------- transforms


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows *before* splitting; returns (df, n_dropped) (ADR 4)."""
    out = df.drop_duplicates(keep="first").reset_index(drop=True)
    return out, len(df) - len(out)


def hour_of_day(time_seconds: pd.Series | np.ndarray) -> np.ndarray:
    """Hour in ``[0, 24)`` from seconds since the first transaction (two-day window, ADR 5)."""
    return (np.asarray(time_seconds, dtype=float) % SECONDS_PER_DAY) / 3600.0


def engineer_features(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Return the model input frame (no ``Class``).

    * ``Amount`` → ``log1p(Amount)`` when ``cfg.log_amount``
    * ``Time``   → ``hour_sin``, ``hour_cos`` of hour-of-day when ``cfg.time_feature == "sincos"``,
      dropped otherwise
    * ``V1..V28`` unchanged
    """
    X = df[V_COLUMNS].copy()
    amount = df["Amount"].to_numpy(dtype=float)
    X["Amount"] = np.log1p(amount) if cfg.log_amount else amount
    if cfg.time_feature == "sincos":
        angle = 2 * np.pi * hour_of_day(df["Time"]) / 24.0
        X["hour_sin"] = np.sin(angle)
        X["hour_cos"] = np.cos(angle)
    elif cfg.time_feature != "drop":  # pragma: no cover – guarded by config validation
        raise ValueError(f"unknown time_feature {cfg.time_feature!r}")
    assert X.shape[1] == cfg.n_features
    return X


def split_indices(y: np.ndarray, cfg: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded split. Legit rows follow ``cfg.legit_fractions``; frauds go to val/test only
    according to ``cfg.fraud_val_fraction``. Returns sorted (idx_train, idx_val, idx_test)."""
    y = np.asarray(y)
    rng = np.random.default_rng(cfg.seed)
    legit = rng.permutation(np.flatnonzero(y == 0))
    fraud = rng.permutation(np.flatnonzero(y == 1))

    f_train, f_val, _ = cfg.legit_fractions
    n_train = int(round(f_train * len(legit)))
    n_val = int(round(f_val * len(legit)))
    legit_train, legit_val, legit_test = np.split(legit, [n_train, n_train + n_val])

    n_fraud_val = int(round(cfg.fraud_val_fraction * len(fraud)))
    fraud_val, fraud_test = fraud[:n_fraud_val], fraud[n_fraud_val:]

    idx_train = np.sort(legit_train)
    idx_val = np.sort(np.r_[legit_val, fraud_val])
    idx_test = np.sort(np.r_[legit_test, fraud_test])
    return idx_train, idx_val, idx_test


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a ``StandardScaler`` on the training (legit-only) matrix."""
    return StandardScaler().fit(np.asarray(X_train, dtype=float))


# --------------------------------------------------------------------------- report


def _hourly_fraud_rate(df: pd.DataFrame) -> dict[str, list]:
    hours = np.floor(hour_of_day(df["Time"])).astype(int)
    counts = np.bincount(hours, minlength=24)
    frauds = np.bincount(hours, weights=df["Class"].to_numpy(), minlength=24)
    rate = np.divide(frauds, counts, out=np.zeros(24), where=counts > 0)
    return {
        "transactions": counts.tolist(),
        "frauds": frauds.astype(int).tolist(),
        "fraud_rate": rate.tolist(),
    }


def _amount_summary(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, part in (
        ("all", df),
        ("legit", df[df["Class"] == 0]),
        ("fraud", df[df["Class"] == 1]),
    ):
        a = part["Amount"]
        out[name] = {
            "min": float(a.min()),
            "median": float(a.median()),
            "mean": float(a.mean()),
            "p99": float(a.quantile(0.99)),
            "max": float(a.max()),
            "zero_amount_rows": int((a == 0).sum()),
        }
    return out


def build_report(
    df: pd.DataFrame,
    n_raw: int,
    n_dup: int,
    idx: tuple[np.ndarray, np.ndarray, np.ndarray],
    scaler: StandardScaler,
    feature_names: list[str],
    cfg: DataConfig,
    sha256: str,
) -> dict[str, Any]:
    y = df["Class"].to_numpy()
    splits = {}
    for name, ids in zip(("train", "val", "test"), idx, strict=True):
        n_fraud = int(y[ids].sum())
        splits[name] = {"legit": int(len(ids) - n_fraud), "fraud": n_fraud, "total": int(len(ids))}
    n_fraud_total = int(y.sum())
    return {
        "source": {"raw_csv": cfg.raw_csv, "zip_path": cfg.zip_path, "sha256": sha256},
        "rows": {"raw": int(n_raw), "duplicates_dropped": int(n_dup), "after_dedupe": int(len(df))},
        "class_balance": {
            "legit": int(len(df) - n_fraud_total),
            "fraud": n_fraud_total,
            "fraud_rate": n_fraud_total / len(df),
        },
        "features": feature_names,
        "splits": splits,
        "amount": _amount_summary(df),
        "by_hour": _hourly_fraud_rate(df),
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "config": asdict(cfg),
    }


def split_table_markdown(report: dict[str, Any]) -> str:
    """Markdown table of split sizes for the report."""
    s = report["splits"]
    lines = [
        "| Split | Legit | Fraud | Total | Fraud share |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("train", "val", "test"):
        r = s[name]
        share = r["fraud"] / r["total"] if r["total"] else 0.0
        lines.append(f"| {name} | {r['legit']:,} | {r['fraud']:,} | {r['total']:,} | {share:.3%} |")
    cb = report["class_balance"]
    lines.append(
        f"| **all** | {cb['legit']:,} | {cb['fraud']:,} | {cb['legit'] + cb['fraud']:,} "
        f"| {cb['fraud_rate']:.3%} |"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- pipeline


def prepare(
    cfg: DataConfig, project_root: Path, figures_dir: Path | None = None
) -> tuple[Splits, dict[str, Any]]:
    """Run the full Phase 1 pipeline, write the processed artifacts and EDA figures.

    Returns ``(splits, report)``. ``figures_dir`` defaults to ``<processed_dir>/figures``.
    """
    root = Path(project_root)
    raw_csv = root / cfg.raw_csv
    if cfg.zip_path:
        extract_dataset(root / cfg.zip_path, raw_csv)
    elif not raw_csv.exists():
        raise DataError(f"{raw_csv} does not exist and data.zip_path is empty")

    digest = sha256_of(raw_csv)
    if cfg.expected_sha256 and digest != cfg.expected_sha256.lower():
        raise DataError(
            f"sha256 mismatch for {raw_csv}: got {digest}, expected {cfg.expected_sha256}"
        )

    raw = load_raw(raw_csv)
    n_raw = len(raw)
    df, n_dup = drop_duplicates(raw) if cfg.drop_duplicates else (raw.reset_index(drop=True), 0)

    X_df = engineer_features(df, cfg)
    feature_names = list(X_df.columns)
    X_all = X_df.to_numpy(dtype=np.float64)
    y = df["Class"].to_numpy(dtype=np.int64)

    idx_train, idx_val, idx_test = split_indices(y, cfg)
    assert y[idx_train].sum() == 0, "leakage: fraud rows in the training split"

    scaler = fit_scaler(X_all[idx_train])
    tf = lambda a: scaler.transform(a).astype(np.float32)  # noqa: E731
    splits = Splits(
        X_train=tf(X_all[idx_train]),
        X_val=tf(X_all[idx_val]),
        y_val=y[idx_val],
        X_test=tf(X_all[idx_test]),
        y_test=y[idx_test],
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        feature_names=tuple(feature_names),
    )

    out = root / cfg.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / SPLITS_FILE,
        X_train=splits.X_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )
    joblib.dump(scaler, out / SCALER_FILE)
    (out / FEATURES_FILE).write_text(json.dumps(feature_names, indent=2))

    report = build_report(
        df, n_raw, n_dup, (idx_train, idx_val, idx_test), scaler, feature_names, cfg, digest
    )
    from .plots import eda_figures  # local import keeps matplotlib out of the hot path

    figs = eda_figures(df, Path(figures_dir) if figures_dir else out / "figures")
    report["figures"] = {k: _relative(v, root) for k, v in figs.items()}
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2))
    return splits, report


def _relative(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def load_splits(processed_dir: Path) -> Splits:
    """Load ``splits.npz`` + ``feature_names.json`` written by :func:`prepare`."""
    processed_dir = Path(processed_dir)
    npz_path = processed_dir / SPLITS_FILE
    if not npz_path.exists():
        raise DataError(f"{npz_path} not found – run `fraud-ae prepare` first")
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files}
    names = tuple(json.loads((processed_dir / FEATURES_FILE).read_text()))
    return Splits(**arrays, feature_names=names)


def load_report(processed_dir: Path) -> dict[str, Any]:
    return json.loads((Path(processed_dir) / REPORT_FILE).read_text())
