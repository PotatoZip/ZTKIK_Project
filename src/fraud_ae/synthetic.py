"""Tiny synthetic stand-in for ``creditcard.csv`` (ADR 9).

The frame has the same 31 columns (``Time``, ``V1..V28``, ``Amount``, ``Class``) and the same
quirks that Phase 1 has to handle: exact duplicate rows, heavy-tailed amounts, a two-day ``Time``
axis in seconds and an hour-of-day dependence of the fraud rate. Frauds are shifted in a subset
of the V-features so that a sane detector can find them, but they are not trivially separable.

It exists so that the pipeline and its invariants can be checked in seconds without the 150 MB
download (``pytest``, ``fraud-ae prepare --synthetic``). It is **not** used for any reported
number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
SECONDS_PER_DAY = 86_400


def make_synthetic_frame(
    n_legit: int = 2000, n_fraud: int = 40, n_duplicates: int = 10, seed: int = 0
) -> pd.DataFrame:
    """Return a creditcard-like frame with ``n_legit + n_fraud + n_duplicates`` rows.

    The last ``n_duplicates`` rows are exact copies of the first legitimate rows.
    """
    if n_duplicates > n_legit:
        raise ValueError("n_duplicates must not exceed n_legit")
    rng = np.random.default_rng(seed)
    n = n_legit + n_fraud
    y = np.r_[np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)]

    V = rng.normal(size=(n, 28))
    # frauds: mean shift and wider spread on half of the (already PCA-like) components
    V[n_legit:, :14] = rng.normal(loc=2.0, scale=1.5, size=(n_fraud, 14))

    # hour-of-day: legit traffic peaks in the afternoon, frauds are spread around the clock,
    # which makes the night-time fraud *rate* higher (as in the real data)
    hour_legit = rng.normal(loc=14.0, scale=4.0, size=n_legit) % 24
    hour_fraud = rng.uniform(0, 24, size=n_fraud)
    hour = np.r_[hour_legit, hour_fraud]
    day = rng.integers(0, 2, size=n)
    time_s = np.round(day * SECONDS_PER_DAY + hour * 3600 + rng.uniform(0, 3600, size=n))

    amount = np.where(
        y == 1,
        rng.lognormal(mean=3.5, sigma=1.5, size=n),
        rng.lognormal(mean=3.0, sigma=1.2, size=n),
    ).round(2)
    amount[rng.random(n) < 0.01] = 0.0  # the real data contains zero-amount rows

    df = pd.DataFrame(V, columns=V_COLUMNS)
    df.insert(0, "Time", time_s)
    df["Amount"] = amount
    df["Class"] = y
    dup = df.iloc[:n_duplicates]
    return pd.concat([df, dup], ignore_index=True)


def write_synthetic_csv(path: Path, **kwargs) -> Path:
    """Write :func:`make_synthetic_frame` to ``path`` (parents created) and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_frame(**kwargs).to_csv(path, index=False)
    return path
