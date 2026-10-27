# Project 12 – Fraud Detection in Bank Transactions with a Deep Autoencoder

ZTKIK course project. A PyTorch autoencoder is trained **only on legitimate** credit-card
transactions; a transaction is flagged as fraud when its reconstruction error (MSE) exceeds a
threshold chosen on legitimate validation data. Success criterion from the abstract:
**recall ≥ 85 % at false-positive rate < 1 %** on a held-out test set.

* Abstract: `tmp.md` · Roadmap: `ROADMAP.md` · Design and decisions: `ARCHITECTURE.md`
* Dataset: Kaggle *Credit Card Fraud Detection* (ULB MLG), 284 807 transactions, 492 frauds
  (283 726 / 473 after dropping 1 081 exact duplicates). Place the Kaggle zip at `dataset.zip`
  (git-ignored); `fraud-ae prepare` extracts it and verifies the CSV's sha256 pinned in the configs.

## Setup (CPU only, Python 3.11, [uv](https://docs.astral.sh/uv/))

```bash
uv sync --group dev          # creates .venv with the CPU torch wheel
uv run fraud-ae --help
uv run pytest                # unit tests use a synthetic dataset, no download needed
uv run ruff check src tests
```

## Running the four project phases

```bash
uv run fraud-ae prepare   --config configs/base.toml                      # 1. preprocessing
uv run fraud-ae prepare   --config configs/base.toml --synthetic          #    same pipeline on a 2 k-row generated set
uv run fraud-ae train     --config configs/base.toml --run runs/ae_base   # 2. train AE on legit rows (~30 s CPU)
uv run fraud-ae train     --config configs/base.toml --run runs/ae_synthetic --synthetic --epochs 5   # smoke run
uv run fraud-ae threshold --run runs/ae_base                              # 3. score val/test, threshold = val-legit quantile
uv run fraud-ae threshold --run runs/ae_base --method mahalanobis --strategy youden   # variants (val only)
uv run fraud-ae evaluate  --run runs/ae_base                              # 4. recall / FPR on test, figures
uv run fraud-ae baselines --config configs/base.toml                      # PCA + Isolation Forest
uv run fraud-ae all       --config configs/base.toml --run runs/ae_base   # phases 1-4
```

`prepare` writes `data/processed/{splits.npz, scaler.joblib, feature_names.json, data_report.json}`
and the EDA figures to `report/figures/eda_*.png`; `--synthetic` writes to
`data/processed_synthetic/` instead so the real artifacts are never overwritten.

Each run directory contains `config.toml` (snapshot), `model.pt` (best checkpoint),
`history.csv`, `train_summary.json`, `score_calibration.npz`, `scores_{val,test}.npz`,
`threshold.json`, `metrics.json` (includes `meets_target`) and `figures/`. `--synthetic` on
`train` / `threshold` reads the splits written by `prepare --synthetic`. `threshold` reads the
run's own `config.toml` snapshot, so scoring always matches the trained model.

## Layout

```
configs/        TOML experiments (base + ablations)
src/fraud_ae/   package: config, data, synthetic, model, train, scoring, threshold, evaluate, baselines, plots, cli
tests/          pytest suite on the synthetic creditcard-like frame from fraud_ae.synthetic
data/           raw/ (extracted CSV) and processed/ (splits, scaler, data_report.json) – git-ignored
runs/           one directory per experiment – git-ignored
report/         report.md (English), prompt_log.md, figures/
```

## Status

Phases 0 (scaffold), 1 (data), 2 (training) and 3 (threshold) done; phases 4–6 are tracked
in `ROADMAP.md` and every change in `report/CHANGELOG.md`. CLI sub-commands for unimplemented phases exit with
code 2 and a "not implemented" message.

### Phase 1 – data (`fraud-ae prepare`, seed 42)

| Split | Legit | Fraud | Total | Fraud share |
|---|---:|---:|---:|---:|
| train | 169,952 | 0 | 169,952 | 0.000% |
| val | 56,651 | 236 | 56,887 | 0.415% |
| test | 56,650 | 237 | 56,887 | 0.417% |
| **all** | 283,253 | 473 | 283,726 | 0.167% |

* Pipeline: unzip → sha256 → drop 1 081 exact duplicates (19 of them frauds) → `log1p(Amount)`,
  hour-of-day `sin`/`cos`, `V1..V28` unchanged (31 inputs) → seeded split (legit 60/20/20 %,
  frauds 50/50 val/test, none in train) → `StandardScaler` fitted on train only.
* EDA (`report/figures/eda_*.png`): fraud share 0.167 %; median amount 22.00 (legit) vs 9.82
  (fraud), 1 808 zero-amount rows; the fraud *rate* peaks at night (hour 2: 1.45 %, hour 4:
  1.04 % vs 0.167 % overall) while traffic is lowest, which motivates the hour-of-day feature.
* Invariants under test (`tests/test_data.py`): no fraud in train, splits disjoint and covering,
  seeded split reproducible, scaler statistics equal train statistics, no duplicate rows across
  splits, sha256 mismatch rejected, artifact round-trip.

### Phase 2 – training (`fraud-ae train --run runs/ae_base`, seed 42)

| | |
|---|---|
| Model | 31 → 20 → 10 → 5 → 10 → 20 → 31, ELU, linear output, 1 836 parameters |
| Optimiser | Adam lr 1e-3, weight decay 1e-5, batch 512, `ReduceLROnPlateau` |
| Selection | lowest MSE on the 56 651 legitimate validation rows (no fraud label used) |
| Result | val-legit MSE 0.770 (epoch 1) → **0.420** (best, epoch 99); train MSE 0.418; 100/100 epochs, 31 s on CPU |

The 100-epoch cap binds before early stopping (patience 10) does: the curve is flat but still
creeping down, so a longer schedule is a candidate for the Phase 4 ablations. Figure:
`report/figures/training_curves_ae_base.png`. Determinism: two runs with the same seed give
identical losses in `history.csv` (only the wall-clock `seconds` column differs) and a
byte-identical `model.pt` (unit-tested on the synthetic split and checked on the real data).

### Phase 3 – scoring and threshold (`fraud-ae threshold --run runs/ae_base`)

Scores are computed from the reconstruction residual `x − AE(x)`; the calibration statistics for
the non-plain methods come from legitimate validation rows only. The primary threshold is the
`1 − 0.009` quantile of the legitimate validation scores, so the FPR criterion holds by
construction without reading any fraud label. **Validation numbers only** – the test set is
scored and stored here but first evaluated in Phase 4.

| Score method | Threshold (quantile 0.9 %) | Val FPR | Val recall (236 frauds) |
|---|---:|---:|---:|
| `mse` (primary) | 1.862 | 0.900 % | 84.3 % |
| `standardized` | 155.7 | 0.900 % | 84.7 % |
| `mahalanobis` | 259.0 | 0.900 % | 85.2 % |

The semi-supervised strategies (`youden`, `max_f1`) use val fraud labels and are reported for
comparison only: Youden lands at 2.1–2.5 % FPR (87 % recall), max-F1 at ≈ 0.2 % FPR (74–77 %
recall). Figure: `report/figures/score_histogram_val_ae_base.png`.
