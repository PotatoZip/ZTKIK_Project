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
uv run fraud-ae evaluate  --run runs/ae_base                              # 4. test-set metrics + CIs, meets_target, figures
uv run fraud-ae baselines --config configs/base.toml                      # PCA + Isolation Forest
uv run fraud-ae all       --config configs/base.toml --run runs/ae_base   # phases 1-4 (--synthetic for the toy set)
scripts/run_ablations.sh                                                  # decision-gate runs → results/ablations_phase4.md
```

`prepare` writes `data/processed/{splits.npz, scaler.joblib, feature_names.json, data_report.json}`
and the EDA figures to `report/figures/eda_*.png`; `--synthetic` writes to
`data/processed_synthetic/` instead so the real artifacts are never overwritten.

Each run directory contains `config.toml` (snapshot; the *effective* config if `--epochs` /
`--seed` overrides were used), `model.pt` (best checkpoint), `history.csv`,
`train_summary.json`, `score_calibration.npz`, `scores_{val,test}.npz`, `threshold.json`,
`metrics.json` (includes `meets_target`) and `figures/`. `--synthetic` on `train` /
`threshold` / `evaluate` reads the splits written by `prepare --synthetic`. `threshold` and
`evaluate` read the run's own `config.toml` snapshot, so scoring always matches the trained
model. Test labels are read for the first time by `evaluate`.

## Layout

```
configs/        TOML experiments (base + ablations: latent size, no time feature, score method)
scripts/        run_ablations.sh (decision-gate runs via the CLI), summarize_runs.py (table from runs/*/metrics.json)
src/fraud_ae/   package: config, data, synthetic, model, train, scoring, threshold, evaluate, baselines, plots, cli
tests/          pytest suite on the synthetic creditcard-like frame from fraud_ae.synthetic
data/           raw/ (extracted CSV) and processed/ (splits, scaler, data_report.json) – git-ignored
runs/           one directory per experiment – git-ignored
results/        Markdown tables generated from runs/*/metrics.json (tracked)
report/         report.md (English), prompt_log.md, figures/
```

## Status

Phases 0 (scaffold), 1 (data), 2 (training), 3 (threshold) and 4 (evaluation, milestone M2)
done; phases 5–6 are tracked in `ROADMAP.md` and every change in `report/CHANGELOG.md`.
`fraud-ae baselines` (Phase 5) still exits with code 2 and a "not implemented" message.

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

### Phase 4 – evaluation on the test set (`fraud-ae evaluate --run runs/ae_base`, milestone M2)

The threshold is the one chosen on validation data in Phase 3 (1.862, `mse`); test labels are
read here for the first time. Intervals are Wilson 95 %; the 1 000-sample stratified bootstrap
agrees within 0.3 pp.

| Metric (56 650 legit / 237 fraud test rows) | Value |
|---|---:|
| ROC-AUC / PR-AUC (average precision) | 0.938 / 0.538 |
| Recall at oracle FPR 0.1 % / 0.5 % / 1 % / 2 % | 47.3 % / 74.7 % / 77.2 % / 78.9 % |
| At the val-chosen threshold: TP / FP / TN / FN | 183 / 541 / 56 109 / 54 |
| Recall | **77.2 %** [71.5 %, 82.1 %] |
| FPR | 0.955 % [0.878 %, 1.038 %] |
| Precision / F1 | 25.3 % / 0.381 |
| Criterion recall ≥ 85 % and FPR < 1 % (TP ≥ 202, FP ≤ 566) | **not met** (FPR part met) |

The recall-vs-FPR curve plateaus at 77–79 % between 0.3 % and 2 % FPR and only crosses 85 %
near 7 % FPR: roughly a fifth of the test frauds reconstruct as well as legitimate rows, and
no threshold fixes that. The largest fraud/legit residual ratios are on V17, V14, V12, V10 and
V16. Figures: `report/figures/{roc_pr,recall_vs_fpr,threshold_sweep,score_histogram,per_feature_residuals}_test_ae_base.png`.

**Decision gate** (`scripts/run_ablations.sh`, full table in `results/ablations_phase4.md`,
one run directory per row, threshold always the 0.9 % val-legit quantile):

| Variant | Val recall | Test FPR | Test recall [95 %] |
|---|---:|---:|---|
| base (`mse`, latent 5, seed 42) | 84.3 % | 0.955 % | 77.2 % [71.5, 82.1] |
| score `standardized` / `mahalanobis` | 84.7 % / 85.2 % | 0.990 % / 0.934 % | 76.4 % / 77.6 % |
| latent 3 / 8 | 83.9 % / 84.7 % | 0.923 % / 0.987 % | 78.5 % / 77.6 % |
| no hour-of-day feature (29 inputs) | 85.6 % | 0.932 % | 78.5 % [72.8, 83.2] |
| 200 epochs | 84.3 % | 0.999 % | 77.2 % |
| seeds 0–3 (+ 42) | 83.1–84.3 % | 0.88–0.94 % | 75.9–77.2 % (mean 76.5 %) |

No variant meets the criterion: the test recall of this model family is 76–79 % with upper
confidence bounds ≤ 83 %, while the validation half of the frauds is consistently 6–8 pp easier.
Seed-to-seed spread is ±0.7 pp, i.e. the gap is not noise. Selecting on validation (the honest
protocol) would pick `mahalanobis`, which gives 77.6 % on test. Phases 5–6 therefore compare
against the PCA and Isolation Forest baselines and report the shortfall with its causes rather
than tuning further on the test set.
