# Project 12 – Fraud Detection in Bank Transactions with a Deep Autoencoder

ZTKIK course project. A PyTorch autoencoder is trained **only on legitimate** credit-card
transactions; a transaction is flagged as fraud when its reconstruction error (MSE) exceeds a
threshold chosen on legitimate validation data. Success criterion from the abstract:
**recall ≥ 85 % at false-positive rate < 1 %** on a held-out test set.

* **Report: `report/report.md`** (rendered from `report/report_template.md` by `fraud-ae report`;
  every number comes from a JSON artifact) · Abstract: `tmp.md` · Roadmap: `ROADMAP.md` ·
  Design and decisions: `ARCHITECTURE.md` · Changelog: `report/CHANGELOG.md` · Prompt log:
  `report/prompt_log.md`
* **Outcome in one line:** test recall **77.2 %** [71.5 %, 82.1 %] at FPR 0.955 % – the FPR half
  of the criterion holds, the 85 % recall half does not, for the base model and all eleven
  variants; the AE still beats PCA (70.0 % at best) and Isolation Forest (56.1 %) at the same FPR.
* Dataset: Kaggle *Credit Card Fraud Detection* (ULB MLG), 284 807 transactions, 492 frauds
  (283 726 / 473 after dropping 1 081 exact duplicates). Place the Kaggle zip at `dataset.zip`
  (git-ignored); `fraud-ae prepare` extracts it and verifies the CSV's sha256 pinned in the configs.

## Setup (CPU only, Python 3.11, [uv](https://docs.astral.sh/uv/))

```bash
uv sync --group dev          # creates .venv with the CPU torch wheel
uv run fraud-ae --help
uv run pytest                # unit tests use a synthetic dataset, no download needed
uv run ruff check src tests
scripts/check.sh             # format check + lint + tests + version (the hardening gate)
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
uv run fraud-ae baselines --config configs/base.toml                      # 5. PCA k=5/10/15 + Isolation Forest → runs/baseline_*
uv run fraud-ae compare   --out results/comparison.md --figures-dir report/figures   #    table + figures from runs/*/metrics.json
uv run fraud-ae report                                                    # 6. render report/report.md from the artifacts
uv run fraud-ae all       --config configs/base.toml --run runs/ae_base   # phases 1-4 (--synthetic for the toy set)
scripts/run_ablations.sh                                                  # every AE variant + baselines + comparison + report (≈ 10 min)
```

**Reproduce the whole study from a clean checkout** (CPU, ≈ 10 min): `uv sync --group dev`,
place `dataset.zip`, then `uv run fraud-ae all --config configs/base.toml --run runs/ae_base`
followed by `scripts/run_ablations.sh`. The last step rewrites `results/comparison.md`,
`report/figures/*` and `report/report.md`; `uv run pytest` then verifies that the committed
report equals what the artifacts render to.

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
configs/        TOML experiments (base + ablations: latent size, no time feature, score method, BatchNorm, temporal split)
scripts/        run_ablations.sh (every run, baselines, comparison, report), check.sh (format + lint + tests)
src/fraud_ae/   package: config, data, synthetic, model, train, scoring, threshold, evaluate, baselines, plots, report, cli
tests/          pytest suite on the synthetic creditcard-like frame from fraud_ae.synthetic
data/           raw/ (extracted CSV) and processed/ (splits, scaler, data_report.json) – git-ignored
runs/           one directory per experiment – git-ignored
results/        comparison.md and ablations_phase4.md, generated from runs/*/metrics.json (tracked)
report/         report.md (rendered), report_template.md (source of truth), CHANGELOG.md, prompt_log.md, figures/
```

## Status

All six phases are done (milestone M3, submission package, version 1.0.0). The sections below
are the per-phase working notes; the report in `report/report.md` is the deliverable. Phase
outcomes are in `ROADMAP.md`, every change in `report/CHANGELOG.md`.

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

### Phase 5 – baselines and ablations (`fraud-ae baselines`, `fraud-ae compare`)

PCA(k) reconstruction error and a 200-tree Isolation Forest are fitted on the same 169 952
legitimate training rows, thresholded with the same 0.9 % val-legit quantile and scored by the
same evaluator, so every row below is one `runs/<name>/metrics.json`. Full table with ROC/PR-AUC,
oracle recall at 1 % FPR and precision: `results/comparison.md`; figures
`report/figures/comparison_recall.png` and `report/figures/seed_variance.png`.

| Model (random split, 237 test frauds) | ROC-AUC | Test FPR | Test recall [Wilson 95 %] |
|---|---:|---:|---|
| **AE 31-20-10-5, `mse`** (`ae_base`) | 0.938 | 0.955 % | **77.2 %** [71.5, 82.1] |
| AE, latent 3 / no hour feature (best variants) | 0.931 / 0.929 | 0.92 % / 0.93 % | 78.5 % [72.8, 83.2] |
| AE + BatchNorm (`ae_batchnorm`) | 0.938 | 1.001 % | 75.9 % [70.1, 80.9] |
| PCA k = 15 (`baseline_pca_k15`, 55.8 % variance) | 0.940 | 0.953 % | 70.0 % [63.9, 75.5] |
| PCA k = 10 | 0.934 | 0.950 % | 59.9 % [53.6, 65.9] |
| PCA k = 5 (same bottleneck as the AE) | 0.934 | 0.909 % | 58.6 % [52.3, 64.7] |
| Isolation Forest (`baseline_iforest`) | 0.940 | 0.967 % | 56.1 % [49.8, 62.3] |

* The non-linear AE beats the linear autoencoder with the same bottleneck by 19 pp and the
  Isolation Forest by 21 pp at the same FPR, although all ROC-AUCs are ≈ 0.93–0.94: the
  difference is concentrated in the low-FPR region that matters. No model meets the criterion.
* BatchNorm (ADR 6) costs 1.3 pp and pushes the FPR to 1.001 %, i.e. it also fails the FPR half.
* Seed variance (`ae_base`, seeds 0–3 and 42): val recall 83.7 % ± 0.6 %, test recall
  76.5 % ± 0.6 % (75.9–77.2 %). The val–test gap of 7 pp is therefore systematic, not seed noise.

**Temporal split** (`configs/ablation_temporal.toml`: train on the earliest 60 % of legitimate
traffic, validate on the next 20 %, test on the last 20 %; frauds keep their own time, 342 of
473 fall into the training window and are dropped → 57 val / 74 test frauds):

| Model (temporal split, 74 test frauds) | ROC-AUC | Test FPR | Test recall [95 %] | Recall @ oracle 1 % FPR |
|---|---:|---:|---|---:|
| AE 31-20-10-5 (`ae_temporal`) | 0.951 | 0.528 % | 73.0 % [61.9, 81.8] | 79.7 % |
| PCA k = 15 (`baseline_pca_k15_temporal`) | 0.963 | 0.558 % | 60.8 % [49.4, 71.1] | 78.4 % |
| PCA k = 5 / Isolation Forest | 0.954 / 0.951 | 0.55 % / 0.64 % | 43.2 % / 39.2 % | 55.4 % / 55.4 % |

The val-legit threshold transfers to a *lower* FPR on later traffic (0.53 % instead of 0.9 %),
so the AE loses 7 pp of recall to threshold drift (73.0 % at the threshold vs 79.7 % at the
oracle 1 % FPR); the ranking itself is as good as on the random split. Intervals are wide with
74 frauds. The AE stays ahead of every baseline under both protocols.

### Phase 6 – hardening and report (`fraud-ae report`, `scripts/check.sh`)

* `report/report.md` is **rendered** from `report/report_template.md`: every number is a
  placeholder such as `{{ ${runs/ae_base:at_threshold.recall} | pct1 }}` resolved from
  `metrics.json`, `threshold.json`, `train_summary.json` or `data_report.json`
  (`src/fraud_ae/report.py`; simple arithmetic and `mean/std/min/max` across runs are allowed,
  e.g. the seed-variance table). Edit the template, not the report.
* `tests/test_report.py` checks the placeholder engine, that the template only references
  artifact locations, and – when the real runs are present – that the committed `report.md`
  is byte-identical to what the artifacts render to. `scripts/check.sh` runs format check,
  lint and the full suite (109 tests).
* The report follows the outline in `ROADMAP.md`: problem and threat model with the precision
  consequence of the criterion, data and leakage controls, model, threshold, evaluation
  protocol, results (primary verdict, ranking analysis, ablations, seed variance, baselines,
  temporal split, per-feature residuals), discussion, limitations, conclusions,
  reproducibility and references.
