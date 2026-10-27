# Changelog – Project 12 (Autoencoder fraud detection)

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the project phases in
`ROADMAP.md`. Dates are the dates of the work sessions.

## [Unreleased]

Planned (Roadmap v2 in `ROADMAP.md`, decided 2026-10-27): Phase 7 pooled-fraud protocol +
repeated splits + `fraud-ae diagnose`; Phase 8 unsupervised ceiling (VAE, latent density,
hour-segmented thresholds); Phase 9 fully supervised classifiers (logistic regression,
gradient boosting, MLP), weakly-supervised AE and hybrid stacking with 5-fold CV over frauds
and a few-label curve; Phase 10 rolling threshold, variance, evasion probe; Phase 11 report v2.

## [1.0.0] – 2026-10-27 – Phase 6: hardening and report (milestone M3)

### Added
- `fraud_ae.report`: placeholder engine (`${path:key.path}` refs into `metrics.json`,
  `threshold.json`, `train_summary.json`, `data_report.json`; `+ - * /`, `mean/std/min/max`
  via an `ast` whitelist; formats `pct0..3`, `pp1`, `f0..4`, `int`, `verdict`, `str`),
  `render`, `render_report`, `references`; CLI `fraud-ae report [--template] [--out]`.
- `report/report_template.md` (source of truth, ≈ 310 artifact references) and the rendered
  `report/report.md`: abstract, threat model and criterion consequences, data and leakage
  controls, model and training, scoring and threshold, evaluation protocol, results (primary
  verdict, ranking analysis, ablations, seed variance, baselines, temporal split, per-feature
  residuals), discussion, limitations, conclusions, reproducibility, references; 10 figures.
- `scripts/check.sh` (format check, lint, tests, version); `scripts/run_ablations.sh` now
  copies the base-run figures and renders the report at the end.
- Tests: `tests/test_report.py` (7: formats, expressions, error collection, file output, CLI
  failure path, static template check, committed report ≡ fresh render when runs exist); suite
  109 tests.

### Changed
- README: outcome summary and report link at the top, reproduction recipe, Phase 6 section,
  status "all six phases done". ROADMAP Phase 6 outcome; ARCHITECTURE report section and ADR 23.
- Version 0.5.0 → 1.0.0.

## [0.5.0] – 2026-10-27 – Phase 5: baselines and ablations

### Added
- `fraud_ae.baselines`: `fit_pca` / `pca_reconstruction_error` / `pca_scores` (k ∈ {5, 10, 15},
  same MSE units as the AE), `fit_iforest` / `iforest_scores` (200 trees, seeded),
  `run_baselines` writing the AE run-directory contract (minus `model.pt` / `history.csv`)
  with the same val-legit quantile threshold and evaluator; `comparison_rows`,
  `seed_families`, `comparison_markdown`, `write_comparison` → `results/comparison.md`.
- `plots.baseline_comparison` (bars with Wilson whiskers, oracle recall@1 % FPR markers) and
  `plots.seed_variance` (box + points per seed family).
- CLI: `fraud-ae baselines --config <toml> [--synthetic]` (run names carry `_temporal` /
  `_synthetic`), `fraud-ae compare [--runs …] [--out] [--figures-dir]`.
- Temporal split: `data.split = "temporal"` (`data.temporal_split_indices`; frauds in the
  training window are dropped and counted in `data_report.json` together with the time
  windows); configs `ablation_temporal.toml` (own `data/processed_temporal`) and
  `ablation_batchnorm.toml`.
- `scripts/run_ablations.sh` now also runs BatchNorm, temporal, both baseline sets and the
  comparison.
- Tests: `tests/test_baselines.py` (8), three temporal-split tests in `tests/test_data.py`, a
  baselines + compare CLI test; suite 102 tests.
- Real results (`results/comparison.md`, 21 runs): random split PCA k = 5 / 10 / 15 → 58.6 % /
  59.9 % / 70.0 %, Isolation Forest 56.1 %, AE 77.2 %; BatchNorm 75.9 % at FPR 1.001 %;
  temporal split AE 73.0 % (oracle 79.7 %) vs PCA k = 15 60.8 %; seed variance 76.5 % ± 0.6 %.
  Figures `report/figures/{comparison_recall,seed_variance}.png`. ADRs 19–22.

### Changed
- `train.snapshot_config` is public (used by the baselines).
- `scripts/summarize_runs.py` removed; `fraud-ae compare` writes the tables instead
  (`results/ablations_phase4.md` is kept as the Phase 4 record).
- `.gitignore`: `data/processed_temporal/`.
- Version 0.4.0 → 0.5.0.

## [0.4.0] – 2026-10-27 – Phase 4: evaluation and decision gate (milestone M2)

### Added
- `fraud_ae.evaluate`: `wilson_interval`, `threshold_for_fpr` / `recall_at_fpr` (oracle
  operating points on test legit), `confusion_at_threshold`, `rates_from_confusion`,
  `bootstrap_ci` (stratified, seeded), `evaluate` → `metrics.json` with ROC-AUC, PR-AUC,
  recall@FPR grid, confusion matrix and rates at the val-chosen threshold, Wilson and bootstrap
  CIs, `criterion` (`fp_allowed`, `tp_required`) and `meets_target`; `per_feature_summary`;
  `save_metrics` / `load_metrics`.
- `plots`: `roc_pr_curves` (log FPR axis), `recall_vs_fpr` (criterion region + operating point),
  `threshold_sweep`, `per_feature_residuals`.
- CLI: `fraud-ae evaluate --run <dir> [--synthetic]` (first and only reader of test labels),
  `train --seed`, `all --synthetic`; `fraud-ae all` now runs phases 1–4.
- `train` rejects processed splits whose feature count disagrees with the config.
- Configs `ablation_score_standardized.toml`, `ablation_score_mahalanobis.toml`;
  `ablation_no_time.toml` now uses its own `data/processed_no_time`.
- `scripts/run_ablations.sh` (decision-gate runs through the CLI) and
  `scripts/summarize_runs.py` → `results/ablations_phase4.md` (tracked).
- Tests: `tests/test_evaluate.py` (17, the two Phase 4 xfails flipped to pass), four CLI tests
  (evaluate end-to-end, missing threshold → exit 1, `all --synthetic`, feature-count guard).
- Real results: `runs/ae_base` test recall 77.2 % [71.5 %, 82.1 %] at FPR 0.955 %, criterion
  not met; 11 gate runs at 75.9–78.5 %. Figures `report/figures/*_test_ae_base.png`.
  ADRs 15–18 in `ARCHITECTURE.md`.

### Changed
- `config.toml` run snapshot is the *effective* config (with a header) when `--epochs` /
  `--seed` overrides were used, instead of a verbatim copy that contradicts the run.
- `.gitignore`: `data/processed_no_time/`.
- Version 0.3.0 → 0.4.0.

## [0.3.0] – 2026-10-27 – Phase 3: scoring and threshold

### Added
- `fraud_ae.scoring`: `residuals`, `calibrate` (val-legit residual mean / std / population
  covariance with ridge), `score` with `mse` / `standardized` / `mahalanobis`,
  `score_calibration.npz` and `scores_{val,test}.npz` I/O.
- `fraud_ae.threshold`: `quantile_threshold` (primary, val-legit only), `youden_threshold`,
  `max_f1_threshold` (semi-supervised, comparison only), `choose_threshold` → `ThresholdResult`
  (now also records score method and val class counts), `threshold.json` I/O.
- `plots.score_histogram`: log-score densities per class with the threshold, legit FPR and recall.
- CLI: `fraud-ae threshold --run <dir> [--method] [--strategy] [--target-fpr] [--synthetic]`;
  reads the run's `config.toml` snapshot; `fraud-ae all` now runs phases 1–3.
- Tests: `tests/test_scoring.py` (11) and `tests/test_threshold.py` (10, the Phase 3 xfail
  flipped to pass), two CLI end-to-end tests; `synthetic_splits` fixture moved to `conftest.py`.
- Real run `runs/ae_base`: `mse` + quantile → threshold 1.862, val FPR 0.900 %, val recall
  84.3 %; `standardized` 84.7 %, `mahalanobis` 85.2 % at the same val FPR. Figure copied to
  `report/figures/score_histogram_val_ae_base.png`. ADRs 13–14 in `ARCHITECTURE.md`.

### Changed
- CLI exits with code 1 (not a traceback) on missing run artifacts or invalid overrides.
- Version 0.2.0 → 0.3.0.

## [0.2.0] – 2026-10-26 – Phase 2: training

### Added
- `fraud_ae.train`: `seed_everything` (random/numpy/torch + deterministic algorithms), manual
  seeded mini-batching, Adam + `ReduceLROnPlateau`, `EarlyStopping` on val-legit MSE, best
  checkpoint `model.pt`, `history.csv`, `train_summary.json`, verbatim `config.toml` snapshot,
  `load_model` / `load_history` / `load_summary`, `mse_per_sample`.
- `plots.training_curves`: train vs val-legit MSE (log scale) with the best epoch marked and the
  learning-rate schedule in a lower panel.
- CLI: `fraud-ae train --run <dir> [--synthetic] [--epochs N]` with a one-line summary.
- `tests/test_train.py` (11 tests): early-stopping logic, run-directory artifacts, checkpoint
  round-trip, convergence, fraud-vs-legit reconstruction gap on the synthetic split, identical
  losses and weights for the same seed, early-stop trigger, shape validation; two CLI
  end-to-end tests.
- First real run `runs/ae_base` (milestone M1): val-legit MSE 0.770 → 0.420 in 100 epochs / 31 s;
  figure copied to `report/figures/training_curves_ae_base.png`.
- `report/CHANGELOG.md` (this file); ADRs 11–12 in `ARCHITECTURE.md`.

### Changed
- `.gitignore`: editor/OS block added; comment on the `*_log.md` rule (it does not affect the
  already-tracked `report/prompt_log.md`).
- Version 0.1.0 → 0.2.0 (`pyproject.toml`, `fraud_ae.__version__`, `uv.lock`).

## [0.1.0] – 2026-10-26 – Phase 1: data

### Added
- `fraud_ae.data`: zip extraction, sha256 pin check, schema check, duplicate removal before the
  split, `log1p(Amount)`, hour-of-day sin/cos, seeded legit-only split (60/20/20, frauds 50/50
  val/test), train-only `StandardScaler`, artifacts `splits.npz` / `scaler.joblib` /
  `feature_names.json` / `data_report.json`, `split_table_markdown`.
- `fraud_ae.synthetic`: 2 050-row creditcard-like frame shared by the tests and
  `fraud-ae prepare --synthetic` (writes to `data/processed_synthetic/`).
- EDA figures (`plots.eda_*`): class ratio, amount distribution per class, traffic and fraud
  rate by hour → `report/figures/eda_*.png`.
- `tests/test_data.py` (19 tests): split invariants, leakage controls, determinism, artifact
  round-trip, checksum and schema errors.
- Real dataset processed: 1 081 duplicates dropped, 473 frauds, split 169 952 / 56 887 / 56 887;
  CSV sha256 pinned in all configs.

### Changed
- `cli.py` resolves paths against the current working directory at call time; config and data
  errors exit with code 1.
- README, ROADMAP and ARCHITECTURE updated with the observed numbers (criterion now ≤ 566 false
  alarms and ≥ 202 of 237 test frauds).

## [0.0.1] – 2026-10-25 – Phase 0: scaffold

### Added
- Package layout `src/fraud_ae/` with module stubs, `uv` project with CPU-only torch index,
  TOML configs (`base` + ablations) parsed into frozen dataclasses with validation, `fraud-ae`
  CLI skeleton, pytest suite on a synthetic frame, `ROADMAP.md`, `ARCHITECTURE.md`,
  `report/prompt_log.md`.
