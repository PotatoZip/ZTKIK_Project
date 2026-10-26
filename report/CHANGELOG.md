# Changelog – Project 12 (Autoencoder fraud detection)

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the project phases in
`ROADMAP.md`. Dates are the dates of the work sessions.

## [Unreleased]

Planned: Phase 3 (scoring + threshold), Phase 4 (evaluation + criterion check), Phase 5
(baselines + ablations), Phase 6 (hardening + report).

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
