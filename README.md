# Project 12 – Fraud Detection in Bank Transactions with a Deep Autoencoder

ZTKIK course project. A PyTorch autoencoder is trained **only on legitimate** credit-card
transactions; a transaction is flagged as fraud when its reconstruction error (MSE) exceeds a
threshold chosen on legitimate validation data. Success criterion from the abstract:
**recall ≥ 85 % at false-positive rate < 1 %** on a held-out test set.

* Abstract: `tmp.md` · Roadmap: `ROADMAP.md` · Design and decisions: `ARCHITECTURE.md`
* Dataset: Kaggle *Credit Card Fraud Detection* (ULB MLG), 284 807 transactions, 492 frauds.
  Place the Kaggle zip at `dataset.zip` (git-ignored); `fraud-ae prepare` extracts it.

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
uv run fraud-ae train     --config configs/base.toml --run runs/ae_base   # 2. train AE on legit rows
uv run fraud-ae threshold --run runs/ae_base                              # 3. MSE threshold (val-legit quantile)
uv run fraud-ae evaluate  --run runs/ae_base                              # 4. recall / FPR on test, figures
uv run fraud-ae baselines --config configs/base.toml                      # PCA + Isolation Forest
uv run fraud-ae all       --config configs/base.toml --run runs/ae_base   # phases 1-4
```

Each run directory contains `config.toml` (snapshot), `model.pt`, `history.csv`,
`threshold.json`, `metrics.json` (includes `meets_target`) and `figures/`.

## Layout

```
configs/        TOML experiments (base + ablations)
src/fraud_ae/   package: config, data, model, train, scoring, threshold, evaluate, baselines, plots, cli
tests/          pytest suite on a synthetic creditcard-like frame
data/           raw/ (extracted CSV) and processed/ (splits, scaler) – git-ignored
runs/           one directory per experiment – git-ignored
report/         report.md (English), prompt_log.md, figures/
```

## Status

Phase 0 (scaffold) done. Phases 1–6 are tracked in `ROADMAP.md`; CLI sub-commands for
unimplemented phases exit with code 2 and a "not implemented" message.
