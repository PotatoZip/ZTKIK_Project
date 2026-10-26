# Architecture – Project 12 (Autoencoder fraud detection)

## 1. Why the design looks like this

The abstract asks for four phases (preprocess → train on legit → threshold → evaluate) and a hard
criterion (recall ≥ 85 % at FPR < 1 %). Three facts shape everything below:

1. **Extreme imbalance** (492 frauds in 284 807 rows): only ROC-AUC, PR-AUC and recall-at-FPR are
   meaningful; every recall figure needs a confidence interval.
2. **The threshold must come from legitimate validation data only**, otherwise the 85 % / 1 % claim
   is leaked from the test set.
3. **The criterion is borderline for a plain MSE autoencoder** (literature: recall 0.80–0.90 at
   1 % FPR). The design therefore keeps scoring and architecture configurable so the gap can be
   closed by disclosed, reproducible ablations rather than by hand-tuning.

Quantified criterion (after Phase 1, seed 42): with 56 650 legitimate test rows, FPR < 1 %
allows ≤ 566 false alarms; with 237 test frauds, recall ≥ 85 % means ≥ 202 detected. Precision
at this operating point is ~26–40 %, which is normal for an alerting system and should be stated
in the report.

## 2. Data flow

```
dataset.zip ─unzip→ data/raw/creditcard.csv ─sha256→ DataFrame
  → drop exact duplicates (1 081 rows, 19 of them frauds; dedupe BEFORE splitting)
  → features:  Amount → log1p(Amount)
               Time   → hour-of-day → sin, cos      (config data.time_feature = "sincos" | "drop")
               V1..V28 unchanged
  → seeded split:  legit 60 / 20 / 20 % → train / val / test
                   fraud  0 / 50 / 50 %             (config data.fraud_val_fraction)
  → StandardScaler fit on TRAIN only
  → data/processed/{splits.npz, scaler.joblib, feature_names.json, data_report.json}
    + report/figures/eda_{class_ratio,amount_distribution,fraud_rate_by_hour}.png
  → train.py  → runs/<name>/{config.toml, model.pt, history.csv}
  → scoring.py + threshold.py → scores_{val,test}.npz, threshold.json
  → evaluate.py + plots.py    → metrics.json (meets_target), figures/*.png
  → baselines.py              → runs/baseline_*/…, results/comparison.md
```

Invariants enforced by `tests/test_data.py`: no fraud in train; index sets pairwise disjoint and
covering; split reproducible from the seed; scaler statistics equal train statistics; no identical
feature rows across splits; sha256 mismatch and schema errors rejected; `splits.npz` round-trips.

`data_report.json` holds everything the report needs from this phase: sha256, row counts,
duplicates, class balance, per-split sizes, amount summary per class, per-hour traffic and fraud
rate, scaler mean/scale, figure paths and the `[data]` config snapshot. `idx_*` are positions in
the de-duplicated frame, so any split row can be traced back to the CSV.

## 3. Components (`src/fraud_ae/`)

| Module | Responsibility | Key API |
|---|---|---|
| `config.py` | frozen dataclasses from TOML (stdlib `tomllib`), cross-field validation | `load_config(path) -> Config` |
| `data.py` | Phase 1 pipeline and artifact I/O | `prepare(cfg, root, figures_dir) -> (Splits, report)`, `load_splits(dir)`, `split_table_markdown(report)` |
| `synthetic.py` | 2 k-row creditcard-like frame for tests and `prepare --synthetic` (ADR 9) | `make_synthetic_frame(n_legit, n_fraud, n_duplicates, seed)` |
| `model.py` | symmetric MLP autoencoder | `Autoencoder`, `build_model(input_dim, cfg)` |
| `train.py` | seeding, manual seeded mini-batching, early stopping on val-legit MSE, best checkpoint, `ReduceLROnPlateau` | `train(cfg, X_train, X_val_legit, run_dir) -> Autoencoder`, `load_model(run_dir)`, `mse_per_sample(model, X)` |
| `scoring.py` | residual-based anomaly scores | `score(model, X, cfg, calib)` with `mse` / `standardized` / `mahalanobis` |
| `threshold.py` | threshold strategies | `choose_threshold(scores_val, y_val, cfg) -> ThresholdResult` |
| `evaluate.py` | metrics + CIs + criterion check | `evaluate(scores, y, threshold, cfg) -> dict` |
| `baselines.py` | PCA reconstruction, Isolation Forest on the same splits | `run_baselines(cfg, splits, runs_root)` |
| `plots.py` | all figures, headless matplotlib, shared two-class palette | one function per figure; `eda_figures(df, out_dir)` |
| `cli.py` | `fraud-ae prepare [--synthetic]|train|threshold|evaluate|baselines|all` | `main(argv)`; paths resolve against the cwd |

### Model
`input_dim (31 or 29) → 20 → 10 → 5 → 10 → 20 → input_dim`, ELU, linear output, 1 836 parameters.
Adam (lr 1e-3, wd 1e-5), MSE loss, batch 512, ≤ 100 epochs, early stopping patience 10,
`ReduceLROnPlateau(factor 0.5, patience 5)`. One epoch on the 170 k training rows takes well under
a second on CPU, so sweeps are cheap.

### Training loop (Phase 2)
* Only `X_train` (legit) is fitted; model selection uses the MSE on the **legitimate** part of
  the validation split (`X_val[y_val == 0]`). No fraud label is read during training.
* Determinism: `seed_everything` seeds `random`/`numpy`/`torch` and turns on
  `torch.use_deterministic_algorithms(True)`; batches come from `torch.randperm` driven by a
  dedicated seeded `Generator` instead of a `DataLoader`, so two runs with the same seed produce
  identical losses and bit-identical weights (tested; only the wall-clock column of
  `history.csv` differs).
* The best state dict (lowest val-legit MSE) is kept in memory and written once at the end,
  together with `history.csv` and `train_summary.json` (epochs run, best epoch, parameter
  count, wall time). The training-curve figure is drawn from `history.csv`.

### Scoring
* `mse` – per-sample mean squared residual (abstract's method, primary).
* `standardized` – residuals scaled by per-feature residual std (val-legit), squared, summed.
* `mahalanobis` – residual vector vs covariance of val-legit residuals.

### Threshold
* `quantile` (primary) – `(1 − target_fpr)` quantile of **val-legit** scores, `target_fpr = 0.009`
  to leave margin so the test FPR lands below 1 %.
* `youden`, `max_f1` – semi-supervised (use val frauds); reported for comparison only.

### Evaluation
Threshold-free: ROC-AUC, average precision, recall at FPR ∈ {0.1, 0.5, 1, 2} %.
At threshold: confusion matrix, precision, recall, FPR, F1, Wilson CIs, 1 000-sample bootstrap CI
for recall@1 % FPR, `meets_target`.

### Run-directory contract
```
runs/<name>/ config.toml  model.pt  history.csv  train_summary.json
             scores_val.npz  scores_test.npz  threshold.json  metrics.json  figures/*.png
```
Baselines write the same files (minus `model.pt`/`history.csv`) so one comparison routine works
for everything.

## 4. Architecture decision records

| ADR | Decision | Rationale |
|---|---|---|
| 1 | Python package + CLI; notebooks optional and logic-free | reproducible, testable, maps 1:1 onto the course phases |
| 2 | Threshold = val-legit quantile at 0.9 % | satisfies "FPR < 1 %" by construction, no label leakage, margin for sampling noise |
| 3 | Frauds split 50/50 val/test (`fraud_val_fraction`) | honest model comparison on val, untouched test; set to 0 to put all frauds in test |
| 4 | Dedupe before split | identical rows would otherwise sit in both train and test |
| 5 | `Time` → hour-of-day sin/cos, ablated | raw seconds over a 2-day window are a clock, not a feature |
| 6 | No BatchNorm by default | inference-time batch statistics distort per-sample anomaly scores |
| 7 | CPU-only torch via pinned uv index | no GPU on the machine; model is tiny |
| 8 | TOML configs via stdlib `tomllib`, frozen dataclasses | zero extra dependency on Python 3.11; immutable run config |
| 9 | Tests and `prepare --synthetic` share one generated 2 k-row frame (`synthetic.py`) | suite needs no 150 MB download and runs in seconds; the CLI path can be exercised end-to-end on any machine; synthetic outputs go to `data/processed_synthetic/` so real artifacts are never overwritten |
| 10 | Dataset sha256 pinned in every config, checked by `prepare` | a different CSV (re-download, partial extract) fails loudly instead of silently changing the numbers |
| 11 | Manual seeded mini-batching (`torch.randperm` + `Generator`) instead of `DataLoader` | tensors fit in memory; no worker processes, no shuffling non-determinism, one fewer moving part for the "same seed → same losses" exit criterion |
| 12 | `--synthetic` flag on `prepare` and `train` | every phase can be smoke-tested end-to-end in seconds on the 2 k-row generated set before touching the real data |

## 5. Threat model and limitations (to be expanded in the report)
* Adversary: executes unauthorized transactions that break the card-holder's usual pattern.
  The AE models "usual" as low reconstruction error on the legitimate manifold.
* The features V1–V28 are PCA-obfuscated and there is no account identifier, so per-account
  velocity/limit features cannot be built; the detector is a **point-anomaly** detector.
* Adversarial evasion (crafting transactions with low reconstruction error) is discussed, not
  implemented.
* A random split is the literature-comparable protocol; a temporal split (train on earlier
  transactions) is run as an ablation because it better reflects concept drift in production.
