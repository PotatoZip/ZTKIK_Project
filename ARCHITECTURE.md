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

Quantified criterion: with ~56.9 k legitimate test rows, FPR < 1 % allows ≤ ~569 false alarms;
with 246 test frauds, recall ≥ 85 % means ≥ 209 detected. Precision at that operating point is
~27–42 %, which is normal for an alerting system and should be stated in the report.

## 2. Data flow

```
dataset.zip ─unzip→ data/raw/creditcard.csv ─sha256→ DataFrame
  → drop exact duplicates (the dataset has ~1 k; dedupe BEFORE splitting)
  → features:  Amount → log1p(Amount)
               Time   → hour-of-day → sin, cos      (config data.time_feature = "sincos" | "drop")
               V1..V28 unchanged
  → seeded split:  legit 60 / 20 / 20 % → train / val / test
                   fraud  0 / 50 / 50 %             (config data.fraud_val_fraction)
  → StandardScaler fit on TRAIN only
  → data/processed/{splits.npz, scaler.joblib, feature_names.json, data_report.json}
  → train.py  → runs/<name>/{config.toml, model.pt, history.csv}
  → scoring.py + threshold.py → scores_{val,test}.npz, threshold.json
  → evaluate.py + plots.py    → metrics.json (meets_target), figures/*.png
  → baselines.py              → runs/baseline_*/…, results/comparison.md
```

Invariants enforced by `tests/test_data.py`: no fraud in train; index sets pairwise disjoint and
covering; scaler statistics equal train statistics.

## 3. Components (`src/fraud_ae/`)

| Module | Responsibility | Key API |
|---|---|---|
| `config.py` | frozen dataclasses from TOML (stdlib `tomllib`), cross-field validation | `load_config(path) -> Config` |
| `data.py` | Phase 1 pipeline and artifact I/O | `prepare(cfg, root) -> Splits`, `load_splits(dir)` |
| `model.py` | symmetric MLP autoencoder | `Autoencoder`, `build_model(input_dim, cfg)` |
| `train.py` | seeding, early stopping on val-legit MSE, checkpoint | `train(cfg, X_train, X_val_legit, run_dir)` |
| `scoring.py` | residual-based anomaly scores | `score(model, X, cfg, calib)` with `mse` / `standardized` / `mahalanobis` |
| `threshold.py` | threshold strategies | `choose_threshold(scores_val, y_val, cfg) -> ThresholdResult` |
| `evaluate.py` | metrics + CIs + criterion check | `evaluate(scores, y, threshold, cfg) -> dict` |
| `baselines.py` | PCA reconstruction, Isolation Forest on the same splits | `run_baselines(cfg, splits, runs_root)` |
| `plots.py` | all figures, headless matplotlib | one function per figure |
| `cli.py` | `fraud-ae prepare|train|threshold|evaluate|baselines|all` | `main(argv)` |

### Model
`input_dim (31 or 29) → 20 → 10 → 5 → 10 → 20 → input_dim`, ELU, linear output, ~2 k parameters.
Adam (lr 1e-3, wd 1e-5), MSE loss, batch 512, ≤ 100 epochs, early stopping patience 10,
`ReduceLROnPlateau`. One epoch ≈ 1–2 s on CPU, so sweeps are cheap.

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
runs/<name>/ config.toml  model.pt  history.csv  scores_val.npz  scores_test.npz
             threshold.json  metrics.json  figures/*.png
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
| 9 | Tests run on a synthetic 2 k-row frame | suite needs no 150 MB download and runs in seconds |

## 5. Threat model and limitations (to be expanded in the report)
* Adversary: executes unauthorized transactions that break the card-holder's usual pattern.
  The AE models "usual" as low reconstruction error on the legitimate manifold.
* The features V1–V28 are PCA-obfuscated and there is no account identifier, so per-account
  velocity/limit features cannot be built; the detector is a **point-anomaly** detector.
* Adversarial evasion (crafting transactions with low reconstruction error) is discussed, not
  implemented.
* A random split is the literature-comparable protocol; a temporal split (train on earlier
  transactions) is run as an ablation because it better reflects concept drift in production.
