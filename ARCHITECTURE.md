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
     or temporal:  legit ordered by Time, cut 60 / 20 / 20 %; frauds by their own Time,
                   train-window frauds dropped      (config data.split = "temporal", ablation)
  → StandardScaler fit on TRAIN only
  → data/processed/{splits.npz, scaler.joblib, feature_names.json, data_report.json}
    + report/figures/eda_{class_ratio,amount_distribution,fraud_rate_by_hour}.png
  → train.py  → runs/<name>/{config.toml, model.pt, history.csv}
  → scoring.py + threshold.py → scores_{val,test}.npz, threshold.json
  → evaluate.py + plots.py    → metrics.json (meets_target), figures/*_test.png
  → baselines.py              → runs/baseline_{pca_k5,pca_k10,pca_k15,iforest}[_temporal]/…
  → baselines.write_comparison (fraud-ae compare) → results/comparison.md,
                                 report/figures/{comparison_recall,seed_variance}.png
  → report.py (fraud-ae report) → report/report.md rendered from report/report_template.md
  → scripts/run_ablations.sh  → all AE variants + baselines + comparison + report in one go
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
| `scoring.py` | residual-based anomaly scores + score/calibration I/O | `residuals`, `calibrate(model, X_val_legit)`, `score(model, X, cfg, calib)` with `mse` / `standardized` / `mahalanobis`, `save_scores` / `load_scores` |
| `threshold.py` | threshold strategies, alarm rule `score > threshold` | `quantile_threshold`, `youden_threshold`, `max_f1_threshold`, `choose_threshold(scores_val, y_val, cfg, score_method) -> ThresholdResult`, `save_threshold` / `load_threshold` |
| `evaluate.py` | metrics + CIs + criterion check | `evaluate(scores, y, threshold, cfg, seed) -> dict`, `recall_at_fpr`, `threshold_for_fpr`, `wilson_interval`, `bootstrap_ci`, `per_feature_summary`, `save_metrics` / `load_metrics` |
| `baselines.py` | PCA reconstruction, Isolation Forest on the same splits; comparison table across all runs | `pca_scores`, `iforest_scores`, `run_baselines(cfg, splits, runs_root, suffix)`, `comparison_rows`, `seed_families`, `write_comparison(run_dirs, out_md, figures_dir)` |
| `plots.py` | all figures, headless matplotlib, shared two-class palette | one function per figure; `eda_figures(df, out_dir)` |
| `report.py` | render `report/report.md` from `report/report_template.md`; placeholders resolve to JSON artifact values with arithmetic and `mean/std/min/max` | `render(template, root)`, `render_report(root, template, out)`, `references(template)` |
| `cli.py` | `fraud-ae prepare|train [--epochs] [--seed]|threshold|evaluate|baselines|all` (all with `--synthetic`) `compare [--runs] [--out] [--figures-dir]` and `report [--template] [--out]` | `main(argv)`; paths resolve against the cwd |
| `scripts/` | `run_ablations.sh`: every AE variant, both baseline sets, the comparison and the report through the CLI; `check.sh`: format check + lint + tests | – |

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
* `standardized` – residuals centred and scaled by the per-feature residual mean/std of
  val-legit, squared, summed (≈ χ² with `d` degrees of freedom on legit rows).
* `mahalanobis` – squared Mahalanobis distance of the residual vector w.r.t. the mean and
  (population, ridge 1e-6) covariance of val-legit residuals; equals `standardized` when the
  residual covariance is diagonal (tested).
* Calibration statistics are computed on `X_val[y_val == 0]` only and stored in
  `score_calibration.npz`; `scores_{val,test}.npz` hold `scores` + `y` for Phase 4.

### Threshold
* `quantile` (primary) – `(1 − target_fpr)` quantile of **val-legit** scores, `target_fpr = 0.009`
  to leave margin so the test FPR lands below 1 %. Observed on `runs/ae_base`: val FPR 0.900 %.
* `youden`, `max_f1` – semi-supervised (use val frauds); reported for comparison only. sklearn's
  curve thresholds mean `score >= t`; they are mapped to the project-wide rule `score > t` by
  stepping to the largest score strictly below `t`.
* `threshold.json` records strategy, target, value, val FPR / recall, score method and the val
  class counts. The `threshold` command reads the run's `config.toml` snapshot (not the global
  config) and lets `--method / --strategy / --target-fpr` override single fields.

### Evaluation (Phase 4)
* Test labels are read here for the first time (`scores_test.npz` written in Phase 3).
* Threshold-free, i.e. ranking quality of the score: ROC-AUC, average precision (PR-AUC) and
  recall at FPR ∈ `eval.fpr_grid` = {0.1, 0.5, 1, 2} %. For the grid the threshold is set on the
  **test** legit scores so that exactly `floor(fpr · n_legit)` legit rows alarm
  (`threshold_for_fpr`); this is an oracle operating point and is reported as such.
* At the Phase 3 threshold (chosen on val-legit, never adjusted here): confusion matrix,
  precision, recall, FPR, F1; Wilson 95 % intervals for recall, FPR and precision; a stratified
  bootstrap (`eval.bootstrap_samples` = 1 000 resamples of legit and fraud rows separately,
  seeded) giving percentile CIs for recall and FPR at the threshold and for recall@`target_fpr`.
* `meets_target` = recall ≥ `eval.target_recall` **and** FPR < `eval.target_fpr` on the point
  estimates, exactly as the abstract states the criterion; `criterion` also records
  `fp_allowed` and `tp_required` so the verdict is checkable by hand. The CIs qualify the verdict
  but do not change it.
* Explainability: `per_feature` = mean squared residual per input feature for legit vs fraud
  rows and the five features with the largest fraud/legit ratio.
* Figures (`figures/*_test.png`): ROC (log FPR axis) + PR, recall-vs-FPR with the criterion
  region and the val-chosen operating point, threshold sweep (recall / FPR / precision),
  test-score histogram, per-feature residuals.
* Decision gate: when `meets_target` is false, `scripts/run_ablations.sh` produces one run
  directory per variant (score method, latent size, no time feature, 200 epochs, five seeds)
  through the same three CLI commands, and `scripts/summarize_runs.py` tabulates them. Score
  variants are retrained from their own config rather than re-scoring `runs/ae_base`: training
  is deterministic, so the checkpoint is byte-identical and each run stays self-contained.

### Baselines and comparison (Phase 5)
* PCA(k), k ∈ {5, 10, 15}: `sklearn.decomposition.PCA(svd_solver="full")` fitted on train
  legit; score = mean squared reconstruction error per row, i.e. the same units as the AE's
  `mse`. A linear autoencoder with the same information bottleneck as the AE's latent 5.
* Isolation Forest: 200 trees, `max_samples="auto"` (256), seeded, fitted on train legit;
  score = `−score_samples` (higher = more anomalous).
* Both use the same `choose_threshold` (0.9 % val-legit quantile) and the same `evaluate`, and
  write the AE run-directory contract minus `model.pt` / `history.csv`, so `write_comparison`
  needs no special cases. `fraud-ae baselines --config configs/ablation_temporal.toml` gives the
  temporal-split baselines (`runs/baseline_*` from that config's `processed_dir`).
* `fraud-ae compare` reads every `runs/*/metrics.json` (synthetic runs excluded, `--runs` to
  pick), sorts by test recall, prints the Wilson intervals and the oracle recall@1 % FPR, groups
  `<family>_seed<n>` directories into a seed-variance table (mean ± sample std), and draws the
  bar chart (`comparison_recall.png`) and the seed box plot (`seed_variance.png`).
* Temporal split (`data.split = "temporal"`): legit rows sorted by `Time`, cut at 60 / 80 %;
  frauds assigned to val/test by their own `Time`; frauds in the train window are dropped and
  counted in `data_report.json` (`rows.frauds_dropped_train_window`, `time_windows`). It is the
  production-like protocol (train on the past, test on the future) and is run for the AE and
  the baselines; it has fewer frauds per split, so wider intervals.

### Report rendering (Phase 6)
`report/report_template.md` is the source of truth; `report/report.md` is generated and
committed so that readers need no artifacts. A placeholder `{{ expr | fmt }}` contains refs
`${path:key.path}` (`path` = a run directory → its `metrics.json`, or an explicit JSON file);
`expr` may combine refs with `+ - * /` and `mean/std/min/max/abs` (evaluated through an
`ast` whitelist, no `eval`); `fmt` ∈ `pct0..3`, `pp1`, `f0..4`, `int`, `verdict`, `str`.
Rendering fails as a whole and lists every unresolved placeholder. The committed report is
compared against a fresh render in the test suite whenever the real runs are present.

### Run-directory contract
```
runs/<name>/ config.toml  model.pt  history.csv  train_summary.json
             score_calibration.npz  scores_val.npz  scores_test.npz  threshold.json
             metrics.json
             figures/{training_curves, score_histogram_val, roc_pr_test, recall_vs_fpr_test,
                      threshold_sweep_test, score_histogram_test, per_feature_residuals_test}.png
```
`config.toml` is a verbatim copy of the config the run was started from; when a CLI override
(`--epochs`, `--seed`) changed it, the file is instead the *effective* config with a header
naming the source file, so a run directory never claims settings it did not use.
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
| 12 | `--synthetic` flag on `prepare`, `train` and `threshold` | every phase can be smoke-tested end-to-end in seconds on the 2 k-row generated set before touching the real data |
| 13 | `threshold` reads the run's `config.toml` snapshot, not `configs/*.toml` | scoring and threshold always refer to the model that was actually trained; variants are explicit CLI overrides recorded in `threshold.json` |
| 14 | Test scores are written in Phase 3 but test labels are first read in Phase 4 | one scoring pass per run; the "test set touched once" leakage control still holds because no metric is computed here |
| 15 | The criterion verdict uses the val-chosen threshold; the recall@FPR grid uses oracle thresholds on test legit | the verdict must reflect what a deployment would do (threshold fixed before seeing test data); the grid separates ranking quality from threshold-transfer error |
| 16 | Wilson intervals for every proportion plus a stratified bootstrap | 237 test frauds make a ±5 pp interval unavoidable; Wilson is closed-form and exact enough, the bootstrap also covers the re-derived threshold in recall@FPR |
| 17 | `config.toml` snapshot = effective config when CLI overrides are used | `--seed` / `--epochs` runs would otherwise carry a snapshot that contradicts `train_summary.json` |
| 18 | One run directory per experiment; ablations are configs + `scripts/run_ablations.sh`, not flags that mutate `runs/ae_base` | every table row maps to one `metrics.json` produced by the same three commands; nothing in the base run is overwritten by a variant |
| 19 | Baselines reuse the AE's threshold rule, evaluator and run-directory contract | a like-for-like comparison: same splits, same 0.9 % val-legit quantile, same CIs; the comparison code has one code path |
| 20 | PCA score = mean squared reconstruction error (same units as AE `mse`) | PCA is the linear special case of the AE; identical scoring isolates the effect of non-linearity |
| 21 | Temporal split drops train-window frauds instead of moving them to val/test | moving them would evaluate on frauds older than the training data; dropping keeps "train on the past, test on the future" honest at the cost of fewer test frauds |
| 22 | `fraud-ae compare` replaces the Phase 4 `summarize_runs.py` script | one Markdown writer for ablations, baselines and the Phase 6 report tables; figures and seed statistics come from the same rows |
| 23 | The report is rendered from a template whose numbers are artifact references, and the rendered file is committed and tested against the artifacts | "every number traceable to a `metrics.json`" becomes a mechanical property instead of a reviewing effort; derived figures (differences, seed means) are computed at render time, never typed |

## 5. Threat model and limitations (to be expanded in the report)
* Adversary: executes unauthorized transactions that break the card-holder's usual pattern.
  The AE models "usual" as low reconstruction error on the legitimate manifold.
* The features V1–V28 are PCA-obfuscated and there is no account identifier, so per-account
  velocity/limit features cannot be built; the detector is a **point-anomaly** detector.
* Adversarial evasion (crafting transactions with low reconstruction error) is discussed, not
  implemented.
* A random split is the literature-comparable protocol; a temporal split (train on earlier
  transactions) is run as an ablation because it better reflects concept drift in production.
