# Roadmap – Project 12 (Autoencoder fraud detection)

Goal: a reproducible pipeline and an English report demonstrating **recall ≥ 85 % at FPR < 1 %**
on the Credit Card Fraud dataset, with a deep autoencoder trained on legitimate transactions only,
compared against PCA-reconstruction and Isolation Forest baselines.

## Phases

| # | Phase | Scope | Deliverables | Exit criteria | Effort |
|---|---|---|---|---|---|
| 0 | **Scaffold** ✅ | layout, `pyproject`/uv, configs, package skeleton, tests, docs | this repo | `uv sync`, `fraud-ae --help`, `pytest`, `ruff` all green | 1 session |
| 1 | **Data** (abstract phase 1) ✅ | unzip + sha256, dedupe, `log1p(Amount)`, hour-of-day sin/cos, seeded split (train = legit only), scaler fit on train, EDA | `data.py`, `synthetic.py`, `prepare [--synthetic]`, `data_report.json`, EDA figures (class ratio, Amount distribution, fraud rate by hour) | split invariants tested (`tests/test_data.py`, 19 tests pass); split-size table in README; sha256 pinned in configs | 1 session |
| 2 | **Train** (abstract phase 2) ✅ | training loop, early stopping on val-legit MSE, checkpoints, seeding | `train.py`, `train [--synthetic] [--epochs]`, `runs/ae_base/{config.toml,model.pt,history.csv,train_summary.json}`, `report/figures/training_curves_ae_base.png` | val MSE converges (0.770 → 0.420); same seed → identical losses and byte-identical `model.pt` (tested) | 1 session |
| 3 | **Threshold** (abstract phase 3) ✅ | scoring (`mse`, `standardized`, `mahalanobis`), threshold strategies | `scoring.py`, `threshold.py`, `threshold [--method] [--strategy] [--target-fpr] [--synthetic]`, `runs/ae_base/{scores_val,scores_test,score_calibration}.npz`, `threshold.json`, `report/figures/score_histogram_val_ae_base.png` | val FPR = 0.900 % with the quantile strategy; quantile property tests pass (24 Phase 3 tests) | 0.5 session |
| 4 | **Evaluate** (abstract phase 4) ✅ | ROC/PR-AUC, recall@FPR grid, confusion matrix, Wilson + bootstrap CIs, figures, criterion check | `evaluate.py`, `plots.py`, `evaluate [--synthetic]`, `train --seed`, `runs/ae_base/metrics.json`, `report/figures/*_test_ae_base.png`, `scripts/run_ablations.sh`, `results/ablations_phase4.md` | **Decision gate taken:** `meets_target` false on test (recall 77.2 % [71.5, 82.1] at FPR 0.955 %); scoring variants, latent 3/8, no-time, 200 epochs and 5 seeds all land at 75.9–78.5 % → shortfall documented, Phase 5 compares against baselines | 1 session |
| 5 | **Baselines & ablations** ✅ | PCA(k ∈ {5,10,15}), Isolation Forest; ablations: scoring method, latent size, Time feature, BatchNorm, 5 seeds, temporal split | `baselines.py`, `baselines [--synthetic]`, `compare`, `configs/ablation_{batchnorm,temporal}.toml`, `data.split = "temporal"`, `results/comparison.md` (21 runs), `report/figures/{comparison_recall,seed_variance}.png` | comparison table complete with Wilson CIs; seed variance 76.5 % ± 0.6 % (5 seeds); AE 77.2 % vs best baseline PCA k=15 70.0 % at the same FPR | 1 session |
| 6 | **Hardening & report** ✅ | tests green, README final, `report/report.md` (English), `report/prompt_log.md`, limitations + threat-model discussion | `report.py`, `fraud-ae report`, `report/report_template.md` → `report/report.md`, `scripts/check.sh`, README final, CHANGELOG 1.0.0 | every number in the report is a placeholder resolved from a JSON artifact (tested: committed report ≡ render of the artifacts); `fraud-ae all` + `scripts/run_ablations.sh` reproduce everything in ≈ 10 min; 109 tests, ruff clean | 1 session |

Milestones: **M1** first trained AE (end of Phase 2) · **M2** criterion verdict (end of Phase 4) ·
**M3** submission package (end of Phase 6). Total ≈ 6–7 working sessions.

### Phase 1 outcome (2026-10-26)

* Real data: 284 807 rows → 1 081 exact duplicates dropped (19 frauds among them) → 283 726 rows,
  473 frauds (0.167 %). Split (seed 42): train 169 952 legit / 0 fraud; val 56 651 / 236;
  test 56 650 / 237. `prepare` runs in ≈ 5 s on CPU.
* Consequence for the criterion: FPR < 1 % on test allows ≤ 566 false alarms; recall ≥ 85 %
  means ≥ 202 of 237 test frauds detected.
* EDA supports ADR 5: the fraud rate at hours 2–4 is 6–9× the overall rate while traffic is
  lowest; fraud amounts are smaller in median (9.82 vs 22.00) but heavier-tailed.
* `fraud-ae prepare --synthetic` runs the identical pipeline on a 2 050-row generated frame
  (`fraud_ae.synthetic`) into `data/processed_synthetic/`; the test suite uses the same frame.

### Phase 2 outcome (2026-10-26, milestone M1)

* `runs/ae_base`: 1 836-parameter AE trained on 169 952 legit rows; val-legit MSE
  0.770 → 0.420 (best epoch 99 of 100), train MSE 0.418, 31 s on CPU. The epoch cap binds before
  early stopping – the curve is flat but still creeping; `train.epochs = 200` is a cheap Phase 4
  ablation.
* Synthetic smoke run (`--synthetic`, 1 200 rows): 100 epochs in 0.2 s; frauds reconstruct with
  more than 2× the median MSE of legit rows (tested), i.e. the premise holds on the toy set.
* Determinism: unit test on the synthetic split plus two 3-epoch runs on the real data give
  identical train/val losses and byte-identical checkpoints (only the timing column differs).

### Phase 3 outcome (2026-10-27)

* `runs/ae_base`, `mse` + quantile (target 0.9 %): threshold 1.862, val FPR 0.900 %
  (510 of 56 651 legit rows), val recall 84.3 % (199 of 236). Plain MSE therefore sits exactly in
  the borderline band anticipated in the risk table.
* Score variants at the same val FPR: `standardized` 84.7 %, `mahalanobis` 85.2 %
  (recall@val-FPR 0.1 % jumps from 55.5 % to 69.5 % with Mahalanobis). Semi-supervised
  strategies: Youden 2.1–2.5 % FPR / 87 % recall, max-F1 ≈ 0.2 % FPR / 74–77 % recall.
* These are validation numbers; the Phase 4 decision gate is taken on the untouched test set.
  Expected: plain MSE narrowly misses 85 %, Mahalanobis is the first candidate to try.
* Synthetic smoke run: val FPR 1.0 % (granularity 1/400), val recall 100 %; all three methods
  reach ≥ 50 % recall at 5 % FPR in the tests.

### Phase 4 outcome (2026-10-27, milestone M2)

* `runs/ae_base` on the untouched test set (56 650 legit / 237 fraud) at the val-chosen
  threshold 1.862: TP 183, FP 541 → recall **77.2 %** (Wilson [71.5 %, 82.1 %], bootstrap
  [71.7 %, 82.3 %]), FPR 0.955 % [0.878 %, 1.038 %], precision 25.3 %. ROC-AUC 0.938, PR-AUC
  0.538. Criterion **not met** (needs TP ≥ 202, FP ≤ 566; FPR part holds).
* Ranking quality, not threshold transfer, is the limit: recall at oracle test FPR 1 % is also
  77.2 %, the curve plateaus at 77–79 % from 0.3 % to 2 % FPR and crosses 85 % only near 7 %.
  About a fifth of the test frauds reconstruct like legit rows. Largest fraud/legit residual
  ratios: V17, V14, V12, V10, V16.
* Decision gate executed (`scripts/run_ablations.sh`, `results/ablations_phase4.md`, 11 runs,
  ≈ 40 s each): `standardized` 76.4 %, `mahalanobis` 77.6 %, latent 3 / 8 → 78.5 % / 77.6 %,
  no hour-of-day feature 78.5 %, 200 epochs 77.2 %, seeds 0–3 → 75.9–76.8 % (5-seed mean
  76.5 %, spread ±0.7 pp). Every upper confidence bound is ≤ 83.2 %. Validation recall is
  83–86 % for all variants, so the val half of the frauds is systematically easier than the
  test half (6–8 pp); with 236 / 237 frauds per half this is within the sampling variance the
  risk table anticipated, but it means the 85 % claim cannot be made for this model family.
* Found while running the gate: `ablation_no_time.toml` shared `data/processed` with base, so
  its first run silently trained on 31 features. Fixed with a per-config `processed_dir` and a
  feature-count guard in `train` (tested).
* Synthetic smoke run (`evaluate --synthetic`): ROC-AUC 1.0, recall 100 %, FPR 1.5 %
  (6 of 400 – the 0.9 % quantile on 400 val rows has 0.25 % granularity), i.e. the toy set
  exercises the code path but is too small to pass the FPR part by construction.
* Consequence for Phases 5–6: the report presents the plain and the best configuration with
  CIs, compares against PCA / Isolation Forest, and discusses the shortfall (point-anomaly
  detector on PCA-obfuscated features, fraud subpopulation indistinguishable by reconstruction
  error) instead of tuning on the test set.

### Phase 5 outcome (2026-10-27)

* Baselines on the random split (`fraud-ae baselines --config configs/base.toml`, < 1 s each):
  PCA k = 5 / 10 / 15 → test recall 58.6 % / 59.9 % / 70.0 % at FPR 0.91–0.95 %; Isolation
  Forest (200 trees) 56.1 % at 0.97 %. The AE (77.2 %) beats the linear autoencoder with the
  same bottleneck (PCA k = 5) by 19 pp and the forest by 21 pp, while all ROC-AUCs sit at
  0.93–0.94 – the advantage lives entirely in the low-FPR region.
* Remaining ablations: BatchNorm 75.9 % at FPR 1.001 % (fails both halves; ADR 6 confirmed);
  temporal split (`data.split = "temporal"`, 342 of 473 frauds fall into the training window
  and are dropped → 57 val / 74 test frauds): AE 73.0 % [61.9, 81.8] at FPR 0.528 %, oracle
  recall@1 % 79.7 %; PCA k = 15 60.8 %, PCA k = 5 43.2 %, Isolation Forest 39.2 %. The val-legit
  quantile transfers to roughly half the FPR on later traffic (drift), which costs the AE 7 pp
  of recall; the ranking is unchanged.
* Seed variance (5 seeds): val 83.7 % ± 0.6 %, test 76.5 % ± 0.6 %; the 7 pp val–test gap is
  systematic. `results/comparison.md` (21 runs) and the two figures are generated by
  `fraud-ae compare`; `scripts/run_ablations.sh` reproduces every run from scratch
  (≈ 10 min on CPU).
* Found and fixed: the temporal baselines initially overwrote the random-split ones (same run
  names); baseline run names now carry the split (`_temporal`).
* Conclusion for the report: no model in this study reaches 85 % recall at 1 % FPR on this
  split; the deep AE is the best of the compared unsupervised detectors by a wide margin, and
  the shortfall is a property of the fraud subpopulation that is reconstructable from legitimate
  structure, not of threshold choice, bottleneck size, time encoding, normalisation or seed.

### Phase 6 outcome (2026-10-27, milestone M3)

* `src/fraud_ae/report.py` + `fraud-ae report`: `report/report.md` is rendered from
  `report/report_template.md`; placeholders reference `runs/<run>/{metrics,threshold,
  train_summary}.json` and `data/processed*/data_report.json`, with arithmetic and
  `mean/std/min/max` for derived figures (seed statistics, differences in pp). The template holds
  about 310 references; the unit test `test_tracked_report_matches_artifacts` fails if the
  committed report and the artifacts ever disagree.
* Report (≈ 3 900 words, 10 figures, 7 tables) follows the planned outline: threat model and the
  precision consequence of the criterion, data/leakage controls, model, threshold, protocol,
  primary verdict, ranking analysis, ablations, seed variance, baselines, temporal split,
  per-feature residuals, discussion, limitations, conclusions, reproducibility, references.
* Hardening: `scripts/check.sh` (format check, lint, tests, version); `scripts/run_ablations.sh`
  ends with the figure copy and `fraud-ae report`, so one script regenerates every table,
  figure and the report from scratch. 109 tests pass; ruff format/check clean.
* Final verdict carried into the report: recall 77.2 % [71.5 %, 82.1 %] at FPR 0.955 % –
  criterion not met; AE best of the compared detectors (+7.2 pp over PCA k = 15, +21.1 pp over
  Isolation Forest at the same FPR).

## Roadmap v2 – reaching the criterion (Phases 7–11)

v1 stopped at 77.2 % test recall and showed that no cheap knob moves it. Before planning v2 a
read-only diagnosis of the v1 artifacts was run (ad-hoc analysis of 2026-10-27 on the v1 test
half; **reproduced on all 473 frauds by `fraud-ae diagnose` in Phase 7, see the Phase 7
outcome below** – the numbers in this table are the v1-half originals):

| Finding | Evidence (v1 artifacts, `runs/ae_base` test split) |
|---|---|
| The 54 missed frauds lie on the legitimate manifold | means of V14 / V17 / V12 / V10 ≈ −1.8 / +1.7 / −0.7 / −0.5 for missed frauds vs ≈ −8 for detected frauds and ≈ 0 for legit rows; missed-fraud score median 0.61 vs legit median 0.29 (threshold 1.86) |
| No unsupervised detector catches them | at their own 1 % FPR: PCA k = 15 catches 1 of 54, Isolation Forest 5, Mahalanobis 2, latent 3 → 4, no-time → 4, another seed 2 |
| Fusion and ensembles do not help | rank-average AE + PCA + IF 77.6 %, 5-seed MSE ensemble 76.4 %; the *union* of three detectors reaches 79.7 % only at 1.77 % FPR |
| Missed frauds are daytime, small amounts | flat over hours (detected frauds cluster at 02–04 h and 11 h); median amount 4.79 vs 14.46 for detected |
| Pooled estimate over all 473 frauds (legit-only threshold) | 80.8 % – the val half (84.3 %) and the test half (77.2 %) are two draws of the same detector |
| Labels help, but the test half is hard | logistic regression / gradient boosting trained on train-legit + the 236 val frauds: 83.5 % / 82.3 % recall at 1 % FPR on the test half; with 100 labelled frauds 78.8 %, 50 → 75.9 %, 25 → 70.3 % |

Consequences: (1) a strictly unsupervised point-anomaly detector on these features has a
ceiling near 80 % at 1 % FPR, so the honest expectation for legit-only methods is "document the
ceiling", not "reach 85 %"; (2) crossing 85 % needs labelled frauds *inside* the model,
evaluated out-of-fold, plus the right measurement protocol (all frauds evaluated, repeated
splits). Decisions taken with the project owner: labels may be used; the autoencoder stays the
central model of the narrative, and **fully supervised classifiers are first-class candidates**
(the dataset is labelled, so the question "what do labels buy over the legit-only detector" is
answered head-on rather than with a reference ceiling); evaluation moves to pooled frauds with
repeated legit splits – the threshold never uses a fraud label, so all 473 frauds can be test
frauds (ADR 3, `fraud_val_fraction = 0`). Every method, unsupervised or supervised, is
thresholded the same way (legit quantile of its own score) and judged by the same criterion.

| # | Phase | Scope | Deliverables | Exit criteria / decision gate | Effort |
|---|---|---|---|---|---|
| 7 | **Diagnosis & measurement protocol** ✅ | pooled-fraud protocol (`fraud_val_fraction = 0`), repeated legit splits (`--data-seed`, 5 seeds), `fraud-ae diagnose` (missed-fraud profiles, overlap matrix, fusion upper bounds), literature-protocol replication (all frauds in test, no dedupe) | `configs/v2_{pooled,literature}.toml`, `--data-seed` on `prepare/train/baselines/all` (`<processed_dir>_dseed<N>`; `threshold`/`evaluate` follow the run snapshot), `baselines --suffix`, `compare --figure-prefix`, `diagnose.py` + `fraud-ae diagnose` → `diagnostics.json` + `figures/missed_frauds.png`, `scripts/run_v2_protocol.sh`, `results/v2_protocol.md` | **G7 taken:** pooled recall 80.8 % [77.0 %, 84.1 %] at FPR 0.955 % (382 of 473; needs 403); 5 data seeds 80.5 % ± 1.0 %; diagnosis confirms "missed frauds are on the manifold" (91 missed, legit-percentile median 83.7 %, V14/V17 ≈ −1.9/+1.6 vs −8.0/−8.4 for detected); `meets_target` false → Phase 8 | 1 session |
| 8 | **Unsupervised ceiling** (legit-only training) | more training data (80/10/10), VAE reconstruction probability, latent-density score (GMM on codes), hour-segmented thresholds; negative results documented | `model.kind = "vae"`, `scoring` methods `recon_prob` / `latent_gmm`, `threshold.strategy = "quantile_by_hour"`, `configs/v2_unsup_*.toml`, `results/v2_unsupervised.md` | **G8:** ceiling table with CIs; selection on data seed 0, confirmation on seeds 1–4; pooled `meets_target` on the confirmation seeds → Phase 11, else (expected) → Phase 9 with the best unsupervised score as the AE component | 1 session |
| 9 | **Supervised & weakly-supervised learning** | 5-fold CV over the 473 frauds; fully supervised classifiers (logistic regression, gradient boosting, small MLP; class weighting vs undersampling); negative-example AE (margin / deviation loss, Deep-SAD-style latent term); hybrid stacking `[x, residual, AE score]`; few-label curve (25 / 50 / 100 / all) for every method | `fraud_ae/supervised.py`, `fraud_ae/semi.py`, `folds.json`, `fraud-ae supervised` / `fraud-ae semi --config configs/v2_*.toml`, `results/v2_supervised.md`, few-label figure | **G9 (milestone M5):** pooled out-of-fold recall ≥ 85 % at FPR < 1 % with CI for at least one method; the table shows supervised, hybrid and AE-based rows side by side with their label budgets, so the report can state what labels buy; if no method reaches it, the ceiling and the few-label curve are the result | 1.5–2 sessions |
| 10 | **Robustness & deployment realism** | rolling-origin temporal validation with rolling threshold re-estimation (fixes the Phase 5 drift), seed and data-seed variance of the winner, adversarial evasion probe (minimal perturbation to cross below the threshold) | `results/v2_robustness.md`, evasion figure | rolling threshold holds FPR within [0.8 %, 1.0 %] on every window; variance tables; median evasion budget reported | 0.5–1 session |
| 11 | **Report v2** | sections 6.8–6.10, updated abstract and conclusions, every number via placeholders | `report/report_template.md` → `report/report.md`, CHANGELOG 2.0.0, README outcome line | **M6:** report rendered from artifacts, committed report ≡ fresh render (existing test) | 0.5 session |

Milestones: **M4** pooled protocol and diagnosis (end of Phase 7, reached 2026-10-29) · **M5**
criterion reached or ceiling proven (end of Phase 9) · **M6** report v2 (end of Phase 11).
Total ≈ 4–5 sessions.

### Phase 7 – methods

* **Pooled-fraud protocol.** `data.fraud_val_fraction = 0.0`: validation holds legit rows only
  (`ThresholdResult.val_recall = None`, `evaluate` already handles zero val frauds; the
  semi-supervised strategies `youden` / `max_f1` are excluded by design). Threshold, calibration
  and model selection stay legit-only; recall is measured once per method on all 473 frauds.
* **Repeated legit splits.** `--data-seed N` on `prepare` / `train` / `baselines` writes and
  reads `<processed_dir>_dseed<N>` (same suffix pattern as `--synthetic`, same feature-count
  guard); the override is recorded in the run's effective config, so `threshold` / `evaluate`
  follow it without a flag. Five data seeds × base config → pooled recall mean ± std across
  seeds plus the Wilson interval on the pooled counts of each run.
* **`fraud-ae diagnose --run <dir> [--against <dirs…>]`.** Reuses `scoring.load_scores`,
  `evaluate.threshold_for_fpr` and `idx_test` from `data.load_splits` to map missed frauds back
  to raw rows: score percentile among legit rows, amount, hour, deltas on the top residual
  features; Jaccard overlap of the missed sets across runs; union and rank-fusion recall at a
  given FPR as upper bounds for ensembles. Writes `diagnostics.json` and a figure.
* **Literature-protocol replication.** One run with all frauds in test, no duplicate removal
  and the same quantile threshold – a comparability row for the report, not a candidate.

### Phase 7 outcome (2026-10-29, milestone M4)

* **Pooled protocol** (`configs/v2_pooled.toml`, `runs/ae_pooled`): the legit split is the v1
  split, so the checkpoint is byte-identical to `runs/ae_base` (verified with `cmp`); only the
  measurement changes. On 56 650 legit + **all 473 frauds** at the v1 threshold 1.862: TP 382,
  FP 541 → recall **80.8 %** [77.0 %, 84.1 %] (bootstrap [77.4 %, 84.4 %]), FPR 0.955 %,
  precision 41.4 %, ROC-AUC 0.951, PR-AUC 0.660. Criterion needs TP ≥ 403 → **not met**; the
  oracle recall at exactly 1 % FPR is the same 80.8 %. The v1 val half (84.3 %) and test half
  (77.2 %) are confirmed as two draws of one 80.8 % detector.
* **Repeated legit splits** (`--data-seed 0–3` + the config seed 42, train seed fixed at 42):
  79.7 / 79.7 / 80.5 / 80.8 / 82.0 % → **80.5 % ± 1.0 %** (min–max 79.7–82.0 %); FPR
  0.87–1.01 % – data seed 2 lands at 1.008 % with the 0.9 % val quantile, i.e. the threshold
  margin covers split variance only marginally (input for the Phase 10 rolling-threshold work).
  No data seed reaches 403 TP; the best single split is 388.
* **Literature protocol** (`configs/v2_literature.toml`: no dedupe, 284 807 rows, 492 frauds
  all in test, 170 589 train legit): recall 82.3 % [78.7 %, 85.4 %] at FPR 0.865 %. About
  1.5 pp of the published-looking gain comes from duplicated frauds and from a lower FPR; it is
  a comparability row, not a candidate.
* **Pooled baselines** (`--suffix _pooled`): PCA k = 15 73.2 %, k = 10 61.5 %, k = 5 60.5 %,
  Isolation Forest 57.5 % – the ranking and the ≈ 8 pp AE lead over the best baseline hold on
  473 frauds with intervals half as wide as v1.
* **Diagnosis** (`fraud-ae diagnose --run runs/ae_pooled --against` the three pooled baselines,
  `runs/ae_pooled/diagnostics.json`, `report/figures/missed_frauds_ae_pooled.png`): 91 missed
  frauds. Their score median is 0.617 (legit 0.292, detected 11.4) – the 83.7th percentile of
  legit scores while the threshold sits at the 99.1st; median amount 4.49 (detected 12.31,
  legit 22.51); night share (00–06 h) 13 % vs 27 % for detected and 8 % for legit, i.e. the
  missed frauds are spread over the day. The five most separating raw features (V17, V14, V12,
  V10, V16; effect sizes 6–11 σ) have detected-fraud means of −8.4 / −8.0 / −7.4 / −6.6 / −5.1
  but missed-fraud means of +1.6 / −1.9 / −0.6 / −0.5 / +0.6 against ≈ 0 for legit rows: **the
  missed frauds lie on the legitimate manifold – confirmed.** Of the 91, PCA k = 15 alarms on 2,
  PCA k = 5 on 1, Isolation Forest on 7 (identical at their oracle 1 % thresholds); Jaccard of
  the AE's and PCA k = 15's missed sets 0.69. Ensemble ceilings: union of the four detectors
  82.5 % at 1.77 % FPR, rank-average fusion 79.1 % at 1.00 % – **no unsupervised combination
  reaches 85 %**, matching the planning diagnosis.
* **Tooling:** `--data-seed` (recorded in the effective config snapshot, ADR 24),
  `baselines --suffix` (ADR 25), `compare --figure-prefix` + "Data seed" / "Test frauds"
  columns + `_dseed` families, `fraud_ae/diagnose.py` (ADRs 26–27), `scripts/run_v2_protocol.sh`
  (≈ 5 min CPU), `results/v2_protocol.md` (11 rows). `scripts/run_ablations.sh` now passes the
  explicit v1 run list to `compare`, so `results/comparison.md` keeps its 21 v1 rows. 122 tests.
* **Decision G7:** pooled `meets_target` is false and the ceiling of unsupervised combinations
  is ≈ 82 % → **Phase 8** (unsupervised ceiling: VAE, latent density, hour-segmented
  thresholds, more training data), selection on data seed 0, confirmation on seeds 1–4 as
  pre-registered. The pooled protocol with `fraud_val_fraction = 0` is the measurement for every
  v2 phase from here on.

### Phase 8 – methods (expected gain in parentheses)

1. **More training data** – `legit_fractions = [0.8, 0.1, 0.1]`; 28 k legit val rows still
   resolve the 0.9 % quantile to ±0.05 % (+0–1 pp).
2. **Variational AE with reconstruction probability** (An & Cho 2015, the abstract's own
   reference) – `model.kind = "vae"`, score = −log p(x | z) averaged over L = 10 samples,
   new `scoring` method `recon_prob` (+0–3 pp).
3. **Latent-density score** (DAGMM-lite) – Gaussian mixture (k ∈ {4, 8}) fitted on encoder codes
   of val-legit; score = rank-average of MSE and latent negative log-likelihood; aims at frauds
   with an ordinary residual but an unusual code; calibration stored in `score_calibration.npz`
   (+1–3 pp).
4. **Hour-segmented thresholds** – `threshold.strategy = "quantile_by_hour"`: night (00–06 h)
   and day legit quantiles at the same 0.9 %, identical global FPR, recall reallocated toward
   the night-time fraud cluster (±1 pp, cheap).
5. **Documented negative results, no new runs** – seed / bottleneck ensembles and score fusion
   (≤ 0 pp in the diagnosis above).

Selection rule, pre-registered: choose the best method on data seed 0, confirm on data seeds
1–4, never re-pick on the confirmation seeds. All variants use the v1 run-directory contract so
`fraud-ae compare` and `fraud-ae report` work unchanged.

### Phase 9 – methods

Protocol: the 473 frauds are split into 5 fixed folds (`folds.json`); a method sees the
train-fold frauds, the held-out fold is pooled for evaluation → out-of-fold recall over all 473
frauds with a Wilson interval. The legit split is unchanged and the threshold is still the legit
quantile of the method's own score (probability or anomaly score alike), so FPR < 1 % holds by
construction for every method. Hyper-parameters are fixed in advance or chosen inside the
training folds, never on the held-out fold. A unit test asserts that no fraud index appears in
a train fold and in its own evaluation fold. The few-label curve (25 / 50 / 100 / all
train-fold frauds) is a deliverable for every method, because ~380 training labels per fold are
optimistic for a real deployment.

0. **Fully supervised classifiers (first-class candidates).** Trained on train-legit + train-fold
   frauds: class-weighted logistic regression (linear reference), `HistGradientBoostingClassifier`
   (expected strongest tabular model), and a small MLP in PyTorch with the AE's encoder
   architecture; imbalance handled by class weights and, as a variant, random undersampling of
   legit rows (Dal Pozzolo et al. 2015). The diagnosis gives 82–84 % on the hard v1 test half
   with 236 labels; pooled over folds 85–88 % is expected. Deliverable `fraud_ae/supervised.py`
   + `fraud-ae supervised`, same run-directory contract.
1. **Negative-example AE** – loss = MSE on legit + λ · max(0, m − MSE) on train-fold frauds,
   with the margin m taken from the legit score quantile; variant with a Deep-SAD-style term
   pulling legit codes toward a centre and pushing fraud codes away. `train.labelled_frauds =
   true`. Expected: the AE itself crosses the plateau for frauds similar to the labelled ones.
2. **Hybrid stacking** – features `[x, residual vector, AE score]` → class-weighted logistic
   regression and `HistGradientBoostingClassifier` (scikit-learn is already a dependency);
   threshold = legit quantile of the classifier score. Expected: most of the label-driven gain.
3. **AE as a feature for the supervised models** – the hybrid of item 2 is also the test of
   whether the legit-only AE adds anything once labels exist: compare gradient boosting on `x`
   alone (item 0) with gradient boosting on `[x, residual, AE score]` at equal label budgets.
   If the AE score adds nothing, the report says so.

### Phase 10 – methods

* **Rolling threshold.** Re-estimate the legit quantile on a sliding window of the most recent
  legit traffic and evaluate rolling-origin (train on the past, score the next window). Target:
  FPR within [0.8 %, 1.0 %] on every window (v1 temporal split drifted to 0.53 %).
* **Variance.** Five model seeds and five data seeds of the winning method; report mean ± std.
* **Evasion probe.** Gradient descent on the input of each detected fraud to minimise its score
  below the threshold under an L2 budget; report the median budget and which features move.
  Turns the threat-model discussion into a measured limitation.

## Risks

| Risk | Mitigation |
|---|---|
| Plain-MSE AE lands at 80–84 % recall @ 1 % FPR (literature band 0.80–0.90) | **materialised, worse than feared (77 % on test, 84 % on val)**: scoring variants, bottleneck sweep, drop `Time`, 5 seeds all 76–79 %; report both plain and best configuration with CIs, no further test-set tuning |
| Test FPR slightly above 1 % from sampling noise | threshold targets 0.9 % on val; report CI |
| Leakage (scaler, duplicates, threshold tuned on test) | unit-tested invariants; test set touched once in Phase 4 |
| Non-determinism | seeds + deterministic torch; seed variance reported |
| torch install pulls CUDA wheels | CPU index pinned in `pyproject.toml` |
| v2: label leakage in Phase 9 (a fraud in both a train fold and its own evaluation fold) | folds fixed in `folds.json`; unit test on fold disjointness; threshold still legit-only |
| v2: selection on the pooled test numbers (test-set tuning in disguise) | pre-registered rule: pick on data seed 0, confirm on data seeds 1–4; both shown in the results table |
| v2: few labels available in practice | few-label curve (25 / 50 / 100 / all) is a Phase 9 deliverable for every method, supervised included |
| v2: supervised models learn the two-day fraud campaigns rather than fraud in general | Phase 10 rolling-origin validation applies to the supervised winner as well; the report presents supervised and legit-only results as complementary (labels buy recall on known patterns, the AE covers departures from normal traffic) |
| v2: VAE / margin-loss training instability | seeded, early stopping on the legit validation loss, fall back to the plain AE score on NaN |
| v2: pooled numbers read as "the same test set as v1" | the report states that pooled frauds are evaluated once per method; v1 numbers stay in the report for comparability |

## Report outline (Phase 6)
1. Problem, threat model and the 85 % / 1 % criterion (with its precision consequence)
2. Dataset and preprocessing (duplicates, features, split protocol, leakage controls)
3. Model and training
4. Threshold selection (why val-legit quantile)
5. Results: AE vs baselines, CIs, figures, ablations, seed variance
6. Limitations (point-anomaly detector, PCA-obfuscated features, no account history, adversarial evasion) and conclusions

Report v2 delta (Phase 11): 6.8 pooled-fraud protocol and repeated splits (why the v1 val/test
halves differ, pooled estimate with CI, literature-protocol row); 6.9 unsupervised ceiling
(VAE, latent density, hour-segmented thresholds, negative results); 6.10 supervised,
hybrid and weakly-supervised AE results out-of-fold with the few-label curve, rolling-threshold
and evasion results; abstract and conclusions rewritten around "what labels buy" – the AE as
the legit-only baseline, the supervised classifier as the labelled upper line.
