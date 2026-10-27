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

## Risks

| Risk | Mitigation |
|---|---|
| Plain-MSE AE lands at 80–84 % recall @ 1 % FPR (literature band 0.80–0.90) | **materialised, worse than feared (77 % on test, 84 % on val)**: scoring variants, bottleneck sweep, drop `Time`, 5 seeds all 76–79 %; report both plain and best configuration with CIs, no further test-set tuning |
| Test FPR slightly above 1 % from sampling noise | threshold targets 0.9 % on val; report CI |
| Leakage (scaler, duplicates, threshold tuned on test) | unit-tested invariants; test set touched once in Phase 4 |
| Non-determinism | seeds + deterministic torch; seed variance reported |
| torch install pulls CUDA wheels | CPU index pinned in `pyproject.toml` |

## Report outline (Phase 6)
1. Problem, threat model and the 85 % / 1 % criterion (with its precision consequence)
2. Dataset and preprocessing (duplicates, features, split protocol, leakage controls)
3. Model and training
4. Threshold selection (why val-legit quantile)
5. Results: AE vs baselines, CIs, figures, ablations, seed variance
6. Limitations (point-anomaly detector, PCA-obfuscated features, no account history, adversarial evasion) and conclusions
