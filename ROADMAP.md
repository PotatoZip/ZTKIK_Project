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
| 4 | **Evaluate** (abstract phase 4) | ROC/PR-AUC, recall@FPR grid, confusion matrix, Wilson + bootstrap CIs, figures, criterion check | `evaluate.py`, `plots.py`, `metrics.json`, ROC/PR/recall-vs-FPR/threshold-sweep/per-feature figures | **Decision gate:** `meets_target` true → Phase 5; false → run scoring variants, latent sweep (`configs/ablation_latent_*`), `ablation_no_time`, 5 seeds, then re-evaluate | 1 session |
| 5 | **Baselines & ablations** | PCA(k ∈ {5,10,15}), Isolation Forest; ablations: scoring method, latent size, Time feature, BatchNorm, 5 seeds, temporal split | `baselines.py`, `results/comparison.md`, box/bar plots | comparison table complete with CIs; seed variance reported | 1–1.5 sessions |
| 6 | **Hardening & report** | tests green, README final, `report/report.md` (English), `report/prompt_log.md`, limitations + threat-model discussion | submission package | every number in the report traceable to a `metrics.json`; `fraud-ae all` + `baselines` reproduce it | 1 session |

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

## Risks

| Risk | Mitigation |
|---|---|
| Plain-MSE AE lands at 80–84 % recall @ 1 % FPR (literature band 0.80–0.90) | scoring variants, bottleneck sweep, drop `Time`, multi-seed averaging; report both plain and best configuration |
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
