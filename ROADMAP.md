# Roadmap – Project 12 (Autoencoder fraud detection)

Goal: a reproducible pipeline and an English report demonstrating **recall ≥ 85 % at FPR < 1 %**
on the Credit Card Fraud dataset, with a deep autoencoder trained on legitimate transactions only,
compared against PCA-reconstruction and Isolation Forest baselines.

## Phases

| # | Phase | Scope | Deliverables | Exit criteria | Effort |
|---|---|---|---|---|---|
| 0 | **Scaffold** ✅ | layout, `pyproject`/uv, configs, package skeleton, tests, docs | this repo | `uv sync`, `fraud-ae --help`, `pytest`, `ruff` all green | 1 session |
| 1 | **Data** (abstract phase 1) ✅ | unzip + sha256, dedupe, `log1p(Amount)`, hour-of-day sin/cos, seeded split (train = legit only), scaler fit on train, EDA | `data.py`, `synthetic.py`, `prepare [--synthetic]`, `data_report.json`, EDA figures (class ratio, Amount distribution, fraud rate by hour) | split invariants tested (`tests/test_data.py`, 19 tests pass); split-size table in README; sha256 pinned in configs | 1 session |
| 2 | **Train** (abstract phase 2) | training loop, early stopping on val-legit MSE, checkpoints, seeding | `train.py`, `runs/ae_base/{model.pt,history.csv}`, training-curve figure | val MSE converges; two runs with the same seed give identical losses | 1 session |
| 3 | **Threshold** (abstract phase 3) | scoring (`mse`, `standardized`, `mahalanobis`), threshold strategies | `scoring.py`, `threshold.py`, `threshold.json`, score histogram | val FPR ≈ 0.9 % with the quantile strategy; quantile property test passes | 0.5 session |
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
