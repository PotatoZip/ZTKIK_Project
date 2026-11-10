<!-- TEMPLATE: every number inside a double-brace placeholder is read from a JSON artifact by `fraud-ae report`.
     Edit this file, not report.md. See src/fraud_ae/report.py for the placeholder syntax. -->
# Fraud Detection in Bank Transactions with a Deep Autoencoder

**ZTKIK – Project 12.** Deep autoencoder trained on legitimate credit-card transactions only;
fraud is flagged by a high reconstruction error. Code, configs and every artifact behind this
report: repository root (`README.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `report/CHANGELOG.md`).
Version 2 of the report: Sections 6.8–6.11, the discussion and the conclusions cover Roadmap v2
(pooled-fraud protocol, the legit-only ceiling, labels inside the model, robustness); Sections
1–6.7 are the v1 study and are unchanged.

## Abstract

We train a 31-20-10-5-10-20-31 autoencoder (AE) in PyTorch on {{ ${runs/ae_base/train_summary.json:n_train} | int }}
legitimate transactions of the Kaggle *Credit Card Fraud Detection* dataset and flag a
transaction as fraudulent when its reconstruction error exceeds a threshold chosen on
legitimate validation data only. The project criterion is **recall ≥ 85 % at a false-positive
rate (FPR) < 1 %** on a held-out test set. On the pre-registered v1 split
({{ ${runs/ae_base:n_fraud} }} test frauds) the AE reaches **{{ ${runs/ae_base:at_threshold.recall} | pct1 }}**
recall at an FPR of {{ ${runs/ae_base:at_threshold.fpr} | pct3 }}; measured on all
{{ ${runs/ae_pooled:n_fraud} }} frauds under the pooled protocol (the threshold never reads a
fraud label, so every fraud can be a test fraud) the same model reaches
**{{ ${runs/ae_pooled:at_threshold.recall} | pct1 }}** (Wilson 95 % interval
{{ ${runs/ae_pooled:at_threshold.recall_ci_wilson.0} | pct1 }} – {{ ${runs/ae_pooled:at_threshold.recall_ci_wilson.1} | pct1 }})
at an FPR of {{ ${runs/ae_pooled:at_threshold.fpr} | pct3 }}, and
{{ mean(${runs/ae_pooled_dseed0:at_threshold.recall}, ${runs/ae_pooled_dseed1:at_threshold.recall}, ${runs/ae_pooled_dseed2:at_threshold.recall}, ${runs/ae_pooled_dseed3:at_threshold.recall}, ${runs/ae_pooled_dseed4:at_threshold.recall}) | pct1 }} ±
{{ std(${runs/ae_pooled_dseed0:at_threshold.recall}, ${runs/ae_pooled_dseed1:at_threshold.recall}, ${runs/ae_pooled_dseed2:at_threshold.recall}, ${runs/ae_pooled_dseed3:at_threshold.recall}, ${runs/ae_pooled_dseed4:at_threshold.recall}) | pp1 }}
over five re-drawn legitimate splits. The FPR half of the criterion holds, the recall half does
not, and the shortfall is systematic: eleven v1 variants, a variational AE scored by
reconstruction probability, latent-density scores, hour-segmented thresholds and a third more
training data all stay at about 80–82 %. The diagnosis explains why: the
{{ ${runs/ae_pooled/diagnostics.json:n_missed} }} missed frauds reconstruct like legitimate
rows (median score at the {{ ${runs/ae_pooled/diagnostics.json:groups.missed.legit_percentile_median} * 100 | f1 }}th
percentile of legitimate scores), and no unsupervised ensemble reaches 85 % either. The AE
still beats PCA reconstruction ({{ ${runs/baseline_pca_k15_pooled:at_threshold.recall} | pct1 }})
and an Isolation Forest ({{ ${runs/baseline_iforest_pooled:at_threshold.recall} | pct1 }}) at
the same FPR. Labels change the picture. Evaluated out-of-fold over all 473 frauds with
legit-only thresholds, a class-weighted gradient-boosting classifier reaches
**{{ ${runs/sup_hgb:at_threshold.recall} | pct1 }}** ({{ ${runs/sup_hgb:at_threshold.recall_ci_wilson.0} | pct1 }} – {{ ${runs/sup_hgb:at_threshold.recall_ci_wilson.1} | pct1 }})
at an FPR of {{ ${runs/sup_hgb:at_threshold.fpr} | pct3 }}, and the same autoencoder
fine-tuned with a margin loss on the labelled frauds
**{{ ${runs/sup_neg_ae:at_threshold.recall} | pct1 }}**; about 50 labelled frauds per fold
suffice for a tree model, and the AE as an extra feature adds nothing measurable. Robustness
checks on the temporal split show that a fixed threshold drifts within two days (the classifier
would run at {{ ${results/v2_robustness.json:rolling.models.hgb.summary.fixed.fpr_mean} | pct2 }} FPR) and that
re-estimating the legitimate quantile on a rolling window restores it
({{ ${results/v2_robustness.json:rolling.models.hgb.summary.rolling_legit.fpr_mean} | pct2 }}); both labelled
winners keep {{ ${results/v2_robustness.json:rolling.models.hgb.summary.rolling_legit.pooled_recall} | pct1 }}
recall on the {{ ${results/v2_robustness.json:rolling.models.hgb.summary.rolling_legit.n_fraud} }} later frauds and
stay above 85 % across model seeds and legitimate splits. A white-box attacker needs a median
L2 budget of {{ ${results/v2_robustness.json:evasion.median_minimal_budget} | f0 }} standardised units (median
fraud norm {{ ${results/v2_robustness.json:evasion.fraud_l2_norm_median} | f1 }}) to push a detected fraud below
the AE threshold.

## 1. Problem, threat model and success criterion

**Setting.** Automated anti-fraud systems in electronic banking score every card transaction
and raise alarms that analysts or rule engines act upon. Labels are scarce, delayed and
expensive, so the project follows the *unsupervised* route: learn what legitimate traffic looks
like and treat departures from it as suspicious.

**Threat model.** The adversary (a "financial fraud / anomaly attacker") executes unauthorized
transactions with stolen card data that break the card-holder's usual pattern – amount, time of
day, merchant profile. The attacker does not see the model; the defence is a detector that,
given a single transaction's feature vector, must decide whether it is anomalous. Two
operational costs matter: a missed fraud (direct loss) and a false alarm (friction for a
legitimate customer, analyst time). The criterion from the project abstract fixes the trade-off:
**detect at least 85 % of frauds while raising false alarms on fewer than 1 % of legitimate
transactions.**

**What the criterion implies on this data.** With {{ ${runs/ae_base:n_legit} | int }}
legitimate and {{ ${runs/ae_base:n_fraud} }} fraudulent test transactions, FPR < 1 % allows at
most {{ ${runs/ae_base:criterion.fp_allowed} }} false alarms and recall ≥ 85 % requires at least
{{ ${runs/ae_base:criterion.tp_required} }} detected frauds. Even a detector that meets both
would have a precision of only about
{{ ${runs/ae_base:criterion.tp_required} / (${runs/ae_base:criterion.tp_required} + ${runs/ae_base:criterion.fp_allowed}) | pct0 }}
– roughly three alarms out of four would be legitimate transactions. This is the normal
operating regime of an alerting system on a {{ ${data/processed/data_report.json:class_balance.fraud_rate} | pct3 }}
base rate and must be stated alongside the recall figure.

## 2. Data and preprocessing

**Dataset.** Kaggle *Credit Card Fraud Detection* (ULB Machine Learning Group): two days of
European card transactions, {{ ${data/processed/data_report.json:rows.raw} | int }} rows with 28
PCA-transformed features `V1..V28`, the raw `Amount`, the elapsed `Time` in seconds and the
label `Class`. The CSV is verified against a pinned SHA-256 before use.

**Duplicates.** {{ ${data/processed/data_report.json:rows.duplicates_dropped} | int }} rows are
exact duplicates of another row; they are removed *before* splitting, otherwise identical
rows could appear in both training and test data. {{ ${data/processed/data_report.json:rows.after_dedupe} | int }}
rows remain, {{ ${data/processed/data_report.json:class_balance.fraud} }} of them frauds
({{ ${data/processed/data_report.json:class_balance.fraud_rate} | pct3 }}).

**Features (31 model inputs).** `V1..V28` unchanged; `Amount` → `log1p(Amount)` (heavy right
tail: median {{ ${data/processed/data_report.json:amount.all.median} | f2 }}, maximum
{{ ${data/processed/data_report.json:amount.all.max} | int }}); `Time` → hour of day →
`sin`/`cos` pair. The raw `Time` is a clock over a two-day window and would act as a row
identifier; the hour of day is a genuine signal: the fraud *rate* at 02:00 is
{{ ${data/processed/data_report.json:by_hour.fraud_rate.2} | pct2 }} and at 04:00
{{ ${data/processed/data_report.json:by_hour.fraud_rate.4} | pct2 }}, against
{{ ${data/processed/data_report.json:class_balance.fraud_rate} | pct3 }} overall, while traffic is
lowest at night (Figure 1). Fraud amounts are smaller in median
({{ ${data/processed/data_report.json:amount.fraud.median} | f2 }} vs
{{ ${data/processed/data_report.json:amount.legit.median} | f2 }}) but heavier-tailed.

**Split protocol (seed 42).** Legitimate rows are shuffled and divided 60 / 20 / 20 % into
train / validation / test; frauds are divided 50 / 50 between validation and test and **never
enter training**. The `StandardScaler` is fitted on the training rows only.

| Split | Legitimate | Fraud | Total |
|---|---:|---:|---:|
| train | {{ ${data/processed/data_report.json:splits.train.legit} | int }} | {{ ${data/processed/data_report.json:splits.train.fraud} }} | {{ ${data/processed/data_report.json:splits.train.total} | int }} |
| validation | {{ ${data/processed/data_report.json:splits.val.legit} | int }} | {{ ${data/processed/data_report.json:splits.val.fraud} }} | {{ ${data/processed/data_report.json:splits.val.total} | int }} |
| test | {{ ${data/processed/data_report.json:splits.test.legit} | int }} | {{ ${data/processed/data_report.json:splits.test.fraud} }} | {{ ${data/processed/data_report.json:splits.test.total} | int }} |

**Leakage controls** (all unit-tested): no fraud in training; the three index sets are disjoint
and cover the de-duplicated frame; the split is reproducible from the seed; scaler statistics
equal training statistics; no identical feature rows across splits; the threshold is chosen on
validation data only; test labels are read exactly once, by the evaluation step.

![Figure 1 – traffic and fraud rate by hour of day](figures/eda_fraud_rate_by_hour.png)

*Figure 1. Traffic volume (top) and fraud rate (bottom) by hour of day. The fraud rate peaks
when traffic is lowest.*

## 3. Model and training

**Architecture.** Symmetric multilayer-perceptron autoencoder 31 → 20 → 10 → **5** → 10 → 20 →
31 with ELU activations and a linear output layer ({{ ${runs/ae_base/train_summary.json:n_parameters} | int }}
parameters). BatchNorm is off by default because inference-time batch statistics distort
per-sample anomaly scores (tested as an ablation in Section 6.3).

**Training.** Mean-squared-error loss on the {{ ${runs/ae_base/train_summary.json:n_train} | int }}
legitimate training rows; Adam (learning rate 10⁻³, weight decay 10⁻⁵), batch size 512, up to
100 epochs with early stopping (patience 10) and a plateau learning-rate scheduler. Model
selection uses the MSE on the {{ ${runs/ae_base/train_summary.json:n_val_legit} | int }}
*legitimate* validation rows, so no fraud label influences training. Everything is
deterministic for a given seed (seeded manual mini-batching, deterministic PyTorch kernels):
two runs with the same seed give identical losses and a byte-identical checkpoint.

**Result.** Validation MSE falls to {{ ${runs/ae_base/train_summary.json:best_val_legit_mse} | f3 }}
(best epoch {{ ${runs/ae_base/train_summary.json:best_epoch} }} of
{{ ${runs/ae_base/train_summary.json:epochs_run} }}; training MSE
{{ ${runs/ae_base/train_summary.json:final_train_mse} | f3 }}) in
{{ ${runs/ae_base/train_summary.json:wall_seconds} | f0 }} s on a CPU (Figure 2). The epoch cap
binds before early stopping; a 200-epoch run (Section 6.3) lowers the MSE to
{{ ${runs/ae_epochs_200/train_summary.json:best_val_legit_mse} | f3 }} without changing the
detection result.

![Figure 2 – training curves](figures/training_curves_ae_base.png)

*Figure 2. Training and validation (legitimate rows) MSE per epoch, learning rate below.*

## 4. Scoring and threshold selection

**Anomaly score.** The primary score is the per-transaction mean squared reconstruction
residual, `mse(x) = ‖x − AE(x)‖² / 31`, as prescribed by the project. Two variants that
whiten the residual with statistics of legitimate validation residuals were also implemented:
`standardized` (per-feature z-scores, squared and summed) and `mahalanobis` (full residual
covariance). Higher is more anomalous for all three.

**Threshold.** The alarm rule is `score > τ`. τ is the (1 − 0.009) quantile of the scores of
the *legitimate validation rows*: by construction about 0.9 % of legitimate traffic alarms,
leaving a margin below the 1 % limit for sampling noise, and no fraud label is consulted. On
the validation split this gives τ = {{ ${runs/ae_base/threshold.json:value} | f3 }}, FPR
{{ ${runs/ae_base/threshold.json:val_fpr} | pct3 }} and – read only for information, not for
tuning – a validation recall of {{ ${runs/ae_base/threshold.json:val_recall} | pct1 }} on
{{ ${runs/ae_base/threshold.json:n_val_fraud} }} validation frauds (Figure 3). Two
semi-supervised alternatives (Youden's J, maximum F1) use validation fraud labels and were
evaluated for comparison only; Youden's J lands at 2–2.5 % FPR and therefore cannot satisfy
the criterion.

![Figure 3 – validation score distribution](figures/score_histogram_val_ae_base.png)

*Figure 3. Distribution of log₁₀(MSE) for legitimate and fraudulent validation transactions
with the chosen threshold.*

## 5. Evaluation protocol

All test-set numbers come from one evaluation pass per run (`metrics.json`) at the
validation-chosen threshold:

* **At the threshold:** confusion matrix, recall, FPR, precision, F1; Wilson 95 % intervals for
  every proportion; a stratified 1 000-sample bootstrap (legitimate and fraudulent rows resampled
  separately) as a cross-check.
* **Threshold-free (ranking quality):** ROC-AUC, average precision (PR-AUC) and recall at an
  *oracle* FPR of 0.1 / 0.5 / 1 / 2 % (threshold set on the test legitimate scores themselves).
  Comparing the oracle recall at 1 % with the recall at the transferred threshold separates
  "the score cannot rank the frauds" from "the threshold did not transfer".
* **Verdict:** `meets_target` = recall ≥ 85 % and FPR < 1 % on the point estimates, as the
  criterion is stated; the intervals qualify the verdict.
* **Decision gate.** The protocol fixed in advance: if the pre-registered configuration fails,
  run the disclosed variants (scoring method, bottleneck 3/8, no time feature, five seeds) and
  report all of them; never tune on the test set.

## 6. Results

### 6.1 Primary result: the criterion is {{ ${runs/ae_base:meets_target} | verdict }}

| Test set: {{ ${runs/ae_base:n_legit} | int }} legitimate / {{ ${runs/ae_base:n_fraud} }} fraudulent | Value |
|---|---:|
| True positives / false positives / true negatives / false negatives | {{ ${runs/ae_base:at_threshold.tp} }} / {{ ${runs/ae_base:at_threshold.fp} }} / {{ ${runs/ae_base:at_threshold.tn} | int }} / {{ ${runs/ae_base:at_threshold.fn} }} |
| **Recall** (Wilson 95 %) | **{{ ${runs/ae_base:at_threshold.recall} | pct1 }}** [{{ ${runs/ae_base:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_base:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| Recall, bootstrap 95 % | [{{ ${runs/ae_base:bootstrap.recall_at_threshold.ci_lo} | pct1 }}, {{ ${runs/ae_base:bootstrap.recall_at_threshold.ci_hi} | pct1 }}] |
| **FPR** (Wilson 95 %) | **{{ ${runs/ae_base:at_threshold.fpr} | pct3 }}** [{{ ${runs/ae_base:at_threshold.fpr_ci_wilson.0} | pct3 }}, {{ ${runs/ae_base:at_threshold.fpr_ci_wilson.1} | pct3 }}] |
| Precision / F1 | {{ ${runs/ae_base:at_threshold.precision} | pct1 }} / {{ ${runs/ae_base:at_threshold.f1} | f3 }} |
| ROC-AUC / PR-AUC | {{ ${runs/ae_base:roc_auc} | f3 }} / {{ ${runs/ae_base:pr_auc} | f3 }} |
| Recall at oracle FPR 0.1 % / 0.5 % / 1 % / 2 % | {{ ${runs/ae_base:recall_at_fpr.0.recall} | pct1 }} / {{ ${runs/ae_base:recall_at_fpr.1.recall} | pct1 }} / {{ ${runs/ae_base:recall_at_fpr.2.recall} | pct1 }} / {{ ${runs/ae_base:recall_at_fpr.3.recall} | pct1 }} |
| Criterion (TP ≥ {{ ${runs/ae_base:criterion.tp_required} }}, FP ≤ {{ ${runs/ae_base:criterion.fp_allowed} }}) | **{{ ${runs/ae_base:meets_target} | verdict }}** (FPR part: {{ ${runs/ae_base:criterion.meets_fpr} | verdict }}; recall part: {{ ${runs/ae_base:criterion.meets_recall} | verdict }}) |

The threshold transferred as designed – the test FPR of {{ ${runs/ae_base:at_threshold.fpr} | pct3 }}
is below 1 % – but the detector finds {{ ${runs/ae_base:at_threshold.tp} }} of the
{{ ${runs/ae_base:n_fraud} }} test frauds where {{ ${runs/ae_base:criterion.tp_required} }} were
required. The upper end of the confidence interval
({{ ${runs/ae_base:at_threshold.recall_ci_wilson.1} | pct1 }}) is below 85 %, so the miss is not
a sampling accident.

### 6.2 Ranking analysis: a fifth of the frauds look legitimate

Recall at an oracle 1 % FPR is {{ ${runs/ae_base:recall_at_fpr.2.recall} | pct1 }} – the same as
at the transferred threshold – and the recall-vs-FPR curve (Figure 4) is almost flat between
0.3 % and 2 % FPR ({{ ${runs/ae_base:recall_at_fpr.1.recall} | pct1 }} at 0.5 %,
{{ ${runs/ae_base:recall_at_fpr.3.recall} | pct1 }} at 2 %); it crosses 85 % only near 7 % FPR.
The limitation is therefore the *ranking*: about
{{ 1 - ${runs/ae_base:recall_at_fpr.3.recall} | pct0 }} of the test frauds obtain
reconstruction errors indistinguishable from legitimate traffic, and no threshold choice can
recover them. The threshold sweep (Figure 5) shows the same plateau from the other side, and
the test-score histogram (Figure 6) the bimodal fraud distribution behind it.

![Figure 4 – recall vs FPR](figures/recall_vs_fpr_test_ae_base.png)

*Figure 4. Recall as a function of FPR on the test set (log axis). The shaded box is the
criterion region; the dot is the validation-chosen operating point.*

![Figure 5 – threshold sweep](figures/threshold_sweep_test_ae_base.png)

*Figure 5. Recall, FPR and precision as functions of the threshold.*

![Figure 6 – test score distribution](figures/score_histogram_test_ae_base.png)

*Figure 6. Distribution of log₁₀(MSE) on the test set with the validation-chosen threshold.*

![Figure 7 – ROC and precision-recall curves](figures/roc_pr_test_ae_base.png)

*Figure 7. ROC (log FPR axis) and precision-recall curves of the base model on the test set.*

### 6.3 Ablations (decision gate)

Every variant is a separate run directory produced by the same three commands; the threshold is
always the 0.9 % validation-legitimate quantile of the variant's own scores.

| Variant | Val recall | Test FPR | Test recall [Wilson 95 %] |
|---|---:|---:|---|
| Base: `mse`, latent 5, 31 inputs, seed 42 | {{ ${runs/ae_base/threshold.json:val_recall} | pct1 }} | {{ ${runs/ae_base:at_threshold.fpr} | pct3 }} | **{{ ${runs/ae_base:at_threshold.recall} | pct1 }}** [{{ ${runs/ae_base:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_base:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| Score `standardized` | {{ ${runs/ae_score_standardized/threshold.json:val_recall} | pct1 }} | {{ ${runs/ae_score_standardized:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_score_standardized:at_threshold.recall} | pct1 }} [{{ ${runs/ae_score_standardized:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_score_standardized:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| Score `mahalanobis` | {{ ${runs/ae_score_mahalanobis/threshold.json:val_recall} | pct1 }} | {{ ${runs/ae_score_mahalanobis:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_score_mahalanobis:at_threshold.recall} | pct1 }} [{{ ${runs/ae_score_mahalanobis:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_score_mahalanobis:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| Latent 3 | {{ ${runs/ae_latent_3/threshold.json:val_recall} | pct1 }} | {{ ${runs/ae_latent_3:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_latent_3:at_threshold.recall} | pct1 }} [{{ ${runs/ae_latent_3:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_latent_3:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| Latent 8 | {{ ${runs/ae_latent_8/threshold.json:val_recall} | pct1 }} | {{ ${runs/ae_latent_8:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_latent_8:at_threshold.recall} | pct1 }} [{{ ${runs/ae_latent_8:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_latent_8:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| No hour-of-day feature (29 inputs) | {{ ${runs/ae_no_time/threshold.json:val_recall} | pct1 }} | {{ ${runs/ae_no_time:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_no_time:at_threshold.recall} | pct1 }} [{{ ${runs/ae_no_time:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_no_time:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| BatchNorm | {{ ${runs/ae_batchnorm/threshold.json:val_recall} | pct1 }} | {{ ${runs/ae_batchnorm:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_batchnorm:at_threshold.recall} | pct1 }} [{{ ${runs/ae_batchnorm:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_batchnorm:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| 200 epochs | {{ ${runs/ae_epochs_200/threshold.json:val_recall} | pct1 }} | {{ ${runs/ae_epochs_200:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_epochs_200:at_threshold.recall} | pct1 }} [{{ ${runs/ae_epochs_200:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_epochs_200:at_threshold.recall_ci_wilson.1} | pct1 }}] |

No variant meets the criterion. Selecting on validation – the only honest selection rule –
would pick the Mahalanobis score ({{ ${runs/ae_score_mahalanobis/threshold.json:val_recall} | pct1 }}
on validation), which yields {{ ${runs/ae_score_mahalanobis:at_threshold.recall} | pct1 }} on
test; the two best test numbers (latent 3, no time feature) are within one interval width of
the base model and were not chosen by any validation signal. BatchNorm lowers recall and pushes
the FPR to {{ ${runs/ae_batchnorm:at_threshold.fpr} | pct3 }}, failing both halves, which
supports keeping it off. Longer training changes the reconstruction error but not the
detection result.

### 6.4 Seed variance

The base configuration was trained with five seeds (0, 1, 2, 3, 42):

| | Mean ± sample std | Min – max |
|---|---:|---|
| Validation recall | {{ mean(${runs/ae_base_seed0/threshold.json:val_recall}, ${runs/ae_base_seed1/threshold.json:val_recall}, ${runs/ae_base_seed2/threshold.json:val_recall}, ${runs/ae_base_seed3/threshold.json:val_recall}, ${runs/ae_base/threshold.json:val_recall}) | pct1 }} ± {{ std(${runs/ae_base_seed0/threshold.json:val_recall}, ${runs/ae_base_seed1/threshold.json:val_recall}, ${runs/ae_base_seed2/threshold.json:val_recall}, ${runs/ae_base_seed3/threshold.json:val_recall}, ${runs/ae_base/threshold.json:val_recall}) | pct1 }} | {{ min(${runs/ae_base_seed0/threshold.json:val_recall}, ${runs/ae_base_seed1/threshold.json:val_recall}, ${runs/ae_base_seed2/threshold.json:val_recall}, ${runs/ae_base_seed3/threshold.json:val_recall}, ${runs/ae_base/threshold.json:val_recall}) | pct1 }} – {{ max(${runs/ae_base_seed0/threshold.json:val_recall}, ${runs/ae_base_seed1/threshold.json:val_recall}, ${runs/ae_base_seed2/threshold.json:val_recall}, ${runs/ae_base_seed3/threshold.json:val_recall}, ${runs/ae_base/threshold.json:val_recall}) | pct1 }} |
| Test recall | {{ mean(${runs/ae_base_seed0:at_threshold.recall}, ${runs/ae_base_seed1:at_threshold.recall}, ${runs/ae_base_seed2:at_threshold.recall}, ${runs/ae_base_seed3:at_threshold.recall}, ${runs/ae_base:at_threshold.recall}) | pct1 }} ± {{ std(${runs/ae_base_seed0:at_threshold.recall}, ${runs/ae_base_seed1:at_threshold.recall}, ${runs/ae_base_seed2:at_threshold.recall}, ${runs/ae_base_seed3:at_threshold.recall}, ${runs/ae_base:at_threshold.recall}) | pct1 }} | {{ min(${runs/ae_base_seed0:at_threshold.recall}, ${runs/ae_base_seed1:at_threshold.recall}, ${runs/ae_base_seed2:at_threshold.recall}, ${runs/ae_base_seed3:at_threshold.recall}, ${runs/ae_base:at_threshold.recall}) | pct1 }} – {{ max(${runs/ae_base_seed0:at_threshold.recall}, ${runs/ae_base_seed1:at_threshold.recall}, ${runs/ae_base_seed2:at_threshold.recall}, ${runs/ae_base_seed3:at_threshold.recall}, ${runs/ae_base:at_threshold.recall}) | pct1 }} |
| Test FPR | {{ mean(${runs/ae_base_seed0:at_threshold.fpr}, ${runs/ae_base_seed1:at_threshold.fpr}, ${runs/ae_base_seed2:at_threshold.fpr}, ${runs/ae_base_seed3:at_threshold.fpr}, ${runs/ae_base:at_threshold.fpr}) | pct3 }} ± {{ std(${runs/ae_base_seed0:at_threshold.fpr}, ${runs/ae_base_seed1:at_threshold.fpr}, ${runs/ae_base_seed2:at_threshold.fpr}, ${runs/ae_base_seed3:at_threshold.fpr}, ${runs/ae_base:at_threshold.fpr}) | pct3 }} | {{ min(${runs/ae_base_seed0:at_threshold.fpr}, ${runs/ae_base_seed1:at_threshold.fpr}, ${runs/ae_base_seed2:at_threshold.fpr}, ${runs/ae_base_seed3:at_threshold.fpr}, ${runs/ae_base:at_threshold.fpr}) | pct3 }} – {{ max(${runs/ae_base_seed0:at_threshold.fpr}, ${runs/ae_base_seed1:at_threshold.fpr}, ${runs/ae_base_seed2:at_threshold.fpr}, ${runs/ae_base_seed3:at_threshold.fpr}, ${runs/ae_base:at_threshold.fpr}) | pct3 }} |

Seed-to-seed variation is below one percentage point, whereas validation recall exceeds test
recall by about
{{ mean(${runs/ae_base_seed0/threshold.json:val_recall}, ${runs/ae_base_seed1/threshold.json:val_recall}, ${runs/ae_base_seed2/threshold.json:val_recall}, ${runs/ae_base_seed3/threshold.json:val_recall}, ${runs/ae_base/threshold.json:val_recall}) - mean(${runs/ae_base_seed0:at_threshold.recall}, ${runs/ae_base_seed1:at_threshold.recall}, ${runs/ae_base_seed2:at_threshold.recall}, ${runs/ae_base_seed3:at_threshold.recall}, ${runs/ae_base:at_threshold.recall}) | pp1 }}
for every seed and every variant (Figure 8): the random half of the frauds assigned to
validation is systematically easier than the half assigned to test. With 236 / 237 frauds per
half this is within the expected sampling variation of a 50/50 fraud split, but it is the
reason a validation-based estimate of "about 84 %" could not be confirmed on the test set.

![Figure 8 – seed variance](figures/seed_variance.png)

*Figure 8. Validation and test recall of the base configuration across five seeds.*

### 6.5 Baselines

PCA reconstruction error (a linear autoencoder) with k = 5, 10, 15 components and a 200-tree
Isolation Forest were fitted on the same training rows, thresholded by the same
validation-legitimate quantile and scored by the same evaluator.

| Model (random split, {{ ${runs/ae_base:n_fraud} }} test frauds) | ROC-AUC | Test FPR | Test recall [Wilson 95 %] |
|---|---:|---:|---|
| **Autoencoder** (base) | {{ ${runs/ae_base:roc_auc} | f3 }} | {{ ${runs/ae_base:at_threshold.fpr} | pct3 }} | **{{ ${runs/ae_base:at_threshold.recall} | pct1 }}** [{{ ${runs/ae_base:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_base:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| PCA k = 15 ({{ ${runs/baseline_pca_k15/train_summary.json:explained_variance_ratio} | pct1 }} variance) | {{ ${runs/baseline_pca_k15:roc_auc} | f3 }} | {{ ${runs/baseline_pca_k15:at_threshold.fpr} | pct3 }} | {{ ${runs/baseline_pca_k15:at_threshold.recall} | pct1 }} [{{ ${runs/baseline_pca_k15:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/baseline_pca_k15:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| PCA k = 10 ({{ ${runs/baseline_pca_k10/train_summary.json:explained_variance_ratio} | pct1 }} variance) | {{ ${runs/baseline_pca_k10:roc_auc} | f3 }} | {{ ${runs/baseline_pca_k10:at_threshold.fpr} | pct3 }} | {{ ${runs/baseline_pca_k10:at_threshold.recall} | pct1 }} [{{ ${runs/baseline_pca_k10:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/baseline_pca_k10:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| PCA k = 5 ({{ ${runs/baseline_pca_k5/train_summary.json:explained_variance_ratio} | pct1 }} variance; same bottleneck as the AE) | {{ ${runs/baseline_pca_k5:roc_auc} | f3 }} | {{ ${runs/baseline_pca_k5:at_threshold.fpr} | pct3 }} | {{ ${runs/baseline_pca_k5:at_threshold.recall} | pct1 }} [{{ ${runs/baseline_pca_k5:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/baseline_pca_k5:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| Isolation Forest (200 trees) | {{ ${runs/baseline_iforest:roc_auc} | f3 }} | {{ ${runs/baseline_iforest:at_threshold.fpr} | pct3 }} | {{ ${runs/baseline_iforest:at_threshold.recall} | pct1 }} [{{ ${runs/baseline_iforest:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/baseline_iforest:at_threshold.recall_ci_wilson.1} | pct1 }}] |

At the same FPR the non-linear AE detects
{{ ${runs/ae_base:at_threshold.recall} - ${runs/baseline_pca_k5:at_threshold.recall} | pp1 }} more
of the frauds than the linear autoencoder with the same five-dimensional bottleneck and
{{ ${runs/ae_base:at_threshold.recall} - ${runs/baseline_iforest:at_threshold.recall} | pp1 }} more
than the Isolation Forest, although all ROC-AUCs lie between
{{ min(${runs/ae_base:roc_auc}, ${runs/baseline_pca_k5:roc_auc}, ${runs/baseline_pca_k10:roc_auc}, ${runs/baseline_pca_k15:roc_auc}, ${runs/baseline_iforest:roc_auc}) | f3 }} and
{{ max(${runs/ae_base:roc_auc}, ${runs/baseline_pca_k5:roc_auc}, ${runs/baseline_pca_k10:roc_auc}, ${runs/baseline_pca_k15:roc_auc}, ${runs/baseline_iforest:roc_auc}) | f3 }}.
The ROC-AUC averages over all operating points and hides the difference; the advantage of the
AE is concentrated in the low-FPR region that matters for an alerting system (Figure 9). None
of the baselines approaches the criterion either.

![Figure 9 – comparison](figures/comparison_recall.png)

*Figure 9. Test recall at the validation-chosen threshold for every run, with Wilson 95 %
intervals; hollow markers show the oracle recall at 1 % FPR.*

### 6.6 Temporal split (concept drift)

The random split is the protocol used in the literature on this dataset, but a deployed
system is trained on the past and applied to the future. In the temporal variant the
legitimate transactions are ordered by `Time` and cut 60 / 20 / 20 %; frauds are assigned to
the validation or test window by their own time stamp, and the
{{ ${data/processed_temporal/data_report.json:rows.frauds_dropped_train_window} }} frauds that
fall into the training window are dropped (training stays legitimate-only). This leaves
{{ ${data/processed_temporal/data_report.json:splits.val.fraud} }} validation and
{{ ${data/processed_temporal/data_report.json:splits.test.fraud} }} test frauds, so intervals
are wide.

| Model (temporal split, {{ ${runs/ae_temporal:n_fraud} }} test frauds) | ROC-AUC | Test FPR | Test recall [Wilson 95 %] | Recall at oracle 1 % FPR |
|---|---:|---:|---|---:|
| Autoencoder | {{ ${runs/ae_temporal:roc_auc} | f3 }} | {{ ${runs/ae_temporal:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_temporal:at_threshold.recall} | pct1 }} [{{ ${runs/ae_temporal:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_temporal:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/ae_temporal:recall_at_fpr.2.recall} | pct1 }} |
| PCA k = 15 | {{ ${runs/baseline_pca_k15_temporal:roc_auc} | f3 }} | {{ ${runs/baseline_pca_k15_temporal:at_threshold.fpr} | pct3 }} | {{ ${runs/baseline_pca_k15_temporal:at_threshold.recall} | pct1 }} [{{ ${runs/baseline_pca_k15_temporal:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/baseline_pca_k15_temporal:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/baseline_pca_k15_temporal:recall_at_fpr.2.recall} | pct1 }} |
| PCA k = 10 | {{ ${runs/baseline_pca_k10_temporal:roc_auc} | f3 }} | {{ ${runs/baseline_pca_k10_temporal:at_threshold.fpr} | pct3 }} | {{ ${runs/baseline_pca_k10_temporal:at_threshold.recall} | pct1 }} [{{ ${runs/baseline_pca_k10_temporal:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/baseline_pca_k10_temporal:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/baseline_pca_k10_temporal:recall_at_fpr.2.recall} | pct1 }} |
| PCA k = 5 | {{ ${runs/baseline_pca_k5_temporal:roc_auc} | f3 }} | {{ ${runs/baseline_pca_k5_temporal:at_threshold.fpr} | pct3 }} | {{ ${runs/baseline_pca_k5_temporal:at_threshold.recall} | pct1 }} [{{ ${runs/baseline_pca_k5_temporal:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/baseline_pca_k5_temporal:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/baseline_pca_k5_temporal:recall_at_fpr.2.recall} | pct1 }} |
| Isolation Forest | {{ ${runs/baseline_iforest_temporal:roc_auc} | f3 }} | {{ ${runs/baseline_iforest_temporal:at_threshold.fpr} | pct3 }} | {{ ${runs/baseline_iforest_temporal:at_threshold.recall} | pct1 }} [{{ ${runs/baseline_iforest_temporal:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/baseline_iforest_temporal:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/baseline_iforest_temporal:recall_at_fpr.2.recall} | pct1 }} |

Two effects appear. First, the validation-legitimate threshold transfers to an FPR of only
{{ ${runs/ae_temporal:at_threshold.fpr} | pct3 }} on later traffic: the score distribution of
legitimate transactions drifts downwards within two days, so a fixed quantile becomes
conservative and costs the AE
{{ ${runs/ae_temporal:recall_at_fpr.2.recall} - ${runs/ae_temporal:at_threshold.recall} | pp1 }}
of recall relative to the oracle 1 % operating point. In production the quantile would be
re-estimated on a rolling window of recent traffic. Second, the ranking itself is unaffected:
the oracle recall at 1 % FPR ({{ ${runs/ae_temporal:recall_at_fpr.2.recall} | pct1 }}) is in line
with the random-split figure ({{ ${runs/ae_base:recall_at_fpr.2.recall} | pct1 }}), and the AE stays ahead of every baseline.

### 6.7 Which features drive the detection

The mean squared residual per input feature (Figure 10) is largest, relative to legitimate
traffic, on {{ ${runs/ae_base:per_feature.top_features.0} }},
{{ ${runs/ae_base:per_feature.top_features.1} }}, {{ ${runs/ae_base:per_feature.top_features.2} }},
{{ ${runs/ae_base:per_feature.top_features.3} }} and {{ ${runs/ae_base:per_feature.top_features.4} }}
– the components that supervised studies on this dataset also identify as the most
discriminative. `Amount` and the hour-of-day pair contribute little, which is consistent with
the no-time ablation.

![Figure 10 – per-feature residuals](figures/per_feature_residuals_test_ae_base.png)

*Figure 10. Mean squared reconstruction residual per feature for legitimate and fraudulent test
transactions, sorted by their ratio.*

### 6.8 Pooled-fraud protocol, repeated splits and diagnosis (v2)

The v1 protocol put half of the frauds into validation although the threshold never reads a
fraud label (Section 4). Under the *pooled protocol* the legitimate split is unchanged, every
fraud is a test fraud and the checkpoint is byte-identical to `runs/ae_base`; only the
measurement changes. Re-drawing the legitimate split with five data seeds (train seed fixed)
separates split variance from the sampling interval.

| Measurement of the same legit-only AE | Test frauds | Test FPR | Test recall [Wilson 95 %] |
|---|---:|---:|---|
| v1 split (Section 6.1) | {{ ${runs/ae_base:n_fraud} }} | {{ ${runs/ae_base:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_base:at_threshold.recall} | pct1 }} [{{ ${runs/ae_base:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_base:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| pooled protocol, same split | {{ ${runs/ae_pooled:n_fraud} }} | {{ ${runs/ae_pooled:at_threshold.fpr} | pct3 }} | **{{ ${runs/ae_pooled:at_threshold.recall} | pct1 }}** [{{ ${runs/ae_pooled:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_pooled:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| pooled, five re-drawn legit splits (mean ± std) | 473 | {{ min(${runs/ae_pooled_dseed0:at_threshold.fpr}, ${runs/ae_pooled_dseed1:at_threshold.fpr}, ${runs/ae_pooled_dseed2:at_threshold.fpr}, ${runs/ae_pooled_dseed3:at_threshold.fpr}, ${runs/ae_pooled_dseed4:at_threshold.fpr}) | pct2 }} – {{ max(${runs/ae_pooled_dseed0:at_threshold.fpr}, ${runs/ae_pooled_dseed1:at_threshold.fpr}, ${runs/ae_pooled_dseed2:at_threshold.fpr}, ${runs/ae_pooled_dseed3:at_threshold.fpr}, ${runs/ae_pooled_dseed4:at_threshold.fpr}) | pct2 }} | {{ mean(${runs/ae_pooled_dseed0:at_threshold.recall}, ${runs/ae_pooled_dseed1:at_threshold.recall}, ${runs/ae_pooled_dseed2:at_threshold.recall}, ${runs/ae_pooled_dseed3:at_threshold.recall}, ${runs/ae_pooled_dseed4:at_threshold.recall}) | pct1 }} ± {{ std(${runs/ae_pooled_dseed0:at_threshold.recall}, ${runs/ae_pooled_dseed1:at_threshold.recall}, ${runs/ae_pooled_dseed2:at_threshold.recall}, ${runs/ae_pooled_dseed3:at_threshold.recall}, ${runs/ae_pooled_dseed4:at_threshold.recall}) | pp1 }} |
| literature protocol (no duplicate removal, all frauds in test) | {{ ${runs/ae_literature:n_fraud} }} | {{ ${runs/ae_literature:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_literature:at_threshold.recall} | pct1 }} [{{ ${runs/ae_literature:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_literature:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| PCA k = 15, pooled | 473 | {{ ${runs/baseline_pca_k15_pooled:at_threshold.fpr} | pct3 }} | {{ ${runs/baseline_pca_k15_pooled:at_threshold.recall} | pct1 }} [{{ ${runs/baseline_pca_k15_pooled:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/baseline_pca_k15_pooled:at_threshold.recall_ci_wilson.1} | pct1 }}] |
| Isolation Forest, pooled | 473 | {{ ${runs/baseline_iforest_pooled:at_threshold.fpr} | pct3 }} | {{ ${runs/baseline_iforest_pooled:at_threshold.recall} | pct1 }} [{{ ${runs/baseline_iforest_pooled:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/baseline_iforest_pooled:at_threshold.recall_ci_wilson.1} | pct1 }}] |

The v1 validation ({{ ${runs/ae_base/threshold.json:val_recall} | pct1 }}) and test
({{ ${runs/ae_base:at_threshold.recall} | pct1 }}) halves are two draws of one
{{ ${runs/ae_pooled:at_threshold.recall} | pct1 }} detector; the interval is half as wide
and the criterion ({{ ${runs/ae_pooled:criterion.tp_required} }} of 473 frauds) is still
{{ ${runs/ae_pooled:meets_target} | verdict }}. The literature row gains about 1.5 pp from
duplicated frauds and a lower FPR and is shown for comparability only.

**Diagnosis of the missed frauds** (`fraud-ae diagnose`, Figure 11). The
{{ ${runs/ae_pooled/diagnostics.json:n_missed} }} missed frauds have a median score of
{{ ${runs/ae_pooled/diagnostics.json:groups.missed.score_median} | f3 }} (legitimate
{{ ${runs/ae_pooled/diagnostics.json:groups.legit.score_median} | f3 }}, detected frauds
{{ ${runs/ae_pooled/diagnostics.json:groups.detected.score_median} | f1 }}), i.e. the
{{ ${runs/ae_pooled/diagnostics.json:groups.missed.legit_percentile_median} * 100 | f1 }}th
percentile of legitimate scores while the threshold sits at the 99.1st; their median amount is
{{ ${runs/ae_pooled/diagnostics.json:groups.missed.amount_median} | f2 }} and they are spread
over the day. On the five most separating features
({{ ${runs/ae_pooled/diagnostics.json:top_features.features.0} }},
{{ ${runs/ae_pooled/diagnostics.json:top_features.features.1} }},
{{ ${runs/ae_pooled/diagnostics.json:top_features.features.2} }},
{{ ${runs/ae_pooled/diagnostics.json:top_features.features.3} }},
{{ ${runs/ae_pooled/diagnostics.json:top_features.features.4} }}) detected frauds average
{{ ${runs/ae_pooled/diagnostics.json:top_features.detected_mean.0} | f1 }} to
{{ ${runs/ae_pooled/diagnostics.json:top_features.detected_mean.4} | f1 }} standard
deviations, missed frauds {{ ${runs/ae_pooled/diagnostics.json:top_features.missed_mean.1} | f1 }} to
{{ ${runs/ae_pooled/diagnostics.json:top_features.missed_mean.0} | f1 }} – they lie on the
legitimate manifold. The other detectors do not see them either: at their own thresholds PCA
k = 15 alarms on {{ ${runs/ae_pooled/diagnostics.json:against.0.caught_of_missed_own} }} of
the {{ ${runs/ae_pooled/diagnostics.json:n_missed} }}, the Isolation Forest on
{{ ${runs/ae_pooled/diagnostics.json:against.2.caught_of_missed_own} }}; the union of all
four detectors reaches {{ ${runs/ae_pooled/diagnostics.json:fusion.union_own.recall} | pct1 }}
at {{ ${runs/ae_pooled/diagnostics.json:fusion.union_own.fpr} | pct2 }} FPR and rank-average
fusion {{ ${runs/ae_pooled/diagnostics.json:fusion.rank_fusion.recall} | pct1 }} at
{{ ${runs/ae_pooled/diagnostics.json:fusion.rank_fusion.fpr} | pct2 }}. No unsupervised
combination reaches 85 %.

![Figure 11 – missed frauds](figures/missed_frauds_ae_pooled.png)

*Figure 11. Missed vs detected frauds under the pooled protocol: score densities, hour of day,
amount and the means of the most separating features.*

### 6.9 The legit-only ceiling (v2)

Six legit-only candidates were run on data seed 0 under a pre-registered rule (best pooled
recall with FPR < 1 % is re-run on data seeds 1–4 next to the base AE; configs
`configs/v2_unsup_*.toml`, table `results/v2_unsupervised.md`).

| Candidate (data seed 0; base AE {{ ${runs/ae_pooled_dseed0:at_threshold.recall} | pct1 }} at {{ ${runs/ae_pooled_dseed0:at_threshold.fpr} | pct3 }}) | Test FPR | Test recall [Wilson 95 %] | Recall at oracle 0.1 % FPR |
|---|---:|---|---:|
| legit split 80 / 10 / 10 (a third more training rows) | {{ ${runs/ae_train80_dseed0:at_threshold.fpr} | pct3 }} | **{{ ${runs/ae_train80_dseed0:at_threshold.recall} | pct1 }}** [{{ ${runs/ae_train80_dseed0:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_train80_dseed0:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/ae_train80_dseed0:recall_at_fpr.0.recall} | pct1 }} |
| night / day legit quantiles (hour-segmented threshold) | {{ ${runs/ae_by_hour_dseed0:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_by_hour_dseed0:at_threshold.recall} | pct1 }} [{{ ${runs/ae_by_hour_dseed0:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_by_hour_dseed0:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/ae_by_hour_dseed0:recall_at_fpr.0.recall} | pct1 }} |
| variational AE, reconstruction probability [3] | {{ ${runs/ae_vae_dseed0:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_vae_dseed0:at_threshold.recall} | pct1 }} [{{ ${runs/ae_vae_dseed0:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_vae_dseed0:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/ae_vae_dseed0:recall_at_fpr.0.recall} | pct1 }} |
| z(log MSE) + z(latent GMM NLL), 8 components | {{ ${runs/ae_latent_gmm8_dseed0:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_latent_gmm8_dseed0:at_threshold.recall} | pct1 }} [{{ ${runs/ae_latent_gmm8_dseed0:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_latent_gmm8_dseed0:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/ae_latent_gmm8_dseed0:recall_at_fpr.0.recall} | pct1 }} |
| latent GMM NLL alone | {{ ${runs/ae_latent_nll_dseed0:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_latent_nll_dseed0:at_threshold.recall} | pct1 }} [{{ ${runs/ae_latent_nll_dseed0:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_latent_nll_dseed0:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/ae_latent_nll_dseed0:recall_at_fpr.0.recall} | pct1 }} |

The 80 / 10 / 10 split won the selection but not the confirmation: over the five splits it
averages {{ mean(${runs/ae_train80_dseed0:at_threshold.recall}, ${runs/ae_train80_dseed1:at_threshold.recall}, ${runs/ae_train80_dseed2:at_threshold.recall}, ${runs/ae_train80_dseed3:at_threshold.recall}, ${runs/ae_train80_dseed4:at_threshold.recall}) | pct1 }} ±
{{ std(${runs/ae_train80_dseed0:at_threshold.recall}, ${runs/ae_train80_dseed1:at_threshold.recall}, ${runs/ae_train80_dseed2:at_threshold.recall}, ${runs/ae_train80_dseed3:at_threshold.recall}, ${runs/ae_train80_dseed4:at_threshold.recall}) | pp1 }}
against {{ mean(${runs/ae_pooled_dseed0:at_threshold.recall}, ${runs/ae_pooled_dseed1:at_threshold.recall}, ${runs/ae_pooled_dseed2:at_threshold.recall}, ${runs/ae_pooled_dseed3:at_threshold.recall}, ${runs/ae_pooled_dseed4:at_threshold.recall}) | pct1 }} ±
{{ std(${runs/ae_pooled_dseed0:at_threshold.recall}, ${runs/ae_pooled_dseed1:at_threshold.recall}, ${runs/ae_pooled_dseed2:at_threshold.recall}, ${runs/ae_pooled_dseed3:at_threshold.recall}, ${runs/ae_pooled_dseed4:at_threshold.recall}) | pp1 }}
for the base AE, a paired difference within noise; the best single split is
{{ max(${runs/ae_train80_dseed0:at_threshold.recall}, ${runs/ae_train80_dseed1:at_threshold.recall}, ${runs/ae_train80_dseed2:at_threshold.recall}, ${runs/ae_train80_dseed3:at_threshold.recall}, ${runs/ae_train80_dseed4:at_threshold.recall}) | pct1 }}.
Two negative results are worth stating: the latent code of the 5-D bottleneck carries no
usable density signal for this fraud population (the GMM term costs 25–28 pp when added to the
residual), and the VAE's reconstruction probability equals plain MSE at 1 % FPR but is clearly
worse at 0.1 % FPR because the learned per-feature variance down-weights exactly the
high-residual features. **The ceiling of the legit-only setting on this feature set is
≈ 81–82 % recall at 1 % FPR**, consistent with the diagnosis of Section 6.8.

### 6.10 Labels inside the model (v2)

The {{ ${runs/sup_hgb:n_fraud} }} frauds are cut into five fixed folds. The fold-*k* model is
fitted on all training legitimate rows plus the frauds of the other folds, takes its threshold
from the 0.9 % quantile of its scores on the legitimate validation rows (never a fraud label),
and scores only the fold-*k* rows; scores are divided by the fold threshold, so every method
is evaluated out-of-fold on all 473 frauds at the common threshold 1.0 (`results/v2_supervised.md`).
Hyper-parameters are fixed in `configs/v2_supervised.toml`. Every method is also trained with
25 / 50 / 100 labelled frauds per fold (`results/v2_few_label.md`, Figure 13).

| Method (out-of-fold, all labels) | Test FPR | Recall [Wilson 95 %] | 25 / 50 / 100 labels per fold |
|---|---:|---|---|
| gradient boosting on `[x, AE residual, AE score]` | {{ ${runs/sup_hybrid_hgb:at_threshold.fpr} | pct3 }} | **{{ ${runs/sup_hybrid_hgb:at_threshold.recall} | pct1 }}** [{{ ${runs/sup_hybrid_hgb:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/sup_hybrid_hgb:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/sup_hybrid_hgb/few_label.json:budgets.0.recall} | pct1 }} / {{ ${runs/sup_hybrid_hgb/few_label.json:budgets.1.recall} | pct1 }} / {{ ${runs/sup_hybrid_hgb/few_label.json:budgets.2.recall} | pct1 }} |
| gradient boosting, class-weighted (`hgb`) | {{ ${runs/sup_hgb:at_threshold.fpr} | pct3 }} | **{{ ${runs/sup_hgb:at_threshold.recall} | pct1 }}** [{{ ${runs/sup_hgb:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/sup_hgb:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/sup_hgb/few_label.json:budgets.0.recall} | pct1 }} / {{ ${runs/sup_hgb/few_label.json:budgets.1.recall} | pct1 }} / {{ ${runs/sup_hgb/few_label.json:budgets.2.recall} | pct1 }} |
| logistic regression, class-weighted | {{ ${runs/sup_logreg:at_threshold.fpr} | pct3 }} | **{{ ${runs/sup_logreg:at_threshold.recall} | pct1 }}** [{{ ${runs/sup_logreg:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/sup_logreg:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/sup_logreg/few_label.json:budgets.0.recall} | pct1 }} / {{ ${runs/sup_logreg/few_label.json:budgets.1.recall} | pct1 }} / {{ ${runs/sup_logreg/few_label.json:budgets.2.recall} | pct1 }} |
| gradient boosting, legit under-sampled 20 : 1 | {{ ${runs/sup_hgb_under:at_threshold.fpr} | pct3 }} | **{{ ${runs/sup_hgb_under:at_threshold.recall} | pct1 }}** [{{ ${runs/sup_hgb_under:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/sup_hgb_under:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/sup_hgb_under/few_label.json:budgets.0.recall} | pct1 }} / {{ ${runs/sup_hgb_under/few_label.json:budgets.1.recall} | pct1 }} / {{ ${runs/sup_hgb_under/few_label.json:budgets.2.recall} | pct1 }} |
| logistic regression on `[x, AE residual, AE score]` | {{ ${runs/sup_hybrid_logreg:at_threshold.fpr} | pct3 }} | {{ ${runs/sup_hybrid_logreg:at_threshold.recall} | pct1 }} [{{ ${runs/sup_hybrid_logreg:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/sup_hybrid_logreg:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/sup_hybrid_logreg/few_label.json:budgets.0.recall} | pct1 }} / {{ ${runs/sup_hybrid_logreg/few_label.json:budgets.1.recall} | pct1 }} / {{ ${runs/sup_hybrid_logreg/few_label.json:budgets.2.recall} | pct1 }} |
| **the AE**, fine-tuned with a margin loss on frauds (`neg_ae`) | {{ ${runs/sup_neg_ae:at_threshold.fpr} | pct3 }} | {{ ${runs/sup_neg_ae:at_threshold.recall} | pct1 }} [{{ ${runs/sup_neg_ae:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/sup_neg_ae:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/sup_neg_ae/few_label.json:budgets.0.recall} | pct1 }} / {{ ${runs/sup_neg_ae/few_label.json:budgets.1.recall} | pct1 }} / {{ ${runs/sup_neg_ae/few_label.json:budgets.2.recall} | pct1 }} |
| MLP 31-20-10-1, balanced mini-batches | {{ ${runs/sup_mlp:at_threshold.fpr} | pct3 }} | {{ ${runs/sup_mlp:at_threshold.recall} | pct1 }} [{{ ${runs/sup_mlp:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/sup_mlp:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/sup_mlp/few_label.json:budgets.0.recall} | pct1 }} / {{ ${runs/sup_mlp/few_label.json:budgets.1.recall} | pct1 }} / {{ ${runs/sup_mlp/few_label.json:budgets.2.recall} | pct1 }} |
| the AE with a Deep-SAD latent term [9] | {{ ${runs/sup_sad_ae:at_threshold.fpr} | pct3 }} | {{ ${runs/sup_sad_ae:at_threshold.recall} | pct1 }} [{{ ${runs/sup_sad_ae:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/sup_sad_ae:at_threshold.recall_ci_wilson.1} | pct1 }}] | {{ ${runs/sup_sad_ae/few_label.json:budgets.0.recall} | pct1 }} / {{ ${runs/sup_sad_ae/few_label.json:budgets.1.recall} | pct1 }} / {{ ${runs/sup_sad_ae/few_label.json:budgets.2.recall} | pct1 }} |
| legit-only AE (Section 6.8, no labels) | {{ ${runs/ae_pooled:at_threshold.fpr} | pct3 }} | {{ ${runs/ae_pooled:at_threshold.recall} | pct1 }} [{{ ${runs/ae_pooled:at_threshold.recall_ci_wilson.0} | pct1 }}, {{ ${runs/ae_pooled:at_threshold.recall_ci_wilson.1} | pct1 }}] | – |

**The criterion is {{ ${runs/sup_hgb:meets_target} | verdict }}** by every method that sees
the labelled frauds, the autoencoder itself included; four methods keep the whole interval
above 85 %. Labels buy
+{{ ${runs/sup_hybrid_hgb:at_threshold.recall} - ${runs/ae_pooled:at_threshold.recall} | pp1 }}
for the best classifier and
+{{ ${runs/sup_neg_ae:at_threshold.recall} - ${runs/ae_pooled:at_threshold.recall} | pp1 }}
for the AE; tree models and the fine-tuned AEs degrade gracefully (25 labels per fold still add
1–3 pp over no labels, about 50 reach 85 % with under-sampled gradient boosting), while the
linear and MLP classifiers need ≥ 100 labels to beat the legit-only AE at all. The AE as an
extra feature changes gradient boosting by
{{ ${runs/sup_hybrid_hgb:at_threshold.recall} - ${runs/sup_hgb:at_threshold.recall} | pp1 }}
at all labels and by nothing or less at smaller budgets, and costs the linear model
{{ ${runs/sup_logreg:at_threshold.recall} - ${runs/sup_hybrid_logreg:at_threshold.recall} | pp1 }}:
once labels exist, the reconstruction residual adds nothing that gradient boosting cannot learn
from `x` itself. Precision at this operating point is 44–48 % for every labelled method.

![Figure 12 – labelled methods vs the legit-only AE](figures/v2_sup_comparison_recall.png)

*Figure 12. Out-of-fold recall at FPR < 1 % of the eight labelled methods and the legit-only AE,
all 473 frauds, Wilson 95 % intervals.*

![Figure 13 – few-label curve](figures/v2_few_label.png)

*Figure 13. Out-of-fold recall vs labelled frauds per training fold (25, 50, 100, all);
dashed line = legit-only AE.*

### 6.11 Robustness: rolling threshold, variance, evasion (v2)

**Rolling threshold** (Figure 14). On the temporal split of Section 6.6 the models see the
first 60 % of traffic (the labelled ones also its
{{ ${results/v2_robustness.json:rolling.train_window_frauds} }} frauds); the last 40 % is cut into
{{ ${results/v2_robustness.json:rolling.n_windows} }} windows of equal legitimate count (≈ 14 k rows), window 0
calibrates the fixed 0.9 % legit-quantile threshold (the v1 procedure) and windows 1–7
({{ ${results/v2_robustness.json:rolling.models.hgb.summary.fixed.n_fraud} }} frauds) are evaluated with it and with
the quantile re-estimated on the previous window's legitimate rows.

| Model | Fixed threshold: FPR mean (windows in [0.8 %, 1.0 %]) | Rolling threshold: FPR mean (in band) | Recall on the later frauds, rolling [Wilson 95 %] |
|---|---|---|---|
| legit-only AE | {{ ${results/v2_robustness.json:rolling.models.ae.summary.fixed.fpr_mean} | pct2 }} ({{ ${results/v2_robustness.json:rolling.models.ae.summary.fixed.windows_in_band} }} / 7) | {{ ${results/v2_robustness.json:rolling.models.ae.summary.rolling_legit.fpr_mean} | pct2 }} ({{ ${results/v2_robustness.json:rolling.models.ae.summary.rolling_legit.windows_in_band} }} / 7) | {{ ${results/v2_robustness.json:rolling.models.ae.summary.rolling_legit.pooled_recall} | pct1 }} [{{ ${results/v2_robustness.json:rolling.models.ae.summary.rolling_legit.pooled_recall_ci_wilson.0} | pct1 }}, {{ ${results/v2_robustness.json:rolling.models.ae.summary.rolling_legit.pooled_recall_ci_wilson.1} | pct1 }}] |
| gradient boosting (`hgb`) | **{{ ${results/v2_robustness.json:rolling.models.hgb.summary.fixed.fpr_mean} | pct2 }}** ({{ ${results/v2_robustness.json:rolling.models.hgb.summary.fixed.windows_in_band} }} / 7) | {{ ${results/v2_robustness.json:rolling.models.hgb.summary.rolling_legit.fpr_mean} | pct2 }} ({{ ${results/v2_robustness.json:rolling.models.hgb.summary.rolling_legit.windows_in_band} }} / 7) | {{ ${results/v2_robustness.json:rolling.models.hgb.summary.rolling_legit.pooled_recall} | pct1 }} [{{ ${results/v2_robustness.json:rolling.models.hgb.summary.rolling_legit.pooled_recall_ci_wilson.0} | pct1 }}, {{ ${results/v2_robustness.json:rolling.models.hgb.summary.rolling_legit.pooled_recall_ci_wilson.1} | pct1 }}] |
| AE + margin loss (`neg_ae`) | {{ ${results/v2_robustness.json:rolling.models.neg_ae.summary.fixed.fpr_mean} | pct2 }} ({{ ${results/v2_robustness.json:rolling.models.neg_ae.summary.fixed.windows_in_band} }} / 7) | {{ ${results/v2_robustness.json:rolling.models.neg_ae.summary.rolling_legit.fpr_mean} | pct2 }} ({{ ${results/v2_robustness.json:rolling.models.neg_ae.summary.rolling_legit.windows_in_band} }} / 7) | {{ ${results/v2_robustness.json:rolling.models.neg_ae.summary.rolling_legit.pooled_recall} | pct1 }} [{{ ${results/v2_robustness.json:rolling.models.neg_ae.summary.rolling_legit.pooled_recall_ci_wilson.0} | pct1 }}, {{ ${results/v2_robustness.json:rolling.models.neg_ae.summary.rolling_legit.pooled_recall_ci_wilson.1} | pct1 }}] |

The fixed threshold is the failure mode, and worst for the supervised model, whose legitimate
scores drift upward after calibration; the AE drifts the other way (Section 6.6).
Re-estimating the quantile on the previous window removes the systematic error for all three
models. The remaining per-window spread (± 0.3 pp) is what binomial noise on 14 k rows allows
(standard error 0.08 pp), so "every window inside a 0.2-pp band" is not an attainable target
and the mean is the right criterion. Trained on the frauds of the first 60 % of traffic, both
labelled winners catch {{ ${results/v2_robustness.json:rolling.models.hgb.summary.rolling_legit.pooled_recall} | pct1 }}
of the later frauds – the supervised models did not merely memorise the two-day campaigns –
with a wide interval.

**Variance of the winners** (all labels, out-of-fold; model seeds and re-drawn legitimate
splits, each split with its own legit-only AE):

| Family | Group | Runs | Recall mean ± std (min – max) | FPR min – max | Every run ≥ 85 % |
|---|---|---:|---|---|:---:|
| `hgb` | model seed | {{ ${results/v2_robustness.json:variance.sup_hgb.model_seed.n} }} | {{ ${results/v2_robustness.json:variance.sup_hgb.model_seed.recall_mean} | pct1 }} ± {{ ${results/v2_robustness.json:variance.sup_hgb.model_seed.recall_std} | pp1 }} ({{ ${results/v2_robustness.json:variance.sup_hgb.model_seed.recall_min} | pct1 }} – {{ ${results/v2_robustness.json:variance.sup_hgb.model_seed.recall_max} | pct1 }}) | {{ ${results/v2_robustness.json:variance.sup_hgb.model_seed.fpr_min} | pct2 }} – {{ ${results/v2_robustness.json:variance.sup_hgb.model_seed.fpr_max} | pct2 }} | {{ ${results/v2_robustness.json:variance.sup_hgb.model_seed.all_meet_recall} }} |
| `hgb` | legit split | {{ ${results/v2_robustness.json:variance.sup_hgb.data_seed.n} }} | {{ ${results/v2_robustness.json:variance.sup_hgb.data_seed.recall_mean} | pct1 }} ± {{ ${results/v2_robustness.json:variance.sup_hgb.data_seed.recall_std} | pp1 }} ({{ ${results/v2_robustness.json:variance.sup_hgb.data_seed.recall_min} | pct1 }} – {{ ${results/v2_robustness.json:variance.sup_hgb.data_seed.recall_max} | pct1 }}) | {{ ${results/v2_robustness.json:variance.sup_hgb.data_seed.fpr_min} | pct2 }} – {{ ${results/v2_robustness.json:variance.sup_hgb.data_seed.fpr_max} | pct2 }} | {{ ${results/v2_robustness.json:variance.sup_hgb.data_seed.all_meet_recall} }} |
| `neg_ae` | model seed | {{ ${results/v2_robustness.json:variance.sup_neg_ae.model_seed.n} }} | {{ ${results/v2_robustness.json:variance.sup_neg_ae.model_seed.recall_mean} | pct1 }} ± {{ ${results/v2_robustness.json:variance.sup_neg_ae.model_seed.recall_std} | pp1 }} ({{ ${results/v2_robustness.json:variance.sup_neg_ae.model_seed.recall_min} | pct1 }} – {{ ${results/v2_robustness.json:variance.sup_neg_ae.model_seed.recall_max} | pct1 }}) | {{ ${results/v2_robustness.json:variance.sup_neg_ae.model_seed.fpr_min} | pct2 }} – {{ ${results/v2_robustness.json:variance.sup_neg_ae.model_seed.fpr_max} | pct2 }} | {{ ${results/v2_robustness.json:variance.sup_neg_ae.model_seed.all_meet_recall} }} |
| `neg_ae` | legit split | {{ ${results/v2_robustness.json:variance.sup_neg_ae.data_seed.n} }} | {{ ${results/v2_robustness.json:variance.sup_neg_ae.data_seed.recall_mean} | pct1 }} ± {{ ${results/v2_robustness.json:variance.sup_neg_ae.data_seed.recall_std} | pp1 }} ({{ ${results/v2_robustness.json:variance.sup_neg_ae.data_seed.recall_min} | pct1 }} – {{ ${results/v2_robustness.json:variance.sup_neg_ae.data_seed.recall_max} | pct1 }}) | {{ ${results/v2_robustness.json:variance.sup_neg_ae.data_seed.fpr_min} | pct2 }} – {{ ${results/v2_robustness.json:variance.sup_neg_ae.data_seed.fpr_max} | pct2 }} | {{ ${results/v2_robustness.json:variance.sup_neg_ae.data_seed.all_meet_recall} }} |

Gradient boosting without early stopping or subsampling is deterministic, so its model-seed
spread is zero by construction; the split-to-split spread of both winners is far inside the
Wilson half-width, and every run clears the criterion.

**Evasion probe** (Figure 15). For each of the {{ ${results/v2_robustness.json:evasion.n_detected} }} test frauds the
legit-only AE detects, projected gradient descent [10] on the input minimises the score inside
an L2 ball of radius ε in standardised feature units (white-box, two starts, best iterate; no
feasibility constraints). Within ε = 1 – about one standard deviation of a single feature –
{{ ${results/v2_robustness.json:evasion.per_budget.5.evaded_fraction} | pct1 }} of the detected frauds cross below the
threshold, within ε = 4 {{ ${results/v2_robustness.json:evasion.per_budget.9.evaded_fraction} | pct1 }}, within
ε = 8 {{ ${results/v2_robustness.json:evasion.per_budget.11.evaded_fraction} | pct1 }}; the median minimal budget is
**{{ ${results/v2_robustness.json:evasion.median_minimal_budget} | f0 }}** against a median L2 norm of
{{ ${results/v2_robustness.json:evasion.fraud_l2_norm_median} | f1 }} for a detected fraud, and
{{ ${results/v2_robustness.json:evasion.never_evaded} }} frauds are not evaded within ε = 24. The perturbation moves
{{ ${results/v2_robustness.json:evasion.movement.features.0} }}, {{ ${results/v2_robustness.json:evasion.movement.features.1} }} and
{{ ${results/v2_robustness.json:evasion.movement.features.2} }} most – the components that separate frauds in the
first place (Section 6.7). An attacker with the gradients therefore has to move a typical fraud
more than half of the way to the mean legitimate transaction; since `V1..V28` are PCA
components of undisclosed fields, the cost in the real feature space is higher still.

![Figure 14 – rolling threshold](figures/v2_rolling_fpr.png)

*Figure 14. FPR per window on the temporal split with the fixed window-0 threshold and with the
quantile re-estimated on the previous window; shaded band = [0.8 %, 1.0 %].*

![Figure 15 – evasion probe](figures/v2_evasion.png)

*Figure 15. Share of detected frauds pushed below the AE threshold as a function of the L2
budget, and the features that move at the median budget.*

## 7. Discussion

**Why 85 % is out of reach for the legit-only detector family.** Reconstruction-error detectors flag
transactions that lie off the manifold of legitimate traffic. The frauds the AE misses are
those that lie *on* it: their `V`-profile, amount and hour are unremarkable one transaction at a
time. The ablations show that this is not a matter of capacity (latent 3 vs 8), optimisation
(100 vs 200 epochs), normalisation, score whitening or seed: every configuration misses the
same fifth. Distinguishing these frauds requires information that the row does not carry –
the card-holder's history, velocity, merchant, device – i.e. *contextual* rather than *point*
anomaly detection – or labels (Section 6.10).

**What labels buy, and what the AE buys.** With the labelled frauds inside the model every
method clears the criterion out-of-fold, including the autoencoder with a margin loss; the
Phase 8 ceiling was a ceiling of the *legit-only* setting, not of the feature set. The missed
frauds are not separable by distance from the legitimate manifold but they are separable by a
decision boundary learned from examples. Conversely, once labels exist the AE residual adds
nothing to gradient boosting: the two routes are complementary in *what they need* (labels
versus none) rather than in *what they see*. A deployment would run the supervised classifier
for recall on known fraud patterns next to the legit-only AE for departures from normal traffic
that no label has described yet, both behind a rolling threshold.

**Validation versus test.** The project's pre-registered protocol (threshold and model
selection on validation, a single evaluation pass on test) exposed a 7-point gap between the
two fraud halves that would have been invisible had the threshold been tuned on test or the
frauds pooled. We report it as a finding rather than re-splitting until the gap disappears.

**Against the literature.** Published reconstruction-error autoencoders on this dataset reach
recall of roughly 0.80–0.90 at about 1 % FPR under random splits with all frauds in the test
set, pooled thresholds and without duplicate removal. Our setting is stricter on each point;
the literature-protocol row of Section 6.8 ({{ ${runs/ae_literature:at_threshold.recall} | pct1 }}
on {{ ${runs/ae_literature:n_fraud} }} frauds) reproduces the lower end of that band and shows
where the difference comes from (duplicated frauds, a lower FPR).

## 8. Limitations and threat-model discussion

* **Point-anomaly detector.** Each transaction is scored in isolation. An attacker who keeps
  individual transactions inside the card-holder's usual envelope (small amounts, usual hours)
  is invisible by construction; such patterns are only detectable from account history.
* **Obfuscated features.** `V1..V28` are PCA components of undisclosed original fields; no
  merchant, country, channel or account identifier is available, so velocity and limit-based
  features cannot be built and the results do not transfer directly to a bank's raw data.
* **Two days of data, one geography.** Section 6.11 measures the drift: a fixed legit-quantile
  threshold runs the supervised winner at {{ ${results/v2_robustness.json:rolling.models.hgb.summary.fixed.fpr_mean} | pct2 }}
  FPR within the second day; rolling re-estimation on the previous window restores the mean
  but cannot remove the ± 0.3 pp sampling spread of 14 k-row windows. Seasonal effects and the
  stability of the supervised models beyond the two recorded days are outside what this dataset
  can show; the labelled winners were checked on {{ ${results/v2_robustness.json:rolling.models.hgb.summary.fixed.n_fraud} }}
  later frauds only.
* **Adversarial evasion – measured.** The AE is differentiable, and Section 6.11 gives the
  white-box cost: a median L2 budget of {{ ${results/v2_robustness.json:evasion.median_minimal_budget} | f0 }}
  standardised units per detected fraud, {{ ${results/v2_robustness.json:evasion.per_budget.5.evaded_fraction} | pct1 }}
  of them within one unit. The probe is an upper bound on attacker power (full gradient access,
  PCA components set directly, no feasibility constraints); the gradient-boosting winner is not
  differentiable and was not probed. Hardening (score randomisation, ensembles, rate-limited
  scoring, retraining) is discussed in the literature and not implemented here.
* **Label noise and base rate.** The dataset labels are themselves the output of a fraud
  process; the base rate ({{ ${data/processed/data_report.json:class_balance.fraud_rate} | pct3 }})
  means that even at the criterion three of four alarms would be false, which bounds the
  operational value of any detector at this operating point.
* **Statistical power.** {{ ${runs/ae_base:n_fraud} }} test frauds give ±5-point intervals;
  differences smaller than that between variants are not resolvable here.

## 9. Conclusions

1. A 1 836-parameter autoencoder trained only on legitimate transactions, with a threshold set
   on legitimate validation data, detects **{{ ${runs/ae_base:at_threshold.recall} | pct1 }}** of
   the v1 test frauds and **{{ ${runs/ae_pooled:at_threshold.recall} | pct1 }}**
   ({{ ${runs/ae_pooled:at_threshold.recall_ci_wilson.0} | pct1 }} – {{ ${runs/ae_pooled:at_threshold.recall_ci_wilson.1} | pct1 }})
   of all {{ ${runs/ae_pooled:n_fraud} }} frauds at an FPR below 1 %. The FPR criterion is met;
   the 85 % recall criterion is **{{ ${runs/ae_pooled:criterion.meets_recall} | verdict }}** in
   the legit-only setting, and the interval excludes it.
2. The shortfall is a ceiling of the legit-only approach, not of this model: v1 variants, more
   training data, a VAE, latent-density scores and hour-segmented thresholds all stay at
   ≈ 80–82 %. The {{ ${runs/ae_pooled/diagnostics.json:n_missed} }} missed frauds lie on the
   legitimate manifold and no unsupervised ensemble reaches 85 %
   (union {{ ${runs/ae_pooled/diagnostics.json:fusion.union_own.recall} | pct1 }} at
   {{ ${runs/ae_pooled/diagnostics.json:fusion.union_own.fpr} | pct2 }} FPR).
3. The autoencoder is nonetheless the best unsupervised detector in the study by a wide margin:
   +{{ ${runs/ae_pooled:at_threshold.recall} - ${runs/baseline_pca_k15_pooled:at_threshold.recall} | pp1 }}
   over the best PCA baseline and
   +{{ ${runs/ae_pooled:at_threshold.recall} - ${runs/baseline_iforest_pooled:at_threshold.recall} | pp1 }}
   over the Isolation Forest at the same FPR on 473 frauds, and ahead on the temporal split.
4. **With labels the criterion is met.** Evaluated out-of-fold over all 473 frauds with
   legit-only thresholds, gradient boosting reaches
   **{{ ${runs/sup_hgb:at_threshold.recall} | pct1 }}** ({{ ${runs/sup_hgb:at_threshold.recall_ci_wilson.0} | pct1 }} – {{ ${runs/sup_hgb:at_threshold.recall_ci_wilson.1} | pct1 }})
   and the autoencoder fine-tuned with a margin loss
   **{{ ${runs/sup_neg_ae:at_threshold.recall} | pct1 }}**; about 50 labelled frauds per fold
   suffice for a tree model. The AE as an extra feature adds nothing measurable
   ({{ ${runs/sup_hybrid_hgb:at_threshold.recall} - ${runs/sup_hgb:at_threshold.recall} | pp1 }}).
5. A fixed legit-quantile threshold does not survive two days of traffic (the supervised
   winner would run at {{ ${results/v2_robustness.json:rolling.models.hgb.summary.fixed.fpr_mean} | pct2 }}
   FPR); re-estimating the quantile on a rolling window of recent legitimate traffic is part of
   the detector, not an operational detail. Both labelled winners hold
   {{ ${results/v2_robustness.json:rolling.models.hgb.summary.rolling_legit.pooled_recall} | pct1 }}
   recall on later frauds and ≥ 85 % across model seeds and legitimate splits.
6. Evading the legit-only AE costs a white-box attacker a median L2 budget of
   {{ ${results/v2_robustness.json:evasion.median_minimal_budget} | f0 }} standardised units, more
   than half of a fraud's distance to the mean legitimate transaction, moved along the very
   features that make frauds detectable.
7. Recommendation: deploy the supervised classifier for recall on known fraud patterns together
   with the legit-only autoencoder for departures from normal traffic, both behind a rolling
   threshold, and treat account-level context as the next source of recall.

## 10. Reproducibility

Every number above is rendered from a JSON artifact by `fraud-ae report` (template:
`report/report_template.md`); a unit test fails if the committed report and the artifacts
disagree. The v1 study is reproduced on a CPU in about ten minutes, the v2 study in about one
hour more:

```bash
uv sync --group dev
uv run fraud-ae all --config configs/base.toml --run runs/ae_base   # phases 1–4
scripts/run_ablations.sh                                           # all v1 variants, baselines, comparison
scripts/run_v2_protocol.sh                                         # 6.8: pooled protocol, data seeds, diagnosis (≈ 5 min)
scripts/run_v2_unsupervised.sh                                     # 6.9: legit-only candidates, pre-registered pick (≈ 8 min)
scripts/run_v2_supervised.sh                                       # 6.10: eight labelled methods, few-label curve (≈ 25 min)
scripts/run_v2_robustness.sh                                       # 6.11: variance runs, rolling threshold, evasion (≈ 12 min)
uv run fraud-ae report && scripts/check.sh                         # render this report; format, lint, tests
```

Artifact map: `data/processed*/data_report.json` (Section 2), `runs/<run>/train_summary.json`
(Section 3), `runs/<run>/threshold.json` (Section 4, validation columns),
`runs/<run>/metrics.json` (Sections 5–6), `results/comparison.md` (all 21 v1 runs in one table),
`runs/ae_pooled/diagnostics.json` (Section 6.8), `results/v2_{protocol,unsupervised,supervised,few_label}.md`
and `runs/sup_<method>/{folds.json,few_label.json}` (Sections 6.9–6.10),
`results/v2_robustness.json` (Section 6.11).
The prompt log of the AI-assisted development is in `report/prompt_log.md`; design decisions
are recorded in `ARCHITECTURE.md`.

## 11. References

1. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability
   with undersampling for unbalanced classification. *IEEE Symposium Series on Computational
   Intelligence (SSCI)*, 159–166.
2. ULB Machine Learning Group. *Credit Card Fraud Detection* dataset. Kaggle,
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
3. An, J., & Cho, S. (2015). Variational autoencoder based anomaly detection using
   reconstruction probability. *Special Lecture on IE*, 2(1), 1–18.
4. Sakurada, M., & Yairi, T. (2014). Anomaly detection using autoencoders with nonlinear
   dimensionality reduction. *Proceedings of the MLSDA Workshop*, 4–11.
5. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *IEEE International
   Conference on Data Mining (ICDM)*, 413–422.
6. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing
   Surveys*, 41(3), 1–58.
7. Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
   *Journal of the American Statistical Association*, 22(158), 209–212.
8. PyTorch documentation, https://pytorch.org/docs/ ; scikit-learn documentation,
   https://scikit-learn.org/stable/
9. Ruff, L., Vandermeulen, R. A., Görnitz, N., Binder, A., Müller, E., Müller, K.-R., &
   Kloft, M. (2020). Deep semi-supervised anomaly detection. *International Conference on
   Learning Representations (ICLR)*.
10. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning
    models resistant to adversarial attacks. *International Conference on Learning
    Representations (ICLR)*.
