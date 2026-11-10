#!/usr/bin/env bash
# Roadmap v2 Phase 10 (ROADMAP.md): robustness of the Phase 9 winners (hgb, neg_ae).
# 1. model-seed variance (train.seed 0–3; seed 42 = the Phase 9 runs), 2. data-seed variance
# (legit splits 0–4 with their own legit-only AE), 3. rolling-origin validation with a rolling
# threshold on the temporal split + the evasion probe on the legit-only AE.
# Usage: scripts/run_v2_robustness.sh   (≈ 12 min on CPU)
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=configs/v2_supervised.toml
POOLED=configs/v2_pooled.toml
run_ae() {  # run_ae <run-name> <config> [extra train args...]
  local name=$1 cfg=$2; shift 2
  uv run fraud-ae train     --config "$cfg" --run "runs/$name" "$@"
  uv run fraud-ae threshold --run "runs/$name"
  uv run fraud-ae evaluate  --run "runs/$name"
}

# 0. prerequisites from earlier phases
[ -f runs/ae_pooled/metrics.json ] || uv run fraud-ae all --config $POOLED --run runs/ae_pooled
[ -f runs/sup_hgb/metrics.json ] && [ -f runs/sup_neg_ae/metrics.json ] \
  || uv run fraud-ae supervised --config $CFG --methods hgb neg_ae
[ -f data/processed_temporal/splits.npz ] || uv run fraud-ae prepare --config configs/ablation_temporal.toml
[ -f runs/ae_temporal/metrics.json ] || run_ae ae_temporal configs/ablation_temporal.toml

# 1. model-seed variance on the base pooled split (all labels only)
for s in 0 1 2 3; do
  echo "=== model seed $s"
  uv run fraud-ae supervised --config $CFG --methods hgb neg_ae --budgets --seed "$s" --suffix "_seed$s"
done

# 2. data-seed variance: re-drawn legit splits with their own legit-only AE (Phases 7–8)
for s in 0 1 2 3 4; do
  echo "=== data seed $s"
  [ -f "data/processed_pooled_dseed$s/splits.npz" ] || uv run fraud-ae prepare --config $POOLED --data-seed "$s"
  [ -f "runs/ae_pooled_dseed$s/metrics.json" ] || run_ae "ae_pooled_dseed$s" $POOLED --data-seed "$s"
  uv run fraud-ae supervised --config $CFG --methods hgb neg_ae --budgets --data-seed "$s" \
    --ae-run "runs/ae_pooled_dseed$s"
done

# 3. rolling threshold, variance tables, evasion probe → results/v2_robustness.{md,json}
uv run fraud-ae robustness --config $CFG --out results/v2_robustness.md --figures-dir report/figures
