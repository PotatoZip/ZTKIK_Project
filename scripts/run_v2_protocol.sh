#!/usr/bin/env bash
# Roadmap v2 Phase 7 (ROADMAP.md): pooled-fraud protocol, repeated legitimate splits, the
# literature-protocol row, pooled baselines, the missed-fraud diagnosis and the protocol table.
# Nothing here touches runs/ae_base, data/processed or the v1 comparison / report.
# Usage: scripts/run_v2_protocol.sh   (≈ 5 min on CPU: 6 × prepare, 6 × AE training)
set -euo pipefail
cd "$(dirname "$0")/.."

run() {  # run <run-name> <config> [extra train args...]
  local name=$1 cfg=$2; shift 2
  echo "=== $name  ($cfg $*)"
  uv run fraud-ae train     --config "$cfg" --run "runs/$name" "$@"
  uv run fraud-ae threshold --run "runs/$name"
  uv run fraud-ae evaluate  --run "runs/$name"
}

POOLED=configs/v2_pooled.toml
LIT=configs/v2_literature.toml

# 1. splits: pooled (data seed 42 from the config + four re-drawn legit splits) and literature
uv run fraud-ae prepare --config $POOLED --figures-dir data/processed_pooled/figures
for s in 0 1 2 3; do uv run fraud-ae prepare --config $POOLED --data-seed "$s"; done
uv run fraud-ae prepare --config $LIT --figures-dir data/processed_literature/figures

# 2. the base AE under the pooled protocol (same legit split as ae_base → identical checkpoint)
run ae_pooled $POOLED
for s in 0 1 2 3; do run "ae_pooled_dseed$s" $POOLED --data-seed "$s"; done
run ae_literature $LIT

# 3. pooled baselines (explicit suffix so the v1 runs/baseline_* stay untouched)
uv run fraud-ae baselines --config $POOLED --suffix _pooled

# 4. diagnosis of the pooled base run against the baselines on the same test split
#    (a --data-seed run has a different legit test split and is rejected by `diagnose`)
uv run fraud-ae diagnose --run runs/ae_pooled \
  --against runs/baseline_pca_k15_pooled runs/baseline_pca_k5_pooled runs/baseline_iforest_pooled
cp runs/ae_pooled/figures/missed_frauds.png report/figures/missed_frauds_ae_pooled.png

# 5. protocol table: v1 base run next to the pooled runs, the literature row and pooled baselines
uv run fraud-ae compare \
  --runs runs/ae_base runs/ae_pooled runs/ae_pooled_dseed0 runs/ae_pooled_dseed1 \
         runs/ae_pooled_dseed2 runs/ae_pooled_dseed3 runs/ae_literature \
         runs/baseline_pca_k5_pooled runs/baseline_pca_k10_pooled runs/baseline_pca_k15_pooled \
         runs/baseline_iforest_pooled \
  --out results/v2_protocol.md --figures-dir report/figures --figure-prefix v2_protocol_
