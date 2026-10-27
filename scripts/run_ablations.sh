#!/usr/bin/env bash
# Phase 4 decision-gate + Phase 5 ablation runs (ROADMAP.md): one run directory per experiment,
# each produced by the plain CLI (train → threshold → evaluate), then the baselines and the
# comparison table. Usage: scripts/run_ablations.sh   (≈ 40 s per AE run on CPU, ≈ 10 min total)
set -euo pipefail
cd "$(dirname "$0")/.."

run() {  # run <run-name> <config> [extra train args...]
  local name=$1 cfg=$2; shift 2
  echo "=== $name  ($cfg $*)"
  uv run fraud-ae train     --config "$cfg" --run "runs/$name" "$@"
  uv run fraud-ae threshold --run "runs/$name"
  uv run fraud-ae evaluate  --run "runs/$name"
}

NAMES=()
run_all() {  # run_all <run-name> <config> [extra train args...]
  NAMES+=("runs/$1"); run "$@"
}

uv run fraud-ae prepare --config configs/ablation_no_time.toml    # 29-feature splits → data/processed_no_time
uv run fraud-ae prepare --config configs/ablation_temporal.toml   # chronological split → data/processed_temporal
run_all ae_base                configs/base.toml
run_all ae_score_standardized  configs/ablation_score_standardized.toml
run_all ae_score_mahalanobis   configs/ablation_score_mahalanobis.toml
run_all ae_latent_3            configs/ablation_latent_3.toml
run_all ae_latent_8            configs/ablation_latent_8.toml      # latent 5 == base
run_all ae_no_time             configs/ablation_no_time.toml
run_all ae_batchnorm           configs/ablation_batchnorm.toml
run_all ae_temporal            configs/ablation_temporal.toml
run_all ae_epochs_200          configs/base.toml --epochs 200
for s in 0 1 2 3; do                                               # + seed 42 == base → 5 seeds
  run_all "ae_base_seed$s"     configs/base.toml --seed "$s"
done

# Phase 5: baselines on the random split and on the temporal split, then the comparison
uv run fraud-ae baselines --config configs/base.toml
uv run fraud-ae baselines --config configs/ablation_temporal.toml
uv run fraud-ae compare --out results/comparison.md --figures-dir report/figures
