#!/usr/bin/env bash
# Roadmap v2 Phase 8 (ROADMAP.md): the unsupervised ceiling under the pooled protocol.
# Candidates (legit-only training, one config each): more training data (80/10/10), VAE with
# reconstruction probability, latent-density scores (GMM k=4 / k=8 and latent NLL alone), and
# hour-segmented thresholds. Pre-registered selection: every candidate is run on data seed 0;
# the candidate with the highest pooled recall among those with FPR < 1 % is confirmed on data
# seeds 1–4 next to the base AE. Nothing here touches the v1 runs or the Phase 7 runs.
# Usage: scripts/run_v2_unsupervised.sh   (≈ 8 min on CPU)
set -euo pipefail
cd "$(dirname "$0")/.."

run() {  # run <run-name> <config> [extra train args...]
  local name=$1 cfg=$2; shift 2
  echo "=== $name  ($cfg $*)"
  uv run fraud-ae train     --config "$cfg" --run "runs/$name" "$@"
  uv run fraud-ae threshold --run "runs/$name"
  uv run fraud-ae evaluate  --run "runs/$name"
}
ensure() {  # like run, but skip a run directory that already has metrics.json (deterministic anyway)
  if [ -f "runs/$1/metrics.json" ]; then echo "=== $1 (exists, kept)"; else run "$@"; fi
}
split() {  # split <config> <data-seed>: prepare once
  local dir; dir=$(uv run python -c "import tomllib,sys;print(tomllib.load(open('$1','rb'))['data']['processed_dir'])")
  [ -f "${dir}_dseed$2/splits.npz" ] || uv run fraud-ae prepare --config "$1" --data-seed "$2"
}

POOLED=configs/v2_pooled.toml
VARIANTS="train80 vae latent_gmm4 latent_gmm8 latent_nll by_hour"
SEL=0
CONF="1 2 3 4"

# 1. selection split: every candidate and the base AE on data seed 0
split $POOLED $SEL
split configs/v2_unsup_train80.toml $SEL
ensure "ae_pooled_dseed$SEL" $POOLED --data-seed $SEL
for v in $VARIANTS; do run "ae_${v}_dseed$SEL" "configs/v2_unsup_$v.toml" --data-seed $SEL; done

# 2. pre-registered selection rule (no human in the loop): highest pooled recall with FPR < 1 %
WINNER=$(uv run python - "$VARIANTS" "$SEL" <<'PY'
import json, sys
rows = []
for v in sys.argv[1].split():
    m = json.load(open(f"runs/ae_{v}_dseed{sys.argv[2]}/metrics.json"))
    rows.append((m["criterion"]["meets_fpr"], m["at_threshold"]["recall"], -m["at_threshold"]["fpr"], v))
ok = [r for r in rows if r[0]] or rows
print(max(ok)[3])
PY
)
echo "selected on data seed $SEL: $WINNER"

# 3. confirmation on data seeds 1–4 (base AE on the same splits for reference)
RUNS=()
for s in $CONF; do
  split $POOLED "$s"
  ensure "ae_pooled_dseed$s" $POOLED --data-seed "$s"
  [ "$WINNER" = train80 ] && split configs/v2_unsup_train80.toml "$s"
  run "ae_${WINNER}_dseed$s" "configs/v2_unsup_$WINNER.toml" --data-seed "$s"
  RUNS+=("runs/ae_pooled_dseed$s" "runs/ae_${WINNER}_dseed$s")
done

# 4. ceiling table: base + every candidate on seed 0, winner + base on the confirmation seeds
for v in $VARIANTS; do RUNS+=("runs/ae_${v}_dseed$SEL"); done
uv run fraud-ae compare --runs "runs/ae_pooled_dseed$SEL" "${RUNS[@]}" \
  --out results/v2_unsupervised.md --figures-dir report/figures --figure-prefix v2_unsup_
{
  echo
  echo "## Pre-registered selection (Phase 8)"
  echo
  echo "Candidates on data seed $SEL: \`${VARIANTS// /\`, \`}\` (configs \`configs/v2_unsup_*.toml\`), reference \`ae_pooled_dseed$SEL\`."
  echo "Selected: **\`$WINNER\`** – highest pooled recall among candidates with FPR < 1 % on data seed $SEL – and confirmed on data seeds ${CONF// /, } (family \`ae_$WINNER\` in the seed-variance table) next to the base AE (family \`ae_pooled\`)."
} >> results/v2_unsupervised.md
echo "done: results/v2_unsupervised.md (selected: $WINNER)"
