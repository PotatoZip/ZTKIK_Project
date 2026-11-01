#!/usr/bin/env bash
# Roadmap v2 Phase 9 (ROADMAP.md): labelled frauds inside the model, evaluated out-of-fold on the
# pooled split (configs/v2_supervised.toml): fully supervised classifiers, hybrid stacking on the
# legit-only AE's residual, and the fine-tuned autoencoders, each with the few-label curve
# (25 / 50 / 100 / all labelled frauds per training fold). Needs runs/ae_pooled (Phase 7).
# Usage: scripts/run_v2_supervised.sh   (≈ 30 min on CPU: 8 methods × 4 budgets × 5 folds)
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=configs/v2_supervised.toml
if [ ! -f runs/ae_pooled/metrics.json ]; then
  uv run fraud-ae all --config configs/v2_pooled.toml --run runs/ae_pooled
fi

# 1. every method of the config at every budget → runs/sup_<method>/{metrics,few_label,folds}.json
uv run fraud-ae supervised --config $CFG

# 2. few-label table + figure, legit-only AE as the reference line
RUNS=()
for m in logreg hgb hgb_under mlp hybrid_logreg hybrid_hgb neg_ae sad_ae; do RUNS+=("runs/sup_$m"); done
uv run fraud-ae few-label --runs "${RUNS[@]}" --out results/v2_few_label.md \
  --figure report/figures/v2_few_label.png --reference runs/ae_pooled

# 3. side-by-side table: legit-only AE, supervised, hybrid and fine-tuned AE rows (all labels)
uv run fraud-ae compare --runs runs/ae_pooled "${RUNS[@]}" \
  --out results/v2_supervised.md --figures-dir report/figures --figure-prefix v2_sup_
