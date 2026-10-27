"""Print a Markdown table from ``runs/*/metrics.json`` (Phase 4 decision gate; Phase 5 extends
this into ``results/comparison.md``). Usage: ``python scripts/summarize_runs.py runs/ae_*``."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def row(run: Path) -> str | None:
    mp = run / "metrics.json"
    if not mp.exists():
        return None
    m = json.loads(mp.read_text())
    s = (
        json.loads((run / "train_summary.json").read_text())
        if (run / "train_summary.json").exists()
        else {}
    )
    at = m["at_threshold"]
    r1 = next((g for g in m["recall_at_fpr"] if abs(g["fpr_target"] - 0.01) < 1e-12), None)
    lo, hi = at["recall_ci_wilson"]
    return (
        f"| `{run.name}` | {m.get('score_method', '?')} | {s.get('seed', '?')} | "
        f"{s.get('best_val_legit_mse', float('nan')):.3f} | {m['roc_auc']:.4f} | "
        f"{m['pr_auc']:.3f} | "
        f"{at['fpr']:.3%} | **{at['recall']:.1%}** [{lo:.1%}, {hi:.1%}] | "
        f"{(r1['recall'] if r1 else float('nan')):.1%} | {at['precision']:.1%} | "
        f"{'✅' if m['meets_target'] else '❌'} |"
    )


def main(paths: list[str]) -> None:
    print(
        "| Run | Score | Seed | Val-legit MSE | ROC-AUC | PR-AUC | Test FPR | Test recall "
        "[Wilson 95 %] | Recall @ 1 % FPR | Precision | Meets target |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---|---:|---:|:---:|")
    for p in sorted(Path(x) for x in paths):
        line = row(p)
        if line:
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
