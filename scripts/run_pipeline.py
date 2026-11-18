"""Run the full ScamShield AZ pipeline: data -> features -> models -> decisions -> evaluation.

Usage:
    python scripts/run_pipeline.py                  # full run incl. sensitivity analysis
    python scripts/run_pipeline.py --skip-sensitivity
    python scripts/run_pipeline.py --figures-only     # re-draw figures from saved results
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from scamshield.config import ARTIFACTS_DIR, load_intervention_yaml, load_model_config
from scamshield.pipeline import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--intervention-config", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--skip-sensitivity", action="store_true", help="skip the (slower) sensitivity analyses")
    parser.add_argument("--figures-only", action="store_true", help="re-draw figures from saved metrics/predictions")
    args = parser.parse_args()

    if args.figures_only:
        import pandas as pd

        from scamshield.decision.optimizer import load_intervention_config
        from scamshield.evaluation.plots import make_all_figures

        metrics = json.loads((args.artifacts_dir / "metrics.json").read_text())
        preds = pd.read_parquet(args.artifacts_dir / "predictions_test.parquet")
        make_all_figures(metrics, preds, load_intervention_config(args.intervention_config), args.artifacts_dir / "figures")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    metrics = run_full_pipeline(
        load_model_config(args.model_config),
        load_intervention_yaml(args.intervention_config),
        args.artifacts_dir,
        with_sensitivity=not args.skip_sensitivity,
    )

    m, e2e = metrics["models"], metrics["end_to_end"]
    summary = {
        "test_transactions": metrics["data"]["test"],
        "pr_auc": {"baseline": round(m["baseline_lr"]["pr_auc"], 3), "scamshield": round(m["scamshield"]["pr_auc"], 3)},
        "total_cost_AZN": {k: round(e2e[k]["total_cost"]) for k in ("baseline", "scamshield")},
        "cost_reduction_pct": round(100 * e2e["savings"]["cost_reduction_pct"], 1),
        "runtime_s": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
