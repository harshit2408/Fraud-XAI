"""
scripts/run_phase9_eval.py

Thin CLI over `src.evaluation.phase9_report` — the fixed evaluation battery PRD
Phase 9 re-runs after every step.

Reads `reports/ensemble_test_probabilities.npz` (produced by
`scripts/export_test_probabilities.py` against whichever model is currently
frozen in `models/ensemble.json`) and prints the standard Phase 9 row:
PR-AUC / ROC-AUC, precision_at_recall at 70/80/90%, confusion matrices at the
deployed and (optionally) a re-derived threshold, business-impact both
conventions, and the target-band verdict.

Usage:
    python scripts/run_phase9_eval.py --step "9.0 baseline"
    python scripts/run_phase9_eval.py --step "9.1 re-derived" --rederived-threshold 0.0102
    python scripts/run_phase9_eval.py --step "9.0 baseline" --json reports/phase9_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings  # noqa: E402
from src.evaluation.business_impact import CostModel  # noqa: E402
from src.evaluation.phase9_report import build_report, format_report  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_NPZ = PROJECT_ROOT / "reports" / "ensemble_test_probabilities.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9 evaluation battery.")
    parser.add_argument("--step", required=True, help="Label for this run.")
    parser.add_argument(
        "--probabilities",
        default=str(DEFAULT_NPZ),
        help="Path to the ensemble test-probabilities .npz.",
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--rederived-threshold",
        type=float,
        default=None,
        help="Second threshold to also report a confusion matrix at (9.1+). "
        "Ignored if --derive-threshold-from is given.",
    )
    parser.add_argument(
        "--derive-threshold-from",
        default=None,
        help="Path to a VALIDATION-split probabilities .npz. When given, the "
        "re-derived threshold is computed here (argmax of net annual value "
        "under --convention) instead of being passed as a magic constant.",
    )
    parser.add_argument(
        "--convention",
        default="net_of_principal",
        choices=["prd", "net_of_principal"],
        help="Cost-model convention for --derive-threshold-from "
        "(ADR-001 §7: net_of_principal is the resolved default).",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional path to also write the full report as JSON.",
    )
    args = parser.parse_args()

    npz_path = Path(args.probabilities)
    if not npz_path.exists():
        raise SystemExit(
            f"{npz_path} not found — run scripts/export_test_probabilities.py "
            "against the currently frozen model first."
        )

    data = np.load(npz_path)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    deployed_threshold = float(data["threshold"])

    # Stamp the report with the frozen ensemble it describes, so a stale .npz
    # (scored against an older models/ensemble.json) is visible in the output.
    spec_path = PROJECT_ROOT / "models" / "ensemble.json"
    spec_stamp = None
    if spec_path.exists():
        spec_json = json.loads(spec_path.read_text(encoding="utf-8"))
        run_id = spec_json.get("mlflow_run_id", "unknown-run")
        mtime = spec_path.stat().st_mtime
        spec_stamp = f"ensemble.json run={run_id} mtime={mtime:.0f}"

    config = load_settings(args.config).model_dump()
    cost_model = CostModel.from_config(config)
    logger.info(
        "Cost model: cost_fn=%s cost_fp=%s revenue_tp=%s (implied fraud swing $%s)",
        cost_model.cost_fn,
        cost_model.cost_fp,
        cost_model.revenue_tp,
        cost_model.implied_fraud_swing,
    )

    rederived = args.rederived_threshold
    if args.derive_threshold_from:
        from src.evaluation.business_impact import (
            CONVENTION_NET,
            CONVENTION_PRD,
            best_threshold,
            threshold_sweep,
        )

        val_path = Path(args.derive_threshold_from)
        if not val_path.exists():
            raise SystemExit(f"{val_path} not found.")
        val = np.load(val_path)
        conv = CONVENTION_NET if args.convention == "net_of_principal" else CONVENTION_PRD
        sweep = threshold_sweep(
            val["y_true"], val["y_prob"], cost_model, convention=conv
        )
        rederived = float(best_threshold(sweep).threshold)
        logger.info(
            "Re-derived threshold on %s (%s convention): %.6f",
            val_path.name,
            args.convention,
            rederived,
        )

    report = build_report(
        y_true=y_true,
        y_prob=y_prob,
        deployed_threshold=deployed_threshold,
        cost_model=cost_model,
        step=args.step,
        rederived_threshold=rederived,
        ensemble_spec_stamp=spec_stamp,
    )

    print(format_report(report))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
