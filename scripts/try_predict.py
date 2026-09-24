"""
scripts/try_predict.py

Quick manual smoke of the built serving path — NO Docker, NO Kafka, NO uvicorn.

Builds the same `InferenceService` the FastAPI lifespan and the Kafka consumer
use, reads a few RAW rows from `data/raw/test_transaction.csv`, scores each one,
and prints the decision plus the top SHAP factors.

    conda run -n fraudx python scripts/try_predict.py --n 5
    conda run -n fraudx python scripts/try_predict.py --n 20 --skip 100000

Note: until `python src/data/preprocess.py` has been re-run to persist the
`null_count_cols` / `card_agg_state` the current on-disk transformers predate,
this script pads each raw row to the full V1..V339 schema itself so the PCA
step accepts it. A real Kafka/HTTP client does not need to — the serving
transform does that once the re-run lands. Cold-start card aggregates
(`tx_count_per_card=0` etc.) are expected here for the same reason.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import build_inference_service
from src.data.feature_engineering import V_FEATURE_COLS
from src.streaming.producer import _json_safe


def main() -> int:
    parser = argparse.ArgumentParser(description="Score raw test transactions locally.")
    parser.add_argument("--n", type=int, default=5, help="How many rows to score.")
    parser.add_argument("--skip", type=int, default=0, help="Rows to skip first.")
    args = parser.parse_args()

    raw_path = PROJECT_ROOT / "data" / "raw" / "test_transaction.csv"
    if not raw_path.exists():
        print(f"Missing {raw_path}", file=sys.stderr)
        return 1

    rows = pd.read_csv(
        raw_path,
        skiprows=range(1, args.skip + 1) if args.skip else None,
        nrows=args.n,
    )
    # Interim pad (see module docstring): ensure every V-column is present so
    # the fitted PCA's n_features check passes against pre-re-run artifacts.
    for col in V_FEATURE_COLS:
        if col not in rows.columns:
            rows[col] = float("nan")

    print("Loading the ensemble + SHAP explainer (a few seconds)...")
    service = build_inference_service()
    print(f"model_version = {service.models.model_version}")
    print(f"models        = {sorted(service.models.trainers)}\n")

    # Interim pad (see module docstring): the current transformers predate the
    # persisted raw-schema list, so `_restore_raw_schema` is a no-op and the
    # transform will not re-add raw passthrough columns (dist2, D6-D14, M4…)
    # the model consumes directly. Add them as NaN here; imputation fills them
    # the same way batch did. A preprocessing re-run removes the need for this.
    raw_passthrough = [
        c
        for c in service.models.feature_names
        if not c.startswith(("pca_v_", "amount_", "hour_", "day_", "tx_", "time_"))
    ]

    fraud = 0
    for record in rows.to_dict(orient="records"):
        txn = _json_safe(record)
        txn["TransactionID"] = str(txn["TransactionID"])
        for col in raw_passthrough:
            txn.setdefault(col, None)
        result = service.predict(txn)
        fraud += result.decision == "FRAUD"
        print(
            f"txn {txn['TransactionID']:>9}  amt={txn.get('TransactionAmt'):>10.2f}  "
            f"p_fraud={result.fraud_probability:.4f}  -> {result.decision}"
            f"  (threshold={result.threshold:.4f}, mode={result.mode}, "
            f"{result.latency_ms:.0f} ms)"
        )
        for factor in result.explanation[:3]:
            print(f"      {factor['feature']:<24} {factor['contribution']:+.4f}")

    print(
        f"\n{fraud}/{len(rows)} flagged FRAUD at the frozen cost-optimal threshold "
        "(it is deliberately low — see RESULTS.md §6)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
