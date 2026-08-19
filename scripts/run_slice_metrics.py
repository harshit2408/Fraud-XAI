"""
scripts/run_slice_metrics.py

Phase C5: per-slice validation metrics for the frozen XGBoost baseline,
broken out by ProductCD, hour-of-day bucket, and card-tenure bucket.

Selects the business-value-optimal threshold on the VALIDATION split (same
cost model and method as `src/training/train_xgb.py`) and scores every
slice at that single threshold, so the slice table reflects the same
operating point the model would actually run at — never a
per-slice-optimized threshold, which would hide exactly the kind of
subgroup weakness this table exists to surface.

Output: reports/slice_metrics.csv (one row per slice per slice dimension)
and reports/slice_metrics.md (human-readable tables).

Usage:
    conda run -n fraudx python scripts/run_slice_metrics.py
"""

import logging
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_engineering import FeatureEngineer
from src.evaluation.evaluator import ModelEvaluator
from src.training.train_xgb import XGBTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
TRANSFORMERS_DIR = Path("data/processed/transformers")


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Minimal GFM table renderer (avoids adding a `tabulate` dependency
    for `DataFrame.to_markdown`, which is not in requirements.txt)."""
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda v: f"{v:.4f}")
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body_rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in formatted.itertuples(index=False)
    ]
    return "\n".join([header, separator, *body_rows])


def load_product_cd_classes(transformers_dir: Path) -> List[str]:
    """Read ProductCD's fitted category order via the safe transformer
    loader (Phase D6: joblib + checksum verification, not raw pickle.load)
    rather than a hand-copied constant, so a re-fit encoder (category
    added/dropped) can't silently desync the slice labels from what
    X_val["ProductCD"]'s integer codes actually mean."""
    fe = FeatureEngineer()
    fe.load_transformers(str(transformers_dir))
    return list(fe._label_encoders["ProductCD"].classes_)


def resolve_predict_and_threshold(
    trainer: XGBTrainer,
    evaluator: ModelEvaluator,
    y_val: np.ndarray,
    y_prob_val_fn: Callable[[], np.ndarray],
    thresh_cfg: dict,
) -> Tuple[float, str]:
    """Prefer the threshold frozen into the model artifact (Phase C4); only
    fall back to re-deriving one on validation for pre-C4 artifacts that
    were saved with no `threshold` key, and say so loudly so this script
    doesn't silently diverge from the actual serving operating point once
    C4 artifacts become the norm."""
    frozen_threshold = trainer.threshold
    if frozen_threshold is not None:
        logger.info(f"Using frozen threshold from model artifact: {frozen_threshold:.4f}")
        return frozen_threshold, "frozen (model artifact)"

    logger.warning(
        "Model artifact has no frozen threshold (pre-Phase-C4 artifact) — "
        "recomputing a business-value-optimal threshold on validation. "
        "This threshold may not match whatever is actually deployed."
    )
    threshold = evaluator.find_optimal_threshold(
        y_val, y_prob_val_fn(),
        cost_fn=thresh_cfg.get("cost_fn", 500),
        cost_fp=thresh_cfg.get("cost_fp", 5),
        revenue_tp=thresh_cfg.get("revenue_tp", 0.0),
    )
    return threshold, "recomputed on validation (no frozen threshold in artifact)"


def main() -> None:
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    evaluator = ModelEvaluator()

    logger.info("Loading validation split...")
    X_val = pd.read_parquet(processed_dir / "val_features.parquet")
    y_val = pd.read_parquet(processed_dir / "val_labels.parquet").squeeze().to_numpy()

    # "models/xgb_model.pkl" is a stem, not a literal file (Phase D6):
    # XGBTrainer.load resolves it to xgb_model.ubj/.meta.joblib/.checksums.json.
    logger.info("Loading frozen XGBoost model (models/xgb_model.*)...")
    trainer = XGBTrainer.load("models/xgb_model.pkl")
    y_prob_val = trainer.predict_proba(X_val)

    thresh_cfg = config.get("thresholds", {})
    threshold, threshold_source = resolve_predict_and_threshold(
        trainer, evaluator, y_val, lambda: y_prob_val, thresh_cfg
    )
    logger.info(f"Threshold for slice scoring: {threshold:.4f} ({threshold_source})")

    product_cd_classes = load_product_cd_classes(TRANSFORMERS_DIR)
    slice_dims = {
        "ProductCD": [product_cd_classes[i] for i in X_val["ProductCD"].astype(int)],
        "hour_bucket": evaluator.bucket_hour_of_day(X_val["hour_of_day"].to_numpy()),
        "card_tenure_bucket": evaluator.bucket_card_tenure(X_val["tx_count_per_card"].to_numpy()),
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    all_tables = []
    md_sections = [
        "# Slice Metrics — Validation Split\n",
        f"Threshold: `{threshold:.4f}` ({threshold_source}; cost model: "
        f"cost_fn={thresh_cfg.get('cost_fn', 500)}, "
        f"cost_fp={thresh_cfg.get('cost_fp', 5)}, "
        f"revenue_tp={thresh_cfg.get('revenue_tp', 0.0)})\n",
        f"Model: `models/xgb_model.*` | Rows: {len(X_val)} | "
        f"Fraud rate: {y_val.mean():.4f}\n",
        "Rows flagged `reliable=False` have fewer than 30 validation examples — "
        "treat their metrics as directional, not conclusive.\n",
        "**Reading precision here:** with `revenue_tp` this much larger than `cost_fp` "
        "relative to the ~3.5% fraud base rate, the business-value-optimal policy sits "
        "close to \"flag almost everyone\" — the low precision / high recall seen in "
        "every slice below is an expected consequence of the configured cost model in "
        "`config/config.yaml`, not a defect in slicing. Revisit `thresholds.revenue_tp` "
        "in that config if a higher-precision operating point is wanted.\n",
    ]

    for dim_name, labels in slice_dims.items():
        table = evaluator.compute_slice_metrics(y_val, y_prob_val, labels, threshold=threshold)
        table.insert(0, "dimension", dim_name)
        all_tables.append(table)

        logger.info(f"\n--- {dim_name} ---\n{table.to_string(index=False)}")
        md_sections.append(f"\n## {dim_name}\n")
        md_sections.append(dataframe_to_markdown(table))
        md_sections.append("\n")

    combined = pd.concat(all_tables, ignore_index=True)
    csv_path = REPORTS_DIR / "slice_metrics.csv"
    combined.to_csv(csv_path, index=False)
    logger.info(f"Slice metrics saved to {csv_path}")

    md_path = REPORTS_DIR / "slice_metrics.md"
    md_path.write_text("\n".join(md_sections), encoding="utf-8")
    logger.info(f"Slice metrics report saved to {md_path}")


if __name__ == "__main__":
    main()
