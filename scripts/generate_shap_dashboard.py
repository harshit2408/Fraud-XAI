"""
scripts/generate_shap_dashboard.py

Build the standalone SHAP explainability dashboard (PRD Phase 4, task P4-5).

Wires the real artifacts to `src/explainability/dashboard_generator.py`:

  1. Load the frozen XGBoost model (`models/xgb_model.*`, checksum-verified).
  2. Read its blend weight from `models/ensemble.json` so the page can state
     honestly how much of the ensemble decision it explains (ADR-001 §3.4).
  3. Take a sample of engineered rows from an already-processed feature
     parquet (`data/processed/test_features.parquet` by default — a training
     output, already column-aligned to the model).
  4. Render one self-contained HTML file to `monitoring/shap_dashboard.html`.

The processed parquet is used rather than the raw IEEE-CIS CSV on purpose:
the dashboard explains the exact engineered feature vector the model scores,
which is what `data/processed/*_features.parquet` already is. Running feature
engineering here would just risk drifting from the pipeline that produced the
model.

Usage:
    conda run -n fraudx python scripts/generate_shap_dashboard.py
    conda run -n fraudx python scripts/generate_shap_dashboard.py \
        --features data/processed/val_features.parquet --sample 3000 \
        --out monitoring/shap_dashboard.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.dashboard_generator import (
    DEFAULT_MAX_FEATURES,
    generate_dashboard,
)
from src.explainability.shap_explainer import FraudExplainer
from src.training.train_xgb import XGBTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_FEATURES = "data/processed/test_features.parquet"
DEFAULT_ENSEMBLE_SPEC = "models/ensemble.json"
DEFAULT_MODEL = "models/xgb_model.pkl"  # stem; resolved to .ubj/.meta.joblib
DEFAULT_OUT = "monitoring/shap_dashboard.html"
DEFAULT_SAMPLE = 5_000


def _xgb_blend_weight(spec_path: Path) -> Optional[float]:
    """The XGBoost weight in the default blend, for the "explains N% of the
    decision" line. Read directly from the JSON rather than via `EnsembleSpec`
    to keep this script's dependency surface small; a missing file or key is
    non-fatal — the dashboard just omits the percentage."""
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        mode = payload["modes"][payload["default_mode"]]
        return float(mode["weights"]["xgb"])
    except (OSError, KeyError, ValueError) as exc:
        logger.warning(
            "Could not read the XGBoost blend weight from %s (%s). The "
            "dashboard will render without the 'explains N%% of the decision' "
            "figure.",
            spec_path,
            exc,
        )
        return None


# Above this many rows the SHAP matrix + feature frame get large (~160 MB each
# at 118k x 171 float64) and the render slows without adding signal. `--sample 0`
# ("all rows") is still honoured — this only warns.
SAMPLE_WARN_THRESHOLD = 50_000


def _load_sample(features_path: Path, sample: int, seed: int) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(features_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"Could not read the feature parquet {features_path}: {exc}"
        ) from exc
    if sample and len(frame) > sample:
        frame = frame.sample(n=sample, random_state=seed).reset_index(drop=True)
    if len(frame) > SAMPLE_WARN_THRESHOLD:
        logger.warning(
            "Explaining %d rows — TreeSHAP over this many holds two large "
            "float64 arrays in memory and slows the render. Pass --sample to "
            "cap it.",
            len(frame),
        )
    logger.info("Loaded %d engineered rows from %s", len(frame), features_path)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the standalone SHAP explainability dashboard."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="XGBoost artifact stem.")
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        help="Processed feature parquet (already engineered, model-aligned).",
    )
    parser.add_argument(
        "--ensemble-spec",
        default=DEFAULT_ENSEMBLE_SPEC,
        help="models/ensemble.json — read only for the XGBoost blend weight.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output HTML path.")
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help="Rows to explain (sampled without replacement). 0 = all rows.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help="Features shown in the bars / beeswarm / table.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    args = parser.parse_args()

    features_path = Path(args.features)
    if not features_path.exists():
        logger.error(
            "No processed feature parquet at %s. Run `python src/data/preprocess.py` "
            "to produce the data/processed/*_features.parquet outputs first.",
            features_path,
        )
        return 1

    logger.info("Loading frozen XGBoost model (%s)...", args.model)
    trainer = XGBTrainer.load(args.model)
    if trainer.model is None or trainer.feature_names is None:
        logger.error("Loaded XGBoost artifact has no fitted booster / feature names.")
        return 1

    xgb_weight = _xgb_blend_weight(Path(args.ensemble_spec))
    explainer = FraudExplainer(
        trainer.model,
        feature_names=trainer.feature_names,
        top_k=5,
        explained_weight=xgb_weight,
    )

    sample = _load_sample(features_path, args.sample, args.seed)

    out = generate_dashboard(
        explainer, sample, args.out, max_features=args.max_features
    )
    logger.info("SHAP dashboard: %s", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
