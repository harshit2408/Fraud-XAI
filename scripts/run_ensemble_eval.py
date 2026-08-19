"""
scripts/run_ensemble_eval.py

Evaluates the hybrid ensemble of XGBoost (tabular) and TFT (sequential).
Finds the optimal blending weight on the validation set, then applies it to the test set.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from scipy.optimize import minimize_scalar

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.training.train_xgb import XGBTrainer
from src.training.train_tft import TFTTrainer
from src.evaluation.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def predict_proba_prefer_calibrated(trainer, model_name: str, *args, **kwargs):
    """Prefer each model's calibrated probabilities for ensembling.

    A fixed convex blend (w * xgb + (1-w) * tft) implicitly assumes both
    inputs sit on a comparable probability scale — true for calibrated
    probabilities, not guaranteed for two different model families' raw
    scores. Falls back to raw `predict_proba` (with a loud warning) only
    for artifacts saved before Phase C4 persisted a calibrator, so this
    script degrades gracefully rather than hard-failing on an old artifact.

    Checks `trainer.calibrator is None` directly rather than catching
    `predict_proba_calibrated`'s ValueError by type: that method also
    raises ValueError for unrelated failures (e.g. a genuine feature-schema
    mismatch inside the underlying `predict_proba` call), which a bare
    `except ValueError` would misdiagnose as "no calibrator" and mask the
    real cause behind a misleading warning.
    """
    if trainer.calibrator is None:
        logger.warning(
            f"{model_name} has no frozen calibrator (pre-Phase-C4 artifact) — "
            f"falling back to RAW predict_proba for ensembling. Blending raw "
            f"scores from different model families is not statistically "
            f"principled; re-train {model_name} to pick up calibration before "
            f"trusting these ensemble numbers."
        )
        return trainer.predict_proba(*args, **kwargs)
    return trainer.predict_proba_calibrated(*args, **kwargs)


def evaluate_ensemble():
    config = load_settings("config/config.yaml").model_dump()
    processed_dir = Path(config["data"]["processed_dir"])
    evaluator = ModelEvaluator()

    logger.info("Loading parquet data...")
    X_train = pd.read_parquet(processed_dir / "train_features.parquet")
    X_val = pd.read_parquet(processed_dir / "val_features.parquet")
    y_val = pd.read_parquet(processed_dir / "val_labels.parquet").squeeze()
    X_test = pd.read_parquet(processed_dir / "test_features.parquet")
    y_test = pd.read_parquet(processed_dir / "test_labels.parquet").squeeze()

    xgb_path = config.get("serving", {}).get("model_path", "models/xgb_model.pkl")
    tft_path = config.get("serving", {}).get("tft_model_path", "models/tft_model.pt")

    logger.info(f"Loading XGBoost model from {xgb_path}...")
    xgb_trainer = XGBTrainer.load(xgb_path)

    logger.info(f"Loading TFT model from {tft_path}...")
    tft_trainer = TFTTrainer.load(tft_path, config=config)

    logger.info("Predicting Validation set...")
    # Phase C4 follow-up: blend calibrated probabilities, not raw scores —
    # see predict_proba_prefer_calibrated's docstring.
    xgb_prob_val = predict_proba_prefer_calibrated(xgb_trainer, "XGBoost", X_val)
    # Phase B7: no labels. Phase B6: train supplies real history for val's boundary.
    tft_prob_val = predict_proba_prefer_calibrated(tft_trainer, "TFT", X_val, history_X=X_train)

    # Find optimal ensemble weight 'w' (xgb_weight) to maximize Val PR-AUC
    logger.info("Optimizing ensemble weight on Validation set...")
    def objective(w):
        # w is xgb weight, (1-w) is tft weight
        ens_prob = w * xgb_prob_val + (1 - w) * tft_prob_val
        # negate because minimize_scalar minimizes
        return -evaluator.compute_pr_auc(y_val.values, ens_prob)

    res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    optimal_w = res.x
    best_val_pr_auc = -res.fun

    logger.info(f"Optimal XGBoost Weight: {optimal_w:.4f} (TFT Weight: {1-optimal_w:.4f})")
    logger.info(f"Ensemble Val PR-AUC:    {best_val_pr_auc:.4f}")

    # Baseline comparison on validation
    xgb_val_pr_auc = evaluator.compute_pr_auc(y_val.values, xgb_prob_val)
    tft_val_pr_auc = evaluator.compute_pr_auc(y_val.values, tft_prob_val)
    logger.info(f"Standalone XGB Val:     {xgb_val_pr_auc:.4f}")
    logger.info(f"Standalone TFT Val:     {tft_val_pr_auc:.4f}")

    logger.info("\nPredicting Test set...")
    xgb_prob_test = predict_proba_prefer_calibrated(xgb_trainer, "XGBoost", X_test)
    tft_prob_test = predict_proba_prefer_calibrated(
        tft_trainer, "TFT", X_test,
        history_X=pd.concat([X_train, X_val], axis=0, ignore_index=True),
    )

    ens_prob_test = optimal_w * xgb_prob_test + (1 - optimal_w) * tft_prob_test
    ens_prob_val = optimal_w * xgb_prob_val + (1 - optimal_w) * tft_prob_val

    # Baseline comparison on test
    xgb_test_pr_auc = evaluator.compute_pr_auc(y_test.values, xgb_prob_test)
    tft_test_pr_auc = evaluator.compute_pr_auc(y_test.values, tft_prob_test)
    ens_test_pr_auc = evaluator.compute_pr_auc(y_test.values, ens_prob_test)
    ens_test_roc_auc = evaluator.compute_roc_auc(y_test.values, ens_prob_test)

    logger.info("=" * 60)
    logger.info("FINAL ENSEMBLE TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"  XGBoost Only PR-AUC: {xgb_test_pr_auc:.4f}")
    logger.info(f"  TFT Only PR-AUC:     {tft_test_pr_auc:.4f}")
    logger.info(f"  Ensemble PR-AUC:     {ens_test_pr_auc:.4f}")
    logger.info(f"  Ensemble ROC-AUC:    {ens_test_roc_auc:.4f}")
    logger.info("=" * 60)

    # Phase C1: cost-based threshold is selected on the VALIDATION ensemble
    # probabilities and frozen — test metrics are reported at that
    # threshold, never re-derived from y_test.
    thresh_cfg = config.get("thresholds", {})
    cost_fn = thresh_cfg.get("cost_fn", 500)
    cost_fp = thresh_cfg.get("cost_fp", 5)
    revenue_tp = thresh_cfg.get("revenue_tp", 480)

    optimal_t = evaluator.find_optimal_threshold(y_val.values, ens_prob_val, cost_fn, cost_fp, revenue_tp)
    metrics = evaluator.compute_metrics_at_threshold(y_test.values, ens_prob_test, optimal_t)
    
    logger.info(f"Optimal Threshold: {optimal_t:.4f}")
    y_pred = (ens_prob_test >= optimal_t).astype(int)
    logger.info("\n" + evaluator.generate_classification_report(y_test.values, y_pred))

if __name__ == "__main__":
    evaluate_ensemble()
