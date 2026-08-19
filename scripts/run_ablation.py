"""
scripts/run_ablation.py

Phase 2 imbalance-handling ablation — compares three strategies for the
~3.5% fraud class imbalance on the XGBoost baseline:

  1. `scale_pos_weight` only (natively supported by XGBoost)
  2. SMOTE (Synthetic Minority Oversampling Technique) only
  3. SMOTE + `scale_pos_weight`

Phase C6: all three arms are trained on X_train and scored on X_val — the
winning strategy is selected from validation PR-AUC, never from test. Test
stays completely untouched by this script; it exists only for the final,
frozen report generated elsewhere (see reports/RESULTS.md /
reports/phase2_results.json). This also replaces the near-duplicate
notebooks/02_imbalance_ablation.ipynb, which selected its winner on test —
that notebook has been deleted so this script is the single source of truth
for this experiment.

Usage:
    conda run -n fraudx python scripts/run_ablation.py
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, precision_recall_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")


def compute_optimal_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Best F1 over the PR curve's own threshold grid, and the threshold
    that achieves it. Used here only as a same-split diagnostic of each
    arm's shape, not as a frozen operating threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    best_idx = np.argmax(f1_scores)
    # sklearn's precision_recall_curve returns one fewer threshold than
    # precision/recall points (the last point is recall=0, precision=1 with
    # no corresponding threshold). If the best F1 lands on that last point,
    # there's no real threshold to report; 0.5 is a placeholder for this
    # diagnostic-only value, never used to select the ablation winner or as
    # a frozen operating threshold (see docstring above).
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(f1_scores[best_idx]), float(best_threshold)


def train_and_eval(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
) -> dict:
    """Train one XGBoost arm and score it on the VALIDATION split.

    Both `X_train` and `X_val` are always passed as DataFrames (with the
    original feature names) so every arm gets identical feature-name
    handling — the previous version passed `.values` for the SMOTE arms and
    a DataFrame for the baseline, which is why this signature is typed
    strictly rather than accepting `np.ndarray`.
    """
    logger.info(f"\n--- Running {name} ---")
    start = time.time()

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_prob_val = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_prob_val)
    f1, f1_threshold = compute_optimal_f1(y_val, y_prob_val)

    logger.info(
        f"Time: {train_time:.2f}s | Val PR-AUC: {pr_auc:.4f} | "
        f"Val Best F1: {f1:.4f} @ {f1_threshold:.4f}"
    )
    return {
        "time_s": train_time,
        "val_pr_auc": pr_auc,
        "val_best_f1": f1,
        "val_f1_threshold": f1_threshold,
    }


def main() -> None:
    logger.info("Loading data (train + validation only — test stays untouched)...")
    X_train = pd.read_parquet("data/processed/train_features.parquet")
    y_train = pd.read_parquet("data/processed/train_labels.parquet").squeeze()
    X_val = pd.read_parquet("data/processed/val_features.parquet")
    y_val = pd.read_parquet("data/processed/val_labels.parquet").squeeze()

    # SMOTE requires numeric data. Drop object columns that slipped through.
    obj_cols = X_train.select_dtypes(include=["object"]).columns
    X_train = X_train.drop(columns=obj_cols)
    X_val = X_val.drop(columns=obj_cols)
    logger.info(f"Dropped {len(obj_cols)} object columns. Remaining: {X_train.shape[1]}")

    # Fill NaN for SMOTE (XGBoost tolerates NaN natively, but SMOTE's
    # nearest-neighbor search does not).
    logger.info("Filling NaNs with -999 for SMOTE...")
    X_train = X_train.fillna(-999)
    X_val = X_val.fillna(-999)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count
    logger.info(f"scale_pos_weight (train class ratio) = {scale_weight:.2f}")

    results = {}

    # 1. scale_pos_weight only
    results["scale_pos_weight"] = train_and_eval(
        "scale_pos_weight only", X_train, y_train, X_val, y_val,
        scale_pos_weight=scale_weight,
    )

    # 2. SMOTE only
    logger.info("\nApplying SMOTE...")
    start_smote = time.time()
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE finished in {time.time() - start_smote:.2f}s. New shape: {X_res.shape}")

    results["SMOTE"] = train_and_eval(
        "SMOTE only", X_res, y_res, X_val, y_val, scale_pos_weight=1.0,
    )

    # 3. SMOTE + scale_pos_weight. After SMOTE, the resampled classes are
    # already balanced, so scale_pos_weight=1.0 is the natural choice and
    # equivalent to arm 2 in expectation; this arm instead asks "does
    # *combining* the two levers help", so it reuses the same
    # `scale_weight` computed from the pre-SMOTE class ratio above rather
    # than an unexplained magic constant.
    results["SMOTE + scale_pos_weight"] = train_and_eval(
        f"SMOTE + scale_pos_weight={scale_weight:.2f}", X_res, y_res, X_val, y_val,
        scale_pos_weight=scale_weight,
    )

    logger.info("\n--- Final Results (validation split) ---")
    df = pd.DataFrame(results).T
    logger.info("\n" + df.to_string())

    winner = df["val_pr_auc"].idxmax()
    logger.info(f"\nWinner (by validation PR-AUC): {winner}")

    REPORTS_DIR.mkdir(exist_ok=True)
    output = {
        "split_used_for_selection": "validation",
        "note": "Test split was never loaded by this script; selection is based on val_pr_auc only.",
        "winner": winner,
        "results": results,
    }
    out_path = REPORTS_DIR / "imbalance_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
