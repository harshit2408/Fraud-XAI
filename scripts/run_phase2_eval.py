"""
scripts/run_phase2_eval.py

Phase 2 evaluation — iterative approach.

Strategy: Start from the KNOWN WORKING baseline (~0.56 PR-AUC with 80/20 split)
and incrementally add improvements while monitoring the overfitting gap.

Key learnings from failed experiments:
  1. Dropping raw card features (card1-5, addr1-2) HURTS performance badly
     - These are critical for the model; card aggregates alone can't replace them
  2. Heavy regularization alone can't fix temporal distribution shift
  3. The val-test gap is primarily temporal, not overfitting
  4. Need to go BACK to 80/20 split (the val split just wastes training data)
     and use a portion of training data as validation via time-aware split

Usage:
    conda run -n fraudx python scripts/run_phase2_eval.py
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
)
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def select_f1_threshold(y_true, y_prob):
    """Select the F1-maximizing threshold via precision_recall_curve.

    Phase C1: this is a threshold-selection call — callers must pass the
    VALIDATION split, never y_test. Apply the returned threshold to test
    data with `f1_at_threshold` instead of re-deriving it from test.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(best_threshold)


def f1_at_threshold(y_true, y_prob, threshold):
    """F1 score achieved by a single, already-selected threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def select_threshold_at_fpr(y_true, y_prob, target_fpr=0.05):
    """Select the threshold whose ROC point sits at a given false positive rate.

    Phase C1: this is a threshold-selection call — callers must pass the
    VALIDATION split, never y_test. Apply the returned threshold to test
    data with `recall_at_threshold` instead of re-deriving it from test.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = max(0, np.searchsorted(fpr, target_fpr, side="right") - 1)
    return float(thresholds[idx])


def recall_at_threshold(y_true, y_prob, threshold):
    """Recall achieved by a single, already-selected threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def lgbm_pr_auc(y_true, y_pred):
    """Custom metric for LightGBM early stopping on PR-AUC."""
    score = average_precision_score(y_true, y_pred)
    return "pr_auc", score, True


def run_xgb_experiment(name, params, X_train, y_train, X_val, y_val, X_test, y_test):
    """Run XGBoost experiment with given parameters."""
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT: {name}")
    logger.info(f"{'='*70}")

    model = xgb.XGBClassifier(**params)
    start = time.time()
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=100)
    elapsed = time.time() - start

    return _evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test, elapsed)


def run_lgbm_experiment(name, params, X_train, y_train, X_val, y_val, X_test, y_test,
                        early_stopping_rounds=100):
    """Run LightGBM experiment."""
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT: {name}")
    logger.info(f"{'='*70}")

    model = lgb.LGBMClassifier(**params)
    start = time.time()
    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True)]
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric=lgbm_pr_auc,
              callbacks=callbacks)
    elapsed = time.time() - start

    return _evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test, elapsed)


def _evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test, elapsed):
    """Evaluate model on all splits and return metrics dict."""
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    pr_train = average_precision_score(y_train, y_prob_train)
    pr_val = average_precision_score(y_val, y_prob_val)
    pr_test = average_precision_score(y_test, y_prob_test)
    roc_test = roc_auc_score(y_test, y_prob_test)

    # Phase C1: select the F1 threshold on validation, freeze it, then report
    # what it achieves on test — never re-derive the threshold from y_test.
    f1_thresh = select_f1_threshold(y_val, y_prob_val)
    f1_test = f1_at_threshold(y_test, y_prob_test, f1_thresh)
    # Train-side F1 is a diagnostic of the model's own fit, not a threshold
    # used for reporting, so its threshold may legitimately be selected on train.
    f1_train = f1_at_threshold(y_train, y_prob_train, select_f1_threshold(y_train, y_prob_train))
    fpr5_thresh = select_threshold_at_fpr(y_val, y_prob_val, 0.05)
    rec_5fpr = recall_at_threshold(y_test, y_prob_test, fpr5_thresh)

    gap = pr_train - pr_test
    best_iter = getattr(model, 'best_iteration', getattr(model, 'best_iteration_', -1))

    logger.info(f"  PR-AUC → Train: {pr_train:.4f}  Val: {pr_val:.4f}  Test: {pr_test:.4f}")
    logger.info(f"  Overfit gap: {gap:.4f}  |  ROC-AUC: {roc_test:.4f}")
    logger.info(f"  Best F1: {f1_test:.4f} @ thresh {f1_thresh:.4f}")
    logger.info(f"  Recall@5%FPR: {rec_5fpr:.4f}  |  Time: {elapsed:.1f}s  |  Iters: {best_iter}")

    if gap > 0.15:
        logger.warning("  ⚠ SIGNIFICANT OVERFITTING")
    elif gap > 0.10:
        logger.warning("  ⚠ Moderate overfitting")
    else:
        logger.info("  ✓ Healthy generalization")

    return {
        "name": name, "pr_train": pr_train, "pr_val": pr_val, "pr_test": pr_test,
        "roc_test": roc_test, "f1_test": f1_test, "f1_thresh": f1_thresh,
        "f1_train": f1_train, "rec_5fpr": rec_5fpr, "gap": gap,
        "best_iter": best_iter, "time": elapsed,
        "model": model, "y_prob_test": y_prob_test,
    }


def main():
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])

    logger.info("=" * 70)
    logger.info("PHASE 2 EVALUATION — Iterative Improvement")
    logger.info("=" * 70)

    # ── Load data (use train+val as training, test as holdout) ────────────
    X_train_full = pd.read_parquet(processed_dir / "train_features.parquet")
    y_train_full = pd.read_parquet(processed_dir / "train_labels.parquet").squeeze()
    X_val = pd.read_parquet(processed_dir / "val_features.parquet")
    y_val = pd.read_parquet(processed_dir / "val_labels.parquet").squeeze()
    X_test = pd.read_parquet(processed_dir / "test_features.parquet")
    y_test = pd.read_parquet(processed_dir / "test_labels.parquet").squeeze()

    # Combine train+val for full 80% training, use last 15% of that as validation
    X_combined = pd.concat([X_train_full, X_val], ignore_index=True)
    y_combined = pd.concat([y_train_full, y_val], ignore_index=True)

    # Internal val split: last 15% of training data for early stopping
    val_size = int(len(X_combined) * 0.15)
    X_train = X_combined.iloc[:-val_size]
    y_train = y_combined.iloc[:-val_size]
    X_val_internal = X_combined.iloc[-val_size:]
    y_val_internal = y_combined.iloc[-val_size:]

    logger.info(f"Train: {X_train.shape} | Fraud: {y_train.mean()*100:.2f}%")
    logger.info(f"Val:   {X_val_internal.shape} | Fraud: {y_val_internal.mean()*100:.2f}%")
    logger.info(f"Test:  {X_test.shape} | Fraud: {y_test.mean()*100:.2f}%")
    logger.info(f"Features: {len(X_train.columns)}")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    logger.info(f"scale_pos_weight = {spw:.2f}")

    seed = config.get("project", {}).get("random_seed", 42)
    results = []

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: XGBoost — Moderate depth, moderate reg (baseline++)
    # ══════════════════════════════════════════════════════════════════════════
    r = run_xgb_experiment("XGB depth=6, lr=0.05, reg", {
        "n_estimators": 2000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.1,
        "scale_pos_weight": spw,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 80,
        "random_state": seed,
        "enable_categorical": True,
        "n_jobs": -1,
    }, X_train, y_train, X_val_internal, y_val_internal, X_test, y_test)
    results.append(r)

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: XGBoost — Shallower with stronger regularization
    # ══════════════════════════════════════════════════════════════════════════
    r = run_xgb_experiment("XGB depth=5, lr=0.03, strong_reg", {
        "n_estimators": 3000,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.75,
        "colsample_bytree": 0.7,
        "min_child_weight": 20,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "gamma": 0.5,
        "scale_pos_weight": spw,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 100,
        "random_state": seed,
        "enable_categorical": True,
        "n_jobs": -1,
    }, X_train, y_train, X_val_internal, y_val_internal, X_test, y_test)
    results.append(r)

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: LightGBM — Often generalizes better than XGBoost
    # ══════════════════════════════════════════════════════════════════════════
    r = run_lgbm_experiment("LGBM leaves=63, lr=0.03", {
        "n_estimators": 3000,
        "max_depth": -1,
        "num_leaves": 63,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_data_in_leaf": 30,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": spw,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }, X_train, y_train, X_val_internal, y_val_internal, X_test, y_test,
       early_stopping_rounds=100)
    results.append(r)

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 4: LightGBM — Fewer leaves (more constrained)
    # ══════════════════════════════════════════════════════════════════════════
    r = run_lgbm_experiment("LGBM leaves=31, lr=0.02, strong_reg", {
        "n_estimators": 3000,
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.02,
        "subsample": 0.75,
        "colsample_bytree": 0.6,
        "min_data_in_leaf": 50,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "scale_pos_weight": spw,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }, X_train, y_train, X_val_internal, y_val_internal, X_test, y_test,
       early_stopping_rounds=150)
    results.append(r)

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    best = max(results, key=lambda r: r["pr_test"])

    logger.info("\n\n" + "=" * 80)
    logger.info("EXPERIMENT COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Experiment':<40s} {'PR-AUC':>8s} {'F1':>8s} {'R@5%':>8s} {'Gap':>8s} {'Iters':>6s}")
    logger.info("-" * 80)
    for r in results:
        marker = " ← BEST" if r["name"] == best["name"] else ""
        logger.info(f"{r['name']:<40s} {r['pr_test']:>8.4f} {r['f1_test']:>8.4f} "
                    f"{r['rec_5fpr']:>8.4f} {r['gap']:>8.4f} {r['best_iter']:>6d}{marker}")

    logger.info(f"\n{'='*70}")
    logger.info(f"BEST: {best['name']}")
    logger.info(f"{'='*70}")
    logger.info(f"  Train PR-AUC:      {best['pr_train']:.4f}")
    logger.info(f"  Val PR-AUC:        {best['pr_val']:.4f}")
    logger.info(f"  Test PR-AUC:       {best['pr_test']:.4f}")
    logger.info(f"  Overfit gap:       {best['gap']:.4f}")
    logger.info(f"  ROC-AUC:           {best['roc_test']:.4f}")
    logger.info(f"  Best F1:           {best['f1_test']:.4f}")
    logger.info(f"  Recall @ 5% FPR:   {best['rec_5fpr']:.4f}")

    pr_ok = "✓" if best["pr_test"] >= 0.70 else "✗"
    f1_ok = "✓" if best["f1_test"] >= 0.65 else "✗"
    rec_ok = "✓" if best["rec_5fpr"] >= 0.70 else "✗"
    gap_ok = "✓" if best["gap"] < 0.10 else "✗"
    logger.info(f"\n  PRD TARGETS:")
    logger.info(f"    {pr_ok} PR-AUC ≥ 0.70:        {best['pr_test']:.4f}")
    logger.info(f"    {f1_ok} F1 ≥ 0.65:            {best['f1_test']:.4f}")
    logger.info(f"    {rec_ok} Recall@5%FPR ≥ 0.70: {best['rec_5fpr']:.4f}")
    logger.info(f"    {gap_ok} Overfit gap < 0.10:  {best['gap']:.4f}")

    # Classification report
    y_pred = (best["y_prob_test"] >= best["f1_thresh"]).astype(int)
    logger.info(f"\nClassification Report (threshold={best['f1_thresh']:.4f}):")
    logger.info(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # Feature importance
    model = best["model"]
    importance = model.feature_importances_
    feat_imp = sorted(zip(X_train.columns, importance), key=lambda x: x[1], reverse=True)
    logger.info("Top 25 Features:")
    for i, (fname, imp) in enumerate(feat_imp[:25]):
        logger.info(f"  {i+1:2d}. {fname:<35s} {imp:.4f}")

    # Save
    results_data = {r["name"]: {k: v for k, v in r.items()
                                 if k not in ("model", "y_prob_test")}
                    for r in results}
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(results_data).T.to_json("reports/phase2_results.json", indent=2)
    logger.info(f"\nResults saved to reports/phase2_results.json")


if __name__ == "__main__":
    main()
