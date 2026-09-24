import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
import lightgbm as lgb

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings, load_settings
from src.evaluation.evaluator import ModelEvaluator
from src.training.manifest import build_manifest, write_manifest
from src.training.run_logging import RunLogger
from src.utils.checksums import verify_checksums, write_checksums
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _artifact_paths(base_path: Path) -> Dict[str, Path]:
    """
    Derive the three files an LGBMTrainer artifact is actually split across
    — parity with XGBTrainer/TFTTrainer's `_artifact_paths` (Phase D6).

    Closes finding F2: `models/lgbm_model.pkl` used to be a single raw
    `pickle.dump`/`pickle.load` round-trip — an arbitrary-code-on-load
    surface with no manifest and no checksum, the one artifact D6's
    migration missed. `base_path` (e.g. "models/lgbm_model.pkl") is treated
    purely as a stem, matching the other two trainers:
      - "model":     LightGBM's own native text format
                      (Booster.save_model), not pickle.
      - "metadata":  feature names, config, frozen threshold, calibrator —
                      via joblib. The Booster itself is never put in here,
                      so this file has no reason to ever need arbitrary
                      code execution to load.
      - "checksums": sha256 of the two files above; verified before either
                      is deserialized (see LGBMTrainer.load).
    """
    return {
        "model": base_path.with_name(f"{base_path.stem}.txt"),
        "metadata": base_path.with_name(f"{base_path.stem}.meta.joblib"),
        "checksums": base_path.with_name(f"{base_path.stem}.checksums.json"),
    }


def _make_tb_checkpoint_callback(run_logger: RunLogger, checkpoint_every: int = 100, keep_last: int = 3):
    """
    LightGBM training callback (the `callbacks=[...]` protocol): streams
    every eval-set metric to TensorBoard each boosting round, and
    periodically checkpoints the in-progress booster — parity with
    `TensorBoardCheckpointCallback` in train_xgb.py.
    """

    def _callback(env: "lgb.callback.CallbackEnv") -> None:
        metrics = {
            f"{dataset_name}/{eval_name}": value
            for dataset_name, eval_name, value, _ in env.evaluation_result_list
        }
        run_logger.log_scalars(metrics, step=env.iteration)

        if checkpoint_every and (env.iteration + 1) % checkpoint_every == 0:
            ckpt_path = run_logger.checkpoint_path(env.iteration, suffix="txt")
            env.model.save_model(str(ckpt_path))
            run_logger.prune_checkpoints(keep_last=keep_last)

    _callback.order = 10
    return _callback


class LGBMTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_names = None
        # Phase C4 (parity with XGBTrainer/TFTTrainer): threshold and
        # calibrator are fit once (on validation) during training and
        # frozen into the artifact. Serving must read them via
        # predict()/predict_proba_calibrated(), never recompute.
        self.threshold: Any = None
        self.calibrator: Any = None

    def set_threshold(self, threshold: float) -> None:
        """Freeze the validation-selected decision threshold (Phase C1/C4)."""
        self.threshold = threshold

    def set_calibrator(self, calibrator: Any) -> None:
        """Attach the validation-fitted probability calibrator (Phase C2/C4)."""
        self.calibrator = calibrator

    def build_model(self, scale_pos_weight: float) -> lgb.LGBMClassifier:
        lgbm_cfg = self.config.get("model", {}).get("lightgbm", {}).copy()
        
        # Extract non-constructor params
        early_stopping = lgbm_cfg.pop("early_stopping_rounds", 100)
        lgbm_cfg.pop("eval_metric", None)
        
        self.model = lgb.LGBMClassifier(
            n_estimators=lgbm_cfg.get("n_estimators", 1200),
            max_depth=lgbm_cfg.get("max_depth", -1),
            num_leaves=lgbm_cfg.get("num_leaves", 64),
            learning_rate=lgbm_cfg.get("learning_rate", 0.02),
            subsample=lgbm_cfg.get("subsample", 0.8),
            colsample_bytree=lgbm_cfg.get("colsample_bytree", 0.8),
            min_data_in_leaf=lgbm_cfg.get("min_data_in_leaf", 50),
            reg_alpha=lgbm_cfg.get("reg_alpha", 0.1),
            reg_lambda=lgbm_cfg.get("reg_lambda", 1.0),
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.get("project", {}).get("random_seed", 42),
            n_jobs=-1,
            verbosity=-1
        )
        self._early_stopping_rounds = early_stopping
        return self.model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        run_logger: Optional[RunLogger] = None,
    ) -> None:
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        self.feature_names = list(X_train.columns)

        logger.info(f"Training LightGBM with {len(X_train)} train rows, {len(X_val)} val rows.")

        # early_stopping_rounds <= 0 means "train the full n_estimators" —
        # config/config.yaml currently sets this to 0 (see comment there):
        # the 2026-08-19 audit's run stopped at iteration 4 of 1200 with
        # stopping_rounds=100, which is a training failure to investigate,
        # not a converged model. Passing 0 straight into lgb.early_stopping
        # would stop immediately, so skip the callback entirely instead.
        callbacks = []
        if self._early_stopping_rounds and self._early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=self._early_stopping_rounds, verbose=True))
        else:
            logger.warning(
                "Early stopping disabled (early_stopping_rounds <= 0) — "
                "training the full n_estimators trees."
            )
        if run_logger is not None:
            callbacks.append(_make_tb_checkpoint_callback(run_logger))

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="average_precision",
            callbacks=callbacks
        )

        logger.info(f"Training complete. Best iteration: {self.model.best_iteration_}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        # Ensure column order matches training
        if self.feature_names:
            missing_cols = set(self.feature_names) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in input: {missing_cols}")
            X = X[self.feature_names]

        # After load(), self.model is a raw lgb.Booster (native format, no
        # pickle — see _artifact_paths); right after train(), it is still
        # the sklearn LGBMClassifier wrapper. Booster.predict() on a
        # binary-objective model already returns P(positive class) — the
        # same quantity LGBMClassifier.predict_proba(X)[:, 1] returns — so
        # both branches are equivalent from every caller's point of view.
        if isinstance(self.model, lgb.Booster):
            return np.asarray(self.model.predict(X))
        return self.model.predict_proba(X)[:, 1]

    def predict_proba_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predict_proba passed through the frozen calibrator (Phase C2/C4).

        Raises if no calibrator was attached via `set_calibrator` (or
        restored via `load`) — this must never silently fall back to
        uncalibrated output, which would defeat the point of calibrating.

        CAUTION: `self.threshold` was selected against RAW (uncalibrated)
        probabilities. Do NOT binarize this method's output at
        `self.threshold` — use `predict()` for the frozen operating point;
        use this method only where a calibrated confidence VALUE is needed.
        """
        if self.calibrator is None:
            raise ValueError("No calibrator attached. Call set_calibrator() or load a calibrated artifact.")
        return self.calibrator.predict(self.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binarize RAW predict_proba at the frozen threshold (Phase C1/C4).

        Raises if no threshold was frozen — serving must never invent a
        default or silently recompute one from whatever data it has.
        """
        if self.threshold is None:
            raise ValueError("No threshold frozen. Call set_threshold() or load a thresholded artifact.")
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def save(self, path: str) -> None:
        """Persist the trained model (closes finding F2: no pickle anywhere
        here — parity with XGBTrainer.save()/TFTTrainer.save()).

        Writes three files derived from `path` (see `_artifact_paths`): the
        booster in LightGBM's own native text format, a joblib metadata
        sidecar (feature names, config, frozen threshold, calibrator), and
        a sha256 checksum manifest covering both — verified by `load()`
        before either file is deserialized.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Cannot save.")

        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        paths = _artifact_paths(base_path)

        booster = self.model.booster_ if hasattr(self.model, "booster_") else self.model
        booster.save_model(str(paths["model"]))

        metadata = {
            "feature_names": self.feature_names,
            "config": self.config,
            "threshold": self.threshold,
            "calibrator": self.calibrator,
        }
        joblib.dump(metadata, paths["metadata"])

        write_checksums(
            paths["checksums"], {"model": paths["model"], "metadata": paths["metadata"]}
        )

        logger.info(f"Model saved to {paths['model']} (+ metadata, checksums)")

    @classmethod
    def load(cls, path: str) -> 'LGBMTrainer':
        """Load a model saved by `save()`.

        Verifies the sha256 checksum manifest before deserializing anything
        — a corrupted or tampered artifact must never reach `joblib.load()`
        or LightGBM's model reader. Raises (does not fall back) on a failed
        check. `self.model` after `load()` is a raw `lgb.Booster`, not the
        `LGBMClassifier` sklearn wrapper `train()` builds — see the dispatch
        in `predict_proba()`.
        """
        base_path = Path(path)
        paths = _artifact_paths(base_path)

        verify_checksums(
            paths["checksums"], {"model": paths["model"], "metadata": paths["metadata"]}
        )

        metadata = joblib.load(paths["metadata"])

        trainer = cls(metadata["config"])
        trainer.model = lgb.Booster(model_file=str(paths["model"]))
        trainer.feature_names = metadata["feature_names"]
        # .get() — older artifacts saved before Phase C4 won't have these keys.
        trainer.threshold = metadata.get("threshold")
        trainer.calibrator = metadata.get("calibrator")
        if trainer.threshold is None or trainer.calibrator is None:
            logger.warning(
                "Loaded artifact has no frozen threshold/calibrator — "
                "pre-Phase-C4 artifact. predict()/predict_proba_calibrated() will raise."
            )

        logger.info(f"Model loaded from {paths['model']}")
        return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM model.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    settings = load_settings(args.config)
    config = settings.model_dump()
    seed = set_seed(config.get("project", {}).get("random_seed", 42))
    data_cfg = config["data"]
    processed_dir = Path(data_cfg["processed_dir"])
    
    logger.info("Loading parquet data...")
    X_train = pd.read_parquet(processed_dir / "train_features.parquet")
    y_train = pd.read_parquet(processed_dir / "train_labels.parquet").squeeze()
    
    # Use validation set for early stopping (NOT the test set)
    X_val = pd.read_parquet(processed_dir / "val_features.parquet")
    y_val = pd.read_parquet(processed_dir / "val_labels.parquet").squeeze()
    
    # Test set is held out — only for final evaluation
    X_test = pd.read_parquet(processed_dir / "test_features.parquet")
    y_test = pd.read_parquet(processed_dir / "test_labels.parquet").squeeze()
    
    # scale_pos_weight: config override, else the empirical neg/pos ratio
    # (PRD Phase 9 P9-3). Shared resolver with train_xgb.py.
    from src.training.train_xgb import resolve_scale_pos_weight

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = resolve_scale_pos_weight(
        config, neg_count, pos_count, model_key="lightgbm"
    )
    logger.info(
        "Class imbalance: %d neg / %d pos (ratio %.2f) -> scale_pos_weight = %.4f",
        neg_count,
        pos_count,
        neg_count / pos_count,
        scale_pos_weight,
    )

    # Set up MLflow
    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "fraud_detection"))
    
    with mlflow.start_run(run_name="lgbm_enhanced_features") as run, RunLogger(
        run_type="lightgbm", run_name=f"lgbm_{run.info.run_id[:8]}"
    ) as run_logger:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        trainer = LGBMTrainer(config)
        trainer.build_model(scale_pos_weight=scale_pos_weight)

        # Log params
        lgbm_params = trainer.model.get_params()
        mlflow.log_params({k: v for k, v in lgbm_params.items() if v is not None})
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("imbalance_strategy", "scale_pos_weight")
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("feature_count", len(X_train.columns))
        mlflow.log_param("tensorboard_log_dir", str(run_logger.log_dir))
        mlflow.log_param("checkpoint_dir", str(run_logger.checkpoint_dir))

        # Train with validation set for early stopping
        trainer.train(X_train, y_train, X_val, y_val, run_logger=run_logger)
        
        # Evaluate on all splits for overfitting analysis
        logger.info("Evaluating on all splits...")
        evaluator = ModelEvaluator()
        
        y_prob_train = trainer.predict_proba(X_train)
        y_prob_val = trainer.predict_proba(X_val)
        y_prob_test = trainer.predict_proba(X_test)
        
        # PR-AUC on all splits
        pr_auc_train = evaluator.compute_pr_auc(y_train, y_prob_train)
        pr_auc_val = evaluator.compute_pr_auc(y_val, y_prob_val)
        pr_auc_test = evaluator.compute_pr_auc(y_test, y_prob_test)
        
        roc_auc_test = evaluator.compute_roc_auc(y_test, y_prob_test)
        
        # Overfitting gap analysis
        overfit_gap_train_val = pr_auc_train - pr_auc_val
        overfit_gap_train_test = pr_auc_train - pr_auc_test
        
        logger.info("=" * 60)
        logger.info("OVERFITTING ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"  Train PR-AUC:  {pr_auc_train:.4f}")
        logger.info(f"  Val PR-AUC:    {pr_auc_val:.4f}")
        logger.info(f"  Test PR-AUC:   {pr_auc_test:.4f}")
        logger.info(f"  Gap (train-val):  {overfit_gap_train_val:.4f}")
        logger.info(f"  Gap (train-test): {overfit_gap_train_test:.4f}")
        if overfit_gap_train_test > 0.15:
            logger.warning("⚠️  SIGNIFICANT OVERFITTING DETECTED (gap > 0.15)")
        elif overfit_gap_train_test > 0.10:
            logger.warning("⚠️  Moderate overfitting detected (gap > 0.10)")
        else:
            logger.info("✓ Overfitting gap within acceptable range (< 0.10)")
        logger.info("=" * 60)
        
        # Phase C2 (parity with train_xgb.py/train_tft.py — LGBM previously
        # had no calibration at all): fit isotonic calibration on VALIDATION
        # only, then apply (never refit) to test. Calibration is monotonic
        # so it does not change PR-AUC/ROC-AUC (rank-invariant) — it
        # corrects the probability VALUES the cost-based threshold logic
        # assumes are calibrated. Reported here as diagnostics; the
        # threshold search itself still runs on raw probabilities, same
        # convention as the other two trainers.
        calibrator = evaluator.fit_calibrator(y_val, y_prob_val, method="isotonic")
        y_prob_val_cal = evaluator.apply_calibration(calibrator, y_prob_val)
        y_prob_test_cal = evaluator.apply_calibration(calibrator, y_prob_test)

        brier_val_before = evaluator.compute_brier_score(y_val, y_prob_val)
        brier_val_after = evaluator.compute_brier_score(y_val, y_prob_val_cal)
        brier_test_before = evaluator.compute_brier_score(y_test, y_prob_test)
        brier_test_after = evaluator.compute_brier_score(y_test, y_prob_test_cal)
        mlflow.log_metric("calibration_brier_score_val_before", brier_val_before)
        mlflow.log_metric("calibration_brier_score_val_after", brier_val_after)
        mlflow.log_metric("calibration_brier_score_test_before", brier_test_before)
        mlflow.log_metric("calibration_brier_score_test_after", brier_test_after)
        logger.info(
            f"Calibration (val):  Brier {brier_val_before:.4f} -> {brier_val_after:.4f}"
        )
        logger.info(
            f"Calibration (test): Brier {brier_test_before:.4f} -> {brier_test_after:.4f}"
        )
        if brier_test_after >= brier_test_before:
            logger.warning(
                "⚠️  Calibration did not improve test Brier score — "
                "investigate before relying on predict_proba_calibrated()."
            )

        # Phase C1 (found via mle-reviewer while validating C2 elsewhere in
        # this codebase): threshold must be selected on VALIDATION, never
        # test. This previously read y_test/y_prob_test directly — the same
        # leak already fixed in train_xgb.py and train_tft.py.
        thresh_cfg = config.get("thresholds", {})
        cost_fn = thresh_cfg.get("cost_fn", 500)
        cost_fp = thresh_cfg.get("cost_fp", 5)
        revenue_tp = thresh_cfg.get("revenue_tp", 480)

        optimal_t = evaluator.find_optimal_threshold(y_val, y_prob_val, cost_fn, cost_fp, revenue_tp)
        metrics = evaluator.compute_metrics_at_threshold(y_test, y_prob_test, optimal_t)

        # Best F1 threshold: also selected on validation, then reported on test.
        from sklearn.metrics import precision_recall_curve
        precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_val, y_prob_val)
        f1_scores = np.divide(
            2 * precision_arr * recall_arr,
            precision_arr + recall_arr,
            out=np.zeros_like(precision_arr),
            where=(precision_arr + recall_arr) != 0
        )
        best_f1_idx = np.argmax(f1_scores)
        best_f1_threshold = thresholds_arr[best_f1_idx] if best_f1_idx < len(thresholds_arr) else 0.5
        if best_f1_idx >= len(thresholds_arr):
            logger.warning(
                "⚠️  F1-optimal index fell on precision_recall_curve's trailing "
                "point (no corresponding threshold) — falling back to 0.5."
            )
        best_f1 = evaluator.compute_metrics_at_threshold(y_test, y_prob_test, best_f1_threshold)["f1"]

        # Log metrics
        mlflow.log_metric("pr_auc_train", pr_auc_train)
        mlflow.log_metric("pr_auc_val", pr_auc_val)
        mlflow.log_metric("pr_auc_test", pr_auc_test)
        mlflow.log_metric("roc_auc_test", roc_auc_test)
        mlflow.log_metric("overfit_gap_train_test", overfit_gap_train_test)
        mlflow.log_metric("optimal_threshold", optimal_t)
        mlflow.log_metric("best_f1", best_f1)
        mlflow.log_metric("best_f1_threshold", best_f1_threshold)
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)
            
        logger.info(f"PR-AUC (test): {pr_auc_test:.4f}")
        logger.info(f"ROC-AUC (test): {roc_auc_test:.4f}")
        logger.info(f"Best F1: {best_f1:.4f} at threshold {best_f1_threshold:.4f}")
        logger.info(f"Optimal cost-based threshold: {optimal_t:.4f}")
        logger.info(f"Metrics at optimal threshold: {metrics}")
        
        # Generate and log plots
        reports_dir = Path("reports/figures")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        pr_path = reports_dir / "lgbm_pr_curve.png"
        evaluator.plot_pr_curve(y_test, y_prob_test, "LightGBM", str(pr_path))
        mlflow.log_artifact(str(pr_path))
        
        roc_path = reports_dir / "lgbm_roc_curve.png"
        evaluator.plot_roc_curve(y_test, y_prob_test, "LightGBM", str(roc_path))
        mlflow.log_artifact(str(roc_path))
        
        cm_path = reports_dir / "lgbm_confusion_matrix.png"
        y_pred = (y_prob_test >= optimal_t).astype(int)
        evaluator.plot_confusion_matrix(y_test, y_pred, str(cm_path))
        mlflow.log_artifact(str(cm_path))
        
        val_path = reports_dir / "lgbm_threshold_value.png"
        # Phase C3 parity fix: must pass the frozen validation threshold, or
        # the plotted marker silently re-derives its own argmax from the
        # TEST curve instead of showing the threshold actually used above
        # for cm_path's confusion matrix — the same "two objectives can
        # disagree" bug C3 fixed in train_xgb.py/train_tft.py.
        evaluator.plot_threshold_vs_business_value(
            y_test, y_prob_test, cost_fn, cost_fp, revenue_tp, str(val_path), optimal_threshold=optimal_t
        )
        mlflow.log_artifact(str(val_path))

        # Phase C2: reliability curve, uncalibrated vs. calibrated, on test.
        reliability_path = reports_dir / "lgbm_reliability_curve.png"
        evaluator.plot_reliability_curve(
            y_test, y_prob_test, str(reliability_path), y_prob_calibrated=y_prob_test_cal
        )
        mlflow.log_artifact(str(reliability_path))

        # Phase C4: freeze the validation-selected threshold and the
        # validation-fitted calibrator into the artifact before saving, so
        # serving never has to recompute either.
        trainer.set_threshold(optimal_t)
        trainer.set_calibrator(calibrator)

        # Save model
        model_path = "models/lgbm_model.pkl"
        trainer.save(model_path)
        # save() no longer writes the literal `model_path` (Phase D6 parity
        # with XGBTrainer/TFTTrainer) — it's a stem _artifact_paths()
        # derives three real files from (model, metadata, checksums). Log
        # each of those, not the nonexistent stem.
        for artifact_path in _artifact_paths(Path(model_path)).values():
            mlflow.log_artifact(str(artifact_path))

        # Closes finding F2: LightGBM previously wrote no manifest at all —
        # the only trained model on disk with no traceable link back to the
        # config/dataset/git commit that produced it. Same mechanism as
        # train_xgb.py/train_tft.py.
        manifest = build_manifest(
            model_type="lightgbm",
            mlflow_run_id=run_id,
            settings=settings,
            dataset_dir=processed_dir,
            dataset_files=[
                "train_features.parquet",
                "train_labels.parquet",
                "val_features.parquet",
                "val_labels.parquet",
                "test_features.parquet",
                "test_labels.parquet",
            ],
            random_seed=seed,
            metrics={
                "pr_auc_train": pr_auc_train,
                "pr_auc_val": pr_auc_val,
                "pr_auc_test": pr_auc_test,
                "roc_auc_test": roc_auc_test,
                "overfit_gap_train_test": overfit_gap_train_test,
                "optimal_threshold": optimal_t,
                "best_f1": best_f1,
                "best_f1_threshold": best_f1_threshold,
            },
        )
        manifest_path = write_manifest(model_path, manifest)
        mlflow.log_artifact(str(manifest_path))

        # Print classification report
        logger.info("\n" + evaluator.generate_classification_report(y_test, y_pred))

        logger.info(f"Enhanced feature training complete! MLflow run ID: {run_id}")


if __name__ == "__main__":
    main()
