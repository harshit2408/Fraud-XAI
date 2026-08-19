import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings, load_settings
from src.device import resolve_device
from src.evaluation.evaluator import ModelEvaluator
from src.training.manifest import build_manifest, write_manifest
from src.utils.checksums import verify_checksums, write_checksums
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _artifact_paths(base_path: Path) -> Dict[str, Path]:
    """
    Derive the three files an XGBTrainer artifact is actually split across
    (Phase D6, docs/IMPLEMENTATION_PLAN.md).

    `base_path` (e.g. "models/xgb_model.pkl") is treated purely as a stem —
    the literal path is never written to. This keeps `serving.model_path` in
    config/config.yaml stable across the D6 migration without implying the
    artifact is still a single pickle file:
      - "model":     XGBoost's own JSON/UBJ format (xgb.Booster.save_model),
                      not pickle — no arbitrary-code-on-load surface.
      - "metadata":  everything that isn't the booster itself (feature
                      names, config, frozen threshold, calibrator), via
                      joblib.
      - "checksums": sha256 of the two files above; verified before either
                      is deserialized (see XGBTrainer.load).
    """
    return {
        "model": base_path.with_name(f"{base_path.stem}.ubj"),
        "metadata": base_path.with_name(f"{base_path.stem}.meta.joblib"),
        "checksums": base_path.with_name(f"{base_path.stem}.checksums.json"),
    }


class XGBTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_names = None
        # Phase C4: threshold and calibrator are fit once (on validation)
        # during training and frozen into the artifact. Serving must read
        # them via predict()/predict_proba_calibrated(), never recompute.
        self.threshold: Any = None
        self.calibrator: Any = None

    def set_threshold(self, threshold: float) -> None:
        """Freeze the validation-selected decision threshold (Phase C1/C4)."""
        self.threshold = threshold

    def set_calibrator(self, calibrator: Any) -> None:
        """Attach the validation-fitted probability calibrator (Phase C2/C4)."""
        self.calibrator = calibrator

    def build_model(self, scale_pos_weight: float) -> xgb.XGBClassifier:
        xgb_cfg = self.config.get("model", {}).get("xgboost", {}).copy()
        
        # Extract early_stopping_rounds and eval_metric separately since they 
        # are handled by the fit() method or are specific kwargs
        early_stopping = xgb_cfg.pop("early_stopping_rounds", 50)
        eval_metric = xgb_cfg.pop("eval_metric", "aucpr")
        
        # Override scale_pos_weight
        xgb_cfg["scale_pos_weight"] = scale_pos_weight
        xgb_cfg["random_state"] = self.config.get("project", {}).get("random_seed", 42)
        xgb_cfg["enable_categorical"] = True
        xgb_cfg["n_jobs"] = -1
        # Phase D7: config ships "auto"; XGBoost itself only understands
        # "cpu"/"cuda", so resolve here rather than passing "auto" through.
        xgb_cfg["device"] = resolve_device(xgb_cfg.get("device", "auto"))
        xgb_cfg["eval_metric"] = eval_metric
        xgb_cfg["early_stopping_rounds"] = early_stopping
        
        self.model = xgb.XGBClassifier(**xgb_cfg)
        return self.model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        self.feature_names = list(X_train.columns)
        
        logger.info(f"Training XGBoost with {len(X_train)} train rows, {len(X_val)} val rows.")
        
        # XGBClassifier handles early stopping natively when eval_set is provided
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50,
        )
        
        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
            
        # Ensure column order matches training
        if self.feature_names:
            missing_cols = set(self.feature_names) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in input: {missing_cols}")
            X = X[self.feature_names]
            
        return self.model.predict_proba(X)[:, 1]

    def predict_proba_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predict_proba passed through the frozen calibrator (Phase C2/C4).

        Raises if no calibrator was attached via `set_calibrator` (or
        restored via `load`) — this must never silently fall back to
        uncalibrated output, which would defeat the point of calibrating.

        CAUTION: `self.threshold` was selected against RAW (uncalibrated)
        probabilities (see `find_optimal_threshold` call site in `main()`).
        Do NOT binarize this method's output at `self.threshold` — that
        applies a threshold tuned for the wrong probability scale. Use
        `predict()` (raw probabilities) for the frozen operating point;
        use this method only where a calibrated confidence VALUE is
        needed (e.g. a human-facing "73% likely fraud" explanation).
        """
        if self.calibrator is None:
            raise ValueError("No calibrator attached. Call set_calibrator() or load a calibrated artifact.")
        return self.calibrator.predict(self.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binarize RAW predict_proba at the frozen threshold (Phase C1/C4).

        Raises if no threshold was frozen — serving must never invent a
        default or silently recompute one from whatever data it has.
        `self.threshold` is calibrated for raw probabilities specifically;
        see the caution note on `predict_proba_calibrated`.
        """
        if self.threshold is None:
            raise ValueError("No threshold frozen. Call set_threshold() or load a thresholded artifact.")
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def save(self, path: str) -> None:
        """Persist the trained model (Phase D6: no pickle anywhere here).

        Writes three files derived from `path` (see `_artifact_paths`): the
        booster in XGBoost's native JSON/UBJ format, a joblib metadata
        sidecar (feature names, config, frozen threshold, calibrator), and
        a sha256 checksum manifest covering both — verified by `load()`
        before either file is deserialized.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Cannot save.")

        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        paths = _artifact_paths(base_path)

        self.model.save_model(str(paths["model"]))

        metadata = {
            "feature_names": self.feature_names,
            "config": self.config,
            "threshold": self.threshold,
            "calibrator": self.calibrator,
            # save_model()/load_model() only round-trip the booster itself;
            # XGBClassifier.n_classes_ (predict_proba's dependency; classes_
            # is a read-only property derived from it) is restored on a
            # best-effort basis by XGBoost's own load_model() via an
            # is_classifier() check that is version-sensitive across
            # xgboost/scikit-learn releases — persist it explicitly here
            # instead of depending on that.
            "n_classes_": self.model.n_classes_,
        }
        joblib.dump(metadata, paths["metadata"])

        write_checksums(
            paths["checksums"], {"model": paths["model"], "metadata": paths["metadata"]}
        )

        logger.info(f"Model saved to {paths['model']} (+ metadata, checksums)")

    @classmethod
    def load(cls, path: str) -> 'XGBTrainer':
        """Load a model saved by `save()` (Phase D6).

        Verifies the sha256 checksum manifest before deserializing anything
        — a corrupted or tampered artifact must never reach `joblib.load()`
        or the XGBoost model reader. Raises (does not fall back) on a
        failed check.
        """
        base_path = Path(path)
        paths = _artifact_paths(base_path)

        verify_checksums(
            paths["checksums"], {"model": paths["model"], "metadata": paths["metadata"]}
        )

        metadata = joblib.load(paths["metadata"])

        trainer = cls(metadata["config"])
        trainer.model = xgb.XGBClassifier()
        trainer.model.load_model(str(paths["model"]))
        # Explicitly restore the sklearn-wrapper attribute predict_proba
        # needs — see the comment on `save()` for why this isn't left to
        # XGBoost's own load_model() to reconstruct. classes_ is a
        # read-only property derived from n_classes_, so setting this alone
        # is sufficient.
        trainer.model.n_classes_ = metadata["n_classes_"]
        # Phase D7: force CPU regardless of the device the artifact was
        # trained on — the serving host is not guaranteed to have a GPU.
        trainer.model.set_params(device=resolve_device("auto", force_cpu=True))
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


def load_tuned_params(path: Path) -> Dict[str, Any]:
    """Load a versioned tuned-params YAML written by `tune_xgb.py` (Phase D3).

    Returns the full parsed document (including `mlflow_run_id`, so the
    caller can log which tuning run's params were used) — callers read
    `document["params"]` for the actual hyperparameter overrides.

    Raises:
        FileNotFoundError: `path` does not exist.
        ValueError: the file parses but is missing the required `params` key
            — fail fast rather than silently training with no overrides.
    """
    if not path.exists():
        raise FileNotFoundError(f"Tuned params file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        document = yaml.safe_load(f)

    if not isinstance(document, dict) or "params" not in document:
        raise ValueError(
            f"Tuned params file {path} is missing the required 'params' key "
            "— expected the format written by tune_xgb.py's write_tuned_params()."
        )

    return document


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost model.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--tuned-params",
        default=None,
        help=(
            "Path to a versioned tuned-params YAML from tune_xgb.py "
            "(config/tuned/xgb_<run_id>.yaml, Phase D3). Overrides "
            "config/config.yaml's xgboost hyperparameters when given."
        ),
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    config = settings.model_dump()
    seed = set_seed(config.get("project", {}).get("random_seed", 42))

    tuned_params_source = "config.yaml"
    if args.tuned_params:
        tuned_path = Path(args.tuned_params)
        tuned_document = load_tuned_params(tuned_path)
        config["model"]["xgboost"].update(tuned_document["params"])
        tuned_params_source = str(tuned_path)
        logger.info(
            f"Loaded tuned hyperparameters from {tuned_path} "
            f"(mlflow_run_id={tuned_document.get('mlflow_run_id')})"
        )

        # Phase D4 (mle-reviewer HIGH finding): re-validate the merged
        # config so `settings` — and therefore the manifest's config_hash
        # — reflects the EFFECTIVE hyperparameters actually used for this
        # run, not the pre-override config.yaml. Without this, two runs
        # with genuinely different tuned params got byte-identical
        # config_hash values, defeating D4's rollback-identification
        # acceptance criterion. This also closes D1's fail-fast contract
        # for this write path: a malformed tuned-params file (unexpected
        # key, wrong type) now raises a clear pydantic ValidationError
        # here instead of a late, unclear XGBoost/sklearn TypeError.
        settings = Settings.model_validate(config)
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
    
    # Calculate scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    logger.info(f"Class imbalance: {neg_count} neg / {pos_count} pos -> scale_pos_weight = {scale_pos_weight:.2f}")

    # Set up MLflow
    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "fraud_detection"))
    
    with mlflow.start_run(run_name="xgb_enhanced_features") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        trainer = XGBTrainer(config)
        trainer.build_model(scale_pos_weight=scale_pos_weight)
        
        # Log params
        xgb_params = trainer.model.get_params()
        mlflow.log_params({k: v for k, v in xgb_params.items() if v is not None})
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("imbalance_strategy", "scale_pos_weight")
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("feature_count", len(X_train.columns))
        # Phase D3: which source supplied the hyperparameters this run used.
        mlflow.log_param("tuned_params_source", tuned_params_source)

        # Train with validation set for early stopping
        trainer.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on all splits for overfitting analysis
        logger.info("Evaluating on all splits...")
        evaluator = ModelEvaluator()
        
        y_prob_train = trainer.predict_proba(X_train)
        y_prob_val = trainer.predict_proba(X_val)
        y_prob_test = trainer.predict_proba(X_test)

        # Phase C2: fit isotonic calibration on VALIDATION only, then apply
        # (never refit) to test. Calibration is monotonic so it does not
        # change PR-AUC/ROC-AUC (rank-based); it corrects the probability
        # VALUES that the cost-based threshold logic assumes are calibrated.
        # Reported here as diagnostics; wiring the threshold search itself
        # onto calibrated probabilities is a follow-up decision (Phase C3
        # already unified the objective on raw probabilities and is not
        # touched by this change).
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
                "the isotonic fit on val may not generalize to test."
            )

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
        
        # Phase C1: threshold is selected on the VALIDATION split and then
        # frozen — test metrics below are reported at that frozen threshold,
        # never re-derived from y_test.
        thresh_cfg = config.get("thresholds", {})
        cost_fn = thresh_cfg.get("cost_fn", 500)
        cost_fp = thresh_cfg.get("cost_fp", 5)
        revenue_tp = thresh_cfg.get("revenue_tp", 480)

        optimal_t = evaluator.find_optimal_threshold(y_val, y_prob_val, cost_fn, cost_fp, revenue_tp)
        metrics = evaluator.compute_metrics_at_threshold(y_test, y_prob_test, optimal_t)

        # Also select the F1-optimal threshold on validation, then report
        # what it achieves on test (Phase C1: no threshold-selection call
        # may read y_test).
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
        logger.info(f"Optimal business-value threshold (revenue_tp={revenue_tp}): {optimal_t:.4f}")
        logger.info(f"Metrics at optimal threshold: {metrics}")
        
        # Generate and log plots
        reports_dir = Path("reports/figures")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        pr_path = reports_dir / "xgb_pr_curve.png"
        evaluator.plot_pr_curve(y_test, y_prob_test, "XGBoost", str(pr_path))
        mlflow.log_artifact(str(pr_path))
        
        roc_path = reports_dir / "xgb_roc_curve.png"
        evaluator.plot_roc_curve(y_test, y_prob_test, "XGBoost", str(roc_path))
        mlflow.log_artifact(str(roc_path))
        
        cm_path = reports_dir / "xgb_confusion_matrix.png"
        y_pred = (y_prob_test >= optimal_t).astype(int)
        evaluator.plot_confusion_matrix(y_test, y_pred, str(cm_path))
        mlflow.log_artifact(str(cm_path))
        
        val_path = reports_dir / "xgb_threshold_value.png"
        evaluator.plot_threshold_vs_business_value(
            y_test, y_prob_test, cost_fn, cost_fp, revenue_tp, str(val_path), optimal_threshold=optimal_t
        )
        mlflow.log_artifact(str(val_path))

        # Phase C2: reliability curve, uncalibrated vs. calibrated, on test.
        reliability_path = reports_dir / "xgb_reliability_curve.png"
        evaluator.plot_reliability_curve(
            y_test, y_prob_test, str(reliability_path), y_prob_calibrated=y_prob_test_cal
        )
        mlflow.log_artifact(str(reliability_path))

        # Phase C4: freeze the validation-selected threshold and the
        # validation-fitted calibrator into the artifact before saving, so
        # serving reads them rather than recomputing/refitting either.
        trainer.set_threshold(optimal_t)
        trainer.set_calibrator(calibrator)

        # Save model
        model_path = config.get("serving", {}).get("model_path", "models/xgb_model.pkl")
        trainer.save(model_path)
        # Phase D6: save() no longer writes the literal `model_path` — it's
        # a stem `_artifact_paths()` derives three real files from (model,
        # metadata, checksums). Log each of those, not the nonexistent stem.
        for artifact_path in _artifact_paths(Path(model_path)).values():
            mlflow.log_artifact(str(artifact_path))

        # Phase D4: write a manifest (config hash, git SHA, dataset hash,
        # metrics, timestamp) next to the artifact so it can be traced back
        # to the exact run that produced it, without retraining.
        manifest = build_manifest(
            model_type="xgboost",
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
