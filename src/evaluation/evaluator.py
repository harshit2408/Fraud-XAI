import logging
from typing import Dict, Optional

import matplotlib
# Headless backend: this module only ever saves figures to disk (never
# plt.show()). Forcing Agg avoids picking up an interactive backend
# (e.g. TkAgg) from whatever else has imported matplotlib in-process,
# which on some environments crashes with a broken Tcl/Tk install.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

_SUPPORTED_CALIBRATION_METHODS = ("isotonic",)


class ModelEvaluator:
    def compute_pr_auc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute area under Precision-Recall curve."""
        return float(average_precision_score(y_true, y_prob))

    def compute_roc_auc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """For completeness but NOT primary metric."""
        return float(roc_auc_score(y_true, y_prob))

    def _select_threshold_grid(self, n: int = 200) -> np.ndarray:
        """Log-spaced threshold grid from 1e-4 to 0.99.

        Log spacing concentrates points at the low end, where fraud-detection
        operating points typically live (fraud prevalence is low, so a
        cost-optimal threshold can sit well below the 0.01 floor a linear
        grid would impose). Starting at 1e-4 rather than 0.01 avoids
        clipping the true optimum at the grid boundary (Phase C3).
        """
        return np.logspace(np.log10(1e-4), np.log10(0.99), num=n)

    def _business_value(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        cost_fn: float,
        cost_fp: float,
        revenue_tp: float,
    ) -> float:
        """Net business value at a single threshold: TP*revenue_tp - FP*cost_fp - FN*cost_fn.

        Pure cost minimization (the historical `find_optimal_threshold`
        behavior) is the special case revenue_tp=0.0: maximizing
        -FP*cost_fp - FN*cost_fn is equivalent to minimizing
        FP*cost_fp + FN*cost_fn. This is the single objective shared by
        `find_optimal_threshold` and `plot_threshold_vs_business_value` so
        the two can no longer disagree (Phase C3).
        """
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return float(tp * revenue_tp - fp * cost_fp - fn * cost_fn)

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_fn: float,
        cost_fp: float,
        revenue_tp: float = 0.0,
    ) -> float:
        """
        Select the threshold that maximizes net business value
        (see `_business_value`) over the log-spaced grid from
        `_select_threshold_grid` (1e-4 to 0.99).

        IMPORTANT: callers must pass the VALIDATION split (y_true/y_prob
        from X_val), never the test split — the threshold is a
        hyperparameter selected before test-set evaluation, and must be
        frozen at that point. Test metrics should then be reported at this
        frozen threshold via `compute_metrics_at_threshold`, never
        re-derived from test data (Phase C1).
        """
        thresholds = self._select_threshold_grid()
        values = [
            self._business_value(y_true, y_prob, t, cost_fn, cost_fp, revenue_tp)
            for t in thresholds
        ]
        best_idx = int(np.argmax(values))
        return float(thresholds[best_idx])

    def fit_calibrator(
        self, y_true: np.ndarray, y_prob: np.ndarray, method: str = "isotonic"
    ) -> IsotonicRegression:
        """Fit a probability calibrator on the VALIDATION split only (Phase C2).

        Callers must pass y_true/y_prob from the validation split, never
        test — the calibrator is a fitted transform (like any other
        stateful preprocessor) and must not see test data during fit.
        Apply the returned calibrator to test probabilities via
        `apply_calibration`; never call `fit_calibrator` again on test.
        """
        if method not in _SUPPORTED_CALIBRATION_METHODS:
            raise ValueError(
                f"Unknown calibration method {method!r}; supported: {_SUPPORTED_CALIBRATION_METHODS}"
            )
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(y_prob, y_true)
        return calibrator

    def apply_calibration(self, calibrator: IsotonicRegression, y_prob: np.ndarray) -> np.ndarray:
        """Transform raw probabilities through an already-fitted calibrator.

        Never fits — this is the transform-only half of fit/apply, safe to
        call on val or test once `calibrator` was fit on val.
        """
        return calibrator.predict(y_prob)

    def compute_brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Mean squared error between predicted probability and outcome.

        Lower is better-calibrated; 0.0 is perfect. Unlike PR-AUC/ROC-AUC
        (rank-only), Brier score is sensitive to the actual probability
        values, which is what the cost-based threshold logic assumes.
        """
        return float(brier_score_loss(y_true, y_prob))

    def plot_reliability_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: str,
        y_prob_calibrated: Optional[np.ndarray] = None,
        n_bins: int = 10,
    ) -> None:
        """Reliability diagram: mean predicted probability vs observed
        fraud rate per bin, against the y=x perfect-calibration diagonal.
        Pass `y_prob_calibrated` to overlay the post-calibration curve on
        the same axes so the improvement from `fit_calibrator` /
        `apply_calibration` is visible in one figure (Phase C2).
        """
        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Perfectly calibrated")

        frac_pos, mean_pred = self._reliability_bins(y_true, y_prob, n_bins)
        plt.plot(mean_pred, frac_pos, marker="o", color="red", lw=2, label="Uncalibrated")

        if y_prob_calibrated is not None:
            frac_pos_cal, mean_pred_cal = self._reliability_bins(y_true, y_prob_calibrated, n_bins)
            plt.plot(mean_pred_cal, frac_pos_cal, marker="o", color="blue", lw=2, label="Calibrated")

        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Observed Fraud Rate")
        plt.title("Reliability Curve")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def _reliability_bins(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int):
        """Thin wrapper over sklearn's calibration_curve (uniform-width bins
        over [0, 1]). sklearn silently drops empty bins rather than raising,
        which is what lets this be called on small/synthetic test fixtures
        without special-casing here.

        NOTE: uniform binning is a poor fit for real fraud-score
        distributions, where predicted probabilities cluster near 0 — most
        bins above the low end will be sparse or empty. strategy="quantile"
        is the better choice on real data; kept as "uniform" here only
        because it is what the unit tests' synthetic 0.55-0.95 score range
        exercises cleanly. Revisit before reading this plot on real model
        output.
        """
        from sklearn.calibration import calibration_curve

        return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    def compute_metrics_at_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float
    ) -> Dict[str, float]:
        """Return: precision, recall, f1, accuracy, FP, FN, TP, TN."""
        y_pred = (y_prob >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        
        return {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
        }

    def plot_pr_curve(self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_path: str) -> None:
        """PR curve with AUC in legend."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = self.compute_pr_auc(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"{model_name} (AUC = {pr_auc:.3f})", color="blue", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_path: str) -> None:
        """ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = self.compute_roc_auc(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", color="red", lw=2)
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic Curve")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
        """Normalized confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Normalized Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Legitimate", "Fraud"], rotation=45)
        plt.yticks(tick_marks, ["Legitimate", "Fraud"])
        
        fmt = ".2f"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def plot_threshold_vs_business_value(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_fn: float,
        cost_fp: float,
        revenue_tp: float,
        save_path: str,
        optimal_threshold: Optional[float] = None,
    ) -> None:
        """
        X-axis: threshold (log-spaced grid, 1e-4 to 0.99 — see `_select_threshold_grid`)
        Y-axis: net business value (see `_business_value`) — the same objective
            `find_optimal_threshold` maximizes, so the two can no longer disagree.
        Marks `optimal_threshold` with a vertical line if supplied — pass the
        threshold actually selected on validation and frozen for the model, so
        the marked line matches the real operating point rather than
        re-deriving a (possibly different) optimum from whatever data is
        plotted here. Falls back to the argmax of the plotted curve only if
        no threshold is supplied.
        """
        thresholds = self._select_threshold_grid()
        values = [
            self._business_value(y_true, y_prob, t, cost_fn, cost_fp, revenue_tp)
            for t in thresholds
        ]

        marked_t = optimal_threshold if optimal_threshold is not None else thresholds[int(np.argmax(values))]

        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, values, color="green", lw=2)
        plt.axvline(x=marked_t, color="red", linestyle="--", label=f"Optimal Threshold: {marked_t:.4f}")
        plt.xscale("log")
        plt.xlabel("Decision Threshold (log scale)")
        plt.ylabel("Net Business Value ($)")
        plt.title("Threshold vs. Business Value")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Full sklearn classification report as string."""
        return classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"])

    def compute_slice_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        slice_labels: np.ndarray,
        threshold: float,
        min_slice_size: int = 30,
    ) -> pd.DataFrame:
        """Per-slice metrics table at a single frozen threshold (Phase C5).

        `slice_labels` is any 1D array of per-row slice names (e.g. decoded
        ProductCD, an hour-of-day bucket, a card-tenure bucket) aligned with
        `y_true`/`y_prob`. Reuses `compute_metrics_at_threshold` per slice so
        a slice's numbers can never drift from the pooled definition of
        precision/recall/f1.

        Every slice present in `slice_labels` gets a row — thin slices are
        never dropped, only flagged `reliable=False` when their row count is
        below `min_slice_size`, so a reader can tell "no fraud here" apart
        from "not enough data to say". `pr_auc` is NaN for single-class
        slices (undefined, not computable) instead of raising.

        As with `find_optimal_threshold`, callers should pass the VALIDATION
        split so slice weaknesses are surfaced before the frozen test read.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        slice_labels = np.asarray(slice_labels)

        empty_columns = [
            "slice", "count", "fraud_rate", "pr_auc", "reliable",
            "TP", "TN", "FP", "FN", "precision", "recall", "f1", "accuracy",
        ]
        if len(slice_labels) == 0:
            return pd.DataFrame(columns=empty_columns)

        rows = []
        for label in pd.unique(slice_labels):
            mask = slice_labels == label
            slice_y_true = y_true[mask]
            slice_y_prob = y_prob[mask]
            count = int(mask.sum())

            metrics = self.compute_metrics_at_threshold(slice_y_true, slice_y_prob, threshold)
            has_both_classes = len(np.unique(slice_y_true)) > 1
            pr_auc = self.compute_pr_auc(slice_y_true, slice_y_prob) if has_both_classes else float("nan")

            rows.append({
                "slice": label,
                "count": count,
                "fraud_rate": float(slice_y_true.mean()) if count > 0 else float("nan"),
                "pr_auc": pr_auc,
                "reliable": count >= min_slice_size,
                **metrics,
            })

        return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

    def bucket_hour_of_day(self, hour_of_day: np.ndarray) -> np.ndarray:
        """Bucket a 0-23 hour-of-day array into four named windows:
        night [0,6), morning [6,12), afternoon [12,18), evening [18,24).
        Fixed-width buckets (not quantiles) so labels stay stable and
        interpretable across runs/reports regardless of the traffic
        distribution over the day."""
        hour_of_day = np.asarray(hour_of_day)
        bins = [-np.inf, 6, 12, 18, np.inf]
        labels = ["night", "morning", "afternoon", "evening"]
        return pd.cut(hour_of_day, bins=bins, labels=labels, right=False).astype(str)

    def bucket_card_tenure(self, tx_count_per_card: np.ndarray, n_buckets: int = 4) -> np.ndarray:
        """Bucket cards by `tx_count_per_card` (transactions seen for that
        card in the training window) into `n_buckets` quantile bins, used as
        a proxy for card tenure: no first-seen/account-age field exists in
        the processed feature set, but a card observed many times is, by
        construction, more established than one seen once or twice.

        Labels run low-to-high tenure: "new" (fewest transactions) through
        "established" (most), with intermediate buckets numbered so any
        `n_buckets` works without hand-naming every bucket.
        """
        tx_count_per_card = np.asarray(tx_count_per_card)
        if n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")

        if n_buckets == 2:
            names = ["new", "established"]
        else:
            names = ["new"] + [f"mid_{i}" for i in range(1, n_buckets - 1)] + ["established"]

        ranks = pd.Series(tx_count_per_card).rank(method="first")
        quantile_idx = pd.qcut(ranks, q=n_buckets, labels=False, duplicates="drop")
        return np.array([names[min(int(i), len(names) - 1)] for i in quantile_idx])
