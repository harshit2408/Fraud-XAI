"""
src/data/imbalance_handler.py

Class imbalance mitigation strategies for the fraud detection training pipeline.

The IEEE-CIS dataset has ~3.5% fraud rate. Without correction:
  - XGBoost will predict "not fraud" for everything
  - Achieves 96.5% accuracy but 0% recall — completely useless

Three strategies are supported (from config.imbalance.strategy):
  1. 'smote'         — Synthetic Minority Oversampling Technique
  2. 'class_weight'  — scale_pos_weight for XGBoost
  3. 'focal_loss'    — handled in TFT training, not here

CRITICAL: SMOTE is applied ONLY to training data. Test set is never touched.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


class ImbalanceHandler:
    """
    Applies class imbalance mitigation strategies.

    Designed to be initialized once with the loaded config dict,
    then called with training data.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Loaded config.yaml dict. Reads from config['imbalance'].
        """
        self.config = config
        imbalance_cfg = config.get("imbalance", {})
        self.smote_k_neighbors: int = imbalance_cfg.get("smote_k_neighbors", 5)

    def apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE oversampling to the training set only.

        SMOTE generates synthetic minority-class examples by interpolating
        between existing minority examples in feature space. It is far better
        than random oversampling because it adds information rather than
        duplicating existing rows.

        CRITICAL: Call this ONLY on training data. Never on test or validation.

        Args:
            X_train: Feature DataFrame (training split only).
            y_train: Target Series (training split only).

        Returns:
            Tuple of (X_resampled, y_resampled) with balanced classes.
        """
        minority_before = int((y_train == 1).sum())
        majority_before = int((y_train == 0).sum())
        logger.info(
            f"SMOTE input: {majority_before:,} majority / {minority_before:,} minority "
            f"({minority_before / len(y_train) * 100:.2f}% fraud)"
        )

        smote = SMOTE(
            k_neighbors=self.smote_k_neighbors,
            random_state=42,
        )

        # SMOTE requires numpy arrays; preserve column names after
        X_arr, y_arr = smote.fit_resample(X_train.values, y_train.values)

        X_resampled = pd.DataFrame(X_arr, columns=X_train.columns)
        y_resampled = pd.Series(y_arr, name=y_train.name)

        minority_after = int((y_resampled == 1).sum())
        majority_after = int((y_resampled == 0).sum())
        logger.info(
            f"SMOTE output: {majority_after:,} majority / {minority_after:,} minority "
            f"({minority_after / len(y_resampled) * 100:.2f}% fraud) "
            f"[+{minority_after - minority_before:,} synthetic examples]"
        )

        return X_resampled, y_resampled

    def get_scale_pos_weight(self, y_train: pd.Series) -> float:
        """
        Compute XGBoost's scale_pos_weight parameter.

        XGBoost uses this to up-weight the positive (fraud) class during training.
        Formula: neg_count / pos_count

        For 3.5% fraud rate: scale_pos_weight ≈ 28.6

        Args:
            y_train: Binary target Series from training split.

        Returns:
            Float weight to pass to XGBClassifier(scale_pos_weight=...).
        """
        neg_count = float((y_train == 0).sum())
        pos_count = float((y_train == 1).sum())

        if pos_count == 0:
            raise ValueError("No positive examples in y_train — cannot compute scale_pos_weight.")

        weight = neg_count / pos_count
        logger.info(
            f"scale_pos_weight: {neg_count:.0f} neg / {pos_count:.0f} pos = {weight:.2f}"
        )
        return weight

    def get_class_weights(self, y_train: pd.Series) -> Dict[int, float]:
        """
        Compute sklearn-style class weight dict for neural model training.

        Returns weights inversely proportional to class frequency.
        Example: {0: 0.52, 1: 14.8} for 3.5% fraud rate.

        Args:
            y_train: Binary target Series from training split.

        Returns:
            Dict mapping class label (0/1) to float weight.
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.array([0, 1])
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train.values,
        )
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        logger.info(f"Class weights: {class_weight_dict}")
        return class_weight_dict
