"""
tests/unit/test_imbalance_handler.py

Phase 1 TDD — imbalance handler tests written BEFORE implementation.
Tests run entirely in-memory with synthetic data — no real data required.

Run: pytest tests/unit/test_imbalance_handler.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.imbalance_handler import ImbalanceHandler


@pytest.fixture()
def imbalanced_Xy() -> tuple:
    """
    Synthetic imbalanced dataset: 95% class 0, 5% class 1.
    300 rows, 5 numerical features.
    """
    rng = np.random.default_rng(42)
    n = 300
    y = (rng.random(n) < 0.05).astype(int)
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    return X, pd.Series(y, name="isFraud")


def test_scale_pos_weight_reflects_imbalance(imbalanced_Xy: tuple) -> None:
    """
    scale_pos_weight must equal neg_count / pos_count.
    For 95/5 split: expected ~19.0 (within ±2).
    """
    X, y = imbalanced_Xy
    handler = ImbalanceHandler(config={"imbalance": {"smote_k_neighbors": 5}})

    spw = handler.get_scale_pos_weight(y)

    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    expected = neg_count / pos_count

    assert abs(spw - expected) < 0.01, (
        f"scale_pos_weight={spw:.2f} does not match expected {expected:.2f}"
    )


def test_smote_increases_minority_class(imbalanced_Xy: tuple) -> None:
    """
    After SMOTE, minority class count must be greater than before.
    Total sample count must increase.
    """
    X, y = imbalanced_Xy
    handler = ImbalanceHandler(config={"imbalance": {"smote_k_neighbors": 5}})

    minority_before = (y == 1).sum()
    X_resampled, y_resampled = handler.apply_smote(X, y)
    minority_after = (y_resampled == 1).sum()

    assert minority_after > minority_before, (
        f"SMOTE did not increase minority class: before={minority_before}, after={minority_after}"
    )
    assert len(X_resampled) > len(X), (
        "Total sample count must increase after SMOTE oversampling"
    )


def test_smote_does_not_touch_test_set(imbalanced_Xy: tuple) -> None:
    """
    Critical: SMOTE applied only to X_train must NOT change X_test shape.
    Test set is a separate object and must remain identical.
    """
    X, y = imbalanced_Xy
    handler = ImbalanceHandler(config={"imbalance": {"smote_k_neighbors": 5}})

    # Simulate train/test split: first 240 train, last 60 test
    X_train, X_test = X.iloc[:240].copy(), X.iloc[240:].copy()
    y_train, y_test = y.iloc[:240].copy(), y.iloc[240:].copy()

    original_test_shape = X_test.shape
    original_test_values = X_test.copy()

    # Apply SMOTE only to train
    X_train_res, y_train_res = handler.apply_smote(X_train, y_train)

    # Test set must be completely unchanged
    assert X_test.shape == original_test_shape, (
        f"X_test shape changed after SMOTE: {original_test_shape} → {X_test.shape}"
    )
    pd.testing.assert_frame_equal(X_test, original_test_values), (
        "X_test values were modified after SMOTE — test set contamination!"
    )
