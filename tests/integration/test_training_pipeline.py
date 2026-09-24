"""
tests/integration/test_training_pipeline.py

PRD Phase 10 done-when: "End-to-end train on small subset."

Chains the real feature-engineering orchestration
(`src.data.preprocess._derive_causal_features` plus the same stateful-fit /
stateful-transform sequence `run_pipeline` uses) into a real `XGBTrainer` and
asserts the resulting model actually trains and predicts — no existing test
in the suite chains feature engineering all the way through to a trained,
scoring model. `tests/integration/test_train_serve_equivalence.py` covers the
feature-engineering half in isolation (batch vs. serving-path equivalence)
but stops short of training; this file starts where that one stops.

A small, seeded synthetic raw frame is used rather than the real
`data/raw/*.csv` — `DataLoader.load_raw()` requires ~590k-row Kaggle files
that are not guaranteed to be present in every environment this suite runs
in (they are `.gitignored`), and the PRD's own phrasing ("small subset") asks
for a fast, representative run, not a full-scale one. The synthetic schema
mirrors `test_train_serve_equivalence.py`'s `_raw_frame()` so both files
agree on what a "raw IEEE-CIS-shaped frame" looks like.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_engineering import FeatureEngineer
from src.data.preprocess import _derive_causal_features
from src.training.train_xgb import XGBTrainer

pytestmark = pytest.mark.integration

LABEL_LAG_SECONDS = 30 * 86400  # matches config/config.yaml's production value


def _raw_frame(n: int, seed: int) -> pd.DataFrame:
    """Same schema as `test_train_serve_equivalence.py::_raw_frame` — kept as
    its own copy rather than a shared import so this file's fixture can vary
    row count independently without coupling the two tests' row semantics."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionID": [f"tx{i:05d}" for i in range(n)],
            "TransactionDT": np.sort(rng.integers(0, 4_000_000, n)).astype(float),
            "TransactionAmt": rng.uniform(1.0, 900.0, n).round(2),
            "isFraud": (rng.random(n) < 0.15).astype(int),
            "card1": rng.integers(1000, 1010, n),
            "card2": rng.integers(100, 110, n),
            "card3": rng.integers(140, 152, n),
            "card5": rng.integers(100, 240, n),
            "addr1": rng.integers(200, 340, n),
            "dist1": rng.choice([np.nan, 1.0, 7.0, 33.0], n),
            "ProductCD": rng.choice(["W", "H", "C", "R"], n),
            "card4": rng.choice(["visa", "mastercard"], n),
            "card6": rng.choice(["debit", "credit"], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", np.nan], n),
            "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", np.nan], n),
            "DeviceType": rng.choice(["desktop", "mobile", np.nan], n),
            "DeviceInfo": rng.choice(["Windows", "iOS Device", "SM-G930V", np.nan], n),
            "id_31": rng.choice(["chrome 62.0", "safari 11.0", "mobile safari", np.nan], n),
            "id_33": rng.choice(["1920x1080", "2208x1242", np.nan], n),
            **{f"D{i}": rng.choice([np.nan, 10.0, 50.0, 200.0], n) for i in range(1, 16)},
            **{f"C{i}": rng.integers(0, 6, n).astype(float) for i in range(1, 15)},
            **{f"V{i}": rng.normal(0, 1, n) for i in range(1, 12)},
        }
    )


def _engineer_features(train_raw: pd.DataFrame, val_raw: pd.DataFrame):
    """The same orchestration `src.data.preprocess.run_pipeline` runs per
    split: null counts and PCA over the raw frame (fit on train), a temporal
    sort, the causal feature groups, then the stateful transforms (fit on
    train, transform-only on val)."""
    fe = FeatureEngineer()

    train_out = fe.create_null_count_features(train_raw)
    train_out = fe.reduce_v_features(train_out, fit=True, n_components=5)
    train_out = train_out.sort_values("TransactionDT").reset_index(drop=True)
    train_out = _derive_causal_features(fe, train_out)
    train_out = fe.create_card_hash_features(train_out, fit=True)
    train_out = fe.create_target_encoding(
        train_out, fit=True, update_state=True,
        time_col="TransactionDT", label_lag_seconds=LABEL_LAG_SECONDS,
    )
    train_out = fe.handle_missing_values(train_out, fit=True)
    train_out = fe.encode_categoricals(train_out, fit=True)

    val_out = fe.create_null_count_features(val_raw)
    val_out = fe.reduce_v_features(val_out, fit=False)
    val_out = val_out.sort_values("TransactionDT").reset_index(drop=True)
    val_out = _derive_causal_features(fe, val_out)
    val_out = fe.create_card_hash_features(val_out, fit=False)
    val_out = fe.create_target_encoding(
        val_out, fit=False, update_state=False,
        time_col="TransactionDT", label_lag_seconds=LABEL_LAG_SECONDS,
    )
    val_out = fe.handle_missing_values(val_out, fit=False)
    val_out = fe.encode_categoricals(val_out, fit=False)

    non_feature_cols = ["TransactionID", "TransactionDT", "isFraud"]
    X_train = train_out.drop(columns=non_feature_cols)
    y_train = train_out["isFraud"]
    X_val = val_out.drop(columns=non_feature_cols)
    y_val = val_out["isFraud"]
    return X_train, y_train, X_val, y_val


@pytest.fixture(scope="module")
def small_pipeline_config():
    """A minimal XGBoost config — few trees, shallow depth, CPU-pinned so the
    test is fast and deterministic regardless of the host's GPU."""
    return {
        "project": {"random_seed": 42},
        "model": {
            "xgboost": {
                "n_estimators": 15,
                "max_depth": 3,
                "learning_rate": 0.3,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "eval_metric": "aucpr",
                "early_stopping_rounds": 5,
                "device": "cpu",
            }
        },
    }


@pytest.fixture(scope="module")
def small_engineered_split():
    """A small, seeded raw frame split into train/val, run through the real
    feature-engineering orchestration. `scope="module"` — the pipeline is
    deterministic given its seed, so re-running it per test buys nothing."""
    raw = _raw_frame(n=400, seed=99)
    train_raw = raw.iloc[:300].copy()
    val_raw = raw.iloc[300:].copy().reset_index(drop=True)
    return _engineer_features(train_raw, val_raw)


class TestEndToEndTrainingOnSmallSubset:
    """Raw synthetic frame -> feature engineering -> XGBTrainer -> predictions,
    all through the real production code paths."""

    def test_pipeline_produces_a_trained_model_that_predicts(
        self, small_pipeline_config, small_engineered_split
    ):
        X_train, y_train, X_val, y_val = small_engineered_split
        assert len(X_train) > 0 and len(X_val) > 0
        assert set(y_train.unique()).issubset({0, 1})

        trainer = XGBTrainer(small_pipeline_config)
        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        trainer.build_model(scale_pos_weight=neg / max(pos, 1))
        trainer.train(X_train, y_train, X_val, y_val)

        assert trainer.model is not None
        assert trainer.feature_names == list(X_train.columns)

        val_proba = trainer.predict_proba(X_val)
        assert val_proba.shape == (len(X_val),)
        assert np.all((val_proba >= 0.0) & (val_proba <= 1.0))

    def test_trained_model_beats_predicting_the_base_rate_on_train(
        self, small_pipeline_config, small_engineered_split
    ):
        """A sanity floor, not a performance target: on the SAME small
        synthetic data the model trained on, it must rank better than a
        constant predictor. This catches a broken feature/label wiring (e.g.
        a shuffled label column) without asserting any specific PR-AUC on
        data too small and synthetic to support one."""
        X_train, y_train, X_val, y_val = small_engineered_split

        trainer = XGBTrainer(small_pipeline_config)
        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        trainer.build_model(scale_pos_weight=neg / max(pos, 1))
        trainer.train(X_train, y_train, X_val, y_val)

        from sklearn.metrics import average_precision_score

        train_proba = trainer.predict_proba(X_train)
        base_rate = y_train.mean()
        assert average_precision_score(y_train, train_proba) > base_rate

    def test_predict_proba_rejects_a_frame_missing_trained_columns(
        self, small_pipeline_config, small_engineered_split
    ):
        X_train, y_train, X_val, y_val = small_engineered_split

        trainer = XGBTrainer(small_pipeline_config)
        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        trainer.build_model(scale_pos_weight=neg / max(pos, 1))
        trainer.train(X_train, y_train, X_val, y_val)

        broken = X_val.drop(columns=[X_val.columns[0]])
        with pytest.raises(ValueError, match="Missing columns"):
            trainer.predict_proba(broken)
