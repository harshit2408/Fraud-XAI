"""
tests/unit/test_tft_boundary_history.py

TDD for Phase B6 — sequences must be built once over the full temporally
ordered frame and then assigned to splits by original row index, so that a
card's first transaction in val/test retains its real pre-split history
instead of being truncated at the split boundary.

File: docs/IMPLEMENTATION_PLAN.md Phase B6, src/training/train_tft.py:249-250
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_tft import TFTTrainer


def _minimal_config() -> dict:
    return {
        "project": {"random_seed": 42},
        "data": {"sequence_length": 3},
        "model": {"tft": {"device": "cpu"}},
        "imbalance": {"sampling_strategy": "none", "loss_function": "bce"},
    }


def _card_frame(card_id: int, n: int, start_amt: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "card1": [card_id] * n,
        "TransactionAmt": start_amt + np.arange(n, dtype=float),
    })


class TestBoundaryHistoryPreserved:
    def test_val_first_row_sequence_includes_train_history_for_same_card(self):
        trainer = TFTTrainer(_minimal_config())

        # Card 1 has 5 tx in train, then continues with 2 more tx in val.
        X_train = _card_frame(1, 5, start_amt=0.0)
        y_train = pd.Series([0, 0, 0, 0, 0])
        X_val = _card_frame(1, 2, start_amt=100.0)
        y_val = pd.Series([0, 1])

        splits = trainer._build_sequences_for_splits([
            ("train", X_train, y_train),
            ("val", X_val, y_val),
        ])

        val_seq = splits["val"]
        first_val_mask = val_seq["mask"][0]
        assert first_val_mask.sum() == 3.0, (
            "val's first sequence should be fully populated from train history "
            f"(sequence_length=3), got mask={first_val_mask}"
        )

    def test_without_boundary_fix_first_val_row_would_be_padded(self):
        """Sanity check: building val in isolation (the pre-B6 behavior)
        pads the first row, proving the boundary fix changes real behavior."""
        trainer = TFTTrainer(_minimal_config())
        X_val = _card_frame(1, 2, start_amt=100.0)
        y_val = pd.Series([0, 1])

        isolated = trainer._build_sequences(X_val, y_val, "val")
        assert isolated["mask"][0].sum() == 1.0, (
            "expected the isolated (no-history) build to pad the first row"
        )

    def test_split_sequence_counts_match_input_row_counts(self):
        trainer = TFTTrainer(_minimal_config())
        X_train = _card_frame(1, 5)
        y_train = pd.Series([0] * 5)
        X_val = _card_frame(1, 3, start_amt=100.0)
        y_val = pd.Series([0, 0, 1])

        splits = trainer._build_sequences_for_splits([
            ("train", X_train, y_train),
            ("val", X_val, y_val),
        ])

        assert len(splits["train"]["targets"]) == 5
        assert len(splits["val"]["targets"]) == 3

    def test_targets_correctly_reassigned_per_split(self):
        trainer = TFTTrainer(_minimal_config())
        X_train = _card_frame(1, 4)
        y_train = pd.Series([0, 0, 0, 1])
        X_val = _card_frame(1, 2, start_amt=100.0)
        y_val = pd.Series([1, 0])

        splits = trainer._build_sequences_for_splits([
            ("train", X_train, y_train),
            ("val", X_val, y_val),
        ])

        np.testing.assert_array_equal(splits["train"]["targets"], [0, 0, 0, 1])
        np.testing.assert_array_equal(splits["val"]["targets"], [1, 0])

    def test_three_way_split_preserves_boundary_history_at_each_join(self):
        trainer = TFTTrainer(_minimal_config())
        X_train = _card_frame(1, 4, start_amt=0.0)
        y_train = pd.Series([0, 0, 0, 0])
        X_val = _card_frame(1, 2, start_amt=100.0)
        y_val = pd.Series([0, 0])
        X_test = _card_frame(1, 2, start_amt=200.0)
        y_test = pd.Series([0, 1])

        splits = trainer._build_sequences_for_splits([
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ])

        assert splits["test"]["mask"][0].sum() == 3.0, (
            "test's first sequence should pull real history from val/train"
        )
        assert len(splits["train"]["targets"]) == 4
        assert len(splits["val"]["targets"]) == 2
        assert len(splits["test"]["targets"]) == 2


# ── Phase B7 TDD — TFTTrainer.predict_proba(X) must not require labels ─────────


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "card1": [1] * n,
        "TransactionAmt": np.arange(n, dtype=float),
    })


class TestPredictProbaSignature:
    def test_predict_proba_has_no_required_label_parameter(self):
        import inspect

        sig = inspect.signature(TFTTrainer.predict_proba)
        required = [
            p.name for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty and p.name != "self"
        ]
        assert required == ["X"], (
            f"predict_proba must take X alone (optional kwargs only); got required={required}"
        )

    def test_predict_proba_runs_without_labels(self):
        trainer = TFTTrainer(_minimal_config())
        trainer.build_model(num_numeric_features=1, num_static_features=0)

        X = _frame(6)
        preds = trainer.predict_proba(X)

        assert len(preds) == len(X)
        assert np.isfinite(preds).all()
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_predict_proba_with_history_returns_one_pred_per_input_row(self):
        trainer = TFTTrainer(_minimal_config())
        trainer.build_model(num_numeric_features=1, num_static_features=0)

        history_X = _frame(5)
        X = _frame(2)
        X["TransactionAmt"] += 100

        preds = trainer.predict_proba(X, history_X=history_X)
        assert len(preds) == len(X)
        assert np.isfinite(preds).all()

    def test_predict_proba_history_produces_different_output_than_no_history(self):
        """With sequence_length=3, predicting the first row of X with no
        history pads 2/3 of the window; with history_X supplied, the window
        is fully populated from real prior transactions — the two must not
        be identical (guards against history_X being silently ignored)."""
        trainer = TFTTrainer(_minimal_config())
        trainer.build_model(num_numeric_features=1, num_static_features=0)
        trainer.model.eval()

        X = _frame(1)
        X["TransactionAmt"] = 999.0
        history_X = _card_frame(1, 5, start_amt=0.0)

        pred_no_history = trainer.predict_proba(X)
        pred_with_history = trainer.predict_proba(X, history_X=history_X)

        assert not np.allclose(pred_no_history, pred_with_history), (
            "history_X must change the built sequence (and thus the prediction), "
            "not be silently ignored"
        )
