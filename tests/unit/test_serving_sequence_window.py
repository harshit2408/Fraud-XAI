"""
tests/unit/test_serving_sequence_window.py

Regression tests for a CRITICAL gap found by the mle-reviewer pass on Phase E.

`FeatureStateStore.sequence()` / `append_sequence()` were defined per ADR-002
§5.6 but never called from `InferenceService`. TFT is sequential — it was
trained on a window of each card's last `max_encoder_length` transactions — so
scoring it on the served row alone fed it a length-1 sequence with every other
position masked. For any card with real history that is silently
out-of-distribution: no exception, no `degraded` flag, and the 13.4% TFT
component of the blend quietly wrong on every warm card.

These tests pin the WIRING rather than the numeric outcome, because the numeric
outcome is not a reliable signal here. Verified against the real artifacts
during development: the sequence window genuinely changes TFT's **raw**
probability (0.005952 -> 0.002775 for the same row given six prior
transactions), but the frozen isotonic calibrator mapped both raw values onto
the same calibrated output — isotonic regression is a step function and both
landed on one step. Asserting on calibrated output would therefore assert on a
property of the calibrator's step boundaries, not on whether history reached
the model, and could pass while the wiring was severed again.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.ensemble_spec import parse_ensemble_spec
from src.serving.feature_state import InMemoryFeatureStateStore
from src.serving.inference import FALLBACK_MODE, InferenceService

FEATURES = ["TransactionAmt", "card1", "amount_log"]


def _spec(with_fallback: bool = True):
    modes = {
        "full": {
            "models": ["xgb", "tft"],
            "weights": {"xgb": 0.8, "tft": 0.2},
            "threshold": 0.5,
        }
    }
    if with_fallback:
        modes[FALLBACK_MODE] = {
            "models": ["xgb"],
            "weights": {"xgb": 1.0},
            "threshold": 0.6,
        }
    return parse_ensemble_spec(
        {
            "modes": modes,
            "default_mode": "full",
            "probability_space": "per_model_calibrated",
            "dataset_hash": "abc123",
        }
    )


class _RecordingTFT:
    """Records the `history_X` it was handed, so the test asserts on the
    contract rather than on a model's numeric output."""

    calibrator = "fitted"
    threshold = 0.5

    def __init__(self):
        self.history_seen = []

    def predict_proba_calibrated(self, X, history_X=None):
        self.history_seen.append(None if history_X is None else len(history_X))
        return np.array([0.4])


class _StubXGB:
    calibrator = "fitted"
    threshold = 0.5
    feature_names = FEATURES

    def predict_proba_calibrated(self, X, history_X=None):
        return np.array([0.1])


class _ExplodingTFT:
    calibrator = "fitted"
    threshold = 0.5

    def predict_proba_calibrated(self, X, history_X=None):
        raise RuntimeError("TFT forward pass failed")


class _StubTransform:
    """Stands in for ServingFeatureTransformer so these tests stay unit-level."""

    def __init__(
        self,
        feature_engineer=None,
        state_store=None,
        feature_names=None,
        label_lag_seconds=0.0,
    ):
        self.feature_names = feature_names or FEATURES

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        row = raw.iloc[0]
        return pd.DataFrame(
            [[float(row["TransactionAmt"]), float(row["card1"]), 1.0]],
            columns=self.feature_names,
        )


class _StubModels:
    def __init__(self, trainers, spec):
        self.trainers = trainers
        self.ensemble = spec
        self.feature_engineer = None
        self.feature_names = FEATURES
        self.model_version = "ds-cfg-git"


@pytest.fixture()
def service_factory(monkeypatch):
    import src.serving.inference as inference_module

    monkeypatch.setattr(inference_module, "ServingFeatureTransformer", _StubTransform)

    def build(trainers, spec=None, card_state=None):
        store = InMemoryFeatureStateStore(card_state=card_state, sequence_window=10)
        return InferenceService(
            models=_StubModels(trainers, spec or _spec()), state_store=store
        )

    return build


def _tx(transaction_id: str, card1: int = 42, amount: float = 100.0, dt: float = 1000.0):
    return {
        "TransactionID": transaction_id,
        "TransactionDT": dt,
        "TransactionAmt": amount,
        "card1": card1,
    }


class TestSequenceWindowIsWired:
    def test_first_transaction_gets_no_history(self, service_factory):
        """Cold start: nothing stored yet, so TFT takes the zero-padded path —
        exactly what training saw for a card's first transactions."""
        tft = _RecordingTFT()
        service = service_factory({"xgb": _StubXGB(), "tft": tft})

        service.predict(_tx("t1"))

        assert tft.history_seen == [None]

    def test_subsequent_transactions_replay_the_stored_window(self, service_factory):
        """The bug: this used to stay None forever, so TFT never saw history."""
        tft = _RecordingTFT()
        service = service_factory({"xgb": _StubXGB(), "tft": tft})

        for i in range(4):
            service.predict(_tx(f"t{i}", dt=1000.0 + i))

        assert tft.history_seen == [None, 1, 2, 3], (
            "TFT did not receive a growing per-card history window"
        )

    def test_window_is_capped_at_the_encoder_length(self, service_factory):
        tft = _RecordingTFT()
        service = service_factory({"xgb": _StubXGB(), "tft": tft})

        for i in range(15):
            service.predict(_tx(f"t{i}", dt=1000.0 + i))

        assert max(h for h in tft.history_seen if h is not None) == 10

    def test_history_is_per_card_not_global(self, service_factory):
        """A second card must not inherit the first card's sequence."""
        tft = _RecordingTFT()
        service = service_factory({"xgb": _StubXGB(), "tft": tft})

        service.predict(_tx("a1", card1=1))
        service.predict(_tx("a2", card1=1, dt=1001.0))
        tft.history_seen.clear()
        service.predict(_tx("b1", card1=999))

        assert tft.history_seen == [None], "card 999 inherited another card's history"

    def test_stored_vector_matches_the_scored_features(self, service_factory):
        """What is replayed must be the transformed vector, not the raw payload."""
        service = service_factory({"xgb": _StubXGB(), "tft": _RecordingTFT()})
        service.predict(_tx("t1", amount=250.0))

        window = service.state_store.sequence(42)
        assert len(window) == 1
        np.testing.assert_array_equal(
            window[0], np.array([250.0, 42.0, 1.0], dtype=np.float32)
        )


class TestDegradationFallback:
    """ADR-001 §4.6: fall back only to a mode carrying its own frozen
    threshold — never by renormalizing weights."""

    def test_tft_failure_falls_back_to_the_registered_mode(self, service_factory):
        service = service_factory({"xgb": _StubXGB(), "tft": _ExplodingTFT()})

        result = service.predict(_tx("t1"))

        assert result.mode == FALLBACK_MODE
        assert result.degraded is True
        assert "RuntimeError" in (result.degraded_reason or "")
        # The fallback's OWN threshold, not the full blend's.
        assert result.threshold == pytest.approx(0.6)
        assert set(result.model_probabilities) == {"xgb"}

    def test_without_a_registered_fallback_the_error_propagates(self, service_factory):
        """Better to fail loudly than to invent an operating point."""
        service = service_factory(
            {"xgb": _StubXGB(), "tft": _ExplodingTFT()}, spec=_spec(with_fallback=False)
        )

        with pytest.raises(RuntimeError, match="TFT forward pass failed"):
            service.predict(_tx("t1"))

    def test_an_explicitly_requested_mode_is_never_second_guessed(self, service_factory):
        service = service_factory({"xgb": _StubXGB(), "tft": _ExplodingTFT()})

        with pytest.raises(RuntimeError):
            service.predict(_tx("t1"), mode="full")

    def test_healthy_path_is_not_marked_degraded(self, service_factory):
        service = service_factory({"xgb": _StubXGB(), "tft": _RecordingTFT()})

        result = service.predict(_tx("t1"))

        assert result.mode == "full"
        assert result.degraded is False
        assert result.degraded_reason is None
