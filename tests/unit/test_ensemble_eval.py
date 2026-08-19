import numpy as np
import pytest

from scripts.run_ensemble_eval import predict_proba_prefer_calibrated


class _FakeTrainer:
    """Minimal double exercising only the surface
    `predict_proba_prefer_calibrated` depends on: a `calibrator` attribute
    plus `predict_proba`/`predict_proba_calibrated` methods. Avoids the cost
    and complexity of instantiating a real XGBTrainer/TFTTrainer for a test
    that only needs to verify the fallback-selection logic itself."""

    def __init__(self, calibrator, raise_from_predict_proba: Exception = None):
        self.calibrator = calibrator
        self._raise_from_predict_proba = raise_from_predict_proba
        self.calls = []

    def predict_proba(self, X, **kwargs):
        self.calls.append(("predict_proba", X, kwargs))
        if self._raise_from_predict_proba is not None:
            raise self._raise_from_predict_proba
        return np.full(len(X), 0.1)

    def predict_proba_calibrated(self, X, **kwargs):
        self.calls.append(("predict_proba_calibrated", X, kwargs))
        if self.calibrator is None:
            raise ValueError("No calibrator attached. Call set_calibrator() or load a calibrated artifact.")
        return np.full(len(X), 0.2)


def test_prefers_calibrated_probabilities_when_calibrator_present(caplog):
    trainer = _FakeTrainer(calibrator=object())
    X = [0, 1, 2]

    with caplog.at_level("WARNING"):
        result = predict_proba_prefer_calibrated(trainer, "XGBoost", X)

    np.testing.assert_array_equal(result, np.full(3, 0.2))
    assert trainer.calls == [("predict_proba_calibrated", X, {})]
    assert not any("no frozen calibrator" in r.message for r in caplog.records)


def test_falls_back_to_raw_probabilities_with_warning_when_no_calibrator(caplog):
    trainer = _FakeTrainer(calibrator=None)
    X = [0, 1, 2]

    with caplog.at_level("WARNING"):
        result = predict_proba_prefer_calibrated(trainer, "TFT", X)

    np.testing.assert_array_equal(result, np.full(3, 0.1))
    # Never calls predict_proba_calibrated at all — the calibrator=None check
    # short-circuits before that call, so it can't raise on the fallback path.
    assert trainer.calls == [("predict_proba", X, {})]
    assert any("no frozen calibrator" in r.message and "TFT" in r.message for r in caplog.records)


def test_unrelated_value_error_from_predict_proba_propagates_uncaught():
    """A schema-mismatch ValueError inside predict_proba (e.g. XGBTrainer's
    'Missing columns in input') must not be swallowed or misreported as
    'no calibrator' — it should propagate exactly as raised."""
    schema_error = ValueError("Missing columns in input: {'foo'}")
    trainer = _FakeTrainer(calibrator=None, raise_from_predict_proba=schema_error)
    X = [0, 1, 2]

    with pytest.raises(ValueError, match="Missing columns"):
        predict_proba_prefer_calibrated(trainer, "XGBoost", X)


def test_passes_through_kwargs_to_calibrated_predict():
    trainer = _FakeTrainer(calibrator=object())
    X = [0, 1]
    history_X = [9, 9]

    predict_proba_prefer_calibrated(trainer, "TFT", X, history_X=history_X)

    assert trainer.calls == [("predict_proba_calibrated", X, {"history_X": history_X})]
