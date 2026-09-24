"""Unit tests for scripts/run_ensemble_eval.py (the 3-way ensemble driver).

Rewritten for PRD Phase 9: the script was rebuilt from the stale 2-way
XGB+TFT version into a 3-way (XGB+TFT+LGBM) driver that writes
`models/ensemble.json`. The old `predict_proba_prefer_calibrated` helper
(raw-score fallback + warning) was dropped in favour of `_calibrated`, which
hard-fails when a calibrator is missing — matching
`scripts/export_test_probabilities.py`, because `models/ensemble.json`
declares `probability_space=per_model_calibrated` and silently blending raw
scores would put the frozen threshold on the wrong scale.
"""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.run_ensemble_eval import (
    _calibrated,
    _sequence_history,
    apply_min_lift_gate,
    build_spec,
)
from src.serving.ensemble_spec import parse_ensemble_spec


class _FakeTrainer:
    def __init__(self, calibrator, calibrated_value=0.2):
        self.calibrator = calibrator
        self._calibrated_value = calibrated_value
        self.calls = []

    def predict_proba(self, X, **kwargs):
        self.calls.append(("predict_proba", kwargs))
        return np.full(len(X), 0.1)

    def predict_proba_calibrated(self, X, **kwargs):
        self.calls.append(("predict_proba_calibrated", kwargs))
        return np.full(len(X), self._calibrated_value)


def test_calibrated_uses_the_calibrated_path_when_calibrator_present():
    trainer = _FakeTrainer(calibrator=object())
    out = _calibrated(trainer, "XGBoost", [0, 1, 2])
    np.testing.assert_array_equal(out, np.full(3, 0.2))
    assert trainer.calls == [("predict_proba_calibrated", {})]


def test_calibrated_hard_fails_when_no_calibrator():
    """No raw-score fallback — a missing calibrator is a hard error, so the
    3-way blend is never silently fitted on a mixed probability scale."""
    trainer = _FakeTrainer(calibrator=None)
    with pytest.raises(RuntimeError, match="no frozen calibrator"):
        _calibrated(trainer, "LightGBM", [0, 1, 2])
    assert trainer.calls == []


def test_calibrated_passes_kwargs_through():
    trainer = _FakeTrainer(calibrator=object())
    _calibrated(trainer, "TFT", [0, 1], history_X=[9, 9])
    assert trainer.calls == [("predict_proba_calibrated", {"history_X": [9, 9]})]


def test_sequence_history_trims_to_last_seqlen_minus_one_per_card():
    config = {"data": {"sequence_length": 3}}
    X_train = pd.DataFrame(
        {"card1": [1, 1, 1, 1, 2, 2], "TransactionAmt": [10, 11, 12, 13, 20, 21]}
    )
    hist = _sequence_history(X_train, X_train.iloc[:0], config)
    assert (hist.groupby("card1").size() == 2).all()
    assert sorted(hist[hist["card1"] == 1]["TransactionAmt"]) == [12, 13]


def test_sequence_history_no_group_col_returns_concat():
    config = {"data": {"sequence_length": 3}}
    a = pd.DataFrame({"x": [1, 2]})
    b = pd.DataFrame({"x": [3]})
    out = _sequence_history(a, b, config)
    assert list(out["x"]) == [1, 2, 3]


# ─── min-lift gate (the 9.2 decision mechanism) ────────────────────────────


@pytest.mark.parametrize(
    "two, three, min_lift, expect_keep, expect_lift",
    [
        (0.50, 0.51, 0.005, True, pytest.approx(0.01)),   # clear keep
        (0.50, 0.503, 0.005, False, pytest.approx(0.003)),  # below gate -> drop
        (0.50, 0.505, 0.005, True, pytest.approx(0.005)),   # boundary is inclusive
        (0.50, 0.49, 0.005, False, pytest.approx(-0.01)),   # 3-way worse -> drop
    ],
)
def test_apply_min_lift_gate(two, three, min_lift, expect_keep, expect_lift):
    keep, lift = apply_min_lift_gate(two, three, min_lift)
    assert keep is expect_keep
    assert lift == expect_lift


def test_min_lift_gate_call_site_uses_validation_not_test():
    """Regression guard for the 2026-09-09 fix: `apply_min_lift_gate` is a
    generic two-number comparator, so nothing in its own unit test can catch
    a call site that (re-)wires it to the test split. LightGBM's ensemble
    membership is a model-selection decision and must be made on validation;
    test values may only feed the post-hoc diagnostic, never the gate call.
    """
    import inspect

    import scripts.run_ensemble_eval as mod

    source = inspect.getsource(mod.evaluate_ensemble)
    gate_line = next(
        line for line in source.splitlines() if "apply_min_lift_gate(" in line
    )
    assert "val2, val3" in gate_line, (
        f"apply_min_lift_gate call site must pass validation PR-AUC, got: {gate_line!r}"
    )
    assert "test2, test3" not in gate_line


# ─── emitted models/ensemble.json ─────────────────────────────────────────


def test_build_spec_round_trips_through_ensemble_spec_validation():
    spec = build_spec(
        final_w={"xgb": 0.8, "tft": 0.2},
        threshold=0.0106,
        no_tft_threshold=0.02,
        dataset_hash="deadbeef" * 8,
        mlflow_run_id="abc123",
    )
    parsed = parse_ensemble_spec(json.loads(json.dumps(spec)))  # exercises validate()
    assert set(parsed.modes) == {"full", "no_tft"}
    assert parsed.default_mode == "full"
    assert parsed.mode("full").models == ["xgb", "tft"]
    assert parsed.mode("no_tft").models == ["xgb"]
    assert parsed.mode("no_tft").weights == {"xgb": 1.0}
    # both modes carry an in-range, distinct threshold
    assert 0.0 <= parsed.mode("full").threshold <= 1.0
    assert 0.0 <= parsed.mode("no_tft").threshold <= 1.0
    assert parsed.mode("full").threshold != parsed.mode("no_tft").threshold


def test_parse_rejects_an_incompatible_future_schema_version():
    """A spec from a newer major schema must be REFUSED, not silently
    downgraded.

    Without the check, a hypothetical 2.0 spec (one carrying, say, cascade
    stage-2 fields) parses as 1.0 with the unknown fields dropped — serving
    would then score a single stage at what was meant to be a first-stage
    GATE threshold, flagging roughly a third of all traffic as fraud while
    every health check still reported green. Every other integrity check in
    this module fails loud; this one used to read the field and ignore it.
    """
    spec = build_spec({"xgb": 0.8, "tft": 0.2}, 0.0106, 0.02, "b" * 64, None)
    spec["schema_version"] = "2.0"

    with pytest.raises(ValueError, match="schema_version"):
        parse_ensemble_spec(spec)


def test_parse_accepts_a_compatible_minor_schema_bump():
    """Minor versions stay loadable — the guard is on the MAJOR version, so an
    additive 1.x field does not lock out an otherwise-valid artifact."""
    spec = build_spec({"xgb": 0.8, "tft": 0.2}, 0.0106, 0.02, "c" * 64, None)
    spec["schema_version"] = "1.9"

    parsed = parse_ensemble_spec(spec)
    assert parsed.schema_version == "1.9"


def test_build_spec_two_model_full_still_carries_no_tft_fallback():
    """Even when the gate drops LightGBM (full = xgb+tft), the no_tft fallback
    must still be registered so serving can degrade past a TFT failure."""
    spec = build_spec(
        {"xgb": 0.874, "tft": 0.126}, 0.0106, 0.019, "a" * 64, None
    )
    parsed = parse_ensemble_spec(spec)
    assert "no_tft" in parsed.modes
    assert parsed.mlflow_run_id is None  # Optional — validation still passes


def test_build_spec_weights_sum_to_one():
    spec = build_spec({"xgb": 0.7, "tft": 0.1, "lgbm": 0.2}, 0.01, 0.02, "h" * 64, "r")
    for mode in spec["modes"].values():
        assert sum(mode["weights"].values()) == pytest.approx(1.0)
