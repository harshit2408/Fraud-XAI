"""
tests/unit/test_serving_hardening.py

Regression tests for the findings of the 2026-09-01 review pass
(`ecc:mle-reviewer`, `ecc:code-reviewer`, `ecc:python-reviewer`).

Every test here corresponds to a specific reported defect, grouped by the
reviewer that found it so a future reader can trace a test back to its
rationale.
"""

import logging
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.ensemble_spec import parse_ensemble_spec
from src.serving.feature_state import InMemoryFeatureStateStore
from src.serving.inference import (
    FALLBACK_MODE,
    InferenceService,
    ServingInvariantError,
)

FEATURES = ["TransactionAmt", "card1", "amount_log"]


# ── python-reviewer [HIGH]: unlocked read-then-write in the state store ──────


class TestStateStoreConcurrency:
    """`observe()` is a read-compute-write and `_claim()` a check-then-set.
    Individual dict ops are atomic under the GIL, but those *sequences* are
    not — and `/predict` is a sync route served from a threadpool with no
    per-card affinity, so ADR-002 §5.3's "Kafka partitions by card1, one
    writer" guarantee does not cover the HTTP path.

    These run with a near-zero GIL switch interval. Without the lock the
    unlocked versions failed 299/300 and 9/300 respectively; at the default
    interval they almost never fail, which is precisely what made the bug easy
    to miss and dangerous to ship: the aggregates are expanding, so a lost or
    doubled observation is permanent and never self-corrects.
    """

    @pytest.fixture(autouse=True)
    def _aggressive_gil(self):
        original = sys.getswitchinterval()
        sys.setswitchinterval(1e-9)
        yield
        sys.setswitchinterval(original)

    def test_concurrent_observes_on_one_card_lose_nothing(self):
        store = InMemoryFeatureStateStore()
        workers = 16
        barrier = threading.Barrier(workers)

        def observe(i: int) -> None:
            barrier.wait()
            store.observe(card_id=7, amount=10.0, dt=float(i), transaction_id=f"t{i}")

        threads = [threading.Thread(target=observe, args=(i,)) for i in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        state = store.snapshot(7)
        assert state.n == workers, f"lost {workers - state.n} observation(s)"
        assert state.sum_amt == pytest.approx(10.0 * workers)

    def test_only_one_thread_claims_a_repeated_transaction_id(self):
        """Kafka at-least-once redelivery is the motivating case."""
        store = InMemoryFeatureStateStore()
        workers = 16
        barrier = threading.Barrier(workers)
        claims = []

        def observe(_: int) -> None:
            barrier.wait()
            claims.append(
                store.observe(card_id=9, amount=5.0, dt=1.0, transaction_id="SAME")
            )

        threads = [threading.Thread(target=observe, args=(i,)) for i in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(claims) == 1, "the same transaction was folded in more than once"
        assert store.snapshot(9).n == 1

    def test_concurrent_appends_keep_the_window_bounded(self):
        store = InMemoryFeatureStateStore(sequence_window=10)
        barrier = threading.Barrier(16)

        def append(i: int) -> None:
            barrier.wait()
            store.append_sequence(3, np.array([float(i)], dtype=np.float32))

        threads = [threading.Thread(target=append, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(store.sequence(3)) == 10  # bounded, and not corrupted


# ── mle-reviewer [HIGH]: sequence buffer ignored the idempotency verdict ─────


class _StubTrainer:
    calibrator = "fitted"
    threshold = 0.5
    feature_names = FEATURES

    def predict_proba_calibrated(self, X, history_X=None):
        return np.array([0.2])


class _StubTransform:
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


def _spec(with_fallback: bool = True):
    modes = {
        "full": {"models": ["xgb"], "weights": {"xgb": 1.0}, "threshold": 0.5}
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


def _two_model_spec():
    """A blend whose fallback drops to a DIFFERENT model than the default.

    Needed so an invariant violation in the default mode could, if wrongly
    masked, be served successfully by the fallback — which is what makes the
    invariant tests discriminating rather than vacuous.
    """
    return parse_ensemble_spec(
        {
            "modes": {
                "full": {
                    "models": ["xgb", "lgbm"],
                    "weights": {"xgb": 0.7, "lgbm": 0.3},
                    "threshold": 0.5,
                },
                FALLBACK_MODE: {
                    "models": ["lgbm"],
                    "weights": {"lgbm": 1.0},
                    "threshold": 0.6,
                },
            },
            "default_mode": "full",
            "probability_space": "per_model_calibrated",
            "dataset_hash": "abc123",
        }
    )


@pytest.fixture()
def service(monkeypatch):
    import src.serving.inference as inference_module

    monkeypatch.setattr(inference_module, "ServingFeatureTransformer", _StubTransform)
    store = InMemoryFeatureStateStore(sequence_window=10)
    return InferenceService(
        models=_StubModels({"xgb": _StubTrainer()}, _spec()), state_store=store
    )


def _tx(tid: str, card1: int = 42, amount: float = 100.0, dt: float = 1000.0):
    return {
        "TransactionID": tid,
        "TransactionDT": dt,
        "TransactionAmt": amount,
        "card1": card1,
    }


class TestSequenceBufferHonoursIdempotency:
    """The scalar accumulators correctly ignored a redelivered transaction, but
    `append_sequence` ran unconditionally — so a duplicate pushed a second copy
    into the bounded ring buffer, evicting a real historical transaction and
    corrupting the window TFT scores against (ADR-002 §5.3)."""

    def test_duplicate_transaction_does_not_grow_the_sequence(self, service):
        service.predict(_tx("dup"))
        assert len(service.state_store.sequence(42)) == 1

        service.predict(_tx("dup"))  # same TransactionID, redelivered

        assert len(service.state_store.sequence(42)) == 1, (
            "a redelivered transaction was appended to the sequence window"
        )
        assert service.state_store.snapshot(42).n == 1

    def test_distinct_transactions_still_accumulate(self, service):
        for i in range(3):
            service.predict(_tx(f"t{i}", dt=1000.0 + i))

        assert len(service.state_store.sequence(42)) == 3
        assert service.state_store.snapshot(42).n == 3

    def test_accumulators_and_sequence_stay_in_lockstep(self, service):
        for tid in ["a", "b", "b", "c", "a"]:
            service.predict(_tx(tid, dt=1000.0))

        assert service.state_store.snapshot(42).n == len(
            service.state_store.sequence(42)
        )


# ── code-reviewer [MEDIUM]: broad except masked registry-invariant bugs ──────


class TestInvariantViolationsAreNeverDegraded:
    """`_score` raises for "model not loaded" / "no frozen calibrator" — both
    conditions startup already validated. If one fires per-request it is a bug
    in this process, so it must surface, not be downgraded to a 200 carrying
    `degraded: true`."""

    def test_missing_calibrator_raises_instead_of_falling_back(self, monkeypatch):
        import src.serving.inference as inference_module

        monkeypatch.setattr(
            inference_module, "ServingFeatureTransformer", _StubTransform
        )

        class _Uncalibrated:
            calibrator = None
            threshold = 0.5
            feature_names = FEATURES

        # The fallback uses a DIFFERENT, healthy model, so degradation would
        # genuinely succeed if the invariant were masked. Without that the test
        # passes either way (the fallback fails identically and the error
        # propagates regardless), proving nothing — verified by reverting the
        # fix and watching this test still pass.
        service = InferenceService(
            models=_StubModels(
                {"xgb": _Uncalibrated(), "lgbm": _StubTrainer()}, _two_model_spec()
            ),
            state_store=InMemoryFeatureStateStore(),
        )

        with pytest.raises(ServingInvariantError, match="no frozen calibrator"):
            service.predict(_tx("t1"))

    def test_model_missing_from_registry_raises(self, monkeypatch):
        import src.serving.inference as inference_module

        monkeypatch.setattr(
            inference_module, "ServingFeatureTransformer", _StubTransform
        )
        # Same construction: `lgbm` alone would serve the fallback happily.
        service = InferenceService(
            models=_StubModels({"lgbm": _StubTrainer()}, _two_model_spec()),
            state_store=InMemoryFeatureStateStore(),
        )

        with pytest.raises(ServingInvariantError, match="did not load"):
            service.predict(_tx("t1"))

    def test_a_genuine_model_failure_still_degrades(self, monkeypatch):
        """The fallback must remain available for real model-level failures."""
        import src.serving.inference as inference_module

        monkeypatch.setattr(
            inference_module, "ServingFeatureTransformer", _StubTransform
        )

        calls = {"n": 0}

        class _FlakyThenFine:
            calibrator = "fitted"
            threshold = 0.5
            feature_names = FEATURES

            def predict_proba_calibrated(self, X, history_X=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("transient CUDA failure")
                return np.array([0.3])

        service = InferenceService(
            models=_StubModels({"xgb": _FlakyThenFine()}, _spec()),
            state_store=InMemoryFeatureStateStore(),
        )

        result = service.predict(_tx("t1"))

        assert result.degraded is True
        assert result.mode == FALLBACK_MODE
        assert "RuntimeError" in (result.degraded_reason or "")


# ── code-reviewer [MEDIUM]: internal feature names leaked via 422 detail ─────


class TestErrorDetailDoesNotLeakInternals:
    @pytest.fixture()
    def client(self, monkeypatch):
        monkeypatch.setenv("FRAUD_API_SKIP_MODEL_LOAD", "1")
        from src.api.main import create_app
        from src.api.routes.predict import get_inference_service

        class _Exploding:
            def predict(self, transaction, mode=None):
                raise ValueError(
                    "Missing columns in transformed input: "
                    "['pca_v_3', 'amount_zscore_per_card', 'card_hash_freq']"
                )

        app = create_app()
        app.dependency_overrides[get_inference_service] = lambda: _Exploding()
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()

    def test_response_does_not_name_trained_features(self, client, caplog):
        payload = {
            "TransactionID": "tx-1",
            "TransactionDT": 100.0,
            "TransactionAmt": 10.0,
            "card1": 1,
        }
        with caplog.at_level(logging.ERROR):
            response = client.post("/predict", json=payload)

        assert response.status_code == 422
        body = response.text
        for internal in ("pca_v_3", "amount_zscore_per_card", "card_hash_freq"):
            assert internal not in body, f"leaked internal feature name {internal!r}"

        # ...but the operator still gets the full detail server-side.
        assert "pca_v_3" in caplog.text


# ── python-reviewer [LOW]: coercion silently NaN'd unparseable values ────────


class TestNumericCoercionFailsLoudly:
    def _transformer(self):
        from src.data.feature_engineering import FeatureEngineer
        from src.serving.transform import ServingFeatureTransformer

        fe = FeatureEngineer()
        fe._num_fill_values = {"dist2": -999.0}
        return fe, ServingFeatureTransformer(
            feature_engineer=fe, state_store=InMemoryFeatureStateStore()
        )

    def test_unparseable_value_in_a_numeric_column_is_rejected(self):
        fe, transformer = self._transformer()
        frame = pd.DataFrame([{"dist2": "not-a-number"}], dtype=object)

        with pytest.raises(ValueError, match="non-numeric value"):
            transformer._coerce_numeric_dtypes(fe, frame)

    def test_genuinely_null_column_still_coerces(self):
        fe, transformer = self._transformer()
        frame = pd.DataFrame([{"dist2": None}], dtype=object)

        out = transformer._coerce_numeric_dtypes(fe, frame)

        assert out["dist2"].dtype.kind == "f"
        assert out["dist2"].isna().all()


# ── mle-reviewer [MEDIUM/LOW]: ADR-promised observability was never built ────


class TestServingObservability:
    """ADR-002 5.3/5.5 commit to two signals that did not exist, and the
    mle-reviewer flagged three more gaps that are invisible rather than wrong:
    a config_hash mismatch only reachable by grepping logs, TFT scoring on a
    short window after every restart, and out-of-order transactions being
    clamped silently. All of them share a failure mode - the system keeps
    answering while quietly degrading - so each is now a counter or gauge on
    the health endpoint.
    """

    def test_health_exposes_the_serving_observability_block(self, monkeypatch):
        monkeypatch.setenv("FRAUD_API_SKIP_MODEL_LOAD", "1")
        from fastapi.testclient import TestClient

        from src.api.main import create_app

        with TestClient(create_app()) as client:
            body = client.get("/health").json()

        assert "observability" in body
        for key in (
            "cards_scored_with_partial_sequence",
            "out_of_order_transactions",
            "duplicate_transactions",
            "target_encoding_state_age_seconds",
            "config_hash_mismatch",
        ):
            assert key in body["observability"], f"missing signal: {key}"

    def test_partial_sequence_scoring_is_counted(self, service):
        """ADR-002 5.6: the sequence buffer is not persisted, so after a restart
        every card scores TFT on a short window until it refills. That is a real
        quality regression with no error attached, so it must be countable."""
        metrics = service.metrics

        service.predict(_tx("a", card1=1))
        assert metrics.snapshot()["cards_scored_with_partial_sequence"] == 1

        for i in range(12):
            service.predict(_tx(f"b{i}", card1=1, dt=1000.0 + i))

        # Once the window is full the counter must stop advancing.
        before = metrics.snapshot()["cards_scored_with_partial_sequence"]
        service.predict(_tx("c", card1=1, dt=2000.0))
        assert metrics.snapshot()["cards_scored_with_partial_sequence"] == before

    def test_duplicate_transactions_are_counted(self, service):
        service.predict(_tx("dup"))
        service.predict(_tx("dup"))

        assert service.metrics.snapshot()["duplicate_transactions"] == 1

    def test_out_of_order_clamps_are_counted(self):
        """ADR-002 5.3 promised this counter for the clamp path."""
        from src.serving.metrics import ServingMetrics

        metrics = ServingMetrics()
        store = InMemoryFeatureStateStore(metrics=metrics)

        store.observe(card_id=5, amount=10.0, dt=500.0, transaction_id="t1")
        store.observe(card_id=5, amount=10.0, dt=100.0, transaction_id="t2")

        assert metrics.snapshot()["out_of_order_transactions"] == 1

    def test_target_encoding_state_age_grows(self):
        """ADR-002 5.5: the encodings are frozen in this deployment, so their
        staleness must be visible rather than assumed away."""
        from src.serving.metrics import ServingMetrics

        metrics = ServingMetrics()
        first = metrics.snapshot()["target_encoding_state_age_seconds"]
        assert first >= 0.0

        metrics.mark_target_encoding_state(epoch_seconds=0.0)
        assert metrics.snapshot()["target_encoding_state_age_seconds"] > 1e6

    def test_config_hash_mismatch_is_surfaced_not_only_logged(self):
        """mle-reviewer LOW: a genuinely bad mixed-config deploy produced no
        operator-visible signal beyond a log line that may not be scraped."""
        from src.serving.metrics import ServingMetrics

        metrics = ServingMetrics()
        assert metrics.snapshot()["config_hash_mismatch"] is False

        metrics.mark_config_hash_mismatch({"xgb": "a", "lgbm": "b"})
        snapshot = metrics.snapshot()
        assert snapshot["config_hash_mismatch"] is True
        assert "lgbm" in str(snapshot["config_hash_detail"])

