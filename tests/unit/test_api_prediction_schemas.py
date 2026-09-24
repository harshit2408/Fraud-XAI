"""
tests/unit/test_api_prediction_schemas.py

Phase E task **E5**: "Pydantic request/response schemas with range and
staleness validation; include model version in the response" — done when
"invalid input rejected at the boundary".

"At the boundary" is the load-bearing phrase, so these tests assert more than
"a 422 comes back": they assert that a rejected request never reaches the
scoring path at all. A validation layer that rejects a payload *after* the
feature-state store has already been mutated would satisfy the status code and
still corrupt a card's expanding aggregates.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas.prediction import (
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.serving.staleness import (
    MAX_BACKDATE_SECONDS,
    StaleTransactionError,
    check_staleness,
)

VALID = {
    "TransactionID": "tx-0001",
    "TransactionDT": 100_000.0,
    "TransactionAmt": 59.99,
    "card1": 1234,
}


class TestRequestValidation:
    def test_a_well_formed_transaction_is_accepted(self):
        request = TransactionRequest(**VALID)
        assert request.TransactionID == "tx-0001"
        assert request.card1 == 1234

    @pytest.mark.parametrize(
        "field, value, why",
        [
            ("TransactionAmt", -1.0, "negative amount breaks log1p and ratios"),
            ("TransactionAmt", 0.0, "zero amount is not a real transaction"),
            ("TransactionAmt", float("nan"), "NaN passes > comparisons silently"),
            ("TransactionAmt", float("inf"), "infinity passes <= comparisons"),
            ("TransactionDT", -5.0, "negative offset predates the dataset epoch"),
            ("TransactionDT", float("nan"), "NaN timestamp breaks ordering"),
            ("TransactionID", "", "blank id defeats the idempotency guard"),
            ("TransactionID", "   ", "whitespace-only id likewise"),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field, value, why):
        with pytest.raises(Exception):
            TransactionRequest(**{**VALID, field: value})

    @pytest.mark.parametrize(
        "missing", ["TransactionID", "TransactionDT", "TransactionAmt", "card1"]
    )
    def test_required_fields_are_required(self, missing):
        payload = {k: v for k, v in VALID.items() if k != missing}
        with pytest.raises(Exception):
            TransactionRequest(**payload)

    def test_extra_ieee_cis_columns_pass_through(self):
        """V1-V339 and friends are not declared individually, but must survive
        to the transform — dropping them would silently zero real features."""
        request = TransactionRequest(**VALID, V1=0.5, C1=3.0, DeviceInfo="Windows")
        features = request.to_features()
        assert features["V1"] == 0.5
        assert features["C1"] == 3.0
        assert features["DeviceInfo"] == "Windows"

    def test_transaction_id_is_trimmed(self):
        trimmed = TransactionRequest(**{**VALID, "TransactionID": "  tx-9  "})
        assert trimmed.TransactionID == "tx-9"


class TestStalenessRule:
    def test_unseen_card_is_accepted(self):
        """Cold start is the normal path (ADR-002 §5.4), not an error."""
        check_staleness(transaction_dt=10.0, last_seen_dt=None)

    def test_transaction_at_or_after_last_seen_is_accepted(self):
        check_staleness(transaction_dt=1_000.0, last_seen_dt=1_000.0)
        check_staleness(transaction_dt=5_000.0, last_seen_dt=1_000.0)

    def test_slightly_out_of_order_is_tolerated(self):
        check_staleness(
            transaction_dt=1_000.0, last_seen_dt=1_000.0 + MAX_BACKDATE_SECONDS - 1
        )

    def test_materially_backdated_transaction_is_rejected(self):
        with pytest.raises(StaleTransactionError, match="behind this card"):
            check_staleness(
                transaction_dt=1_000.0,
                last_seen_dt=1_000.0 + MAX_BACKDATE_SECONDS + 1,
            )


class TestResponseContract:
    def test_response_carries_the_model_version(self):
        """E5's explicit requirement."""
        response = PredictionResponse(
            transaction_id="tx-1",
            fraud_probability=0.42,
            decision="FRAUD",
            threshold=0.006123,
            model_version="4c1059dcd5d4-24e774db15ac-f73b8af",
            mode="full",
            degraded=False,
        )
        assert response.model_version == "4c1059dcd5d4-24e774db15ac-f73b8af"
        assert response.model_dump()["model_version"]

    @pytest.mark.parametrize("bad_probability", [-0.01, 1.01])
    def test_probability_outside_zero_one_is_rejected(self, bad_probability):
        with pytest.raises(Exception):
            PredictionResponse(
                transaction_id="tx-1",
                fraud_probability=bad_probability,
                decision="FRAUD",
                threshold=0.5,
                model_version="v",
                mode="full",
                degraded=False,
            )

    def test_health_response_exposes_version_and_models(self):
        health = HealthResponse(
            status="healthy",
            model_loaded=True,
            uptime_seconds=12.5,
            model_version="v1",
            models=["lgbm", "tft", "xgb"],
            known_cards=5,
        )
        assert health.model_loaded and health.model_version == "v1"


class TestBoundaryRejectionReachesNothing:
    """The "at the boundary" half of E5's done-when."""

    @pytest.fixture()
    def client_and_spy(self, monkeypatch):
        monkeypatch.setenv("FRAUD_API_SKIP_MODEL_LOAD", "1")

        from src.api.main import create_app
        from src.api.routes.predict import get_inference_service

        calls = []

        class SpyService:
            def predict(self, transaction, mode=None):
                calls.append(transaction)
                raise AssertionError("scoring must not run for an invalid request")

        app = create_app()
        app.dependency_overrides[get_inference_service] = lambda: SpyService()
        with TestClient(app) as client:
            yield client, calls
        app.dependency_overrides.clear()

    @pytest.mark.parametrize(
        "payload",
        [
            {**VALID, "TransactionAmt": -1.0},
            {**VALID, "TransactionID": ""},
            {k: v for k, v in VALID.items() if k != "card1"},
        ],
    )
    def test_invalid_payload_never_reaches_the_service(self, client_and_spy, payload):
        client, calls = client_and_spy
        response = client.post("/predict", json=payload)

        assert response.status_code == 422
        assert calls == [], "an invalid request reached the scoring path"

    def test_health_reports_not_loaded_without_models(self, client_and_spy):
        client, _ = client_and_spy
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is False
        assert body["model_version"] is None
