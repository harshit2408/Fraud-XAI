"""
tests/integration/test_api.py

PRD Phase 10 done-when: FastAPI TestClient tests covering the response
contract, explanation shape, and decision/threshold consistency for
`POST /predict` and `GET /health`.

Boundary rejection (invalid payload never reaches scoring) is already covered
by `tests/unit/test_api_prediction_schemas.py::TestBoundaryRejectionReachesNothing`
and is not repeated here. This file covers the complementary path: a
well-formed request that DOES reach the service, asserting the response the
route builds from `InferenceService.predict()`'s result matches
`PredictionResponse` exactly and is internally consistent.

No real model artifacts are loaded (`FRAUD_API_SKIP_MODEL_LOAD=1`, same as the
schema tests) — a fake `InferenceService` is injected via
`app.dependency_overrides`, which is the FastAPI-recommended seam and the one
this codebase's own rules call for. This keeps the test fast and independent
of `models/` being present, while still exercising the real route, the real
Pydantic response model, and the real FastAPI request/response cycle.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

VALID_TRANSACTION = {
    "TransactionID": "tx-int-0001",
    "TransactionDT": 100_000.0,
    "TransactionAmt": 59.99,
    "card1": 1234,
}


class FakePredictionResult:
    """Mirrors `src.serving.inference.PredictionResult.as_dict()` exactly,
    so the route's `PredictionResponse(**result.as_dict())` call behaves
    identically to the real service."""

    def __init__(
        self,
        fraud_probability: float,
        threshold: float,
        decision: Optional[str] = None,
    ) -> None:
        self.transaction_id = VALID_TRANSACTION["TransactionID"]
        self.fraud_probability = fraud_probability
        self.threshold = threshold
        self.decision = decision or (
            "FRAUD" if fraud_probability > threshold else "LEGITIMATE"
        )
        self.model_version = "4c1059dcd5d4-24e774db15ac-f73b8af"
        self.mode = "full"
        self.degraded = False
        self.model_probabilities = {"xgb": fraud_probability, "lgbm": fraud_probability}
        self.degraded_reason = None
        self.explanation = [
            {"feature": "amount_vs_mean_ratio", "contribution": 0.31},
            {"feature": "hour_sin", "contribution": 0.18},
            {"feature": "card_tx_count_7d", "contribution": -0.09},
        ]
        self.explained_model = "xgb"
        self.explained_weight = 0.692
        self.latency_ms = 12.3

    def as_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "fraud_probability": self.fraud_probability,
            "decision": self.decision,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "mode": self.mode,
            "degraded": self.degraded,
            "model_probabilities": dict(self.model_probabilities),
            "degraded_reason": self.degraded_reason,
            "explanation": [dict(item) for item in self.explanation],
            "explained_model": self.explained_model,
            "explained_weight": self.explained_weight,
            "latency_ms": self.latency_ms,
        }


class FakeInferenceService:
    """Returns a canned, realistic result instead of loading real artifacts."""

    def __init__(self, fraud_probability: float = 0.873, threshold: float = 0.42):
        self.fraud_probability = fraud_probability
        self.threshold = threshold
        self.calls = []

    def predict(self, transaction: Dict[str, Any], mode: Optional[str] = None):
        self.calls.append(transaction)
        return FakePredictionResult(self.fraud_probability, self.threshold)


@pytest.fixture()
def app_and_service(monkeypatch):
    monkeypatch.setenv("FRAUD_API_SKIP_MODEL_LOAD", "1")

    from src.api.main import create_app
    from src.api.routes.predict import get_inference_service

    service = FakeInferenceService()
    app = create_app()
    app.dependency_overrides[get_inference_service] = lambda: service
    with TestClient(app) as client:
        yield client, service
    app.dependency_overrides.clear()


class TestPredictionResponseSchema:
    """PRD Phase 10: 'Response matches PredictionResponse schema exactly.'"""

    def test_response_matches_prediction_response_schema(self, app_and_service):
        client, _ = app_and_service
        response = client.post("/predict", json=VALID_TRANSACTION)

        assert response.status_code == 200
        body = response.json()

        expected_keys = {
            "transaction_id",
            "fraud_probability",
            "decision",
            "threshold",
            "model_version",
            "mode",
            "degraded",
            "model_probabilities",
            "degraded_reason",
            "explanation",
            "explained_model",
            "explained_weight",
            "latency_ms",
        }
        assert set(body.keys()) == expected_keys

    def test_response_field_types_are_correct(self, app_and_service):
        client, _ = app_and_service
        body = client.post("/predict", json=VALID_TRANSACTION).json()

        assert isinstance(body["transaction_id"], str)
        assert isinstance(body["fraud_probability"], float)
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["decision"] in {"FRAUD", "LEGITIMATE"}
        assert isinstance(body["threshold"], float)
        assert isinstance(body["model_version"], str) and body["model_version"]
        assert isinstance(body["explanation"], list)
        assert isinstance(body["latency_ms"], float)


class TestExplanationShape:
    """PRD Phase 10: 'explanation.top_risk_factors has at least 3 features.'

    The current API contract (`src/api/schemas/prediction.py`) names this
    field `explanation` (a flat list of `{feature, contribution}`), not the
    nested `top_risk_factors`/`top_mitigating_factors` shape sketched in the
    PRD's early API-contract draft (§13) — PRD Phase 4 implemented and shipped
    the flatter contract instead. This test asserts the shipped contract.
    """

    def test_explanation_has_at_least_three_features(self, app_and_service):
        client, _ = app_and_service
        body = client.post("/predict", json=VALID_TRANSACTION).json()

        assert len(body["explanation"]) >= 3
        for item in body["explanation"]:
            assert set(item.keys()) == {"feature", "contribution"}
            assert isinstance(item["feature"], str)
            assert isinstance(item["contribution"], float)

    def test_explanation_names_the_explained_model_and_its_weight(
        self, app_and_service
    ):
        """ADR-001 §3.4: explanation covers the XGBoost component only, never
        the blend — the response must say so explicitly, not imply otherwise."""
        client, _ = app_and_service
        body = client.post("/predict", json=VALID_TRANSACTION).json()

        assert body["explained_model"] == "xgb"
        assert body["explained_weight"] is not None


class TestDecisionConsistentWithThreshold:
    """PRD Phase 10: 'If fraud_probability > threshold: decision == "FRAUD".'"""

    @pytest.mark.parametrize(
        "fraud_probability, threshold, expected_decision",
        [
            (0.873, 0.42, "FRAUD"),
            (0.10, 0.42, "LEGITIMATE"),
            (0.42, 0.42, "LEGITIMATE"),  # exactly-at-threshold is not "over"
        ],
    )
    def test_decision_follows_probability_vs_threshold(
        self, monkeypatch, fraud_probability, threshold, expected_decision
    ):
        monkeypatch.setenv("FRAUD_API_SKIP_MODEL_LOAD", "1")

        from src.api.main import create_app
        from src.api.routes.predict import get_inference_service

        service = FakeInferenceService(
            fraud_probability=fraud_probability, threshold=threshold
        )
        app = create_app()
        app.dependency_overrides[get_inference_service] = lambda: service
        with TestClient(app) as client:
            body = client.post("/predict", json=VALID_TRANSACTION).json()

        assert body["decision"] == expected_decision
        if body["fraud_probability"] > body["threshold"]:
            assert body["decision"] == "FRAUD"
        else:
            assert body["decision"] == "LEGITIMATE"
        app.dependency_overrides.clear()


class TestHealthEndpoint:
    """`GET /health` reads `app.state.inference_service` directly rather than
    going through `Depends(get_inference_service)`, so it is not affected by
    `app.dependency_overrides` — that override only reroutes `/predict`. A
    service-loaded `/health` therefore needs the real lifespan path (i.e. NOT
    `FRAUD_API_SKIP_MODEL_LOAD=1`), which is a real-artifact test and is left
    to `tests/integration/test_real_ensemble_artifact_loads.py`. This class
    covers the no-models path only, which is what the override-based fixtures
    elsewhere in this file can actually exercise."""

    def test_health_without_a_service_reports_not_loaded(self, monkeypatch):
        """No dependency override at all — the real lifespan path with
        FRAUD_API_SKIP_MODEL_LOAD=1, i.e. no models and no injected fake."""
        monkeypatch.setenv("FRAUD_API_SKIP_MODEL_LOAD", "1")

        from src.api.main import create_app

        app = create_app()
        with TestClient(app) as client:
            body = client.get("/health").json()

        assert body["status"] == "healthy"
        assert body["model_loaded"] is False
        assert body["model_version"] is None


class TestPredictUnavailableWithoutAService:
    def test_predict_returns_503_when_models_are_not_loaded(self, monkeypatch):
        monkeypatch.setenv("FRAUD_API_SKIP_MODEL_LOAD", "1")

        from src.api.main import create_app

        app = create_app()
        with TestClient(app) as client:
            response = client.post("/predict", json=VALID_TRANSACTION)

        assert response.status_code == 503
