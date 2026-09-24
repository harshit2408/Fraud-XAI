"""
src/api/routes/predict.py

HTTP adapter for the scoring path (Phase E task E5).

Deliberately thin, per `.cursor/rules/python-fastapi.mdc` ("keep routers thin"):
it validates, delegates to `InferenceService`, and maps domain errors onto
status codes. All orchestration lives in `src/serving/`, so the Phase 6 Kafka
consumer produces identical decisions without going through this module.

`def` rather than `async def` on the scoring endpoint is intentional. Scoring
is CPU-bound (a feature transform plus three model forward passes) with no
awaitable I/O; declaring it `async` would run it directly on the event loop and
block every other in-flight request. FastAPI runs a sync handler in its
threadpool instead, which is the correct execution model here.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.schemas.prediction import (
    ErrorResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.serving.inference import InferenceService
from src.serving.staleness import StaleTransactionError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])


def get_inference_service(request: Request) -> InferenceService:
    """Resolve the process-wide service built during lifespan startup.

    A dependency (rather than a module global) so tests can override exactly
    this callable via `app.dependency_overrides`, as the FastAPI rules require.
    """
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded; the service is not ready to score.",
        )
    return service


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        409: {
            "model": ErrorResponse,
            "description": "Stale or out-of-order transaction",
        },
        422: {"model": ErrorResponse, "description": "Schema validation failed"},
        503: {"model": ErrorResponse, "description": "Models not loaded"},
    },
    summary="Score one transaction for fraud",
)
def predict(
    transaction: TransactionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Score a transaction and return the decision with its provenance.

    Schema violations are rejected by FastAPI before this body runs, so an
    invalid payload never reaches the transform or the feature-state store.
    """
    try:
        result = service.predict(transaction.to_features())
    except StaleTransactionError as exc:
        # 409, not 422: the payload is well-formed, but it conflicts with the
        # card history already recorded.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except KeyError as exc:
        # Reachable once `mode` becomes client-supplied; today the default mode
        # is guaranteed to exist by EnsembleSpec.validate() at load time.
        logger.warning("Unservable ensemble mode requested: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Requested scoring mode is not available.",
        ) from exc
    except ValueError as exc:
        # The full message names internal feature-engineering columns (e.g.
        # pca_v_3, amount_zscore_per_card). Logged server-side, never returned:
        # ErrorResponse is documented as carrying no internal detail, and the
        # message would otherwise let a caller enumerate the trained feature set.
        logger.exception(
            "Scoring failed for transaction %s", transaction.TransactionID
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Transaction could not be scored; it is missing required fields.",
        ) from exc

    return PredictionResponse(**result.as_dict())
