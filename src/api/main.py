"""
src/api/main.py

FastAPI application factory with lifespan context manager.

Phase E (E3/E5): startup now builds the `ModelRegistry`, which is where
`FeatureEngineer.load_transformers()` gets its production call site — E3's
done-when. The registry validates that every artifact came from the same
training run and raises otherwise, so a misconfigured deployment fails to come
up rather than serving a mixed-vintage ensemble (ADR-001 §4.2).

Phase 6: the lifespan also starts the Kafka consumer as an asyncio background
task. It is handed the SAME `InferenceService` this module builds — one scoring
path for HTTP and the stream (ADR-001), one shared feature-state store whose
read-then-write / idempotency / RLock hardening (Phase E) already covers the
two writers. A broker that is unreachable at startup is logged and the API
still comes up serving HTTP; set `FRAUD_API_DISABLE_KAFKA=1` to skip it
entirely (tests, HTTP-only deployments).

Uses the modern `lifespan` context manager — NOT deprecated @app.on_event
handlers (per rules.md directive).
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI

from src.api.routes.predict import router as predict_router
from src.api.schemas.prediction import HealthResponse
from src.config import load_settings
from src.monitoring.prediction_log import PredictionLogWriter
from src.serving.feature_state import InMemoryFeatureStateStore
from src.serving.inference import InferenceService
from src.serving.metrics import ServingMetrics
from src.serving.prometheus_metrics import METRICS as PROMETHEUS_METRICS
from src.serving.registry import ModelRegistry
from src.streaming.consumer import FraudDetectionConsumer

logger = logging.getLogger(__name__)

# ── Application state ────────────────────────────────────────────────────────
_start_time: float = 0.0

# Set FRAUD_API_SKIP_MODEL_LOAD=1 to bring the app up without artifacts — used
# by tests that exercise routing and validation with an overridden service
# dependency. Deliberately opt-in: the default path fails loudly when artifacts
# are missing rather than starting a service that cannot score.
SKIP_MODEL_LOAD_ENV = "FRAUD_API_SKIP_MODEL_LOAD"

# Set FRAUD_API_DISABLE_KAFKA=1 to bring the app up without the streaming
# consumer — HTTP `/predict` is unaffected. Used by API tests (no broker) and
# by any deployment that wants HTTP-only serving.
DISABLE_KAFKA_ENV = "FRAUD_API_DISABLE_KAFKA"


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def _should_skip_model_load() -> bool:
    return _env_true(SKIP_MODEL_LOAD_ENV)


def _kafka_disabled() -> bool:
    return _env_true(DISABLE_KAFKA_ENV)


def build_inference_service(
    config: Optional[Dict[str, Any]] = None,
) -> InferenceService:
    """Load every artifact and assemble the scoring service.

    Propagates whatever `ModelRegistry.load()` raises — missing artifacts,
    checksum mismatches, or cross-artifact inconsistency. Startup failure is the
    correct outcome for all three.
    """
    config = config or load_settings().model_dump()

    metrics = ServingMetrics()
    registry = ModelRegistry(config, metrics=metrics)
    models = registry.load()

    # Seed the online store from the accumulators the batch run recorded, so
    # serving continues the card history the model was trained on instead of
    # cold-starting every card (ADR-002 §5.4).
    sequence_window = (
        config.get("model", {}).get("tft", {}).get("max_encoder_length", 10)
    )
    state_store = InMemoryFeatureStateStore(
        card_state=models.feature_engineer.card_agg_state,
        sequence_window=sequence_window,
        metrics=metrics,
    )

    # ADR-002 5.5: with no label feed the target-encoding state is only as
    # fresh as the artifact it was loaded from. Anchor the staleness gauge to
    # that artifact's mtime rather than to process start, which would reset the
    # age to zero on every restart and hide months of drift. Use the SAME
    # registry that loaded the transformers — a second `ModelRegistry(config)`
    # here could resolve `transformer_dir` differently (CWD, env) and silently
    # anchor the gauge to the wrong directory (ecc:mle-reviewer, P4-8).
    _mark_state_age(metrics, registry.transformer_dir)

    label_lag_days = config.get("features", {}).get(
        "target_encoding_label_lag_days", 0.0
    )
    return InferenceService(
        models=models,
        state_store=state_store,
        metrics=metrics,
        prometheus=PROMETHEUS_METRICS,
        sequence_window=sequence_window,
        label_lag_seconds=float(label_lag_days) * 86_400,
        explanation_timeout_ms=registry.explanation_timeout_ms,
        # Closes the E6 loop: `make monitor` reconstructs its drift window from
        # this file, which nothing wrote before Phase E.
        prediction_log=PredictionLogWriter(config["serving"]["log_file"]),
    )


def _mark_state_age(metrics: ServingMetrics, transformer_dir: Path) -> None:
    """Anchor the target-encoding staleness gauge to the artifact on disk."""
    state_file = Path(transformer_dir) / "feature_state.joblib"
    if state_file.exists():
        metrics.mark_target_encoding_state(state_file.stat().st_mtime)
    else:
        logger.warning(
            "No feature_state.joblib under %s — target-encoding staleness "
            "will be reported relative to process start, not the artifact.",
            transformer_dir,
        )


def _start_kafka_consumer(
    app: FastAPI, service: InferenceService, config: Dict[str, Any]
) -> None:
    """Construct the consumer over `service` and launch its async loop.

    A broker that is unreachable here is logged, not raised: HTTP scoring does
    not depend on the stream, and failing the whole container over an absent
    optional dependency is the wrong trade. `/health` then reports
    `kafka_consumer_running: false` so the gap is visible.
    """
    try:
        consumer = FraudDetectionConsumer(
            service, config, prometheus=PROMETHEUS_METRICS
        )
    except Exception as exc:  # noqa: BLE001 - broker down must not kill the API
        logger.error(
            "Kafka consumer not started (%s: %s). HTTP scoring is unaffected; "
            "/health will show kafka_consumer_running=false.",
            type(exc).__name__,
            exc,
        )
        return
    app.state.kafka_consumer = consumer
    app.state.kafka_consumer_task = asyncio.create_task(consumer.run())
    logger.info("Kafka consumer background task started")


async def _stop_kafka_consumer(app: FastAPI) -> None:
    consumer = getattr(app.state, "kafka_consumer", None)
    task = getattr(app.state, "kafka_consumer_task", None)
    if consumer is not None:
        consumer.stop()
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _configure_metrics_endpoint(application: FastAPI) -> None:
    """Attach `prometheus-fastapi-instrumentator` and expose `GET /metrics`.

    Imported here rather than at module top so the rest of the app (and the
    test suite's schema-only imports) do not hard-depend on the package being
    installed. A missing dependency is logged loudly — `/metrics` is a PRD
    done-when, not optional — but does not prevent the API from serving
    `/predict` and `/health`.
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:  # pragma: no cover - dependency is pinned in requirements
        logger.error(
            "prometheus-fastapi-instrumentator is not installed; GET /metrics "
            "will not be available. `pip install -r requirements.txt`."
        )
        return

    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/metrics", "/health"],
    ).instrument(application).expose(
        application, endpoint="/metrics", include_in_schema=True
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Modern FastAPI lifespan event handler.

    Startup:
      - Build the ModelRegistry: loads XGBoost + LightGBM + TFT, the fitted
        feature transformers, and the ensemble spec; validates them against
        each other (E3/E5).
      - Phase 6: start the Kafka consumer as an asyncio background task over
        the same InferenceService.

    Shutdown:
      - Phase 6: stop the Kafka consumer gracefully.
    """
    global _start_time
    _start_time = time.time()
    logger.info("fraud-api starting up...")

    app.state.inference_service = None
    app.state.kafka_consumer = None
    app.state.kafka_consumer_task = None
    if _should_skip_model_load():
        logger.warning(
            "%s is set — starting WITHOUT models. /predict will return 503 "
            "until a service is injected.",
            SKIP_MODEL_LOAD_ENV,
        )
    else:
        app.state.inference_service = build_inference_service()
        model_version = app.state.inference_service.models.model_version
        # PRD 5.7 `model_version_info`: publish the loaded bundle version as a
        # labelled gauge so Grafana's "Model Version" stat panel has a value
        # from the first scrape.
        PROMETHEUS_METRICS.set_model_version(model_version)
        logger.info("Models ready: model_version=%s", model_version)

    if app.state.inference_service is not None and not _kafka_disabled():
        _start_kafka_consumer(
            app, app.state.inference_service, load_settings().model_dump()
        )
    elif _kafka_disabled():
        logger.info("%s is set — Kafka consumer not started.", DISABLE_KAFKA_ENV)

    yield

    logger.info("fraud-api shutting down...")
    await _stop_kafka_consumer(app)


def create_app() -> FastAPI:
    """Application factory (per `.cursor/rules/python-fastapi.mdc`)."""
    application = FastAPI(
        title="Fraud Detection API",
        description="Explainable fraud detection with SHAP explanations",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(predict_router)

    # PRD FR-06 / Phase 5.7: `GET /metrics` in Prometheus text format. The
    # instrumentator adds `http_request_duration_seconds` (the P95 latency
    # panel) automatically; the custom `fraud_*` collectors from
    # `src.serving.prometheus_metrics` are already on the default registry this
    # endpoint serialises. `should_group_status_codes=False` keeps per-status
    # buckets so a 5xx spike is visible.
    _configure_metrics_endpoint(application)

    @application.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Liveness check for the Docker healthcheck and load-balancer probes.

        Reports the loaded `model_version` so a probe can distinguish a warm,
        correctly-versioned instance from one that merely answers.
        """
        service = getattr(application.state, "inference_service", None)
        store = getattr(service, "state_store", None) if service else None
        metrics = getattr(service, "metrics", None) if service else None
        consumer = getattr(application.state, "kafka_consumer", None)
        kafka = consumer.snapshot() if consumer is not None else {}
        return HealthResponse(
            status="healthy",
            model_loaded=service is not None,
            uptime_seconds=round(time.time() - _start_time, 1),
            model_version=service.models.model_version if service else None,
            models=sorted(service.models.trainers) if service else [],
            known_cards=store.known_cards() if hasattr(store, "known_cards") else None,
            kafka_consumer_running=kafka.get("kafka_consumer_running", False),
            kafka_messages_processed=kafka.get("kafka_messages_processed", 0),
            kafka_alerts_published=kafka.get("kafka_alerts_published", 0),
            kafka_consumer_errors=kafka.get("kafka_consumer_errors", 0),
            kafka_consumer_lag=kafka.get("kafka_consumer_lag", -1),
            observability=(
                metrics.snapshot()
                if metrics is not None
                else ServingMetrics().snapshot()
            ),
        )

    return application


app = create_app()
