"""
src/api/main.py

FastAPI application factory with lifespan context manager.

Phase 0 stub: Provides /health endpoint so Docker healthcheck passes.
Full implementation in Phase 5 (model loading, /predict, /metrics, Kafka consumer).

Uses modern `lifespan` context manager — NOT deprecated @app.on_event handlers
(per rules.md directive).
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# ── Application state ────────────────────────────────────────────────────────
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Modern FastAPI lifespan event handler.

    Startup:
      - Phase 5: Load XGBoost + TFT models, SHAP explainer, feature transformers
      - Phase 6: Start Kafka consumer as asyncio background task

    Shutdown:
      - Phase 6: Stop Kafka consumer gracefully
    """
    global _start_time
    _start_time = time.time()
    logger.info("fraud-api starting up...")

    # TODO Phase 5: Load models into ModelRegistry
    # TODO Phase 6: Start Kafka consumer background task

    yield

    # Shutdown
    logger.info("fraud-api shutting down...")
    # TODO Phase 6: Stop Kafka consumer


app = FastAPI(
    title="Fraud Detection API",
    description="Explainable fraud detection with SHAP explanations",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Liveness check for Docker healthcheck and load balancer probes.

    Returns:
        Health status with uptime. Extended in Phase 5 with model_loaded,
        total_predictions, and fraud_rate_last_1000.
    """
    return {
        "status": "healthy",
        "model_loaded": False,  # Phase 5: set True after model loading
        "uptime_seconds": round(time.time() - _start_time, 1),
    }
