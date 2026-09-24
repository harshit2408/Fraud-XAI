"""
src/serving/prometheus_metrics.py

Prometheus instrumentation for the serving path (PRD FR-06 / FR-08 / Phase 5.7).

`ServingMetrics` (src/serving/metrics.py) already counts the *silent-degradation*
signals and surfaces them on `/health` as JSON. That is deliberately not a
Prometheus registry — those integers do not justify a second metrics system, and
`/health` is already polled.

This module is the other half: the metrics `prometheus.yml` scrapes and the
Grafana dashboard (`monitoring/grafana/dashboards/fraud_detection.json`) charts.
The dashboard was committed in Phase 6 already referencing these exact names, so
they are a fixed contract, not a fresh choice:

  - ``fraud_predictions_total{decision=...}`` — Counter. One increment per
    scored transaction, labelled ``FRAUD`` / ``LEGITIMATE``. The dashboard's
    "Fraud Rate" panel is ``rate(fraud_predictions_total{decision="FRAUD"}[5m])
    / rate(fraud_predictions_total[5m])`` and its "Request Volume" panel is
    ``rate(fraud_predictions_total[1m]) * 60``.
  - ``fraud_model_version{version=...}`` — Gauge pinned to ``1`` for the one
    loaded version label (the ``model_version_info`` pattern from PRD 5.7:
    Prometheus has no native info type, so a labelled gauge set to 1 is the
    idiom). Set once at startup.
  - ``fraud_consumer_lag`` — Gauge, the Kafka consumer's total backlog across
    its assigned partitions (sum of ``highwater - position``). PRD §6.4 gates
    the streaming demo on this staying under 500, which is not observable from
    the processed-message counter alone: a consumer that has processed 5000
    messages may still be 5000 behind. Written by
    ``src/streaming/consumer.py`` once per poll; ``-1`` means "not yet known"
    (no partition assignment, or the broker did not report a high-water mark),
    which is deliberately distinct from a genuine ``0`` backlog.
  - ``fraud_drift_detected`` — Gauge, ``1`` when the most recent
    ``drift_scheduler`` run found drift, ``0`` otherwise. Written by
    ``src/monitoring/drift_scheduler.py``; the dashboard's "Feature Drift
    Indicator" stat panel reads it directly.

``http_request_duration_seconds`` (the P95 latency panel) is NOT defined here —
``prometheus-fastapi-instrumentator`` adds it automatically when
``main.create_app()`` wires the instrumentator, and re-declaring it would
collide on the default registry.

**One process-wide registry.** These objects are module-level singletons
registered on ``prometheus_client``'s default ``REGISTRY``, which is what the
instrumentator's ``/metrics`` endpoint serializes. Re-importing the module does
not re-register (Python caches the module); constructing a second set in the
same process would raise ``Duplicated timeseries``. Tests that need isolation
call :func:`reset_for_testing`.
"""

from __future__ import annotations

import logging
from typing import Optional

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge

logger = logging.getLogger(__name__)

# Grafana's fraud_detection.json queries these names verbatim. Changing one is a
# dashboard-breaking change and must be made in both places together.
_PREDICTIONS_TOTAL_NAME = "fraud_predictions_total"
_MODEL_VERSION_NAME = "fraud_model_version"
_DRIFT_DETECTED_NAME = "fraud_drift_detected"
_CONSUMER_LAG_NAME = "fraud_consumer_lag"

# Sentinel for "backlog not yet determined". Prometheus gauges have no null,
# and 0 is a meaningful value (fully caught up), so an out-of-band negative is
# the only way to keep "unknown" distinguishable from "healthy" on the panel.
LAG_UNKNOWN = -1

DECISION_FRAUD = "FRAUD"
DECISION_LEGITIMATE = "LEGITIMATE"


class PrometheusMetrics:
    """Thin wrapper over the three custom collectors.

    Instantiated once in ``main.create_app()`` and handed to the
    ``InferenceService`` so the HTTP route and the Kafka consumer both feed the
    same counters (they already share one ``InferenceService``). The scheduler
    imports the module-level :data:`METRICS` singleton instead — it runs in its
    own process and only writes ``fraud_drift_detected``.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        registry = registry if registry is not None else REGISTRY
        self.predictions_total = Counter(
            _PREDICTIONS_TOTAL_NAME,
            "Total fraud predictions scored, labelled by the decision returned.",
            ["decision"],
            registry=registry,
        )
        self.model_version = Gauge(
            _MODEL_VERSION_NAME,
            "Loaded model bundle version. Pinned to 1 for the active "
            "`version` label (Prometheus info-metric idiom).",
            ["version"],
            registry=registry,
        )
        self.consumer_lag = Gauge(
            _CONSUMER_LAG_NAME,
            "Kafka consumer backlog in messages (sum of highwater - position "
            "over assigned partitions). -1 when the lag is not yet known.",
            registry=registry,
        )
        self.drift_detected = Gauge(
            _DRIFT_DETECTED_NAME,
            "1 when the most recent drift-scheduler run detected feature "
            "drift, 0 otherwise.",
            registry=registry,
        )
        # Initialise both label values so a rate() over them is well-defined
        # from the first scrape rather than starting at NaN.
        self.predictions_total.labels(decision=DECISION_FRAUD)
        self.predictions_total.labels(decision=DECISION_LEGITIMATE)
        self.drift_detected.set(0)
        # -1, not 0: before the consumer polls, the backlog is unknown, and
        # reporting a healthy 0 would let the demo's <500 gate pass vacuously.
        self.consumer_lag.set(LAG_UNKNOWN)

    # ── Serving path ─────────────────────────────────────────────────────────

    def record_decision(self, decision: str) -> None:
        """Count one scored transaction. Unknown labels pass through so a new
        decision value cannot silently drop off the dashboard."""
        self.predictions_total.labels(decision=decision).inc()

    def set_model_version(self, version: str) -> None:
        """Publish the loaded bundle version. Clears any previous label so a
        redeploy in the same process does not leave two versions at 1."""
        self.model_version.clear()
        self.model_version.labels(version=version).set(1)

    # ── Monitoring path ──────────────────────────────────────────────────────

    def set_consumer_lag(self, lag: int) -> None:
        """Publish the consumer backlog. Pass :data:`LAG_UNKNOWN` (-1) when it
        cannot be determined so an unknown backlog is never charted as zero."""
        self.consumer_lag.set(lag)

    def set_drift_detected(self, detected: bool) -> None:
        self.drift_detected.set(1 if detected else 0)


def _build_singleton() -> "PrometheusMetrics":
    try:
        return PrometheusMetrics()
    except ValueError:
        # Already registered on the default REGISTRY (e.g. the module was
        # re-imported under a test runner that reloaded it). Reuse the existing
        # collectors rather than crashing import.
        logger.debug(
            "Prometheus collectors already registered; reusing the default "
            "registry's instances."
        )
        return _from_default_registry()


def _from_default_registry() -> "PrometheusMetrics":
    """Re-wrap collectors already present on the default REGISTRY."""
    instance = PrometheusMetrics.__new__(PrometheusMetrics)
    names_to_collectors = getattr(REGISTRY, "_names_to_collectors", {})
    instance.predictions_total = names_to_collectors[_PREDICTIONS_TOTAL_NAME]
    instance.model_version = names_to_collectors[_MODEL_VERSION_NAME]
    instance.drift_detected = names_to_collectors[_DRIFT_DETECTED_NAME]
    instance.consumer_lag = names_to_collectors[_CONSUMER_LAG_NAME]
    return instance


def reset_for_testing() -> "PrometheusMetrics":
    """Unregister the custom collectors and rebuild them on the default
    registry. For tests only — production has exactly one process-lifetime set.
    """
    global METRICS
    for name in (
        _PREDICTIONS_TOTAL_NAME,
        _MODEL_VERSION_NAME,
        _DRIFT_DETECTED_NAME,
        _CONSUMER_LAG_NAME,
    ):
        collector = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
        if collector is not None:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass
    METRICS = PrometheusMetrics()
    return METRICS


# Process-wide singleton. `main.create_app()` uses this instance; the scheduler
# imports it directly.
METRICS: "PrometheusMetrics" = _build_singleton()
