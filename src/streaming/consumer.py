"""
src/streaming/consumer.py

Kafka consumer for the real-time inference simulation (PRD Phase 6).

**It reuses `src/serving/InferenceService` — by design.** ADR-001 made the
scoring path transport-free precisely so that an HTTP request and a streamed
message cannot drift apart: the consumer does NOT reimplement transform →
sequence → blend → threshold → SHAP, it calls the exact same `InferenceService`
instance the FastAPI lifespan builds (`src/api/main.py:build_inference_service`).

**No separate container.** The consumer runs as an `asyncio` background task
inside `fraud-api` (PRD §7.1, §6.3): same process, same memory, no HTTP hop for
predictions, one feature-state store shared with the HTTP route.

**Concurrency is already handled.** The consumer thread and the HTTP threadpool
mutate one `InMemoryFeatureStateStore`. The read-then-write ordering, the
`TransactionID` idempotency guard, and the `RLock` around every compound
mutation all landed in Phase E for exactly this — see
`src/serving/feature_state.py`. This module adds no new shared state.

**`kafka` is imported lazily.** `kafka-python` is pinned in requirements.txt but
is not always present in a dev env; keeping the import inside `__init__` means
`import src.streaming.consumer` (and the clean-env import-smoke test) succeeds
without a broker, and only *constructing* a consumer needs the dependency.

Usage (wired automatically by the FastAPI lifespan):

    consumer = FraudDetectionConsumer(inference_service, config)
    task = asyncio.create_task(consumer.run())
    ...
    consumer.stop()
    task.cancel()
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from src.serving.inference import InferenceService, PredictionResult
from src.serving.prometheus_metrics import LAG_UNKNOWN, PrometheusMetrics

logger = logging.getLogger(__name__)

# Blocking `KafkaConsumer.poll` timeout, milliseconds. Short enough that
# `stop()` takes effect promptly, long enough not to busy-spin an idle topic.
_POLL_TIMEOUT_MS = 500

# Emit a progress line every N processed messages.
_LOG_EVERY = 100


class FraudDetectionConsumer:
    """Polls `transactions`, scores each row in-process, publishes fraud alerts.

    Lifecycle:
      * ``__init__``  — connect to Kafka (raises if the broker is unreachable).
      * ``run()``     — async loop; offloads the blocking poll to a thread so
                        the FastAPI event loop is never blocked (PRD §6.3).
      * ``stop()``    — flip the run flag and close the clients. Idempotent.

    All counters are plain ints read by ``/health`` (`snapshot()`); the
    consumer never exposes a second metrics registry.
    """

    def __init__(
        self,
        inference_service: InferenceService,
        config: Dict[str, Any],
        prometheus: Optional[PrometheusMetrics] = None,
    ) -> None:
        # Lazy import: see module docstring.
        try:
            from kafka import KafkaConsumer, KafkaProducer
        except ImportError as exc:  # pragma: no cover - exercised via env, not CI
            raise RuntimeError(
                "kafka-python is required to run the streaming consumer. It is "
                "pinned in requirements.txt; install it with "
                "`pip install kafka-python==2.0.2`."
            ) from exc

        self._service = inference_service
        kafka_cfg = config["kafka"]
        self._input_topic: str = kafka_cfg["input_topic"]
        self._output_topic: str = kafka_cfg["output_topic"]
        self._bootstrap = kafka_cfg["bootstrap_servers"]

        self._consumer = KafkaConsumer(
            self._input_topic,
            bootstrap_servers=self._bootstrap,
            group_id=kafka_cfg["consumer_group"],
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            # Non-blocking-ish poll: the loop stays responsive to stop().
            consumer_timeout_ms=_POLL_TIMEOUT_MS,
        )
        self._alert_producer = KafkaProducer(
            bootstrap_servers=self._bootstrap,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
        )

        self._prometheus = prometheus
        self._lag = LAG_UNKNOWN
        self._running = False
        self._messages_processed = 0
        self._alerts_published = 0
        self._errors = 0
        logger.info(
            "FraudDetectionConsumer connected: bootstrap=%s in=%s out=%s",
            self._bootstrap,
            self._input_topic,
            self._output_topic,
        )

    # ── Introspection (surfaced on /health) ──────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    def snapshot(self) -> Dict[str, Any]:
        """Consumer state for the `/health` payload."""
        return {
            "kafka_consumer_running": self._running,
            "kafka_messages_processed": self._messages_processed,
            "kafka_alerts_published": self._alerts_published,
            "kafka_consumer_errors": self._errors,
            "kafka_consumer_lag": self._lag,
        }

    # ── Scoring ──────────────────────────────────────────────────────────────

    def process_message(self, raw_transaction: Dict[str, Any]) -> PredictionResult:
        """Score one transaction through the shared `InferenceService`.

        A FRAUD decision is published to the `fraud_alerts` topic, keyed by
        `card1` so a downstream consumer can co-partition alerts with the
        source stream. Publishing is fire-and-forget: `KafkaProducer.send`
        returns a future we do not block on — a slow alert path must not stall
        the scoring loop.
        """
        result = self._service.predict(raw_transaction)

        if result.decision == "FRAUD":
            self._alert_producer.send(
                self._output_topic,
                key=raw_transaction.get("card1"),
                value=self._alert_payload(result),
            )
            self._alerts_published += 1
            logger.warning(
                "FRAUD ALERT transaction_id=%s p=%.6f mode=%s",
                result.transaction_id,
                result.fraud_probability,
                result.mode,
            )
        return result

    @staticmethod
    def _alert_payload(result: PredictionResult) -> Dict[str, Any]:
        """The alert message body — the audit-relevant slice of the decision."""
        payload = result.as_dict()
        # The full engineered vector is not alert-relevant and would bloat the
        # topic; the prediction log already retains it for drift analysis.
        payload.pop("model_probabilities", None)
        return payload

    # ── Async loop ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Consume until `stop()` is called.

        The blocking `KafkaConsumer.poll` runs via `asyncio.to_thread`, so the
        event loop stays free to service HTTP requests on the same process
        (PRD §6.3). One poisoned message increments the error counter and is
        skipped; it never breaks the loop.
        """
        self._running = True
        logger.info("Kafka consumer loop started (topic=%s)", self._input_topic)
        try:
            while self._running:
                records = await asyncio.to_thread(self._poll_batch)
                if not records:
                    # Idle topic: yield so a tight empty loop cannot starve the
                    # event loop between poll timeouts.
                    await asyncio.sleep(0)
                    continue
                for raw in records:
                    if not self._running:
                        break
                    self._handle_one(raw)
        except asyncio.CancelledError:
            logger.info("Kafka consumer loop cancelled")
            raise
        finally:
            self._running = False
            logger.info(
                "Kafka consumer loop stopped: processed=%d alerts=%d errors=%d",
                self._messages_processed,
                self._alerts_published,
                self._errors,
            )

    def _handle_one(self, raw_transaction: Dict[str, Any]) -> None:
        try:
            self.process_message(raw_transaction)
            self._messages_processed += 1
            if self._messages_processed % _LOG_EVERY == 0:
                logger.info(
                    "Consumer processed %d messages (%d alerts)",
                    self._messages_processed,
                    self._alerts_published,
                )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - one bad message must not kill the loop
            self._errors += 1
            logger.error(
                "Error scoring streamed transaction (id=%s): %s: %s",
                raw_transaction.get("TransactionID"),
                type(exc).__name__,
                exc,
            )

    def _poll_batch(self) -> List[Dict[str, Any]]:
        """Synchronous Kafka poll — always called from a worker thread."""
        batches = self._consumer.poll(timeout_ms=_POLL_TIMEOUT_MS)
        # Sample the backlog on the same thread that just polled: `position()`
        # and `highwater()` read the consumer's own client state, which is not
        # thread-safe to touch from the event loop while a poll is in flight.
        self._refresh_lag()
        return [msg.value for records in batches.values() for msg in records]

    def _refresh_lag(self) -> None:
        """Recompute the total backlog over assigned partitions.

        PRD §6.4 gates the streaming demo on lag staying under 500, and the
        processed-message counter cannot show that — a consumer can be both
        busy and falling behind. Sums ``highwater - position`` across the
        assignment; stays at :data:`LAG_UNKNOWN` while no partition is assigned
        or the broker has not yet reported a high-water mark, so an unknown
        backlog is never charted as a healthy zero.

        Never raises: a failure to read a diagnostic must not stop scoring.
        """
        try:
            partitions = self._consumer.assignment()
            if not partitions:
                self._set_lag(LAG_UNKNOWN)
                return
            total = 0
            for tp in partitions:
                highwater = self._consumer.highwater(tp)
                position = self._consumer.position(tp)
                if highwater is None or position is None:
                    self._set_lag(LAG_UNKNOWN)
                    return
                total += max(0, highwater - position)
            self._set_lag(total)
        except Exception as exc:  # noqa: BLE001 - diagnostics must not kill the loop
            logger.debug("Could not compute consumer lag: %s", exc)
            self._set_lag(LAG_UNKNOWN)

    def _set_lag(self, lag: int) -> None:
        self._lag = lag
        if self._prometheus is not None:
            self._prometheus.set_consumer_lag(lag)

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Stop the loop and release the Kafka clients. Safe to call twice."""
        self._running = False
        # A stopped consumer has no backlog to report; leaving the last sample
        # on the gauge would keep the demo's <500 check "passing" after exit.
        self._set_lag(LAG_UNKNOWN)
        consumer = self._consumer
        producer = self._alert_producer
        self._consumer = None  # type: ignore[assignment]
        self._alert_producer = None  # type: ignore[assignment]
        if consumer is not None:
            try:
                consumer.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                logger.warning("KafkaConsumer.close() failed: %s", exc)
        if producer is not None:
            try:
                producer.flush(timeout=5)
                producer.close(timeout=5)
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                logger.warning("KafkaProducer.close() failed: %s", exc)
        logger.info("FraudDetectionConsumer stopped")
