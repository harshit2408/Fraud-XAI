"""
tests/unit/test_streaming_consumer.py

PRD Phase 6 — `FraudDetectionConsumer`.

No broker: `kafka.KafkaConsumer` / `KafkaProducer` are replaced with in-memory
fakes, and `InferenceService` with a spy. What is under test is the consumer's
own behaviour — that it routes every message through the shared service, that a
FRAUD decision (and only a FRAUD decision) produces an alert, that one poisoned
message does not break the loop, and that `stop()` ends it and releases the
clients.
"""

import asyncio
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.unit


# ── Fakes ────────────────────────────────────────────────────────────────────


@dataclass
class _FakeRecord:
    value: Dict[str, Any]


class FakeKafkaConsumer:
    """Hands out pre-loaded record batches, then empty ones forever."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.closed = False
        self._batches: List[List[_FakeRecord]] = []
        # Lag bookkeeping: `assignment()` / `highwater()` / `position()` are
        # what `_refresh_lag` reads off the real client.
        self._assignment: set = {"transactions-0"}
        self._highwater: Dict[Any, Optional[int]] = {"transactions-0": 0}
        self._position: Dict[Any, Optional[int]] = {"transactions-0": 0}

    def set_lag(self, highwater: Optional[int], position: Optional[int]) -> None:
        self._highwater["transactions-0"] = highwater
        self._position["transactions-0"] = position

    def assignment(self) -> set:
        return self._assignment

    def highwater(self, tp: Any) -> Optional[int]:
        return self._highwater.get(tp)

    def position(self, tp: Any) -> Optional[int]:
        return self._position.get(tp)

    def load(self, messages: List[Dict[str, Any]]) -> None:
        self._batches.append([_FakeRecord(m) for m in messages])

    def poll(self, timeout_ms: int = 0) -> Dict[str, List[_FakeRecord]]:
        if self._batches:
            return {"transactions-0": self._batches.pop(0)}
        return {}

    def close(self) -> None:
        self.closed = True


class FakeKafkaProducer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.sent: List[Dict[str, Any]] = []
        self.closed = False

    def send(self, topic: str, key: Any = None, value: Any = None) -> None:
        self.sent.append({"topic": topic, "key": key, "value": value})

    def flush(self, timeout: Optional[float] = None) -> None:
        pass

    def close(self, timeout: Optional[float] = None) -> None:
        self.closed = True


@dataclass
class _FakeResult:
    """Minimal stand-in for `PredictionResult` (only what the consumer reads)."""

    transaction_id: Optional[str]
    fraud_probability: float
    decision: str
    mode: str = "default"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "fraud_probability": self.fraud_probability,
            "decision": self.decision,
            "mode": self.mode,
            "model_probabilities": {"xgb": self.fraud_probability},
        }


class SpyInferenceService:
    """Records every transaction and returns FRAUD above a probability cutoff."""

    def __init__(self, fraud_if_amount_over: float = 500.0) -> None:
        self.seen: List[Dict[str, Any]] = []
        self._cutoff = fraud_if_amount_over

    def predict(self, transaction: Dict[str, Any]) -> _FakeResult:
        self.seen.append(transaction)
        amt = transaction.get("TransactionAmt", 0.0)
        if amt == "boom":  # sentinel used by the poison-message test
            raise RuntimeError("scoring blew up")
        decision = "FRAUD" if amt > self._cutoff else "LEGITIMATE"
        return _FakeResult(
            transaction_id=str(transaction.get("TransactionID")),
            fraud_probability=0.9 if decision == "FRAUD" else 0.01,
            decision=decision,
        )


@pytest.fixture
def kafka_stub(monkeypatch: pytest.MonkeyPatch):
    """Install a fake `kafka` module so `import kafka` inside the consumer works."""
    fake = types.ModuleType("kafka")
    fake.KafkaConsumer = FakeKafkaConsumer  # type: ignore[attr-defined]
    fake.KafkaProducer = FakeKafkaProducer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kafka", fake)
    yield


CONFIG = {
    "kafka": {
        "bootstrap_servers": "localhost:9092",
        "input_topic": "transactions",
        "output_topic": "fraud_alerts",
        "consumer_group": "fraud_detector",
    }
}


def _make_consumer(service: SpyInferenceService, prometheus: Any = None):
    from src.streaming.consumer import FraudDetectionConsumer

    return FraudDetectionConsumer(service, CONFIG, prometheus=prometheus)


class SpyPrometheus:
    """Captures every `set_consumer_lag` call the consumer makes."""

    def __init__(self) -> None:
        self.lags: List[int] = []

    def set_consumer_lag(self, lag: int) -> None:
        self.lags.append(lag)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_process_message_routes_through_the_shared_service(kafka_stub):
    service = SpyInferenceService()
    consumer = _make_consumer(service)

    consumer.process_message(
        {"TransactionID": "t1", "TransactionAmt": 10.0, "card1": 7}
    )

    assert service.seen == [{"TransactionID": "t1", "TransactionAmt": 10.0, "card1": 7}]


def test_fraud_decision_publishes_an_alert_keyed_by_card1(kafka_stub):
    service = SpyInferenceService(fraud_if_amount_over=100.0)
    consumer = _make_consumer(service)

    consumer.process_message(
        {"TransactionID": "t2", "TransactionAmt": 999.0, "card1": 42}
    )

    assert len(consumer._alert_producer.sent) == 1
    alert = consumer._alert_producer.sent[0]
    assert alert["topic"] == "fraud_alerts"
    assert alert["key"] == 42
    assert alert["value"]["decision"] == "FRAUD"
    # The engineered vector is not alert-relevant and is stripped.
    assert "model_probabilities" not in alert["value"]
    assert consumer.snapshot()["kafka_alerts_published"] == 1


def test_legitimate_decision_publishes_no_alert(kafka_stub):
    service = SpyInferenceService(fraud_if_amount_over=100.0)
    consumer = _make_consumer(service)

    consumer.process_message({"TransactionID": "t3", "TransactionAmt": 5.0, "card1": 1})

    assert consumer._alert_producer.sent == []
    assert consumer.snapshot()["kafka_alerts_published"] == 0


def test_one_poison_message_does_not_break_the_loop(kafka_stub):
    service = SpyInferenceService(fraud_if_amount_over=100.0)
    consumer = _make_consumer(service)

    async def drive() -> None:
        consumer._consumer.load(
            [
                {"TransactionID": "ok1", "TransactionAmt": 10.0, "card1": 1},
                {"TransactionID": "bad", "TransactionAmt": "boom", "card1": 2},
                {"TransactionID": "ok2", "TransactionAmt": 999.0, "card1": 3},
            ]
        )
        task = asyncio.create_task(consumer.run())
        for _ in range(50):
            await asyncio.sleep(0.01)
            if consumer._messages_processed + consumer._errors >= 3:
                break
        consumer.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())

    snap = consumer.snapshot()
    assert snap["kafka_messages_processed"] == 2  # ok1, ok2
    assert snap["kafka_consumer_errors"] == 1  # bad
    assert snap["kafka_alerts_published"] == 1  # ok2 only


def test_stop_ends_the_loop_and_closes_clients(kafka_stub):
    service = SpyInferenceService()
    consumer = _make_consumer(service)
    fake_consumer = consumer._consumer
    fake_producer = consumer._alert_producer

    async def drive() -> None:
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        assert consumer.running is True
        consumer.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())

    assert consumer.running is False
    assert fake_consumer.closed is True
    assert fake_producer.closed is True
    consumer.stop()  # idempotent — must not raise


def test_missing_kafka_dependency_raises_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """With no `kafka` module importable, construction fails with guidance."""
    monkeypatch.setitem(sys.modules, "kafka", None)
    from src.streaming.consumer import FraudDetectionConsumer

    with pytest.raises(RuntimeError, match="kafka-python is required"):
        FraudDetectionConsumer(SpyInferenceService(), CONFIG)


# ── Consumer lag (PRD 6.4) ───────────────────────────────────────────────────


def test_lag_is_unknown_before_the_first_poll(kafka_stub):
    """An unpolled consumer must not report a healthy zero backlog."""
    from src.serving.prometheus_metrics import LAG_UNKNOWN

    consumer = _make_consumer(SpyInferenceService())

    assert consumer.snapshot()["kafka_consumer_lag"] == LAG_UNKNOWN


def test_poll_publishes_the_backlog_to_prometheus_and_health(kafka_stub):
    prometheus = SpyPrometheus()
    consumer = _make_consumer(SpyInferenceService(), prometheus=prometheus)
    consumer._consumer.set_lag(highwater=1200, position=950)

    consumer._poll_batch()

    assert consumer.snapshot()["kafka_consumer_lag"] == 250
    assert prometheus.lags == [250]


def test_lag_reports_unknown_when_the_broker_has_no_highwater(kafka_stub):
    """No high-water mark means the backlog is genuinely unknown, not zero."""
    from src.serving.prometheus_metrics import LAG_UNKNOWN

    prometheus = SpyPrometheus()
    consumer = _make_consumer(SpyInferenceService(), prometheus=prometheus)
    consumer._consumer.set_lag(highwater=None, position=0)

    consumer._poll_batch()

    assert consumer.snapshot()["kafka_consumer_lag"] == LAG_UNKNOWN
    assert prometheus.lags == [LAG_UNKNOWN]


def test_lag_computation_failure_does_not_break_polling(kafka_stub):
    """A diagnostic that raises must never stop the consumer scoring."""
    from src.serving.prometheus_metrics import LAG_UNKNOWN

    consumer = _make_consumer(SpyInferenceService())
    consumer._consumer.load(
        [{"TransactionID": "t1", "TransactionAmt": 1.0, "card1": 1}]
    )

    def boom() -> set:
        raise RuntimeError("broker metadata unavailable")

    consumer._consumer.assignment = boom  # type: ignore[assignment]

    records = consumer._poll_batch()

    assert len(records) == 1
    assert consumer.snapshot()["kafka_consumer_lag"] == LAG_UNKNOWN


def test_stop_clears_a_stale_lag_reading(kafka_stub):
    """A stopped consumer must not leave a passing lag number on the gauge."""
    from src.serving.prometheus_metrics import LAG_UNKNOWN

    prometheus = SpyPrometheus()
    consumer = _make_consumer(SpyInferenceService(), prometheus=prometheus)
    consumer._consumer.set_lag(highwater=10, position=8)
    consumer._poll_batch()
    assert consumer.snapshot()["kafka_consumer_lag"] == 2

    consumer.stop()

    assert consumer.snapshot()["kafka_consumer_lag"] == LAG_UNKNOWN
    assert prometheus.lags[-1] == LAG_UNKNOWN
