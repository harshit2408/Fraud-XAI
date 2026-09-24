"""
tests/unit/test_streaming_producer.py

PRD Phase 6 — `TransactionProducer`.

No broker and no 600 MB CSV: `kafka.KafkaProducer` is a fake, and the raw
transaction file is a tiny synthetic CSV written to a tmp path. What is under
test: that rows are published keyed by `card1`, that `--limit` is honoured,
that the rate limiter is called once per message, and that NaN cells become
absent keys rather than `null`.
"""

import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.unit


class FakeKafkaProducer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.sent: List[Dict[str, Any]] = []
        self.flushed = 0
        self.closed = False

    def send(self, topic: str, key: Any = None, value: Any = None) -> None:
        self.sent.append({"topic": topic, "key": key, "value": value})

    def flush(self, timeout: Optional[float] = None) -> None:
        self.flushed += 1

    def close(self, timeout: Optional[float] = None) -> None:
        self.closed = True


@pytest.fixture
def kafka_stub(monkeypatch: pytest.MonkeyPatch):
    fake = types.ModuleType("kafka")
    fake.KafkaProducer = FakeKafkaProducer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kafka", fake)
    yield


CONFIG = {
    "kafka": {
        "bootstrap_servers": "localhost:9092",
        "input_topic": "transactions",
        "output_topic": "fraud_alerts",
        "consumer_group": "fraud_detector",
        "producer_rate_per_second": 100,
    }
}


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    path = tmp_path / "test_transaction.csv"
    pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [100.0, 200.0, 300.0, 400.0],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            "card1": [7, 7, 9, 9],
            "dist1": [1.0, None, 3.0, None],
        }
    ).to_csv(path, index=False)
    return path


def _make_producer():
    from src.streaming.producer import TransactionProducer

    return TransactionProducer(CONFIG)


def test_publishes_every_row_keyed_by_card1(kafka_stub, raw_csv, monkeypatch):
    monkeypatch.setattr("src.streaming.producer.time.sleep", lambda _s: None)
    producer = _make_producer()

    sent = producer.produce_from_file(raw_csv, rate=1000, limit=None)

    assert sent == 4
    keys = [m["key"] for m in producer._producer.sent]
    assert keys == [7, 7, 9, 9]
    assert all(m["topic"] == "transactions" for m in producer._producer.sent)


def test_limit_is_honoured(kafka_stub, raw_csv, monkeypatch):
    monkeypatch.setattr("src.streaming.producer.time.sleep", lambda _s: None)
    producer = _make_producer()

    sent = producer.produce_from_file(raw_csv, rate=1000, limit=2)

    assert sent == 2
    assert len(producer._producer.sent) == 2


def test_rate_limiter_is_called_once_per_message(kafka_stub, raw_csv, monkeypatch):
    calls: List[float] = []
    monkeypatch.setattr("src.streaming.producer.time.sleep", lambda s: calls.append(s))
    producer = _make_producer()

    producer.produce_from_file(raw_csv, rate=200, limit=3)

    assert calls == [pytest.approx(1 / 200)] * 3


def test_nan_cells_become_absent_keys(kafka_stub, raw_csv, monkeypatch):
    monkeypatch.setattr("src.streaming.producer.time.sleep", lambda _s: None)
    producer = _make_producer()

    producer.produce_from_file(raw_csv, rate=1000, limit=2)

    first, second = (m["value"] for m in producer._producer.sent)
    assert first["dist1"] == 1.0
    assert "dist1" not in second  # NaN dropped, not sent as null


def test_nonpositive_rate_is_rejected(kafka_stub, raw_csv):
    producer = _make_producer()
    with pytest.raises(ValueError, match="rate must be positive"):
        producer.produce_from_file(raw_csv, rate=0)


def test_missing_raw_file_raises_a_clear_error(kafka_stub, tmp_path):
    producer = _make_producer()
    with pytest.raises(FileNotFoundError, match="Raw transaction file not found"):
        producer.produce_from_file(tmp_path / "nope.csv", rate=100)


def test_missing_kafka_dependency_raises_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "kafka", None)
    from src.streaming.producer import TransactionProducer

    with pytest.raises(RuntimeError, match="kafka-python is required"):
        TransactionProducer(CONFIG)
