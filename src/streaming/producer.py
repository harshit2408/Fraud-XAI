"""
src/streaming/producer.py

Kafka producer — reads RAW test transactions and publishes them to the
`transactions` topic at a configurable rate (PRD §6.2).

A CLI one-shot script, NOT a Docker service (PRD §7.1):

    python src/streaming/producer.py --rate 200 --limit 5000

**Why raw CSV, not `test_features.parquet`.** The consumer scores each message
through `src/serving/InferenceService`, which runs the *full* training feature
transform on a raw transaction (`ServingFeatureTransformer`). Publishing
already-engineered parquet rows would double-transform them. The producer
therefore streams `data/raw/test_transaction.csv` (optionally left-joined with
`test_identity.csv` on `TransactionID`, mirroring `DataLoader.load_raw`), which
is exactly the shape `POST /predict` accepts.

**Message key = `card1`.** Kafka partitions by key, so every transaction for a
card lands on one partition and is consumed by one worker in order — which is
what makes the expanding per-card aggregates correct without cross-partition
coordination (ADR-002 §5.3).

`kafka` is imported lazily so `import src.streaming.producer` (and the
clean-env import-smoke test) works without a broker installed.
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import load_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_CHUNK_ROWS = 20_000
_LOG_EVERY = 1_000


class TransactionProducer:
    """Streams raw transactions from a CSV to the Kafka `transactions` topic."""

    def __init__(self, config: Dict[str, Any]) -> None:
        try:
            from kafka import KafkaProducer
        except ImportError as exc:  # pragma: no cover - exercised via env, not CI
            raise RuntimeError(
                "kafka-python is required to run the streaming producer. It is "
                "pinned in requirements.txt; install it with "
                "`pip install kafka-python==2.0.2`."
            ) from exc

        kafka_cfg = config["kafka"]
        self._topic: str = kafka_cfg["input_topic"]
        self._producer = KafkaProducer(
            bootstrap_servers=kafka_cfg["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            linger_ms=20,
        )
        self._sent = 0
        logger.info(
            "TransactionProducer connected: bootstrap=%s topic=%s",
            kafka_cfg["bootstrap_servers"],
            self._topic,
        )

    @property
    def sent(self) -> int:
        return self._sent

    # ── Public API ───────────────────────────────────────────────────────────

    def produce_from_file(
        self,
        transaction_csv: Path,
        identity_csv: Optional[Path] = None,
        rate: int = 100,
        limit: Optional[int] = None,
    ) -> int:
        """Publish up to `limit` transactions at `rate` per second.

        Rate limiting is a per-message `sleep(1 / rate)` — coarse, but this is a
        simulation, not a load test, and it keeps the code readable.
        """
        if rate <= 0:
            raise ValueError(f"--rate must be positive, got {rate}")
        interval = 1.0 / rate

        for transaction in self._iter_transactions(transaction_csv, identity_csv):
            if limit is not None and self._sent >= limit:
                break
            self.produce_single(transaction)
            if self._sent % _LOG_EVERY == 0:
                logger.info("Published %d transactions", self._sent)
            time.sleep(interval)

        self._producer.flush()
        logger.info("Producer finished: %d transactions published", self._sent)
        return self._sent

    def produce_single(self, transaction: Dict[str, Any]) -> None:
        """Publish one transaction, keyed by `card1`."""
        self._producer.send(
            self._topic,
            key=transaction.get("card1"),
            value=transaction,
        )
        self._sent += 1

    def close(self) -> None:
        self._producer.flush()
        self._producer.close(timeout=5)

    # ── Row iteration ────────────────────────────────────────────────────────

    def _iter_transactions(
        self, transaction_csv: Path, identity_csv: Optional[Path]
    ) -> Iterator[Dict[str, Any]]:
        """Yield raw transactions as JSON-safe dicts, chunked to bound memory.

        `test_transaction.csv` is ~600 MB; it is read in `_CHUNK_ROWS` chunks.
        The identity table (~75% of transactions have no identity row) is
        loaded once and left-joined per chunk, matching `DataLoader.load_raw`.
        """
        if not transaction_csv.exists():
            raise FileNotFoundError(
                f"Raw transaction file not found: {transaction_csv}. The "
                "producer streams RAW IEEE-CIS transactions (not the engineered "
                "parquet). Run `python src/data/download_data.py` first."
            )

        identity_df: Optional[pd.DataFrame] = None
        if identity_csv is not None and identity_csv.exists():
            identity_df = pd.read_csv(identity_csv)
            logger.info("Loaded identity table: %d rows", len(identity_df))

        reader = pd.read_csv(transaction_csv, chunksize=_CHUNK_ROWS)
        for chunk in reader:
            if identity_df is not None:
                chunk = chunk.merge(identity_df, on="TransactionID", how="left")
            for record in chunk.to_dict(orient="records"):
                yield _json_safe(record)


def _json_safe(record: Dict[str, Any]) -> Dict[str, Any]:
    """Drop NaNs and coerce numpy scalars so `json.dumps` accepts the row.

    A missing raw column is sent as an absent key rather than `null`; the
    serving transform and the request schema both treat "absent" as "no
    value", and it keeps the message small.
    """
    clean: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
            if isinstance(value, float) and math.isnan(value):
                continue
        clean[key] = value
    return clean


def _resolve_paths(config: Dict[str, Any]) -> Tuple[Path, Optional[Path]]:
    raw_dir = Path(config["data"]["raw_dir"])
    # Config points at the TRAIN files; the streaming sim replays the held-out
    # test transactions, so swap the prefix.
    transaction_csv = raw_dir / config["data"]["train_file"].replace("train_", "test_")
    identity_csv = raw_dir / config["data"]["identity_file"].replace("train_", "test_")
    return transaction_csv, (identity_csv if identity_csv.exists() else None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a real-time transaction stream via Kafka."
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=200,
        help="Transactions per second to publish (default: 200)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Total transactions to publish; 0 means stream all (default: 5000)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    config = load_settings(args.config).model_dump()
    transaction_csv, identity_csv = _resolve_paths(config)
    limit = None if args.limit == 0 else args.limit

    producer = TransactionProducer(config)
    try:
        producer.produce_from_file(
            transaction_csv=transaction_csv,
            identity_csv=identity_csv,
            rate=args.rate,
            limit=limit,
        )
    finally:
        producer.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
