"""
src/streaming/producer.py

Kafka producer script — reads test transactions and publishes to the
'transactions' topic at a configurable rate.

This is a CLI one-shot script, NOT a Docker service (per PRD architecture).

Usage:
    python src/streaming/producer.py --rate 200 --limit 5000

Full implementation in Phase 6. This stub provides the CLI interface
so `make stream` is valid.
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate real-time transaction stream via Kafka."
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
        help="Total number of transactions to publish (default: 5000)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    # TODO Phase 6: Implement full producer logic
    # 1. Load test_features.parquet
    # 2. Connect to Kafka (config.kafka.bootstrap_servers)
    # 3. Publish transactions to config.kafka.input_topic at args.rate/sec
    # 4. Stop after args.limit transactions

    logger.warning(
        f"Producer stub: --rate={args.rate} --limit={args.limit}. "
        "Full implementation in Phase 6."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
