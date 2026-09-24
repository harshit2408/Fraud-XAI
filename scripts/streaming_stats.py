"""
scripts/streaming_stats.py

Print a summary of the streaming run (PRD §6.4).

Reads `GET /health` from the running fraud-api and prints the consumer counters
plus the silent-degradation signals from the `observability` block. Used by
`scripts/run_streaming_demo.sh` after the producer finishes.

No Kafka client and no extra dependency: `/health` already carries everything
the demo needs, and the Docker healthcheck / Prometheus scrape already poll it.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict

DEFAULT_URL = "http://localhost:8000/health"

# PRD 6.4: "Consumer lag stays < 500 messages throughout the demo". Checked
# here because this script is what `run_streaming_demo.sh` runs at the end;
# the same number is the red threshold on the Grafana lag panel.
MAX_ACCEPTABLE_LAG = 500

# `fraud_consumer_lag` sentinel meaning the backlog could not be determined.
LAG_UNKNOWN = -1


def fetch_health(url: str, timeout: float) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - localhost
        return json.loads(resp.read().decode("utf-8"))


def print_summary(health: Dict[str, Any]) -> None:
    obs = health.get("observability", {}) or {}
    rows = [
        ("model_version", health.get("model_version")),
        ("kafka_consumer_running", health.get("kafka_consumer_running")),
        ("messages_processed", health.get("kafka_messages_processed")),
        ("fraud_alerts_published", health.get("kafka_alerts_published")),
        ("consumer_errors", health.get("kafka_consumer_errors")),
        ("consumer_lag", health.get("kafka_consumer_lag")),
        ("known_cards", health.get("known_cards")),
        ("partial_sequence_scorings", obs.get("cards_scored_with_partial_sequence")),
        ("out_of_order_transactions", obs.get("out_of_order_transactions")),
        ("duplicate_transactions", obs.get("duplicate_transactions")),
        ("explanation_failures", obs.get("explanation_failures")),
    ]
    width = max(len(name) for name, _ in rows)
    print("\n=== Streaming run summary ===")
    for name, value in rows:
        print(f"  {name:<{width}} : {value}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a streaming demo run.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"(default: {DEFAULT_URL})")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    try:
        health = fetch_health(args.url, args.timeout)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Could not reach {args.url}: {exc}", file=sys.stderr)
        return 1

    print_summary(health)
    ok = True
    if not health.get("kafka_consumer_running"):
        print("WARNING: kafka_consumer_running is false.", file=sys.stderr)
        ok = False

    lag = health.get("kafka_consumer_lag")
    if lag is None or lag == LAG_UNKNOWN:
        # Not a failure: an idle consumer with no assignment legitimately has
        # no backlog to report, and the demo should say so rather than claim a
        # gate passed on a number it never saw.
        print(
            "NOTE: consumer lag unknown (no partition assignment or no "
            "high-water mark); the <"
            f"{MAX_ACCEPTABLE_LAG} lag check was not evaluated.",
            file=sys.stderr,
        )
    elif lag >= MAX_ACCEPTABLE_LAG:
        print(
            f"WARNING: consumer lag {lag} exceeds the {MAX_ACCEPTABLE_LAG}"
            "-message budget (PRD 6.4).",
            file=sys.stderr,
        )
        ok = False
    else:
        print(f"Consumer lag {lag} is within the {MAX_ACCEPTABLE_LAG}-message budget.")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
