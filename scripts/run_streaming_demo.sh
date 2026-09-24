#!/usr/bin/env bash
#
# scripts/run_streaming_demo.sh
#
# End-to-end real-time inference demo (PRD §6.4).
#
#   1. Bring up the 4-service stack (kafka + fraud-api + prometheus + grafana).
#      The Kafka consumer loop starts automatically inside fraud-api on startup.
#   2. Wait for fraud-api to report healthy.
#   3. Run the producer to simulate a live transaction stream.
#   4. Let the consumer drain, then print a summary via streaming_stats.py.
#
# Done-when target: completes in < 90 seconds (excluding image build / first
# Kafka cold start).

set -euo pipefail

RATE="${RATE:-200}"
LIMIT="${LIMIT:-5000}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"

echo "Starting services (docker compose up -d)..."
docker compose up -d

echo "Waiting for fraud-api to become healthy..."
for _ in $(seq 1 60); do
  if curl -sf "${HEALTH_URL}" > /dev/null; then
    echo "fraud-api is ready."
    break
  fi
  sleep 2
done
curl -sf "${HEALTH_URL}" > /dev/null || {
  echo "fraud-api did not become healthy in time." >&2
  exit 1
}

echo "Producing ${LIMIT} transactions at ${RATE} tx/sec..."
python src/streaming/producer.py --rate "${RATE}" --limit "${LIMIT}"

echo "Producer finished. Waiting 5s for the consumer to drain..."
sleep 5

echo "Fetching stats..."
python scripts/streaming_stats.py --url "${HEALTH_URL}"
