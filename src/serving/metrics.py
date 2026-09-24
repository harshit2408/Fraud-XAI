"""
src/serving/metrics.py

Operational counters for the serving path (ADR-002 §5.3/§5.5, plus the
observability gaps `ecc:mle-reviewer` raised on 2026-09-01).

Every signal here shares one failure mode: **the system keeps answering while
quietly getting worse.** None of them raises, none of them shows up as an error
rate, and none of them would ever appear in a test that only asserts a 200. That
is exactly why they need to be counted:

  - `cards_scored_with_partial_sequence` — the TFT sequence buffer is not
    persisted (ADR-002 §5.6), so after every restart each card scores against a
    short, zero-padded window until it refills. The response is not `degraded`
    (correctly — cold start is a state the model saw in training), so without
    this counter a post-deploy quality dip is invisible.
  - `out_of_order_transactions` — ADR-002 §5.3 promised this and it was never
    built. A backdated transaction is clamped to the no-history sentinel rather
    than emitting a negative gap the model never saw.
  - `duplicate_transactions` — redelivery rejected by the idempotency guard.
    A rising count means an upstream at-least-once source is retrying.
  - `target_encoding_state_age_seconds` — ADR-002 §5.5. The encodings are
    frozen in this deployment (no label feed exists), so their staleness has to
    be *visible* rather than assumed away.
  - `config_hash_mismatch` — the registry downgraded this from a hard failure
    to a warning for a documented reason, which left a genuinely bad
    mixed-config deploy discoverable only by grepping logs.

Deliberately not `prometheus_client`: these are surfaced through `/health`,
which the Docker healthcheck already polls and `prometheus.yml` already scrapes.
Adding a second metrics registry to expose five integers would be more moving
parts than the signal justifies. The shape is a plain mapping, so swapping in a
real Prometheus registry later is a change to this module alone.
"""

import logging
import threading
import time
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


class ServingMetrics:
    """Thread-safe process-lifetime counters.

    Mutated from FastAPI's sync threadpool and (in Phase 6) a Kafka consumer
    thread, so every increment is locked — the same read-then-write hazard the
    feature-state store carries.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {
            "cards_scored_with_partial_sequence": 0,
            "out_of_order_transactions": 0,
            "duplicate_transactions": 0,
            # PRD Phase 4. The score is returned even when SHAP raises, so an
            # explainer that is quietly broken for every request would never
            # show up as an error rate — only as an explanation coverage of
            # zero, which is exactly what this counter makes visible.
            "explanation_failures": 0,
        }
        # Epoch seconds of the target-encoding state the store was seeded from.
        # Defaults to process start: with no label feed the state is only ever
        # as fresh as the artifact it was loaded from.
        self._target_encoding_epoch: float = time.time()
        self._config_hash_mismatch: bool = False
        self._config_hash_detail: Optional[Dict[str, str]] = None

    # ── Counters ─────────────────────────────────────────────────────────────

    def increment(self, name: str, amount: int = 1) -> None:
        """Advance a counter. Unknown names are created, so a new signal does
        not need a schema change here before it can be recorded."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def record_partial_sequence(self, window_length: int, required: int) -> None:
        """Count a request whose TFT window was shorter than the trained one."""
        if window_length < required:
            self.increment("cards_scored_with_partial_sequence")

    def record_out_of_order(self) -> None:
        self.increment("out_of_order_transactions")

    def record_duplicate(self) -> None:
        self.increment("duplicate_transactions")

    def record_explanation_failure(self) -> None:
        """Count a request whose score succeeded but whose SHAP explanation
        raised (PRD Phase 4). Never itself raises — explanation is best-effort.
        """
        self.increment("explanation_failures")

    # ── Gauges ───────────────────────────────────────────────────────────────

    def mark_target_encoding_state(self, epoch_seconds: float) -> None:
        """Record when the loaded target-encoding state was produced."""
        with self._lock:
            self._target_encoding_epoch = epoch_seconds

    def mark_config_hash_mismatch(self, hashes: Mapping[str, str]) -> None:
        """Surface a cross-artifact config disagreement beyond the log line."""
        with self._lock:
            self._config_hash_mismatch = True
            self._config_hash_detail = dict(hashes)

    # ── Read ─────────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Current values, safe to serialize into `/health`."""
        with self._lock:
            age = max(0.0, time.time() - self._target_encoding_epoch)
            return {
                **self._counters,
                "target_encoding_state_age_seconds": round(age, 1),
                "config_hash_mismatch": self._config_hash_mismatch,
                "config_hash_detail": self._config_hash_detail,
            }
