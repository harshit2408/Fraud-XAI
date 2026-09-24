"""
src/serving/feature_state.py

Online per-entity feature state (docs/adr/ADR-002-realtime-feature-state.md §5.1).

ADR-002's audit reduced every card-history feature to five scalars per `card1`
— `(n, sum_amt, sum_amt_sq, max_amt, last_dt)`, modelled by
`CardAggregateState` — which makes exact (not approximate) train/serve
equivalence reachable. This module defines the port that holds them at serving
time, plus the in-process adapter that `docker compose up` ships.

Two rules from the ADR are enforced here rather than left to call sites:

  - **Read-then-write** (§5.2). `snapshot()` returns the state a transaction
    must be scored against; `observe()` folds it in afterwards. Reversing the
    order would leak a transaction's own amount into its own z-score, which is
    exactly what the batch pipeline's exclusive `cumsum`/`cumcount` prevents.
  - **Idempotency** (§5.3). Kafka delivers at least once, and these aggregates
    are *expanding*, so a double-counted transaction inflates `n` and `sum_amt`
    permanently — an error that never self-corrects. `observe()` is keyed by
    `TransactionID` and ignores repeats.

The Redis adapter ADR-002 §5.1 specifies for multi-replica deployments is
deliberately not implemented: the shipped stack is a single `fraud-api`
replica, and the port exists so that stays a deployment choice rather than an
assumption baked into the call sites.
"""

import logging
import threading
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, List, Optional, Protocol

import numpy as np

from src.data.feature_engineering import CardAggregateState, CardWindowState
from src.serving.metrics import ServingMetrics

logger = logging.getLogger(__name__)

# Retained TransactionIDs for the idempotency guard (ADR-002 §5.3). Bounded so
# a long-running consumer cannot grow the set without limit; sized to cover far
# more than any realistic redelivery window.
DEFAULT_DEDUPE_WINDOW = 100_000


class FeatureStateStore(Protocol):
    """Port consumed by `InferenceService` (ADR-001 §4.5 step 2)."""

    def snapshot(self, card_id: Any) -> CardAggregateState:
        """State to score against — never includes the transaction being scored."""
        ...

    def observe(
        self, card_id: Any, amount: float, dt: float, transaction_id: Optional[str] = None
    ) -> bool:
        """Fold a scored transaction in. Returns False if ignored as a duplicate."""
        ...

    def window(self, card_id: Any) -> CardWindowState:
        """Trailing-24h (dt, amount) history behind the P9-4 velocity features.

        ADR-003. Like `snapshot`, must never include the transaction being
        scored.
        """
        ...

    def sequence(self, card_id: Any) -> List[np.ndarray]:
        """Most recent transformed feature vectors for this card, oldest first."""
        ...

    def append_sequence(self, card_id: Any, feature_vector: np.ndarray) -> None:
        """Record the vector just scored, for the next transaction's window."""
        ...

    def bind_metrics(self, metrics: "ServingMetrics") -> None:
        """Adopt the caller's metrics object so counters cannot split.

        Part of the port rather than a private attribute so an alternative
        adapter (ADR-002 5.1's Redis store) cannot silently fail to be wired:
        a store that does not implement this does not satisfy the Protocol.
        """
        ...


class InMemoryFeatureStateStore:
    """Single-process `FeatureStateStore` (ADR-002 §5.1).

    Correct for the shipped `docker compose` topology, where one `fraud-api`
    replica hosts both the HTTP app and the Kafka consumer.

    **Locking.** ADR-002 §5.3 argues that partitioning the Kafka topic by
    `card1` gives each card a single writer, which would make locking
    unnecessary. That holds for the Kafka path — but NOT for HTTP: FastAPI runs
    the sync `/predict` route in a threadpool with no per-card affinity, and
    this same store instance serves both. Every mutation here is a compound
    read-then-write (`observe` reads a card's state, folds one transaction in,
    writes it back; `_claim` checks an id then records it). Individual dict
    operations are atomic under the GIL, but those *sequences* are not, so two
    concurrent same-card requests can lose an update and two racing on one
    `TransactionID` can both pass the dedupe check.

    Measured, not hypothetical: with `sys.setswitchinterval(1e-9)` and 16
    threads, 299/300 trials lost an update and 9/300 double-claimed. At the
    default switch interval it is rare — which makes it worse, not better,
    because the aggregates are *expanding*: a lost or doubled observation is
    permanent, never self-corrects, and will not reproduce on demand.

    One coarse lock is the right granularity here: the critical sections are a
    few dict operations long, so contention is negligible next to the feature
    transform and three model forward passes that dominate a request.
    """

    def __init__(
        self,
        card_state: Optional[Dict[Any, CardAggregateState]] = None,
        card_window: Optional[Dict[Any, CardWindowState]] = None,
        sequence_window: int = 10,
        dedupe_window: int = DEFAULT_DEDUPE_WINDOW,
        metrics: Optional["ServingMetrics"] = None,
    ) -> None:
        # Seeded from `feature_state.joblib` so serving continues the card
        # history the model was trained on rather than cold-starting every
        # card (ADR-002 §5.4).
        self._card_state: Dict[Any, CardAggregateState] = dict(card_state or {})
        # ADR-003 4.5: seeded alongside the scalars so a restart does not pair
        # a card's full expanding history with an empty trailing window.
        self._card_window: Dict[Any, CardWindowState] = dict(card_window or {})
        self._sequence_window = sequence_window
        self._dedupe_window = dedupe_window
        self._sequences: Dict[Any, Deque[np.ndarray]] = {}
        self._seen: "OrderedDict[str, None]" = OrderedDict()
        # Guards every compound read-then-write below. Reentrant so a future
        # caller can hold it across two store calls without self-deadlock.
        self._lock = threading.RLock()
        # Optional so unit tests can build a bare store; when present, the
        # silent-degradation paths below become countable (ADR-002 5.3).
        self._metrics = metrics

    # ── Reads ────────────────────────────────────────────────────────────────

    def bind_metrics(self, metrics: ServingMetrics) -> None:
        """Adopt `metrics`, warning if this store already had a different one.

        Two ServingMetrics instances mean counters split across objects with no
        error and no failing test — the exact silent-divergence this guards
        against. Rebinding is allowed (the service owns the canonical instance)
        but never silent.
        """
        with self._lock:
            existing = self._metrics
            if existing is not None and existing is not metrics:
                logger.warning(
                    "Feature-state store was constructed with a different "
                    "ServingMetrics instance than the InferenceService owns; "
                    "rebinding to the service's so counters stay in one place. "
                    "Pass the same instance to both to avoid this."
                )
            self._metrics = metrics

    def snapshot(self, card_id: Any) -> CardAggregateState:
        """An unseen card returns the zero state, which reproduces exactly the
        values batch emits for a card's first transaction (ADR-002 §5.4) —
        cold start is the normal path, not a degraded one."""
        with self._lock:
            return self._card_state.get(card_id, CardAggregateState())

    def window(self, card_id: Any) -> CardWindowState:
        with self._lock:
            return self._card_window.get(card_id, CardWindowState())

    def sequence(self, card_id: Any) -> List[np.ndarray]:
        with self._lock:
            return list(self._sequences.get(card_id, ()))

    def known_cards(self) -> int:
        with self._lock:
            return len(self._card_state)

    # ── Writes ───────────────────────────────────────────────────────────────

    def observe(
        self, card_id: Any, amount: float, dt: float, transaction_id: Optional[str] = None
    ) -> bool:
        # The claim and the fold must be atomic together: claiming under the
        # lock but folding outside it would still let two same-card requests
        # interleave their read-compute-write and drop one observation.
        with self._lock:
            if transaction_id is not None and not self._claim(transaction_id):
                logger.warning(
                    "Ignoring duplicate TransactionID %s for card %s — already "
                    "folded into the expanding aggregates.",
                    transaction_id,
                    card_id,
                )
                if self._metrics is not None:
                    self._metrics.record_duplicate()
                return False

            current = self._card_state.get(card_id, CardAggregateState())
            # ADR-002 5.3: a backdated transaction is clamped rather than
            # emitting a negative gap the model never saw in training. Silent
            # by design, so it has to be counted to stay discoverable.
            if self._metrics is not None and current.n and dt < current.last_dt:
                self._metrics.record_out_of_order()
            self._card_state[card_id] = current.observe(amount, dt)
            # ADR-003: same event, same lock, same dedupe claim. A backdated
            # transaction is rejected by CardWindowState.observe rather than
            # inserted out of order, which is what keeps the trailing counts
            # equal to the batch ones; it is already counted as out-of-order
            # by the metric above.
            self._card_window[card_id] = self._card_window.get(
                card_id, CardWindowState()
            ).observe(amount, dt)
            return True

    def append_sequence(self, card_id: Any, feature_vector: np.ndarray) -> None:
        with self._lock:
            buffer = self._sequences.get(card_id)
            if buffer is None:
                buffer = deque(maxlen=self._sequence_window)
                self._sequences[card_id] = buffer
            buffer.append(np.asarray(feature_vector, dtype=np.float32))

    def _claim(self, transaction_id: str) -> bool:
        """Record a TransactionID, returning False if it was already applied.

        Caller must hold `self._lock`: this is a check-then-set, and splitting
        the two halves across threads is exactly the double-count this guard
        exists to prevent.
        """
        if transaction_id in self._seen:
            return False
        self._seen[transaction_id] = None
        while len(self._seen) > self._dedupe_window:
            self._seen.popitem(last=False)
        return True
