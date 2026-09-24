"""
src/serving/staleness.py

The staleness rule for an incoming transaction (Phase E task E5).

Lives in `src/serving/` rather than in the API schemas because it is a domain
rule about card history, not an HTTP concern — the Phase 6 Kafka consumer needs
exactly the same check, and `src/serving/` must not import from `src/api/`
(ADR-001 §4.1). Unlike the purely per-request Pydantic constraints, this one
needs the card's last-seen timestamp, so it runs where that state is available.
"""

import math
from typing import Optional

# IEEE-CIS TransactionDT is a seconds offset spanning ~6 months. A served
# transaction should be at or after the history the model carries; rejecting
# anything more than a day behind the card's last-seen timestamp keeps
# out-of-order replays from corrupting per-card expanding aggregates.
MAX_BACKDATE_SECONDS = 86_400.0


class StaleTransactionError(ValueError):
    """Raised when a transaction is too far behind its card's known history."""


def check_staleness(
    transaction_dt: float,
    last_seen_dt: Optional[float],
    max_backdate_seconds: float = MAX_BACKDATE_SECONDS,
) -> None:
    """Reject a transaction that arrives materially out of order for its card.

    The expanding aggregates assume per-card chronological order (ADR-002
    §5.3). A backdated row would produce a negative `time_since_last_tx` — a
    value that never occurs in training, since the batch pipeline sorts
    temporally — and would fold into the accumulators out of sequence.

    `last_seen_dt` of None (an unseen card) is always accepted: cold start is
    the normal path (ADR-002 §5.4).

    Raises:
        StaleTransactionError: the transaction predates the card's last-seen
            timestamp by more than `max_backdate_seconds`.
    """
    if last_seen_dt is None or not math.isfinite(last_seen_dt):
        return
    backdate = last_seen_dt - transaction_dt
    if backdate > max_backdate_seconds:
        raise StaleTransactionError(
            f"Transaction is {backdate:.0f}s behind this card's last seen "
            f"transaction (limit {max_backdate_seconds:.0f}s). Scoring it would "
            "corrupt the card's expanding aggregates."
        )
