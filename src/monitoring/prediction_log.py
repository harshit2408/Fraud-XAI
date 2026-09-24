"""
src/monitoring/prediction_log.py

Structured prediction logging (ADR-001 §4.5 step 8, PRD §7.4).

This is the write half of a contract whose read half is
`drift_reporter.load_prediction_window`. `serving.log_file` was configured from
Phase 0 onward but nothing ever wrote it — `InferenceService` emitted a
`logger.info` line and no more — so `make monitor` had no window to compare.
Both halves live in `src/monitoring/` so the format cannot drift apart
unnoticed; a round-trip test asserts they agree.

**Logging never fails a prediction.** By the time a line is written the
transaction has already been scored and the caller is owed an answer. A full
disk or an unwritable path degrades observability, not the fraud decision, so
`write` swallows its own errors and warns. This is the one place in the serving
path where swallowing an exception is the correct behaviour, which is why it is
confined to this module rather than left to each call site.

**No raw PII is written.** Only the engineered feature vector the model
consumed and the decision metadata are logged (PRD §6.5). The IEEE-CIS dataset
carries no direct identifiers, but the same discipline applies to any real
deployment: prediction logs are joined to delayed labels, so they persist far
longer than a request.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PredictionLogWriter:
    """Append-only JSONL sink for scored predictions.

    One JSON object per line, so the file can be tailed, truncated, rotated, or
    read back incrementally without parsing the whole history.
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Serving runs sync routes in a threadpool; interleaved partial writes
        # would corrupt lines that the reader then has to discard.
        self._lock = threading.Lock()

    def write(
        self,
        prediction: Mapping[str, Any],
        features: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Append one prediction record. Returns False if it could not be written.

        Args:
            prediction: The decision payload — `transaction_id`,
                `fraud_probability`, `decision`, `threshold`, `model_version`,
                `mode`, `degraded`.
            features: The engineered feature vector the model consumed. This is
                what `load_prediction_window` reconstructs the drift window
                from; without it monitoring has nothing to compare.
        """
        record: Dict[str, Any] = {
            **dict(prediction),
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        if features:
            record["features"] = dict(features)

        try:
            line = json.dumps(record, default=_json_safe)
            with self._lock, self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            return True
        except Exception as exc:  # noqa: BLE001 — observability must not break scoring
            logger.warning(
                "Could not append to the prediction log at %s: %s. The "
                "prediction itself was unaffected; drift monitoring will be "
                "missing this record.",
                self.log_path,
                exc,
            )
            return False


def _json_safe(value: Any) -> Any:
    """Last-resort coercion for numpy scalars.

    Targets `np.generic` explicitly rather than duck-typing on a callable
    `.item` attribute. The duck-typed version silently serialized ANY object
    exposing an unrelated zero-argument `item()` method — a dict-like wrapper,
    a test double, a domain object — to whatever that method returned,
    including a plain string, producing a corrupt record that looked valid.
    That contradicted this function's own contract, which is that anything
    genuinely unserialisable must raise so the write is reported and skipped.
    (Found by `ecc:code-reviewer`, repro: a class whose `item()` returns
    "not-a-number" was written out verbatim.)
    """
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
