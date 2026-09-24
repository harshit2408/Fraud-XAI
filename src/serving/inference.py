"""
src/serving/inference.py

Per-request orchestration (ADR-001 §4.5).

Transport-free by design: the FastAPI route and the Phase 6 Kafka consumer both
construct this service, so an HTTP request and a streamed message cannot drift
apart. Nothing here imports from `src/api/`.

The two contracts most easily broken by a later edit, both enforced here:

  - **Calibrated probability space.** The blend weights and the operating
    threshold are fitted over per-model CALIBRATED probabilities
    (`scripts/run_ensemble_eval.py`). This service calls
    `predict_proba_calibrated` and refuses to fall back to raw output — a
    silent fallback would threshold a different quantity while still looking
    entirely healthy.
  - **Read-then-write.** The feature-state store is updated only AFTER the
    score is produced, so a transaction never contributes to its own expanding
    aggregates (ADR-002 §5.2).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.ensemble import blend
from src.serving.feature_state import FeatureStateStore
from src.serving.ensemble_spec import EnsembleMode
from src.serving.metrics import ServingMetrics
from src.serving.prometheus_metrics import PrometheusMetrics
from src.serving.registry import LoadedModels
from src.serving.staleness import check_staleness
from src.monitoring.prediction_log import PredictionLogWriter
from src.serving.transform import ServingFeatureTransformer

logger = logging.getLogger(__name__)

# The degradation mode tried when the default blend cannot be scored. It is
# servable only because run_ensemble_eval.py pre-registers it with its own
# grid-searched weights and its own cost-optimal threshold (ADR-001 3.3).
FALLBACK_MODE = "no_tft"


class ServingInvariantError(RuntimeError):
    """A condition startup already validated has been violated at request time.

    Distinct from a transient model failure on purpose: an unloaded model or a
    missing calibrator means the registry's own guarantees no longer hold, so
    the request must fail loudly rather than be quietly downgraded to a
    fallback blend and returned as a healthy 200.
    """


@dataclass(frozen=True)
class PredictionResult:
    """One scoring decision, with everything needed to audit it later."""

    transaction_id: Optional[str]
    fraud_probability: float
    decision: str
    threshold: float
    model_version: str
    mode: str
    degraded: bool
    model_probabilities: Dict[str, float] = field(default_factory=dict)
    degraded_reason: Optional[str] = None
    latency_ms: float = 0.0
    # PRD Phase 4. `explanation` is the top SHAP contributions as plain dicts
    # ({"feature", "contribution"}), largest magnitude first; empty when the
    # explainer is disabled or itself failed for this row. `explained_model`
    # is always "xgb" when populated — TreeSHAP speaks for the XGBoost
    # component, not the blend (ADR-001 §3.4) — and `explained_weight` is that
    # component's share of the default blend.
    explanation: List[Dict[str, Any]] = field(default_factory=list)
    explained_model: Optional[str] = None
    explained_weight: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "fraud_probability": self.fraud_probability,
            "decision": self.decision,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "mode": self.mode,
            "degraded": self.degraded,
            "model_probabilities": dict(self.model_probabilities),
            "degraded_reason": self.degraded_reason,
            "explanation": [dict(item) for item in self.explanation],
            "explained_model": self.explained_model,
            "explained_weight": self.explained_weight,
            "latency_ms": self.latency_ms,
        }


class InferenceService:
    """Validated transaction in, auditable decision out."""

    def __init__(
        self,
        models: LoadedModels,
        state_store: FeatureStateStore,
        label_lag_seconds: float = 0.0,
        prediction_log: Optional["PredictionLogWriter"] = None,
        metrics: Optional[ServingMetrics] = None,
        sequence_window: int = 10,
        explanation_timeout_ms: int = 0,
        prometheus: Optional[PrometheusMetrics] = None,
    ) -> None:
        self.models = models
        self.state_store = state_store
        self.metrics = metrics or ServingMetrics()
        # PRD FR-08 / Phase 5.7: the scraped `fraud_predictions_total` counter.
        # Optional so unit tests and the training-equivalence harness build the
        # service without touching the process-wide Prometheus registry; the
        # API wires the real singleton in `main.create_app()`.
        self.prometheus = prometheus
        self.sequence_window = sequence_window
        # Per-request wall-clock budget for the SHAP call. TreeSHAP on the
        # 171-feature booster is ~140 ms p50, which would otherwise dominate
        # `/predict` latency (ecc:mle-reviewer, P4-8). 0 = no budget.
        self.explanation_timeout_s: float = max(0, explanation_timeout_ms) / 1000.0
        self._explain_pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="shap-explain")
            if self.explanation_timeout_s > 0
            else None
        )
        # Bind through the port, not a private attribute. The previous version
        # poked `state_store._metrics` and only handled the None case, so a
        # store built with its OWN metrics instance still split the counters,
        # and any adapter naming the attribute differently defeated the check
        # with no error (ecc:code-reviewer).
        binder = getattr(state_store, "bind_metrics", None)
        if callable(binder):
            binder(self.metrics)
        else:
            logger.warning(
                "Feature-state store %s does not implement bind_metrics(); its "
                "counters will not reach /health.",
                type(state_store).__name__,
            )
        # Optional: without it the service still scores, but `make monitor`
        # has no window to compare (E6). None keeps unit tests light.
        self.prediction_log = prediction_log
        self.transformer = ServingFeatureTransformer(
            feature_engineer=models.feature_engineer,
            state_store=state_store,
            feature_names=models.feature_names,
            label_lag_seconds=label_lag_seconds,
        )

    def predict(
        self, transaction: Dict[str, Any], mode: Optional[str] = None
    ) -> PredictionResult:
        """Score one transaction (ADR-001 §4.5 steps 2-8).

        Raises:
            KeyError: `mode` names a blend that was not pre-registered with its
                own frozen threshold — serving never improvises an operating
                point (ADR-001 §3.3).
            ValueError: the transformed vector is missing a trained feature.
        """
        started = time.perf_counter()
        requested = self.models.ensemble.mode(mode)

        # Staleness needs the card's last-seen timestamp, so it is checked here
        # rather than in the Pydantic model - but still BEFORE any state is
        # touched, so a rejected transaction cannot corrupt card history.
        self._check_staleness(transaction)

        raw = pd.DataFrame([transaction])
        features = self.transformer.transform(raw)
        card_id = transaction.get("card1")

        active = requested
        degraded_reason: Optional[str] = None
        try:
            probabilities = self._score(features, active.models, card_id)
        except ServingInvariantError:
            # Startup validated these; if one fires now it is a bug in this
            # process, not a model that needs degrading around. Never masked.
            raise
        except Exception as exc:  # noqa: BLE001 - a model-level failure
            fallback = self._fallback_mode(requested, mode)
            if fallback is None:
                raise
            degraded_reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Mode %r failed (%s); falling back to pre-registered mode %r. "
                "The fallback carries its own frozen threshold - weights are "
                "never renormalized (ADR-001 3.3).",
                requested.name,
                degraded_reason,
                fallback.name,
            )
            active = fallback
            probabilities = self._score(features, active.models, card_id)

        blended = float(blend(probabilities, active.weights)[0])
        decision = "FRAUD" if blended >= active.threshold else "LEGITIMATE"
        if self.prometheus is not None:
            # Counted here, not in the HTTP route, so a streamed message and an
            # HTTP request land on the same counter (ADR-001: one scoring path).
            self.prometheus.record_decision(decision)

        # Read-then-write: only now is the transaction folded into the store -
        # both the scalar accumulators and TFT's per-card sequence window.
        #
        # The sequence buffer is gated on the SAME idempotency decision as the
        # accumulators. A redelivered transaction that `observe` rejects must
        # not be appended either: the ring buffer is bounded, so a duplicate
        # push evicts a real historical transaction and corrupts the window TFT
        # scores against - the accumulators would be right and the sequence
        # quietly wrong (ADR-002 5.3).
        recorded = self._observe(transaction)
        if recorded and card_id is not None:
            self.state_store.append_sequence(card_id, features.to_numpy()[0])

        explanation, explained_model, explained_weight = self._explain(features)

        result = PredictionResult(
            transaction_id=self._transaction_id(transaction),
            fraud_probability=blended,
            decision=decision,
            threshold=active.threshold,
            model_version=self.models.model_version,
            mode=active.name,
            degraded=active.name != requested.name,
            degraded_reason=degraded_reason,
            model_probabilities={k: float(v[0]) for k, v in probabilities.items()},
            explanation=explanation,
            explained_model=explained_model,
            explained_weight=explained_weight,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        logger.info(
            "prediction transaction_id=%s decision=%s p=%.6f mode=%s model_version=%s",
            result.transaction_id,
            result.decision,
            result.fraud_probability,
            result.mode,
            result.model_version,
        )
        if self.prediction_log is not None:
            # The engineered vector, not the raw request: this is exactly what
            # the model consumed, and what drift is measured against (E6).
            self.prediction_log.write(
                result.as_dict(), features=features.iloc[0].to_dict()
            )
        return result

    # ── Internals ────────────────────────────────────────────────────────────

    def _score(
        self,
        features: pd.DataFrame,
        model_names: List[str],
        card_id: Optional[Any] = None,
    ) -> Dict[str, np.ndarray]:
        """Per-model calibrated probabilities for the active mode.

        TFT is sequential: it was trained on a window of the card's last
        `max_encoder_length` transactions, so scoring it on the served row
        alone would feed it a length-1 sequence with every other position
        masked - silently out-of-distribution for any card with real history
        (ADR-002 5.6). Its prior transformed vectors are therefore replayed
        from the store as `history_X`. A card with no stored history keeps the
        zero-padded path, which is exactly what training saw for each card's
        first transactions.
        """
        history = self._sequence_history(features, card_id)

        out: Dict[str, np.ndarray] = {}
        for name in model_names:
            trainer = self.models.trainers.get(name)
            if trainer is None:
                raise ServingInvariantError(
                    f"Ensemble mode requires model '{name}', which the registry "
                    "did not load."
                )
            if getattr(trainer, "calibrator", None) is None:
                # Startup already rejects this; re-checked because a raw
                # fallback here would be invisible in the response.
                raise ServingInvariantError(
                    f"Model '{name}' has no frozen calibrator. The blend is "
                    "defined over calibrated probabilities — refusing to score."
                )
            if name == "tft" and history is not None:
                out[name] = np.asarray(
                    trainer.predict_proba_calibrated(features, history_X=history)
                )
            else:
                out[name] = np.asarray(trainer.predict_proba_calibrated(features))
        return out

    def _explain(
        self, features: pd.DataFrame
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[float]]:
        """Top SHAP contributions for the served feature vector (ADR-001 §3.4).

        Explains the exact engineered frame the model scored, not the raw
        request — the same contract the prediction log takes (E6).

        A raising *or slow* explainer degrades the *explanation*, never the
        score: a missing explanation is recoverable, a wrong or late decision
        is not. A failure is logged (not swallowed) and counted
        (`explanation_failures`), and the response carries an empty
        `explanation`. When `explanation_timeout_ms` is set, the SHAP call runs
        on a worker thread and is abandoned at the budget — TreeSHAP on the
        171-feature booster is ~140 ms, well above a fraud-scoring latency
        target (ecc:mle-reviewer, P4-8).
        """
        # getattr, not attribute access: `LoadedModels` always carries this,
        # but the Kafka-consumer path (Phase 6) and tests may hand in a
        # lighter snapshot type without it — a missing explainer is the same
        # as a disabled one.
        explainer = getattr(self.models, "explainer", None)
        if explainer is None:
            return [], None, None
        try:
            if self._explain_pool is not None:
                future = self._explain_pool.submit(explainer.explain_single, features)
                explained = future.result(timeout=self.explanation_timeout_s)
            else:
                explained = explainer.explain_single(features)
            return (
                explained.as_api_contributions(),
                explained.explained_model,
                explained.explained_weight,
            )
        except FutureTimeout:
            logger.warning(
                "SHAP explanation exceeded the %d ms budget for this "
                "transaction; the score is unaffected and is returned without "
                "an explanation.",
                round(self.explanation_timeout_s * 1000),
            )
            self.metrics.record_explanation_failure()
            return [], None, None
        except Exception as exc:  # noqa: BLE001 - explanation is best-effort
            logger.warning(
                "SHAP explanation failed for this transaction (%s: %s); the "
                "score is unaffected and is returned without an explanation.",
                type(exc).__name__,
                exc,
            )
            self.metrics.record_explanation_failure()
            return [], None, None

    def _sequence_history(
        self, features: pd.DataFrame, card_id: Optional[Any]
    ) -> Optional[pd.DataFrame]:
        """This card's stored prior feature vectors, as a frame TFT consumes."""
        if card_id is None:
            return None
        window = self.state_store.sequence(card_id)
        # A window shorter than the trained encoder length is not an error -
        # it is exactly what training saw for a card's first transactions - but
        # after a restart EVERY card is in this state until the buffer refills,
        # which is a real quality dip with no error attached (ADR-002 5.6).
        self.metrics.record_partial_sequence(len(window), self.sequence_window)
        if not window:
            return None
        return pd.DataFrame(np.vstack(window), columns=features.columns)

    def _fallback_mode(
        self, requested: EnsembleMode, explicit_mode: Optional[str]
    ) -> Optional[EnsembleMode]:
        """The pre-registered mode to retry with, or None to re-raise.

        Only ever a mode the ensemble artifact already registered WITH ITS OWN
        frozen threshold - serving never renormalizes weights over a smaller
        model set, because that changes the score distribution and invalidates
        the threshold (ADR-001 3.3). An explicitly requested mode is never
        second-guessed.
        """
        if explicit_mode is not None or requested.name == FALLBACK_MODE:
            return None
        try:
            return self.models.ensemble.mode(FALLBACK_MODE)
        except KeyError:
            logger.error(
                "Mode %r failed and no %r fallback is registered in the "
                "ensemble artifact - cannot serve without a frozen threshold.",
                requested.name,
                FALLBACK_MODE,
            )
            return None

    def _check_staleness(self, transaction: Dict[str, Any]) -> None:
        card_id = transaction.get("card1")
        dt = transaction.get("TransactionDT")
        if card_id is None or dt is None:
            return
        snapshot = self.state_store.snapshot(card_id)
        last_seen = snapshot.last_dt if snapshot.n else None
        check_staleness(float(dt), last_seen)

    def _observe(self, transaction: Dict[str, Any]) -> bool:
        """Fold the scored transaction in. False if it was not recorded.

        Returns the store's idempotency verdict so the caller can keep the
        sequence buffer in lockstep with the scalar accumulators.
        """
        card_id = transaction.get("card1")
        amount = transaction.get("TransactionAmt")
        dt = transaction.get("TransactionDT")
        if card_id is None or amount is None or dt is None:
            logger.warning(
                "Transaction lacks card1/TransactionAmt/TransactionDT — its "
                "history was not recorded, so later transactions on this card "
                "will under-count."
            )
            return False
        return self.state_store.observe(
            card_id=card_id,
            amount=float(amount),
            dt=float(dt),
            transaction_id=self._transaction_id(transaction),
        )

    @staticmethod
    def _transaction_id(transaction: Dict[str, Any]) -> Optional[str]:
        value = transaction.get("TransactionID")
        return None if value is None else str(value)
