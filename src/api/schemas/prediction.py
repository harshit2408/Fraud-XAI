"""
src/api/schemas/prediction.py

Request/response contract for `POST /predict` (Phase E task **E5**).

E5's done-when is "invalid input rejected at the boundary". Everything below is
therefore expressed as a Pydantic constraint rather than as a hand-written
check inside the route, per `.cursor/rules/python-fastapi.mdc` ("use field
constraints instead of hand-written validation when Pydantic can express the
rule") — a rejected request never reaches the transform, the models, or the
feature-state store, so a malformed payload can never mutate card history.

Two validations are specific to this model and worth calling out:

  - **Range.** `TransactionAmt` must be positive and finite. The batch
    pipeline derives `amount_log = log1p(amt)` and divides by per-card means;
    a negative or NaN amount yields NaN features that tree models silently
    accept, producing a confident-looking score from garbage.
  - **Staleness.** `TransactionDT` is a seconds-offset from the dataset epoch,
    not wall-clock time. A transaction whose timestamp falls behind the card's
    last-seen timestamp cannot be scored correctly: the expanding aggregates
    assume per-card chronological order (ADR-002 §5.3), and an out-of-order row
    produces a negative `time_since_last_tx` the model never saw in training.
    Purely per-request rules (finiteness, ranges) are enforced by the model
    below; the staleness rule needs the card's last-seen timestamp and so is
    applied by `src.serving.staleness.check_staleness` at the point where that
    state is available.
    Either way the request is refused rather than silently clamped.
"""

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Guards against a payload that would blow out the feature frame; the real
# IEEE-CIS schema is ~430 raw columns.
MAX_RAW_FIELDS = 600


class TransactionRequest(BaseModel):
    """One transaction to score.

    Only the fields the feature pipeline actually reads are typed explicitly;
    the remaining IEEE-CIS columns (V1-V339, C1-C14, D1-D15, id_*, M1-M9) are
    accepted through `extra="allow"` so the schema does not need 430
    declarations, but their count is still bounded by `MAX_RAW_FIELDS`.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    TransactionID: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Client-supplied unique id; used for the idempotency guard.",
    )
    TransactionDT: float = Field(
        ...,
        ge=0,
        description="Seconds offset from the dataset epoch (NOT unix time).",
    )
    TransactionAmt: float = Field(
        ...,
        gt=0,
        le=1_000_000,
        description=(
            "Transaction amount. Must be positive: log1p and per-card ratios "
            "are undefined otherwise."
        ),
    )
    card1: int = Field(
        ...,
        description=(
            "Primary card identifier; the entity key for every expanding "
            "aggregate and the Kafka partition key (ADR-002 §5.3)."
        ),
    )

    @field_validator("TransactionAmt", "TransactionDT")
    @classmethod
    def _reject_non_finite(cls, value: float) -> float:
        """NaN and infinity pass `gt`/`ge` comparisons silently in Python, so
        the bound constraints above are not sufficient on their own."""
        if not math.isfinite(value):
            raise ValueError("must be a finite number, got NaN or infinity")
        return value

    @field_validator("TransactionID")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank or whitespace-only")
        return value.strip()

    def model_post_init(self, __context: Any) -> None:
        extras = self.__pydantic_extra__ or {}
        if len(extras) > MAX_RAW_FIELDS:
            raise ValueError(
                f"Transaction carries {len(extras)} extra fields, exceeding the "
                f"{MAX_RAW_FIELDS}-field bound."
            )

    def to_features(self) -> Dict[str, Any]:
        """Flatten to the plain dict the transform layer consumes."""
        return self.model_dump()


class FeatureContribution(BaseModel):
    """One SHAP contribution. Populated in PRD Phase 4."""

    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    """The scoring decision, plus the provenance needed to audit it.

    `model_version` is required by E5 and is the string ADR-001 §4.4 defines:
    ``{dataset_hash[:12]}-{config_hash[:12]}-{git_sha[:7]}``. Because startup
    validated that every artifact shares those hashes, it identifies the whole
    serving stack rather than any single model.
    """

    model_config = ConfigDict(protected_namespaces=())

    transaction_id: Optional[str]
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., description="FRAUD or LEGITIMATE")
    threshold: float = Field(
        ...,
        description=(
            "The frozen operating point for the active mode. Never recomputed "
            "at request time."
        ),
    )
    model_version: str
    mode: str = Field(..., description="Which pre-registered blend scored this.")
    degraded: bool = Field(
        ...,
        description=(
            "True when a fallback mode was used instead of the default blend."
        ),
    )
    model_probabilities: Dict[str, float] = Field(default_factory=dict)
    degraded_reason: Optional[str] = Field(
        default=None,
        description="Why a fallback mode was used, when degraded is true.",
    )
    explanation: List[FeatureContribution] = Field(
        default_factory=list,
        description=(
            "Top SHAP contributions, largest signed magnitude first. Empty "
            "when explainability is disabled or the explainer failed for this "
            "row; when populated it explains the XGBoost component only "
            "(ADR-001 §3.4), not the blend."
        ),
    )
    explained_model: Optional[str] = Field(
        default=None,
        description=(
            "Which model `explanation` speaks for — always 'xgb' when "
            "populated. TreeSHAP is exact for the XGBoost booster; it is not "
            "an explanation of the ensemble decision (ADR-001 §3.4)."
        ),
    )
    explained_weight: Optional[float] = Field(
        default=None,
        description=(
            "The explained model's weight in the default blend, so a client "
            "can state how much of the decision the explanation covers."
        ),
    )
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    """Liveness plus enough state to tell a warm instance from a broken one."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    uptime_seconds: float
    model_version: Optional[str] = None
    models: List[str] = Field(default_factory=list)
    known_cards: Optional[int] = None
    # PRD Phase 6 done-when: `GET /health` confirms `kafka_consumer_running`.
    # False whenever the consumer was never started (tests, models skipped, or
    # the broker was unreachable at startup — HTTP scoring still works in that
    # last case), so a probe can tell a streaming-enabled instance from one
    # serving HTTP only.
    kafka_consumer_running: bool = False
    kafka_messages_processed: int = 0
    kafka_alerts_published: int = 0
    kafka_consumer_errors: int = 0
    # PRD §6.4 budgets the consumer backlog at < 500 during the streaming demo.
    # -1 means "not yet known" (no partition assignment, or no high-water mark
    # from the broker) and is deliberately distinct from a caught-up 0.
    kafka_consumer_lag: int = -1
    observability: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Counters for the silent-degradation paths: partial TFT sequence "
            "windows, out-of-order and duplicate transactions, target-encoding "
            "staleness, and cross-artifact config mismatch (ADR-002 5.3/5.5)."
        ),
    )


class ErrorResponse(BaseModel):
    """Boundary rejection. Deliberately carries no internal detail."""

    detail: str
    error_type: str = "validation_error"
