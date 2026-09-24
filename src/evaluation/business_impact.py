"""
src/evaluation/business_impact.py

Translate confusion-matrix outcomes into annualised money (PRD Phase 8 / FR-09).

`ModelEvaluator._business_value` already scores a threshold in *dataset* units —
it is the objective the operating threshold is selected against. This module is
the reporting counterpart: it scales those same outcomes to an annual book of
transactions and decomposes the result into the lines a business reader expects
(gross benefit, missed-fraud loss, false-alarm cost, net value, and the lift
over a do-nothing baseline).

Kept out of the notebook on purpose. The dollar figures are the headline of the
Phase 8 deliverable, so the arithmetic behind them is worth unit tests; a
notebook cell is not testable and silently rots when the numbers move.

---

**Two accounting conventions, and why both exist.**

The PRD (§8.1) specifies:

    gross_benefit  = annual_tp * revenue_tp     # 480
    fn_loss        = annual_fn * cost_fn        # 500
    fp_cost        = annual_fp * cost_fp        #   5
    net            = gross_benefit - fn_loss - fp_cost

Read literally, a caught fraud earns +480 while a missed one costs -500, so the
swing between catching and missing the *same* $500 fraud is $980 — nearly twice
the amount actually at stake. The recovered principal is counted once as revenue
and again as an avoided loss. That inflates the fraud side of the trade relative
to the $5 false-positive cost, which is why the cost-optimal threshold sits near
0.006 and accepts tens of thousands of false positives per fraud caught.

This is a known open item (ADR-001 §6.3, `docs/IMPLEMENTATION_PLAN.md`) and it
is a *business* question, not an arithmetic bug: whether `revenue_tp` means
recovery net of principal or the principal itself is a modelling choice only the
business can settle.

So both are computed, and neither is hidden:

* :data:`CONVENTION_PRD` reproduces the PRD formula verbatim. It is the
  documented deliverable and the objective the shipped threshold was chosen
  under, so it must remain reportable.
* :data:`CONVENTION_NET` counts each fraud exactly once, measured against a
  world where the fraud happens anyway: a caught fraud recovers `revenue_tp`, a
  missed one recovers nothing (no second charge for the same principal), and
  the do-nothing baseline is therefore 0 rather than the full annual fraud loss.

Reporting both is the point. They rank thresholds differently precisely because
the FN term dominates the FP term under the PRD convention and vanishes under
the other, which is the whole substance of the open question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

# PRD §8.1: the IEEE-CIS training file spans ~180 days of transactions.
DATASET_TRANSACTIONS = 590_540
DATASET_SPAN_DAYS = 180
DAYS_PER_YEAR = 365

CONVENTION_PRD = "prd"
CONVENTION_NET = "net_of_principal"
_CONVENTIONS = (CONVENTION_PRD, CONVENTION_NET)


@dataclass(frozen=True)
class CostModel:
    """The per-outcome dollar amounts, straight from `config.yaml:thresholds`."""

    cost_fn: float = 500.0
    cost_fp: float = 5.0
    revenue_tp: float = 480.0

    @classmethod
    def from_config(cls, config: Dict) -> "CostModel":
        """Build from a loaded `config.yaml` mapping.

        Reads the same `thresholds` block the evaluator uses, so the report and
        the threshold selection can never drift onto different numbers.
        """
        thresholds = config.get("thresholds", {}) or {}
        return cls(
            cost_fn=float(thresholds.get("cost_fn", 500.0)),
            cost_fp=float(thresholds.get("cost_fp", 5.0)),
            revenue_tp=float(thresholds.get("revenue_tp", 480.0)),
        )

    @property
    def implied_fraud_swing(self) -> float:
        """Value difference between catching and missing one fraud, as written.

        ``revenue_tp + cost_fn`` under the PRD convention. Exposed because it is
        the single number that makes the double-count visible: when it exceeds
        `cost_fn`, the same principal is being counted twice.
        """
        return self.revenue_tp + self.cost_fn


@dataclass(frozen=True)
class TransactionVolume:
    """The annual book the dataset-scale confusion matrix is projected onto."""

    annual_transactions: float
    fraud_rate: float

    @property
    def annual_fraud(self) -> float:
        return self.annual_transactions * self.fraud_rate

    @property
    def annual_legitimate(self) -> float:
        return self.annual_transactions - self.annual_fraud

    @classmethod
    def from_dataset(
        cls,
        fraud_rate: float,
        n_transactions: int = DATASET_TRANSACTIONS,
        span_days: int = DATASET_SPAN_DAYS,
    ) -> "TransactionVolume":
        """Scale the dataset's own volume to a year (PRD §8.1).

        `fraud_rate` is a parameter rather than the PRD's hardcoded 0.035 so the
        report describes the split it is actually measuring.
        """
        if span_days <= 0:
            raise ValueError(f"span_days must be positive, got {span_days}")
        daily = n_transactions / span_days
        return cls(annual_transactions=daily * DAYS_PER_YEAR, fraud_rate=fraud_rate)


@dataclass(frozen=True)
class BusinessImpact:
    """Annualised outcome of operating the model at one threshold."""

    threshold: float
    convention: str
    recall: float
    false_positive_rate: float
    precision: float
    annual_transactions: float
    annual_fraud: float
    annual_tp: float
    annual_fn: float
    annual_fp: float
    gross_benefit: float
    false_negative_loss: float
    false_positive_cost: float
    net_annual_value: float
    naive_baseline_value: float

    @property
    def improvement_over_naive(self) -> float:
        """Net value gained versus flagging nothing at all."""
        return self.net_annual_value - self.naive_baseline_value

    @property
    def false_positives_per_fraud_caught(self) -> float:
        """Legitimate customers blocked for each fraud stopped.

        The operational cost the dollar total hides: a strongly net-positive
        model can still be unshippable if this ratio runs to the thousands.
        """
        if self.annual_tp <= 0:
            return float("inf")
        return self.annual_fp / self.annual_tp

    def as_dict(self) -> Dict[str, float]:
        return {
            "threshold": self.threshold,
            "convention": self.convention,
            "recall": self.recall,
            "false_positive_rate": self.false_positive_rate,
            "precision": self.precision,
            "annual_transactions": self.annual_transactions,
            "annual_fraud": self.annual_fraud,
            "annual_tp": self.annual_tp,
            "annual_fn": self.annual_fn,
            "annual_fp": self.annual_fp,
            "gross_benefit": self.gross_benefit,
            "false_negative_loss": self.false_negative_loss,
            "false_positive_cost": self.false_positive_cost,
            "net_annual_value": self.net_annual_value,
            "naive_baseline_value": self.naive_baseline_value,
            "improvement_over_naive": self.improvement_over_naive,
            "false_positives_per_fraud_caught": self.false_positives_per_fraud_caught,
        }


def confusion_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, int]:
    """Confusion counts at one threshold, using the serving comparison.

    `>=` matches `InferenceService`'s decision rule; using `>` here would make
    the report describe a slightly different model than the one deployed.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "tp": int(np.sum((y_pred == 1) & (y_true == 1))),
        "fp": int(np.sum((y_pred == 1) & (y_true == 0))),
        "fn": int(np.sum((y_pred == 0) & (y_true == 1))),
        "tn": int(np.sum((y_pred == 0) & (y_true == 0))),
    }


def compute_impact(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_model: CostModel,
    volume: Optional[TransactionVolume] = None,
    convention: str = CONVENTION_PRD,
) -> BusinessImpact:
    """Project a threshold's confusion matrix onto an annual book of business.

    Rates (recall, FPR) are measured on the supplied split and applied to
    `volume`; raw counts are never scaled directly, so the report does not move
    when the evaluation split changes size.
    """
    if convention not in _CONVENTIONS:
        raise ValueError(
            f"convention must be one of {_CONVENTIONS}, got {convention!r}"
        )

    y_true = np.asarray(y_true).astype(int)
    counts = confusion_at_threshold(y_true, y_prob, threshold)
    n_fraud = counts["tp"] + counts["fn"]
    n_legit = counts["fp"] + counts["tn"]

    recall = counts["tp"] / n_fraud if n_fraud else 0.0
    fpr = counts["fp"] / n_legit if n_legit else 0.0
    flagged = counts["tp"] + counts["fp"]
    precision = counts["tp"] / flagged if flagged else 0.0

    if volume is None:
        volume = TransactionVolume.from_dataset(fraud_rate=float(y_true.mean()))

    annual_tp = volume.annual_fraud * recall
    annual_fn = volume.annual_fraud * (1.0 - recall)
    annual_fp = volume.annual_legitimate * fpr

    gross_benefit = annual_tp * cost_model.revenue_tp
    if convention == CONVENTION_PRD:
        # PRD §8.1 verbatim, double-count included. See the module docstring.
        fn_loss = annual_fn * cost_model.cost_fn
        naive = -(volume.annual_fraud * cost_model.cost_fn)
    else:
        # Each fraud counted once, measured against "the fraud happens anyway".
        # A caught fraud recovers `revenue_tp`; a missed one recovers nothing,
        # so the FN line is zero rather than a second charge for the same money.
        fn_loss = 0.0
        naive = 0.0

    fp_cost = annual_fp * cost_model.cost_fp

    return BusinessImpact(
        threshold=float(threshold),
        convention=convention,
        recall=recall,
        false_positive_rate=fpr,
        precision=precision,
        annual_transactions=volume.annual_transactions,
        annual_fraud=volume.annual_fraud,
        annual_tp=annual_tp,
        annual_fn=annual_fn,
        annual_fp=annual_fp,
        gross_benefit=gross_benefit,
        false_negative_loss=fn_loss,
        false_positive_cost=fp_cost,
        net_annual_value=gross_benefit - fn_loss - fp_cost,
        naive_baseline_value=naive,
    )


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_model: CostModel,
    thresholds: Optional[Sequence[float]] = None,
    volume: Optional[TransactionVolume] = None,
    convention: str = CONVENTION_PRD,
) -> List[BusinessImpact]:
    """Net annual value across a threshold grid (PRD §8.1 Section 3).

    Defaults to a log-spaced grid, mirroring
    `ModelEvaluator._select_threshold_grid`: the operating point sits near
    0.006, so the linear 0.01–0.99 grid the PRD sketches would not contain it
    and the sensitivity plot would miss the peak entirely.
    """
    if thresholds is None:
        thresholds = np.logspace(np.log10(1e-4), np.log10(0.99), 200)
    if volume is None:
        volume = TransactionVolume.from_dataset(
            fraud_rate=float(np.asarray(y_true).astype(int).mean())
        )
    return [
        compute_impact(y_true, y_prob, float(t), cost_model, volume, convention)
        for t in thresholds
    ]


def best_threshold(sweep: Sequence[BusinessImpact]) -> BusinessImpact:
    """The sweep entry with the highest net annual value."""
    if not sweep:
        raise ValueError("sweep is empty")
    return max(sweep, key=lambda impact: impact.net_annual_value)
