"""
src/evaluation/phase9_report.py

The fixed evaluation battery PRD Phase 9 re-runs after every step (§10, 9.0
onward). One vector of blended test probabilities and one vector of labels go
in; out comes the standard row every step's write-up needs:

  - test PR-AUC and ROC-AUC
  - precision_at_recall at recall floors 70 / 80 / 90%  (step 9.0's utility)
  - confusion matrix + precision/recall at the *deployed* operating threshold
  - confusion matrix + precision/recall at a *re-derived* threshold (9.1: the
    cost model can move it) when one is supplied
  - business_impact.compute_impact at the deployed threshold, both accounting
    conventions (PRD double-count and net-of-principal)
  - the target-band verdict: precision >= 30% at recall >= 80% at the operating
    threshold that is actually deployed

Kept as a module, not notebook cells, for the same reason as
`business_impact.py`: the numbers gate whether the phase stops, so the
arithmetic behind them is unit-tested.

This module does no model loading and no retraining. `scripts/run_phase9_eval.py`
is the thin CLI that feeds it `reports/ensemble_test_probabilities.npz`
(regenerated per step by `scripts/export_test_probabilities.py`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

from src.evaluation.business_impact import (
    CONVENTION_NET,
    CONVENTION_PRD,
    CostModel,
    TransactionVolume,
    compute_impact,
)
from src.evaluation.evaluator import ModelEvaluator

# PRD §10 Phase 9 target band: precision at least this, at recall at least
# this, measured at the deployed operating threshold.
TARGET_PRECISION = 0.30
TARGET_RECALL = 0.80

# The recall floors every step reports precision at (Phase 9 done-when:
# "including precision-at-recall for at least recall targets 70%, 80%, 90%").
RECALL_FLOORS: tuple[float, ...] = (0.70, 0.80, 0.90)


@dataclass(frozen=True)
class ThresholdReport:
    """Confusion matrix and rates at one decision threshold."""

    label: str
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "label": self.label,
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class Phase9Report:
    """The full post-step evaluation battery for one candidate model."""

    step: str
    n_rows: int
    n_fraud: int
    pr_auc: float
    roc_auc: float
    deployed_threshold: float
    precision_at_recall: Dict[float, Dict[str, float]]
    at_deployed_threshold: ThresholdReport
    at_rederived_threshold: Optional[ThresholdReport]
    business_impact_prd: Dict[str, float]
    business_impact_net: Dict[str, float]
    target_precision: float = TARGET_PRECISION
    target_recall: float = TARGET_RECALL
    # Provenance of the probability vector: the mlflow_run_id (or a hash) from
    # models/ensemble.json at export time, so a stale .npz scored against an
    # older ensemble is detectable in the report itself (mle-reviewer,
    # 2026-09-08). None when the caller did not supply it.
    ensemble_spec_stamp: Optional[str] = None

    @property
    def meets_target_band(self) -> bool:
        """Phase 9 stop condition, checked at the DEPLOYED operating point.

        The band is defined against "the model's deployed operating threshold"
        (PRD §10), so a re-derived threshold that happens to clear it does not
        stop the sequence unless it is the one actually promoted into
        `models/ensemble.json`.
        """
        r = self.at_deployed_threshold
        return r.precision >= self.target_precision and r.recall >= self.target_recall

    @property
    def target_band_gap(self) -> Dict[str, float]:
        """How far the deployed operating point is from each side of the band.

        Positive = shortfall (still need this much more); <= 0 = that side is
        met. Recorded verbatim in the tracking section per the done-when item
        "the reason for stopping ... or continuing (target not yet met, by how
        much) is recorded".
        """
        r = self.at_deployed_threshold
        return {
            "precision_shortfall": self.target_precision - r.precision,
            "recall_shortfall": self.target_recall - r.recall,
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "ensemble_spec_stamp": self.ensemble_spec_stamp,
            "n_rows": self.n_rows,
            "n_fraud": self.n_fraud,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "deployed_threshold": self.deployed_threshold,
            "precision_at_recall": {
                f"{k:.2f}": v for k, v in self.precision_at_recall.items()
            },
            "at_deployed_threshold": self.at_deployed_threshold.as_dict(),
            "at_rederived_threshold": (
                self.at_rederived_threshold.as_dict()
                if self.at_rederived_threshold is not None
                else None
            ),
            "business_impact_prd": self.business_impact_prd,
            "business_impact_net": self.business_impact_net,
            "target_precision": self.target_precision,
            "target_recall": self.target_recall,
            "meets_target_band": self.meets_target_band,
            "target_band_gap": self.target_band_gap,
        }


def _threshold_report(
    evaluator: ModelEvaluator,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label: str,
) -> ThresholdReport:
    m = evaluator.compute_metrics_at_threshold(y_true, y_prob, threshold)
    return ThresholdReport(
        label=label,
        threshold=float(threshold),
        tp=int(m["TP"]),
        fp=int(m["FP"]),
        fn=int(m["FN"]),
        tn=int(m["TN"]),
        precision=float(m["precision"]),
        recall=float(m["recall"]),
        f1=float(m["f1"]),
    )


def build_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    deployed_threshold: float,
    cost_model: CostModel,
    step: str,
    rederived_threshold: Optional[float] = None,
    volume: Optional[TransactionVolume] = None,
    recall_floors: Sequence[float] = RECALL_FLOORS,
    ensemble_spec_stamp: Optional[str] = None,
) -> Phase9Report:
    """Run the Phase 9 evaluation battery on one probability vector.

    Args:
        y_true: test-split ground truth (0/1).
        y_prob: blended calibrated fraud probability for the same rows.
        deployed_threshold: the operating threshold in `models/ensemble.json`
            — the point the target band is judged at.
        cost_model: from `CostModel.from_config(config)`.
        step: label for this run, e.g. "9.0 baseline" or "9.2 UID features".
        rederived_threshold: optional second threshold to also report a
            confusion matrix at (9.1 re-derives one from the resolved cost
            model; later steps may pass the value they would deploy).
        volume: annual transaction book for the business-impact projection;
            defaults to the dataset-scaled volume at this split's own fraud
            rate (matches `business_impact` defaults).
        recall_floors: recall targets for `precision_at_recall`.

    Returns:
        A frozen `Phase9Report`.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    evaluator = ModelEvaluator()

    if volume is None:
        volume = TransactionVolume.from_dataset(fraud_rate=float(y_true.mean()))

    par: Dict[float, Dict[str, float]] = {}
    for floor in recall_floors:
        try:
            par[float(floor)] = evaluator.precision_at_recall(y_true, y_prob, floor)
        except ValueError as exc:
            # A floor the curve cannot reach is recorded, not fatal — the
            # report still has to render for the other floors and the band
            # check — but it must not pass silently (coding-style.md).
            logger.warning(
                "precision_at_recall floor %.2f unreachable for step %r: %s",
                floor,
                step,
                exc,
            )
            par[float(floor)] = {
                "target_recall": float(floor),
                "achieved_recall": float("nan"),
                "precision": float("nan"),
                "threshold": float("nan"),
                "f1": float("nan"),
                "error": str(exc),
            }

    at_deployed = _threshold_report(
        evaluator, y_true, y_prob, deployed_threshold, "deployed"
    )
    at_rederived = (
        _threshold_report(
            evaluator, y_true, y_prob, rederived_threshold, "re-derived"
        )
        if rederived_threshold is not None
        else None
    )

    impact_prd = compute_impact(
        y_true, y_prob, deployed_threshold, cost_model, volume, CONVENTION_PRD
    ).as_dict()
    impact_net = compute_impact(
        y_true, y_prob, deployed_threshold, cost_model, volume, CONVENTION_NET
    ).as_dict()

    return Phase9Report(
        step=step,
        ensemble_spec_stamp=ensemble_spec_stamp,
        n_rows=int(len(y_true)),
        n_fraud=int(y_true.sum()),
        pr_auc=evaluator.compute_pr_auc(y_true, y_prob),
        roc_auc=evaluator.compute_roc_auc(y_true, y_prob),
        deployed_threshold=float(deployed_threshold),
        precision_at_recall=par,
        at_deployed_threshold=at_deployed,
        at_rederived_threshold=at_rederived,
        business_impact_prd=impact_prd,
        business_impact_net=impact_net,
    )


def format_report(report: Phase9Report) -> str:
    """Human-readable block for stdout and for pasting into RESULTS.md."""
    lines: List[str] = []
    a = lines.append
    a(f"=== Phase 9 evaluation — {report.step} ===")
    if report.ensemble_spec_stamp:
        a(f"  ensemble spec: {report.ensemble_spec_stamp}")
    a(f"  rows: {report.n_rows:,}   fraud: {report.n_fraud:,} "
      f"({report.n_fraud / report.n_rows:.4%})")
    a(f"  test PR-AUC:  {report.pr_auc:.6f}")
    a(f"  test ROC-AUC: {report.roc_auc:.6f}")
    a("  precision_at_recall:")
    for floor, v in sorted(report.precision_at_recall.items()):
        if v.get("error"):
            a(f"    recall>={floor:.0%}: unreachable ({v['error']})")
        else:
            a(f"    recall>={floor:.0%}: precision {v['precision']:.4f} "
              f"@ threshold {v['threshold']:.6f} "
              f"(achieved recall {v['achieved_recall']:.4f})")
    for r in (report.at_deployed_threshold, report.at_rederived_threshold):
        if r is None:
            continue
        a(f"  @ {r.label} threshold {r.threshold:.6f}: "
          f"P={r.precision:.4f} R={r.recall:.4f} F1={r.f1:.4f}  "
          f"TP={r.tp} FP={r.fp} FN={r.fn} TN={r.tn}")
    bi = report.business_impact_prd
    bn = report.business_impact_net
    a("  business impact @ deployed threshold:")
    a(f"    PRD convention:      net ${bi['net_annual_value']:,.0f}  "
      f"({bi['false_positives_per_fraud_caught']:.1f} FP per fraud caught)")
    a(f"    net-of-principal:    net ${bn['net_annual_value']:,.0f}")
    gap = report.target_band_gap
    verdict = "MET" if report.meets_target_band else "NOT MET"
    a(f"  target band (P>={report.target_precision:.0%} @ R>={report.target_recall:.0%} "
      f"at deployed threshold): {verdict}")
    if not report.meets_target_band:
        a(f"    precision shortfall: {gap['precision_shortfall']:+.4f}   "
          f"recall shortfall: {gap['recall_shortfall']:+.4f}")
    return "\n".join(lines)
