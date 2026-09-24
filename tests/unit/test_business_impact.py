"""
tests/unit/test_business_impact.py

PRD Phase 8 — `src/evaluation/business_impact.py`.

The dollar figures in `notebooks/05_business_impact.ipynb` are the headline of
the Phase 8 deliverable, so the arithmetic behind them is pinned here rather
than trusted to a notebook cell.

Fixtures are deliberately tiny and exact — a known confusion matrix at threshold
0.5 — so every expected number below can be checked by hand against the formula
in the module docstring.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.business_impact import (  # noqa: E402
    CONVENTION_NET,
    CONVENTION_PRD,
    BusinessImpact,
    CostModel,
    TransactionVolume,
    best_threshold,
    compute_impact,
    confusion_at_threshold,
    threshold_sweep,
)

pytestmark = pytest.mark.unit


COSTS = CostModel(cost_fn=500.0, cost_fp=5.0, revenue_tp=480.0)

# 1000 annual transactions at a 10% fraud rate: 100 fraud, 900 legitimate.
# Paired with `_split(recall=0.8, fpr=0.1)` this gives 80 TP / 20 FN / 90 FP.
VOLUME = TransactionVolume(annual_transactions=1000.0, fraud_rate=0.1)


def _split(n_fraud: int, n_legit: int, recall: float, fpr: float):
    """Build labels/scores with an exactly known confusion matrix at 0.5.

    Frauds score 0.9 (caught) or 0.1 (missed); legitimate rows score 0.9 (false
    alarm) or 0.1 (correctly passed).
    """
    tp = int(round(n_fraud * recall))
    fp = int(round(n_legit * fpr))
    y_true = np.array([1] * n_fraud + [0] * n_legit)
    y_prob = np.array(
        [0.9] * tp + [0.1] * (n_fraud - tp) + [0.9] * fp + [0.1] * (n_legit - fp)
    )
    return y_true, y_prob


def _standard_split():
    return _split(n_fraud=10, n_legit=90, recall=0.8, fpr=0.1)


# ── Confusion matrix ─────────────────────────────────────────────────────────


class TestConfusionAtThreshold:
    def test_counts_match_the_constructed_split(self):
        y_true, y_prob = _standard_split()

        assert confusion_at_threshold(y_true, y_prob, 0.5) == {
            "tp": 8,
            "fn": 2,
            "fp": 9,
            "tn": 81,
        }

    def test_threshold_is_inclusive_matching_the_serving_rule(self):
        """`InferenceService` flags on `prob >= threshold`; a score sitting
        exactly on the threshold must count as flagged, not passed."""
        counts = confusion_at_threshold(np.array([1, 0]), np.array([0.5, 0.5]), 0.5)

        assert counts == {"tp": 1, "fp": 1, "fn": 0, "tn": 0}


# ── Volume scaling ───────────────────────────────────────────────────────────


class TestTransactionVolume:
    def test_dataset_scales_to_a_year(self):
        """590,540 rows over 180 days, annualised (PRD §8.1)."""
        volume = TransactionVolume.from_dataset(fraud_rate=0.035)

        assert volume.annual_transactions == pytest.approx(590_540 / 180 * 365)
        assert volume.annual_fraud == pytest.approx(volume.annual_transactions * 0.035)
        assert volume.annual_legitimate == pytest.approx(
            volume.annual_transactions * 0.965
        )

    def test_rejects_a_non_positive_span(self):
        with pytest.raises(ValueError, match="span_days must be positive"):
            TransactionVolume.from_dataset(fraud_rate=0.035, span_days=0)


# ── Cost model ───────────────────────────────────────────────────────────────


class TestCostModel:
    def test_reads_the_thresholds_block(self):
        model = CostModel.from_config(
            {"thresholds": {"cost_fn": 400, "cost_fp": 7, "revenue_tp": 390}}
        )

        assert (model.cost_fn, model.cost_fp, model.revenue_tp) == (400.0, 7.0, 390.0)

    def test_falls_back_to_prd_defaults_when_the_block_is_absent(self):
        model = CostModel.from_config({})

        assert (model.cost_fn, model.cost_fp, model.revenue_tp) == (500.0, 5.0, 480.0)

    def test_implied_swing_exposes_the_double_count(self):
        """Catching vs missing the same $500 fraud swings $980 under the PRD
        convention — the documented open question this number surfaces."""
        assert COSTS.implied_fraud_swing == 980.0
        assert COSTS.implied_fraud_swing > COSTS.cost_fn


# ── PRD convention ───────────────────────────────────────────────────────────


class TestPrdConvention:
    def test_annualised_lines_match_the_prd_formula(self):
        y_true, y_prob = _standard_split()

        impact = compute_impact(y_true, y_prob, 0.5, COSTS, VOLUME, CONVENTION_PRD)

        assert impact.annual_tp == pytest.approx(80.0)
        assert impact.annual_fn == pytest.approx(20.0)
        assert impact.annual_fp == pytest.approx(90.0)
        assert impact.gross_benefit == pytest.approx(38_400.0)  # 80 * 480
        assert impact.false_negative_loss == pytest.approx(10_000.0)  # 20 * 500
        assert impact.false_positive_cost == pytest.approx(450.0)  # 90 * 5
        assert impact.net_annual_value == pytest.approx(27_950.0)

    def test_naive_baseline_loses_every_fraud(self):
        y_true, y_prob = _standard_split()

        impact = compute_impact(y_true, y_prob, 0.5, COSTS, VOLUME, CONVENTION_PRD)

        assert impact.naive_baseline_value == pytest.approx(-50_000.0)  # 100 * 500
        assert impact.improvement_over_naive == pytest.approx(77_950.0)

    def test_rates_not_raw_counts_drive_the_projection(self):
        """Doubling the evaluation split must not change the annual report —
        rates are measured, then applied to the annual volume."""
        small = compute_impact(*_standard_split(), 0.5, COSTS, VOLUME)
        large = compute_impact(
            *_split(n_fraud=100, n_legit=900, recall=0.8, fpr=0.1), 0.5, COSTS, VOLUME
        )

        assert large.net_annual_value == pytest.approx(small.net_annual_value)


# ── Net-of-principal convention ──────────────────────────────────────────────


class TestNetOfPrincipalConvention:
    def test_missed_fraud_is_not_charged_twice(self):
        """Under the corrected convention a missed fraud carries no extra
        charge — the loss is the baseline, not an additional cost line."""
        y_true, y_prob = _standard_split()

        impact = compute_impact(y_true, y_prob, 0.5, COSTS, VOLUME, CONVENTION_NET)

        assert impact.false_negative_loss == 0.0
        assert impact.naive_baseline_value == 0.0
        assert impact.net_annual_value == pytest.approx(37_950.0)  # 38400 - 450

    def test_the_two_conventions_differ_by_the_double_counted_book(self):
        """The PRD convention charges the whole annual fraud book a second time
        on the FN line; that constant is the entire discrepancy."""
        y_true, y_prob = _standard_split()

        prd = compute_impact(y_true, y_prob, 0.5, COSTS, VOLUME, CONVENTION_PRD)
        net = compute_impact(y_true, y_prob, 0.5, COSTS, VOLUME, CONVENTION_NET)

        assert net.net_annual_value - prd.net_annual_value == pytest.approx(
            prd.annual_fn * COSTS.cost_fn
        )

    def test_rejects_an_unknown_convention(self):
        y_true, y_prob = _standard_split()

        with pytest.raises(ValueError, match="convention must be one of"):
            compute_impact(y_true, y_prob, 0.5, COSTS, convention="guesswork")


# ── Operational ratio ────────────────────────────────────────────────────────


class TestFalsePositivesPerFraudCaught:
    def test_reports_the_customer_friction_ratio(self):
        y_true, y_prob = _standard_split()

        impact = compute_impact(y_true, y_prob, 0.5, COSTS, VOLUME)

        assert impact.false_positives_per_fraud_caught == pytest.approx(90 / 80)

    def test_is_infinite_when_nothing_is_caught(self):
        """A threshold catching no fraud has no meaningful ratio; it must not
        divide by zero and must not silently report a flattering 0."""
        y_true, y_prob = _split(n_fraud=10, n_legit=90, recall=0.0, fpr=0.1)

        impact = compute_impact(y_true, y_prob, 0.5, COSTS, VOLUME)

        assert impact.false_positives_per_fraud_caught == float("inf")


# ── Sweep ────────────────────────────────────────────────────────────────────


class TestThresholdSweep:
    def test_grid_reaches_below_the_shipped_operating_point(self):
        """The deployed threshold is ~0.006; the PRD's sketched 0.01-0.99 grid
        would not contain it and the sensitivity plot would miss the peak."""
        sweep = threshold_sweep(*_standard_split(), COSTS)

        assert min(i.threshold for i in sweep) <= 0.006
        assert len(sweep) == 200

    def test_best_threshold_maximises_net_value(self):
        sweep = threshold_sweep(*_standard_split(), COSTS)

        assert best_threshold(sweep).net_annual_value == max(
            i.net_annual_value for i in sweep
        )

    def test_empty_sweep_raises_rather_than_returning_none(self):
        with pytest.raises(ValueError, match="sweep is empty"):
            best_threshold([])

    def test_honours_an_explicit_grid(self):
        sweep = threshold_sweep(*_standard_split(), COSTS, thresholds=[0.2, 0.8])

        assert [i.threshold for i in sweep] == [0.2, 0.8]


# ── Serialisation ────────────────────────────────────────────────────────────


class TestAsDict:
    def test_carries_the_derived_fields(self):
        """`as_dict` feeds the notebook's summary table, so the derived
        properties must survive the round-trip, not just the stored fields."""
        payload = compute_impact(*_standard_split(), 0.5, COSTS, VOLUME).as_dict()

        assert payload["improvement_over_naive"] == pytest.approx(77_950.0)
        assert payload["false_positives_per_fraud_caught"] == pytest.approx(90 / 80)
        assert payload["convention"] == CONVENTION_PRD

    def test_impact_is_immutable(self):
        """Report figures get quoted downstream; they must not be editable in
        place after the fact."""
        impact = compute_impact(*_standard_split(), 0.5, COSTS, VOLUME)

        with pytest.raises(Exception):
            impact.net_annual_value = 0.0  # type: ignore[misc]

        assert isinstance(impact, BusinessImpact)
