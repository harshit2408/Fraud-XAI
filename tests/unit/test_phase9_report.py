"""Unit tests for src/evaluation/phase9_report.py (PRD Phase 9 battery)."""

import numpy as np
import pytest

from src.evaluation.business_impact import CostModel
from src.evaluation.phase9_report import (
    RECALL_FLOORS,
    TARGET_PRECISION,
    build_report,
    format_report,
)


@pytest.fixture
def cost_model():
    return CostModel(cost_fn=500.0, cost_fp=5.0, revenue_tp=480.0)


@pytest.fixture
def separable_scores():
    """1,000 rows, 5% fraud, positives ranked strictly above negatives so the
    target band is comfortably met and the numbers are checkable."""
    rng = np.random.default_rng(42)
    n, n_fraud = 1000, 50
    y_true = np.zeros(n, dtype=int)
    y_true[:n_fraud] = 1
    y_prob = np.empty(n, dtype=float)
    y_prob[:n_fraud] = rng.uniform(0.80, 0.99, size=n_fraud)
    y_prob[n_fraud:] = rng.uniform(0.00, 0.40, size=n - n_fraud)
    return y_true, y_prob


@pytest.fixture
def hard_scores():
    """5% fraud with heavy score overlap: at a low deployed threshold recall is
    high but precision is far below the band."""
    rng = np.random.default_rng(0)
    n, n_fraud = 2000, 100
    y_true = np.zeros(n, dtype=int)
    y_true[:n_fraud] = 1
    y_prob = np.empty(n, dtype=float)
    y_prob[:n_fraud] = rng.uniform(0.05, 0.60, size=n_fraud)
    y_prob[n_fraud:] = rng.uniform(0.00, 0.55, size=n - n_fraud)
    return y_true, y_prob


def test_build_report_reports_every_recall_floor(separable_scores, cost_model):
    y_true, y_prob = separable_scores
    report = build_report(y_true, y_prob, 0.5, cost_model, step="unit")
    assert set(report.precision_at_recall) == {float(f) for f in RECALL_FLOORS}
    for v in report.precision_at_recall.values():
        assert "error" not in v
        assert 0.0 <= v["precision"] <= 1.0
        assert v["achieved_recall"] >= v["target_recall"] - 1e-9


def test_meets_target_band_true_on_separable_scores(separable_scores, cost_model):
    y_true, y_prob = separable_scores
    report = build_report(y_true, y_prob, 0.5, cost_model, step="unit")
    assert report.at_deployed_threshold.precision == pytest.approx(1.0)
    assert report.at_deployed_threshold.recall == pytest.approx(1.0)
    assert report.meets_target_band is True
    gap = report.target_band_gap
    assert gap["precision_shortfall"] <= 0
    assert gap["recall_shortfall"] <= 0


def test_meets_target_band_false_and_gap_quantified(hard_scores, cost_model):
    y_true, y_prob = hard_scores
    report = build_report(y_true, y_prob, 0.01, cost_model, step="unit")
    assert report.meets_target_band is False
    gap = report.target_band_gap
    assert gap["precision_shortfall"] == pytest.approx(
        TARGET_PRECISION - report.at_deployed_threshold.precision
    )


def test_band_judged_at_deployed_not_rederived_threshold(hard_scores, cost_model):
    """A re-derived threshold that would clear the band must not flip
    meets_target_band — only the deployed operating point counts (PRD §10)."""
    y_true, y_prob = hard_scores
    report = build_report(
        y_true, y_prob, 0.01, cost_model, step="unit", rederived_threshold=0.55
    )
    assert report.at_rederived_threshold is not None
    assert report.at_rederived_threshold.label == "re-derived"
    assert report.meets_target_band is False


def test_rederived_threshold_report_matches_compute_metrics_at_threshold(
    hard_scores, cost_model
):
    """The re-derived confusion matrix must be the real one at that threshold,
    not a copy of the deployed one or a mislabelled duplicate."""
    from src.evaluation.evaluator import ModelEvaluator

    y_true, y_prob = hard_scores
    t = 0.30
    report = build_report(
        y_true, y_prob, 0.01, cost_model, step="unit", rederived_threshold=t
    )
    r = report.at_rederived_threshold
    assert r.tp + r.fp + r.fn + r.tn == report.n_rows
    assert r.tp + r.fn == report.n_fraud

    direct = ModelEvaluator().compute_metrics_at_threshold(y_true, y_prob, t)
    assert r.tp == direct["TP"]
    assert r.fp == direct["FP"]
    assert r.precision == pytest.approx(direct["precision"])
    assert r.recall == pytest.approx(direct["recall"])
    # And it is genuinely a different operating point than the deployed one.
    assert r.threshold != report.at_deployed_threshold.threshold


def test_unreachable_recall_floor_records_error_row_and_still_builds(cost_model, caplog):
    """If a floor is unreachable, build_report must log a warning, mark that
    floor's row with an 'error' key, and still produce a usable report for
    the other floors and the band check."""
    import logging

    # A model that caps recall well below 0.90: positives are spread but the
    # highest-recall curve point (everything flagged) is forced below 0.9 by
    # making most positives share the negatives' minimum score is not
    # possible (see evaluator test) — instead request a floor of 0.999 which
    # the discrete curve rounds past on a small sample.
    rng = np.random.default_rng(7)
    n, n_fraud = 400, 20
    y_true = np.zeros(n, dtype=int)
    y_true[:n_fraud] = 1
    y_prob = np.empty(n, dtype=float)
    y_prob[:n_fraud] = rng.uniform(0.3, 0.9, size=n_fraud)
    y_prob[n_fraud:] = rng.uniform(0.0, 0.5, size=n - n_fraud)

    with caplog.at_level(logging.WARNING):
        report = build_report(
            y_true, y_prob, 0.2, cost_model, step="unit", recall_floors=(0.80, 1.01)
        )
    # 1.01 is out of range for precision_at_recall -> ValueError -> error row.
    assert "error" in report.precision_at_recall[1.01]
    assert np.isnan(report.precision_at_recall[1.01]["precision"])
    # The reachable floor is unaffected.
    assert "error" not in report.precision_at_recall[0.80]
    # And the warning was logged, not swallowed.
    assert any("unreachable" in m for m in caplog.messages)
    # Report still well-formed.
    assert isinstance(format_report(report), str)


def test_business_impact_both_conventions_present(separable_scores, cost_model):
    y_true, y_prob = separable_scores
    report = build_report(y_true, y_prob, 0.5, cost_model, step="unit")
    assert report.business_impact_prd["convention"] == "prd"
    assert report.business_impact_net["convention"] == "net_of_principal"
    assert (
        report.business_impact_prd["net_annual_value"]
        <= report.business_impact_net["net_annual_value"] + 1e-6
    )


def test_confusion_counts_sum_to_row_count(hard_scores, cost_model):
    y_true, y_prob = hard_scores
    report = build_report(y_true, y_prob, 0.2, cost_model, step="unit")
    r = report.at_deployed_threshold
    assert r.tp + r.fp + r.fn + r.tn == report.n_rows
    assert r.tp + r.fn == report.n_fraud


def test_report_is_wellformed_and_serialisable(separable_scores, cost_model):
    y_true, y_prob = separable_scores
    report = build_report(y_true, y_prob, 0.5, cost_model, step="unit")
    assert report.pr_auc == report.pr_auc  # not NaN
    d = report.as_dict()
    assert d["meets_target_band"] is True
    assert set(d["precision_at_recall"]) == {"0.70", "0.80", "0.90"}
    assert isinstance(format_report(report), str)


def test_format_report_contains_headline_lines(separable_scores, cost_model):
    y_true, y_prob = separable_scores
    text = format_report(
        build_report(y_true, y_prob, 0.5, cost_model, step="9.0 baseline")
    )
    assert "9.0 baseline" in text
    assert "test PR-AUC" in text
    assert "precision_at_recall" in text
    assert "target band" in text
    assert f"{int(TARGET_PRECISION * 100)}%" in text
