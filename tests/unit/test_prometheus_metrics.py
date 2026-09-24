"""
tests/unit/test_prometheus_metrics.py

PRD Phase 5.7 / FR-06: `GET /metrics` in Prometheus format with the custom
`fraud_*` collectors the Grafana dashboard (committed in Phase 6) already
queries by name.

The contract these tests pin:

  - the metric NAMES match `monitoring/grafana/dashboards/fraud_detection.json`
    exactly — a rename here silently blanks a panel there;
  - a scored decision increments `fraud_predictions_total{decision=...}` and the
    two decision labels exist from the first scrape (so `rate()` is not NaN);
  - the model-version gauge is the info-metric idiom (one label at value 1) and
    a redeploy in-process does not leave two versions at 1;
  - the drift gauge defaults to 0 and flips both ways.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from prometheus_client import generate_latest

from src.serving.prometheus_metrics import (
    DECISION_FRAUD,
    DECISION_LEGITIMATE,
    PrometheusMetrics,
    reset_for_testing,
)


@pytest.fixture()
def metrics() -> PrometheusMetrics:
    """A fresh set of collectors on the default registry for each test."""
    return reset_for_testing()


def _scrape() -> str:
    return generate_latest().decode("utf-8")


class TestNamesMatchTheDashboard:
    """fraud_detection.json queries these verbatim; keep them in lockstep."""

    def test_expected_metric_names_are_registered(self, metrics):
        text = _scrape()
        assert "fraud_predictions_total" in text
        assert "fraud_model_version" in text
        assert "fraud_drift_detected" in text


class TestPredictionCounter:
    def test_both_decision_labels_exist_before_any_traffic(self, metrics):
        """rate(fraud_predictions_total{decision="FRAUD"}) must be defined from
        the first scrape, not start at NaN until the first fraud is seen."""
        text = _scrape()
        assert 'fraud_predictions_total{decision="FRAUD"} 0.0' in text
        assert 'fraud_predictions_total{decision="LEGITIMATE"} 0.0' in text

    def test_record_decision_increments_the_labelled_series(self, metrics):
        metrics.record_decision(DECISION_FRAUD)
        metrics.record_decision(DECISION_FRAUD)
        metrics.record_decision(DECISION_LEGITIMATE)

        text = _scrape()
        assert 'fraud_predictions_total{decision="FRAUD"} 2.0' in text
        assert 'fraud_predictions_total{decision="LEGITIMATE"} 1.0' in text

    def test_unknown_decision_label_is_not_dropped(self, metrics):
        """A new decision value must surface as its own series, not vanish."""
        metrics.record_decision("REVIEW")
        assert 'decision="REVIEW"' in _scrape()


class TestModelVersionGauge:
    def test_set_model_version_is_one_label_at_value_one(self, metrics):
        metrics.set_model_version("xgb+lgbm+tft_2026-08-20")
        text = _scrape()
        assert 'fraud_model_version{version="xgb+lgbm+tft_2026-08-20"} 1.0' in text

    def test_redeploy_does_not_leave_two_versions_at_one(self, metrics):
        metrics.set_model_version("v1")
        metrics.set_model_version("v2")
        lines = [
            line
            for line in _scrape().splitlines()
            if line.startswith("fraud_model_version{")
        ]
        assert lines == ['fraud_model_version{version="v2"} 1.0']


class TestDriftGauge:
    def test_defaults_to_zero(self, metrics):
        assert "fraud_drift_detected 0.0" in _scrape()

    def test_set_true_then_false(self, metrics):
        metrics.set_drift_detected(True)
        assert "fraud_drift_detected 1.0" in _scrape()
        metrics.set_drift_detected(False)
        assert "fraud_drift_detected 0.0" in _scrape()


class TestConsumerLagGauge:
    def test_defaults_to_unknown_not_zero(self, metrics):
        """Before the consumer polls, the backlog is unknown. Publishing 0
        would let PRD 6.4's <500 gate pass without a measurement."""
        from src.serving.prometheus_metrics import LAG_UNKNOWN

        assert f"fraud_consumer_lag {float(LAG_UNKNOWN)}" in _scrape()

    def test_set_publishes_the_backlog(self, metrics):
        metrics.set_consumer_lag(342)
        assert "fraud_consumer_lag 342.0" in _scrape()

    def test_can_return_to_unknown(self, metrics):
        from src.serving.prometheus_metrics import LAG_UNKNOWN

        metrics.set_consumer_lag(10)
        metrics.set_consumer_lag(LAG_UNKNOWN)
        assert f"fraud_consumer_lag {float(LAG_UNKNOWN)}" in _scrape()


class TestSingletonReuse:
    def test_reimport_reuses_the_registered_collectors(self, metrics):
        """Constructing a second real set in the same process would raise
        'Duplicated timeseries'; the module-level singleton must tolerate a
        reload by reusing what is already on the default registry."""
        import importlib

        import src.serving.prometheus_metrics as mod

        reloaded = importlib.reload(mod)
        reloaded.METRICS.record_decision(reloaded.DECISION_FRAUD)
        assert "fraud_predictions_total" in _scrape()
