"""
tests/integration/test_monitoring.py

PRD Phase 7 done-when: "tests/test_monitoring.py tests that drift report
generates without error" and "drift_scheduler.py runs ... and produces at least
one report".

These are the integration-level checks that the monitoring *pipeline* holds
together — reporter + scheduler + prediction-log reconstruction + the
`fraud_drift_detected` gauge — not the fine-grained reporter contract, which
`tests/unit/test_drift_reporter.py` already covers.

Evidently is a real dependency here (the drift report is genuinely generated),
so these are marked `integration` and skipped cleanly if it is not installed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("evidently", reason="Evidently not installed")

from src.monitoring.drift_reporter import DriftReporter
from src.monitoring.drift_scheduler import DriftCheckScheduler
from src.monitoring.prediction_log import PredictionLogWriter
from src.serving.prometheus_metrics import reset_for_testing

pytestmark = pytest.mark.integration


def _reference(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionAmt": rng.lognormal(3.0, 1.0, n),
            "amount_log": rng.normal(3.0, 1.0, n),
            "tx_count_per_card": rng.integers(0, 40, n).astype(float),
            "hour_of_day": rng.integers(0, 24, n).astype(float),
        }
    )


def _stable_window(n: int = 200, seed: int = 99) -> pd.DataFrame:
    return _reference(n, seed)


def _drifted_window(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = _reference(n, seed)
    frame["TransactionAmt"] = rng.lognormal(6.5, 1.0, n)
    frame["amount_log"] = rng.normal(9.0, 1.0, n)
    return frame


@pytest.fixture()
def reference_path(tmp_path) -> str:
    path = tmp_path / "train_features.parquet"
    _reference().to_parquet(path, index=False)
    return str(path)


@pytest.fixture()
def scheduler(tmp_path, reference_path) -> DriftCheckScheduler:
    reset_for_testing()
    reporter = DriftReporter(
        reference_data_path=reference_path,
        report_dir=str(tmp_path / "reports"),
    )
    return DriftCheckScheduler(
        reporter=reporter,
        log_file=str(tmp_path / "predictions.jsonl"),
        alerts_dir=str(tmp_path / "alerts"),
        window_rows=5000,
        interval_seconds=0.0,
    )


class TestDriftReportGenerates:
    def test_stable_window_produces_a_report_and_no_alert(self, scheduler):
        report = scheduler.run_once(current=_stable_window())

        assert report is not None
        assert report.drift_detected is False
        assert Path(report.html_path).exists()
        assert Path(report.json_path).exists()
        assert list(scheduler.alerts_dir.glob("drift_alert_*.json")) == []

    def test_drifted_window_writes_a_timestamped_alert_file(self, scheduler):
        report = scheduler.run_once(current=_drifted_window())

        assert report is not None and report.drift_detected is True
        alerts = list(scheduler.alerts_dir.glob("drift_alert_*.json"))
        assert len(alerts) == 1

        payload = json.loads(alerts[0].read_text(encoding="utf-8"))
        assert payload["detected"] is True
        assert payload["drifted_features"]  # an operator needs the names
        assert payload["report_html"].endswith(".html")


class TestDriftGaugeIsUpdated:
    def _gauge_value(self) -> float:
        from prometheus_client import REGISTRY

        return REGISTRY.get_sample_value("fraud_drift_detected")

    def test_gauge_goes_to_one_on_drift_and_back_to_zero_when_stable(self, scheduler):
        scheduler.run_once(current=_drifted_window())
        assert self._gauge_value() == 1.0

        scheduler.run_once(current=_stable_window())
        assert self._gauge_value() == 0.0


class TestSchedulerLoop:
    def test_bounded_run_without_a_log_skips_gracefully(self, scheduler):
        slept = []
        produced = scheduler.run_forever(max_iterations=2, sleep_fn=slept.append)

        assert produced == 0  # no prediction log yet — skipped, not raised
        assert slept == [0.0]  # one sleep between iterations, none after the last

    def test_loop_reads_a_real_prediction_log_and_produces_a_report(self, scheduler):
        writer = PredictionLogWriter(scheduler.log_file)
        window = _drifted_window()
        for i in range(len(window)):
            writer.write(
                {"transaction_id": f"t{i}", "model_version": "v"},
                features=window.iloc[i].to_dict(),
            )

        produced = scheduler.run_forever(max_iterations=1, sleep_fn=lambda _s: None)

        assert produced == 1
        assert list(scheduler.alerts_dir.glob("drift_alert_*.json"))

    def test_one_bad_iteration_does_not_kill_the_loop(self, scheduler, monkeypatch):
        calls = {"n": 0}
        real_run_once = scheduler.run_once

        def flaky_run_once(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("evidently blew up on a degenerate column")
            return real_run_once(current=_stable_window())

        monkeypatch.setattr(scheduler, "run_once", flaky_run_once)
        produced = scheduler.run_forever(max_iterations=2, sleep_fn=lambda _s: None)

        assert calls["n"] == 2  # the loop kept going after the exception
        assert produced == 1
