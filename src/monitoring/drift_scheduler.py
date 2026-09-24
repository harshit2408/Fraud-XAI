"""
src/monitoring/drift_scheduler.py

Automated drift-check loop (PRD Phase 7.4).

`drift_reporter.py` is the one-shot report `make monitor` runs. This module is
the *scheduled* wrapper the PRD asks for:

  1. Read the last N predictions from `logs/predictions.jsonl`
     (`serving.log_file`).
  2. Reconstruct the feature frame and call
     `DriftReporter.generate_data_drift_report()`.
  3. If drift is detected, write `monitoring/alerts/drift_alert_{timestamp}.json`
     and set the `fraud_drift_detected` Prometheus gauge to 1 (0 otherwise).
  4. Sleep `monitoring.drift_check_interval_hours * 3600` seconds, repeat.

**Why the alert file and the gauge both exist.** The gauge is live state for
Grafana's "Feature Drift Indicator" panel; it is lost on restart. The alert
JSON is the durable record — one file per detecting run, timestamped — so the
history survives and can be diffed. Writing only one of them would leave either
the dashboard or the audit trail blind.

**Failure handling.** A single iteration that raises (an empty window right
after a log rotation, Evidently choking on a degenerate column) is logged and
the loop continues to the next interval — a monitoring job that dies on the
first bad window stops monitoring silently, which is the exact failure this
whole area is built to prevent. `--once` / `--max-iterations` let a test or a
cron driver bound the run; the PRD done-when ("runs for 5 minutes and produces
at least one report") is `--max-iterations 2 --interval-seconds <short>`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.monitoring.drift_reporter import (
    DEFAULT_DRIFT_SHARE_THRESHOLD,
    DEFAULT_MISSING_SHARE_THRESHOLD,
    DEFAULT_WINDOW_ROWS,
    DriftReport,
    DriftReporter,
    EmptyWindowError,
    load_prediction_window,
)

logger = logging.getLogger(__name__)


class DriftCheckScheduler:
    """Runs `DriftReporter` on a fixed interval and records the outcome."""

    def __init__(
        self,
        reporter: DriftReporter,
        log_file: str,
        alerts_dir: str,
        window_rows: int = DEFAULT_WINDOW_ROWS,
        interval_seconds: float = 24 * 3600,
    ) -> None:
        self.reporter = reporter
        self.log_file = log_file
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.window_rows = window_rows
        self.interval_seconds = interval_seconds

    # ── One iteration ────────────────────────────────────────────────────────

    def run_once(self, current: Optional[pd.DataFrame] = None) -> Optional[DriftReport]:
        """Execute a single drift check.

        Returns the `DriftReport` on success, or `None` if the window could not
        support a verdict (logged, not raised — the loop must survive it).
        `current` overrides the log-reconstructed window, for tests.
        """
        if current is None:
            try:
                current = load_prediction_window(
                    self.log_file,
                    limit=self.window_rows,
                    reference=self.reporter.reference_data,
                )
            except FileNotFoundError:
                logger.warning(
                    "No prediction log at %s yet — nothing to check this "
                    "interval. This is expected before any traffic is served.",
                    self.log_file,
                )
                return None

        name = f"drift_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        try:
            report = self.reporter.generate_data_drift_report(current, name)
        except EmptyWindowError as exc:
            logger.warning(
                "Drift check skipped: %s. The loop continues; the next "
                "interval will retry with a fuller window.",
                exc,
            )
            return None

        self._record(report)
        return report

    # ── The loop ─────────────────────────────────────────────────────────────

    def run_forever(
        self,
        max_iterations: Optional[int] = None,
        sleep_fn=time.sleep,
    ) -> int:
        """Loop until `max_iterations` (None = forever).

        Returns the count of iterations that produced a report. `sleep_fn` is
        injectable so a test does not actually wait `interval_seconds`.
        """
        produced = 0
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            logger.info("Drift check iteration %d starting", iteration)
            try:
                report = self.run_once()
            except Exception:  # noqa: BLE001 - one bad iteration must not kill the loop
                logger.exception(
                    "Drift check iteration %d raised; the loop continues.",
                    iteration,
                )
                report = None

            if report is not None:
                produced += 1

            if max_iterations is not None and iteration >= max_iterations:
                break
            sleep_fn(self.interval_seconds)
        return produced

    # ── Outcome recording ────────────────────────────────────────────────────

    def _record(self, report: DriftReport) -> None:
        """Set the Prometheus gauge, and write an alert file when drift is up."""
        self._set_gauge(report.drift_detected)
        if not report.drift_detected:
            logger.info(
                "No drift: %.1f%% of %d compared features (threshold %.0f%%).",
                report.drift_share * 100,
                report.compared_features,
                report.threshold * 100,
            )
            return

        alert_path = self.alerts_dir / (
            f"drift_alert_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        )
        payload = {
            "detected": True,
            "generated_at": report.generated_at,
            "drift_share": report.drift_share,
            "threshold": report.threshold,
            "drifted_features": report.drifted_features,
            "missing_features": report.missing_features,
            "schema_incomplete": report.schema_incomplete,
            "alert_reason": report.alert_reason,
            "report_html": report.html_path,
            "report_json": report.json_path,
        }
        alert_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.warning(
            "DRIFT ALERT written to %s — %s",
            alert_path,
            report.alert_reason or f"{report.drift_share:.1%} of features drifted",
        )

    @staticmethod
    def _set_gauge(detected: bool) -> None:
        """Publish the verdict to `fraud_drift_detected`.

        Imported lazily and guarded: the scheduler is useful even where
        `prometheus_client` is absent (it still writes alert files), and a
        metrics-registry import error must not stop a drift check.
        """
        try:
            from src.serving.prometheus_metrics import METRICS

            METRICS.set_drift_detected(detected)
        except Exception as exc:  # noqa: BLE001 - metrics are best-effort here
            logger.warning(
                "Could not update the fraud_drift_detected gauge (%s: %s). "
                "The drift check itself and any alert file are unaffected.",
                type(exc).__name__,
                exc,
            )


def build_scheduler(
    config_path: str = "config/config.yaml",
    interval_seconds: Optional[float] = None,
) -> DriftCheckScheduler:
    """Assemble a scheduler from `config/config.yaml`."""
    config = load_settings(config_path).model_dump()
    monitoring = config["monitoring"]
    reporter = DriftReporter(
        reference_data_path=monitoring["reference_data_path"],
        report_dir=monitoring["evidently_report_dir"],
        drift_share_threshold=monitoring.get(
            "drift_share_threshold", DEFAULT_DRIFT_SHARE_THRESHOLD
        ),
        missing_share_threshold=monitoring.get(
            "missing_share_threshold", DEFAULT_MISSING_SHARE_THRESHOLD
        ),
    )
    interval = (
        interval_seconds
        if interval_seconds is not None
        else monitoring["drift_check_interval_hours"] * 3600
    )
    return DriftCheckScheduler(
        reporter=reporter,
        log_file=config["serving"]["log_file"],
        alerts_dir=monitoring.get("alerts_dir", "monitoring/alerts"),
        window_rows=monitoring.get("drift_window_rows", DEFAULT_WINDOW_ROWS),
        interval_seconds=interval,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run scheduled feature-drift checks (PRD Phase 7.4)."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single check and exit (equivalent to --max-iterations 1).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Stop after this many iterations. Default: run forever.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=None,
        help="Override monitoring.drift_check_interval_hours (useful for a "
        "bounded demo run).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    scheduler = build_scheduler(args.config, interval_seconds=args.interval_seconds)
    max_iterations = 1 if args.once else args.max_iterations
    produced = scheduler.run_forever(max_iterations=max_iterations)
    logger.info("Scheduler stopped after producing %d report(s).", produced)
    # Non-zero only when a bounded run produced nothing at all — a cron wrapper
    # can then alert that monitoring is not seeing data.
    if max_iterations is not None and produced == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
