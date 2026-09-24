"""
src/monitoring/drift_reporter.py

Feature-drift monitoring (Phase E task **E6**, PRD §7.1).

Compares the live feature distribution against the training reference and
writes an Evidently HTML report plus a machine-readable JSON summary. `make
monitor` runs `main()`.

**What this module is defending against.** A monitoring job's worst failure is
not crashing — it is staying green. A reporter that compares an empty window,
or silently drops every feature the current data happens to be missing, emits a
confident "no drift" verdict from data it never actually looked at, and the
first anyone hears of a broken feature pipeline is a fraud loss. Every guard in
`_validate_window` exists to convert that class of silence into a loud failure.

Two scope notes, both consequences of decisions recorded elsewhere:

  - **Data drift only.** `generate_model_performance_report` (PRD §7.1) needs
    `y_true`, and confirmed fraud labels arrive on a ~30-day lag
    (`config.features.target_encoding_label_lag_days`, finding F6). There are
    no labels to score at monitor time, so that report is deliberately not
    implemented here rather than stubbed to look available.
  - **The reference is the training window.** Drift is measured against what
    the model was fitted on, which is what makes "retrain" the actionable
    response to an alert.
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings

logger = logging.getLogger(__name__)

# PRD §7.1: "Alert if > 30% of features show drift (configurable threshold)."
# A monitoring threshold is a product decision — how much distribution shift is
# tolerable before someone is paged — so it is a named constant and a config
# key, never a literal at the call site.
DEFAULT_DRIFT_SHARE_THRESHOLD = 0.30

# Rows of recent prediction history to compare against the reference. Enough
# for a stable per-feature statistical test without loading an unbounded log.
DEFAULT_WINDOW_ROWS = 5_000

# Share of the reference schema that may be absent from the current window
# before that alone raises the alert. `drift_share` is computed over the
# COMPARED columns only, so features silently dropped by a pipeline bug leave
# both numerator and denominator - an entire feature group can vanish without
# ever moving the share. Losing this much of the schema is itself the finding.
DEFAULT_MISSING_SHARE_THRESHOLD = 0.20


class EmptyWindowError(ValueError):
    """The current window cannot support a drift verdict.

    Raised instead of returning "no drift" — a comparison against nothing is
    not evidence of stability, and reporting it as such is the silent failure
    this module exists to prevent.
    """


@dataclass(frozen=True)
class DriftReport:
    """The verdict, plus everything an operator needs to act on it."""

    drift_detected: bool
    drift_share: float
    threshold: float
    drifted_features: List[str]
    compared_features: int
    missing_features: List[str] = field(default_factory=list)
    reference_rows: int = 0
    current_rows: int = 0
    schema_incomplete: bool = False
    alert_reason: str = ""
    html_path: str = ""
    json_path: str = ""
    generated_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DriftReporter:
    """Evidently-backed data-drift reporting against the training reference."""

    def __init__(
        self,
        reference_data_path: str,
        report_dir: str,
        drift_share_threshold: float = DEFAULT_DRIFT_SHARE_THRESHOLD,
        missing_share_threshold: float = DEFAULT_MISSING_SHARE_THRESHOLD,
    ) -> None:
        reference_path = Path(reference_data_path)
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Reference data not found: {reference_path}. Run "
                "`python src/data/preprocess.py` to produce it — drift cannot "
                "be measured without the distribution the model was fitted on."
            )
        self.reference_data = pd.read_parquet(reference_path)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.drift_share_threshold = drift_share_threshold
        self.missing_share_threshold = missing_share_threshold

    def generate_data_drift_report(
        self, current_data: pd.DataFrame, report_name: str
    ) -> DriftReport:
        """Compare `current_data` against the training reference.

        Raises:
            EmptyWindowError: the window has no rows, is entirely null, or
                shares no features with the reference.
        """
        compared = self._validate_window(current_data)
        reference = self.reference_data[compared]
        current = current_data[compared]

        result = self._run_evidently(reference, current, report_name)
        missing = sorted(set(self.reference_data.columns) - set(compared))
        if missing:
            logger.warning(
                "Current window is missing %d reference feature(s): %s. They "
                "were EXCLUDED from the drift comparison and are reported in "
                "`missing_features` — a window missing part of the schema is "
                "usually a pipeline bug, not a quiet no-op.",
                len(missing),
                missing[:10],
            )

        missing_share = (
            len(missing) / len(self.reference_data.columns)
            if len(self.reference_data.columns)
            else 0.0
        )
        schema_incomplete = missing_share > self.missing_share_threshold
        features_drifted = result["share"] > self.drift_share_threshold

        reasons = []
        if features_drifted:
            reasons.append(
                f"{result['share']:.1%} of compared features drifted "
                f"(threshold {self.drift_share_threshold:.0%})"
            )
        if schema_incomplete:
            reasons.append(
                f"{missing_share:.1%} of the reference schema is missing from "
                f"the window (threshold {self.missing_share_threshold:.0%}) - "
                f"{len(missing)} feature(s) absent"
            )

        report = DriftReport(
            # Either condition alerts. A window that lost most of its schema
            # cannot be called stable just because the few surviving columns
            # look unchanged.
            drift_detected=features_drifted or schema_incomplete,
            drift_share=result["share"],
            threshold=self.drift_share_threshold,
            drifted_features=result["drifted"],
            compared_features=len(compared),
            missing_features=missing,
            reference_rows=len(reference),
            current_rows=len(current),
            schema_incomplete=schema_incomplete,
            alert_reason="; ".join(reasons),
            html_path=str(result["html_path"]),
            json_path="",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        report = self._write_summary(report, report_name)

        log = logger.warning if report.drift_detected else logger.info
        log(
            "Drift %s: %.1f%% of %d compared features drifted (threshold %.1f%%)%s",
            "DETECTED" if report.drift_detected else "not detected",
            report.drift_share * 100,
            report.compared_features,
            report.threshold * 100,
            f" — {report.drifted_features[:10]}" if report.drifted_features else "",
        )
        return report

    def check_drift_alert(self, report_path: str) -> bool:
        """Read back a written summary and return its alert verdict (PRD §7.1)."""
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        if payload.get("drift_detected"):
            logger.warning(
                "Drift alert active in %s: %s",
                report_path,
                payload.get("drifted_features", [])[:10],
            )
        return bool(payload.get("drift_detected", False))

    # ── Internals ────────────────────────────────────────────────────────────

    def _validate_window(self, current_data: pd.DataFrame) -> List[str]:
        """Return the comparable columns, or raise if no verdict is possible."""
        if current_data is None or len(current_data) == 0:
            raise EmptyWindowError(
                "Current window has no rows — refusing to report 'no drift' "
                "from an empty comparison."
            )

        shared = [c for c in self.reference_data.columns if c in current_data.columns]
        if not shared:
            raise EmptyWindowError(
                "Current window has no features in common with the reference "
                f"data (reference has {len(self.reference_data.columns)} "
                f"columns, window has {len(current_data.columns)}). Nothing "
                "could be compared."
            )

        usable = [c for c in shared if current_data[c].notna().any()]
        if not usable:
            raise EmptyWindowError(
                "Every shared feature in the current window is entirely null — "
                "there is no distribution to compare."
            )
        return usable

    def _run_evidently(
        self, reference: pd.DataFrame, current: pd.DataFrame, report_name: str
    ) -> Dict[str, Any]:
        """Run the drift preset and extract the share + drifted column names.

        Imported lazily so that importing this module (and the rest of
        `src.monitoring`) does not require Evidently to be installed.
        """
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)

        html_path = self.report_dir / f"{report_name}.html"
        report.save_html(str(html_path))

        payload = report.as_dict()
        summary = payload["metrics"][0]["result"]
        by_column = payload["metrics"][1]["result"]["drift_by_columns"]

        return {
            "share": float(summary["share_of_drifted_columns"]),
            "drifted": sorted(
                name for name, body in by_column.items() if body["drift_detected"]
            ),
            "html_path": html_path,
        }

    def _write_summary(self, report: DriftReport, report_name: str) -> DriftReport:
        """Persist the machine-readable summary `check_drift_alert` reads."""
        json_path = self.report_dir / f"{report_name}.json"
        final = DriftReport(**{**report.as_dict(), "json_path": str(json_path)})
        json_path.write_text(json.dumps(final.as_dict(), indent=2), encoding="utf-8")
        return final


def load_prediction_window(
    log_path: str,
    limit: Optional[int] = DEFAULT_WINDOW_ROWS,
    reference: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Rebuild a feature frame from the serving prediction log (PRD §7.4).

    Each line is one JSON object carrying a `features` mapping. A partially
    written final line is expected — the serving process can be killed
    mid-write — so malformed lines are skipped with a warning rather than
    failing the monitoring run. Anything beyond that (a wholly unreadable log)
    still surfaces, because a silently empty window would be reported as
    "no drift" by a less careful caller.

    **Dtypes are restored from `reference` when it is given.** This matters more
    than it looks: `InferenceService` logs `features.iloc[0].to_dict()`, and a
    pandas Series is homogeneously typed, so taking one row out of a
    mixed-dtype frame upcasts every scalar to Python `float`. `ProductCD=5`
    (int32 in training) is written as `5.0`. **38 of the 171 shipped features
    are low-cardinality ints** - 22% of the vector. Evidently currently rescues
    this with an internal dtype-mismatch fallback, so those columns still get a
    categorical test today, but that is a third-party implementation detail: a
    version bump could silently reclassify a fifth of the feature set from
    chi-square to Wasserstein with nothing failing. Restoring the reference
    dtype here makes the classification this codebase's own contract.
    (`ecc:mle-reviewer`, 2026-09-07.)

    Args:
        log_path: Path to the JSONL prediction log.
        limit: Keep only the most recent N rows. None reads the whole file.
        reference: Frame whose dtypes the window should be cast back to.

    Raises:
        FileNotFoundError: the log does not exist.
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No prediction log at {path}. The API writes it on every scored "
            "request (`serving.log_file`); there is nothing to monitor until "
            "traffic has been served."
        )

    rows: List[Dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            features = record.get("features")
            if isinstance(features, dict) and features:
                rows.append(features)

    if malformed:
        logger.warning(
            "Skipped %d malformed line(s) in %s — expected for a log truncated "
            "mid-write, but a large count suggests a writer problem.",
            malformed,
            path,
        )

    frame = pd.DataFrame(rows)
    if limit is not None and len(frame) > limit:
        frame = frame.iloc[-limit:].reset_index(drop=True)
    if reference is not None and not frame.empty:
        frame = _restore_reference_dtypes(frame, reference)
    return frame


def _restore_reference_dtypes(
    frame: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    """Cast each shared column back to the reference dtype.

    A column that cannot be cast losslessly is left as-is with a warning rather
    than force-coerced: silently truncating 1.5 to 1 would corrupt the very
    distribution being measured.
    """
    restored = frame.copy()
    for column in frame.columns:
        if column not in reference.columns:
            continue
        target = reference[column].dtype
        if restored[column].dtype == target:
            continue
        try:
            if target.kind in "iu":
                values = restored[column]
                if not np.isclose(values, values.round()).all():
                    raise ValueError("non-integral values")
            restored[column] = restored[column].astype(target)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Could not restore column %r to the reference dtype %s (%s). "
                "Leaving it as %s - Evidently may classify it differently than "
                "it did at training time.",
                column,
                target,
                exc,
                restored[column].dtype,
            )
    return restored


def main() -> int:
    """`make monitor` entry point.

    Returns a non-zero exit code when drift is detected, so a scheduler or CI
    job fails loudly instead of needing someone to read the HTML.
    """
    parser = argparse.ArgumentParser(description="Generate a feature-drift report.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--current",
        default=None,
        help="Parquet of the current window. Defaults to reconstructing it "
        "from the serving prediction log.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Most recent N predictions to compare. Defaults to "
        "monitoring.drift_window_rows from config.",
    )
    parser.add_argument("--name", default=None, help="Report basename.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_settings(args.config).model_dump()
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

    if args.current:
        current = pd.read_parquet(args.current)
        logger.info("Loaded %d rows from %s", len(current), args.current)
    else:
        log_file = config["serving"]["log_file"]
        limit = args.limit or monitoring.get("drift_window_rows", DEFAULT_WINDOW_ROWS)
        current = load_prediction_window(
            log_file, limit=limit, reference=reporter.reference_data
        )
        logger.info("Loaded %d predictions from %s", len(current), log_file)

    name = args.name or f"drift_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    report = reporter.generate_data_drift_report(current, name)

    logger.info("Report: %s", report.html_path)
    logger.info("Summary: %s", report.json_path)
    return 1 if report.drift_detected else 0


if __name__ == "__main__":
    raise SystemExit(main())
