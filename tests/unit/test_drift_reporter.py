"""
tests/unit/test_drift_reporter.py

TDD for Phase E task **E6**: "Create `src/monitoring/drift_reporter.py` so
`make monitor` works" — done when "Target runs".

Written before the implementation. The interesting requirement is not "an HTML
file appears": it is that the drift verdict is *trustworthy*. The failure mode
this module exists to prevent is a monitoring job that stays green while
silently comparing nothing — the `silent-failure-hunter` case the mle-workflow
skill calls out ("pipelines can appear green while skipping data, labels, eval
slices, alerts"). So these tests pin down:

  - the alert threshold is read from config, never invented at runtime;
  - an empty or all-missing current window RAISES rather than reporting
    "no drift";
  - features present in reference but absent from the current window are
    reported, not quietly dropped;
  - the verdict is derived from the drift share, and the drifted feature names
    are named so an operator can act.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.drift_reporter import (
    DEFAULT_DRIFT_SHARE_THRESHOLD,
    DriftReport,
    DriftReporter,
    EmptyWindowError,
)


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


def _same_distribution(n: int = 200, seed: int = 99) -> pd.DataFrame:
    return _reference(n, seed)


def _shifted(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Two of four features moved hard — a share of 0.5."""
    rng = np.random.default_rng(seed)
    frame = _reference(n, seed)
    frame["TransactionAmt"] = rng.lognormal(6.0, 1.0, n)  # ~20x larger
    frame["amount_log"] = rng.normal(9.0, 1.0, n)
    return frame


@pytest.fixture()
def reporter(tmp_path) -> DriftReporter:
    reference_path = tmp_path / "train_features.parquet"
    _reference().to_parquet(reference_path, index=False)
    return DriftReporter(
        reference_data_path=str(reference_path),
        report_dir=str(tmp_path / "reports"),
    )


class TestDriftDetection:
    def test_same_distribution_reports_no_drift(self, reporter):
        report = reporter.generate_data_drift_report(_same_distribution(), "stable")

        assert isinstance(report, DriftReport)
        assert report.drift_detected is False
        assert report.drifted_features == []
        assert 0.0 <= report.drift_share <= 1.0

    def test_shifted_distribution_is_detected_and_names_the_features(self, reporter):
        report = reporter.generate_data_drift_report(_shifted(), "shifted")

        assert report.drift_detected is True
        assert report.drift_share >= 0.5
        # An operator needs to know WHICH features moved, not just that some did.
        assert "TransactionAmt" in report.drifted_features
        assert "amount_log" in report.drifted_features

    def test_threshold_comes_from_config_not_a_runtime_guess(self, tmp_path):
        """A monitoring threshold is a product decision. Two reporters over the
        same data must disagree only because their configured thresholds
        differ."""
        reference_path = tmp_path / "ref.parquet"
        _reference().to_parquet(reference_path, index=False)

        strict = DriftReporter(
            str(reference_path), str(tmp_path / "a"), drift_share_threshold=0.1
        )
        lax = DriftReporter(
            str(reference_path), str(tmp_path / "b"), drift_share_threshold=0.9
        )

        current = _shifted()
        assert strict.generate_data_drift_report(current, "strict").drift_detected is True
        assert lax.generate_data_drift_report(current, "lax").drift_detected is False

    def test_default_threshold_matches_the_documented_value(self):
        """PRD §7.1: 'Alert if > 30% of features show drift'."""
        assert DEFAULT_DRIFT_SHARE_THRESHOLD == pytest.approx(0.30)


class TestSilentFailureGuards:
    """Each of these would otherwise produce a confident 'no drift' verdict
    from data that was never actually compared."""

    def test_empty_current_window_raises(self, reporter):
        with pytest.raises(EmptyWindowError, match="no rows"):
            reporter.generate_data_drift_report(pd.DataFrame(), "empty")

    def test_all_null_current_window_raises(self, reporter):
        window = pd.DataFrame({c: [np.nan] * 50 for c in _reference().columns})
        with pytest.raises(EmptyWindowError, match="entirely null"):
            reporter.generate_data_drift_report(window, "null")

    def test_no_overlapping_features_raises(self, reporter):
        window = pd.DataFrame({"totally_unrelated": np.arange(50.0)})
        with pytest.raises(EmptyWindowError, match="no features in common"):
            reporter.generate_data_drift_report(window, "disjoint")

    def test_missing_features_are_reported_not_dropped(self, reporter):
        """A window missing half the schema is a pipeline bug. Comparing only
        the surviving columns and reporting 'no drift' would hide it."""
        window = _same_distribution()[["TransactionAmt", "amount_log"]]

        report = reporter.generate_data_drift_report(window, "partial")

        assert set(report.missing_features) == {"tx_count_per_card", "hour_of_day"}
        assert report.compared_features == 2


class TestArtifacts:
    def test_html_and_json_are_written(self, reporter):
        report = reporter.generate_data_drift_report(_shifted(), "run1")

        html = Path(report.html_path)
        summary = Path(report.json_path)
        assert html.exists() and html.stat().st_size > 0
        assert summary.exists()

        payload = json.loads(summary.read_text(encoding="utf-8"))
        assert payload["drift_detected"] is True
        assert payload["drifted_features"]
        assert "generated_at" in payload
        assert payload["reference_rows"] > 0
        assert payload["current_rows"] > 0

    def test_check_drift_alert_reads_back_the_written_summary(self, reporter):
        report = reporter.generate_data_drift_report(_shifted(), "run2")
        assert reporter.check_drift_alert(report.json_path) is True

        calm = reporter.generate_data_drift_report(_same_distribution(), "run3")
        assert reporter.check_drift_alert(calm.json_path) is False


class TestPredictionLogWindow:
    """`make monitor` has no labelled data; it reads what serving actually
    logged. Reconstructing the window must tolerate a partially-written last
    line (the process can be killed mid-write) without failing the whole run."""

    def test_reads_features_from_a_jsonl_prediction_log(self, tmp_path):
        from src.monitoring.drift_reporter import load_prediction_window

        log = tmp_path / "predictions.jsonl"
        rows = [
            {
                "transaction_id": f"t{i}",
                "features": {"TransactionAmt": float(i), "amount_log": 1.0},
            }
            for i in range(5)
        ]
        log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        window = load_prediction_window(str(log))

        assert len(window) == 5
        assert list(window.columns) == ["TransactionAmt", "amount_log"]

    def test_a_truncated_final_line_is_skipped_with_a_warning(self, tmp_path, caplog):
        from src.monitoring.drift_reporter import load_prediction_window

        log = tmp_path / "predictions.jsonl"
        good = json.dumps({"features": {"TransactionAmt": 1.0}})
        log.write_text(f'{good}\n{good}\n{{"features": {{"Transacti', encoding="utf-8")

        with caplog.at_level("WARNING"):
            window = load_prediction_window(str(log))

        assert len(window) == 2
        assert "malformed" in caplog.text.lower()

    def test_limit_takes_the_most_recent_rows(self, tmp_path):
        from src.monitoring.drift_reporter import load_prediction_window

        log = tmp_path / "predictions.jsonl"
        rows = [{"features": {"TransactionAmt": float(i)}} for i in range(10)]
        log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        window = load_prediction_window(str(log), limit=3)

        assert window["TransactionAmt"].tolist() == [7.0, 8.0, 9.0]

    def test_missing_log_raises_with_an_actionable_message(self, tmp_path):
        from src.monitoring.drift_reporter import load_prediction_window

        with pytest.raises(FileNotFoundError, match="No prediction log"):
            load_prediction_window(str(tmp_path / "absent.jsonl"))


class TestPredictionLogWriter:
    """E6 closes a loop: `make monitor` reads `serving.log_file`, but until now
    nothing wrote it — `InferenceService` only emitted a `logger.info` line, so
    the default monitoring path would have found no log at all. These tests pin
    the writer's contract, which `load_prediction_window` above consumes.
    """

    def test_writes_one_json_object_per_prediction(self, tmp_path):
        from src.monitoring.prediction_log import PredictionLogWriter

        log = tmp_path / "predictions.jsonl"
        writer = PredictionLogWriter(str(log))

        writer.write(
            {"transaction_id": "t1", "fraud_probability": 0.5, "model_version": "v"},
            features={"TransactionAmt": 10.0, "amount_log": 2.3},
        )
        writer.write(
            {"transaction_id": "t2", "fraud_probability": 0.1, "model_version": "v"},
            features={"TransactionAmt": 20.0, "amount_log": 3.0},
        )

        lines = [l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["transaction_id"] == "t1"
        assert first["features"]["TransactionAmt"] == 10.0
        assert "logged_at" in first

    def test_round_trips_through_load_prediction_window(self, tmp_path):
        """The writer and the reader are two halves of one contract; testing
        them apart would let the format drift."""
        from src.monitoring.drift_reporter import load_prediction_window
        from src.monitoring.prediction_log import PredictionLogWriter

        log = tmp_path / "predictions.jsonl"
        writer = PredictionLogWriter(str(log))
        for i in range(4):
            writer.write(
                {"transaction_id": f"t{i}", "model_version": "v"},
                features={"TransactionAmt": float(i), "amount_log": 1.0},
            )

        window = load_prediction_window(str(log))
        assert len(window) == 4
        assert window["TransactionAmt"].tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_creates_the_parent_directory(self, tmp_path):
        from src.monitoring.prediction_log import PredictionLogWriter

        log = tmp_path / "nested" / "dir" / "predictions.jsonl"
        PredictionLogWriter(str(log)).write({"transaction_id": "t"}, features={"a": 1.0})
        assert log.exists()

    def test_a_logging_failure_never_breaks_scoring(self, tmp_path, caplog):
        """Prediction logging is observability, not the decision path. If the
        disk is full or the path is unwritable, the fraud verdict must still be
        returned — the request has already been scored."""
        from src.monitoring.prediction_log import PredictionLogWriter

        writer = PredictionLogWriter(str(tmp_path / "predictions.jsonl"))
        # A value json.dumps cannot serialize, standing in for any write failure.
        with caplog.at_level("WARNING"):
            writer.write({"transaction_id": "t"}, features={"bad": {1, 2, 3}})

        assert "prediction log" in caplog.text.lower()


class TestDtypeFidelityThroughTheRealPath:
    """`ecc:mle-reviewer` [HIGH]: the drift verdict for low-cardinality
    categoricals rested on an untested third-party coercion.

    `InferenceService` logs `features.iloc[0].to_dict()`. A pandas Series is
    homogeneously typed, so pulling one row out of a mixed-dtype frame upcasts
    every scalar to Python `float`: `ProductCD=5` (int32 in training) is written
    to the log as `5.0`. **38 of the 171 shipped features are low-cardinality
    ints** — 22% of the vector. Evidently currently rescues this via a
    dtype-mismatch fallback that reassigns the current column's type to the
    reference's, so the columns still happen to be scored with a categorical
    test. But nothing in this repo asserted that, so a pandas or Evidently
    version bump could silently reclassify a fifth of the feature set from
    chi-square to Wasserstein with no failing test.

    These tests build the window through the ACTUAL writer/reader pair rather
    than from hand-built float fixtures, which is what the previous suite got
    wrong: it was green by construction.
    """

    def _reference_with_categoricals(self, n: int = 300, seed: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "TransactionAmt": rng.lognormal(3.0, 1.0, n),
                "ProductCD": rng.integers(0, 5, n).astype(np.int32),
                "card4": rng.integers(0, 4, n).astype(np.int32),
                "M1": rng.integers(0, 2, n).astype(np.int32),
            }
        )

    def test_window_from_the_real_writer_keeps_reference_dtypes(self, tmp_path):
        """The regression the reviewer identified, end to end."""
        from src.monitoring.drift_reporter import load_prediction_window
        from src.monitoring.prediction_log import PredictionLogWriter

        reference = self._reference_with_categoricals()
        log = tmp_path / "predictions.jsonl"
        writer = PredictionLogWriter(str(log))

        # Exactly what InferenceService does: one row out of a mixed-dtype frame.
        for i in range(20):
            writer.write(
                {"transaction_id": f"t{i}"},
                features=reference.iloc[i].to_dict(),
            )

        window = load_prediction_window(str(log), reference=reference)

        for column in ["ProductCD", "card4", "M1"]:
            assert window[column].dtype == reference[column].dtype, (
                f"{column} came back as {window[column].dtype}, not "
                f"{reference[column].dtype} — categorical drift classification "
                "would depend on a third-party dtype fallback"
            )

    def test_values_survive_the_dtype_restoration(self, tmp_path):
        """Restoring dtypes must not silently corrupt values."""
        from src.monitoring.drift_reporter import load_prediction_window
        from src.monitoring.prediction_log import PredictionLogWriter

        reference = self._reference_with_categoricals()
        log = tmp_path / "predictions.jsonl"
        writer = PredictionLogWriter(str(log))
        for i in range(10):
            writer.write({"transaction_id": f"t{i}"}, features=reference.iloc[i].to_dict())

        window = load_prediction_window(str(log), reference=reference)

        np.testing.assert_array_equal(
            window["ProductCD"].to_numpy(), reference["ProductCD"].head(10).to_numpy()
        )

    def test_a_column_that_cannot_be_restored_is_left_alone_with_a_warning(
        self, tmp_path, caplog
    ):
        """A genuinely non-castable value must not crash the monitoring run —
        but it must not be silently coerced either."""
        from src.monitoring.drift_reporter import load_prediction_window

        reference = self._reference_with_categoricals()
        log = tmp_path / "predictions.jsonl"
        rows = [
            json.dumps({"features": {"ProductCD": 1.5, "TransactionAmt": 3.0}})
            for _ in range(5)
        ]
        log.write_text(chr(10).join(rows), encoding="utf-8")

        with caplog.at_level("WARNING"):
            window = load_prediction_window(str(log), reference=reference)

        assert len(window) == 5
        assert "ProductCD" in caplog.text

    def test_without_a_reference_the_window_is_returned_unchanged(self, tmp_path):
        """Back-compat: the reference argument is optional."""
        from src.monitoring.drift_reporter import load_prediction_window

        log = tmp_path / "predictions.jsonl"
        log.write_text(json.dumps({"features": {"a": 1.0}}), encoding="utf-8")

        assert len(load_prediction_window(str(log))) == 1


class TestMissingFeaturesAreAlertWorthy:
    """`ecc:mle-reviewer` [MEDIUM]: `drift_share` is computed only over the
    COMPARED columns, so a pipeline bug that drops 60 of 171 features excludes
    them from both numerator and denominator. The share cannot cross the
    threshold, and an entire feature group vanishing is demoted to a log line —
    'no drift' over a silently shrunken schema."""

    def test_a_largely_missing_schema_raises_the_alert(self, tmp_path):
        reference_path = tmp_path / "ref.parquet"
        _reference().to_parquet(reference_path, index=False)
        reporter = DriftReporter(str(reference_path), str(tmp_path / "r"))

        # One of four features present: 75% of the schema has vanished, but the
        # one remaining column is drawn from the reference distribution, so
        # feature-level drift alone would report "no drift".
        window = _same_distribution()[["TransactionAmt"]]
        report = reporter.generate_data_drift_report(window, "shrunken")

        assert report.schema_incomplete is True
        assert report.drift_detected is True, (
            "a schema this incomplete must alert even when the surviving "
            "features look stable"
        )
        assert "missing" in (report.alert_reason or "").lower()

    def test_a_few_missing_features_do_not_trip_the_alert(self, tmp_path):
        reference_path = tmp_path / "ref.parquet"
        _reference().to_parquet(reference_path, index=False)
        reporter = DriftReporter(
            str(reference_path), str(tmp_path / "r"), missing_share_threshold=0.5
        )

        window = _same_distribution()[["TransactionAmt", "amount_log", "hour_of_day"]]
        report = reporter.generate_data_drift_report(window, "mostly_there")

        assert report.schema_incomplete is False
        assert report.drift_detected is False

