import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation.evaluator import ModelEvaluator


@pytest.fixture
def evaluator():
    return ModelEvaluator()


@pytest.fixture
def mock_predictions():
    # 10 samples: 2 positive, 8 negative
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # Probabilities that rank positives higher but need a good threshold
    y_prob = np.array([0.9, 0.4, 0.8, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.01])
    return y_true, y_prob


def test_compute_pr_auc(evaluator, mock_predictions):
    y_true, y_prob = mock_predictions
    expected_pr_auc = average_precision_score(y_true, y_prob)
    
    pr_auc = evaluator.compute_pr_auc(y_true, y_prob)
    
    assert isinstance(pr_auc, float)
    assert np.isclose(pr_auc, expected_pr_auc)


def test_compute_roc_auc(evaluator, mock_predictions):
    y_true, y_prob = mock_predictions
    expected_roc_auc = roc_auc_score(y_true, y_prob)
    
    roc_auc = evaluator.compute_roc_auc(y_true, y_prob)
    
    assert isinstance(roc_auc, float)
    assert np.isclose(roc_auc, expected_roc_auc)


def test_find_optimal_threshold(evaluator, mock_predictions):
    y_true, y_prob = mock_predictions

    # Cost scenario 1: High cost for False Negatives (Missed fraud) -> Lower threshold
    cost_fn_1 = 500
    cost_fp_1 = 5
    # threshold 0.35 will catch both frauds (y_prob > 0.35 are idx 0(1), 1(1), 2(0))
    # FN=0, FP=1. Cost = 1*5 = 5.
    # threshold 0.5 will catch one fraud (y_prob > 0.5 are idx 0(1), 2(0))
    # FN=1, FP=1. Cost = 1*500 + 1*5 = 505.
    optimal_thresh_1 = evaluator.find_optimal_threshold(y_true, y_prob, cost_fn_1, cost_fp_1)

    # Expected threshold is in (0.3, 0.4) range depending on steps, maybe ~0.35
    assert 0.3 < optimal_thresh_1 <= 0.4


def test_find_optimal_threshold_grid_is_log_spaced_from_1e_minus_4(evaluator):
    """C3: the search grid must not clip a true optimum at a 0.01 floor."""
    grid = evaluator._select_threshold_grid()

    assert grid.min() == pytest.approx(1e-4, rel=1e-6)
    assert grid.max() < 1.0
    # log-spacing: consecutive ratios are ~constant, unlike linspace
    ratios = grid[1:] / grid[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-3)


def test_find_optimal_threshold_is_not_clipped_when_true_optimum_is_tiny(evaluator):
    """A rare positive scored below the old 0.01 grid floor, with FN vastly
    more expensive than FP, must still be caught: the true optimum requires
    a threshold under 0.01, which the old np.linspace(0.01, 0.99, 99) grid
    could never reach (its own lowest point *is* 0.01)."""
    y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    optimal_t = evaluator.find_optimal_threshold(y_true, y_prob, cost_fn=1_000_000, cost_fp=1)

    # Old grid's floor (0.01) would miss the 0.005-scored positive entirely
    # (FN=1, cost=1,000,000). The new grid reaches below it and finds the
    # true zero-cost optimum (FN=0, FP=0) instead of clipping at 0.01.
    assert optimal_t < 0.01
    achieved_cost = -evaluator._business_value(
        y_true, y_prob, optimal_t, cost_fn=1_000_000, cost_fp=1, revenue_tp=0.0
    )
    assert achieved_cost == 0.0


def test_find_optimal_threshold_and_business_value_plot_share_one_objective(evaluator, mock_predictions, tmp_path):
    """C3: unify the two conflicting business-objective functions.
    find_optimal_threshold(revenue_tp=X) must select the same threshold as
    maximizing the exact curve plot_threshold_vs_business_value draws."""
    y_true, y_prob = mock_predictions
    cost_fn, cost_fp, revenue_tp = 500, 5, 480

    optimal_t = evaluator.find_optimal_threshold(y_true, y_prob, cost_fn, cost_fp, revenue_tp)

    grid = evaluator._select_threshold_grid()
    values = np.array([
        evaluator._business_value(y_true, y_prob, t, cost_fn, cost_fp, revenue_tp)
        for t in grid
    ])
    expected_t = grid[int(np.argmax(values))]

    assert optimal_t == pytest.approx(expected_t)

    # The plot must mark the caller-supplied frozen threshold, not re-derive
    # its own optimum from whatever data it happens to be plotting.
    save_path = tmp_path / "plot.png"
    evaluator.plot_threshold_vs_business_value(
        y_true, y_prob, cost_fn, cost_fp, revenue_tp, str(save_path), optimal_threshold=optimal_t
    )
    assert save_path.exists()


@pytest.fixture
def miscalibrated_predictions():
    """400 samples with a rank-consistent but badly miscalibrated score:
    every predicted probability sits in a narrow high band regardless of
    true prevalence, which is the shape scale_pos_weight-style training
    produces (Phase C2)."""
    rng = np.random.default_rng(42)
    y_true = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])
    y_prob = np.concatenate([
        rng.uniform(0.55, 0.75, 200),  # legitimate scored high
        rng.uniform(0.75, 0.95, 200),  # fraud scored even higher (rank-consistent)
    ])
    return y_true, y_prob


def test_fit_calibrator_returns_fitted_isotonic_model(evaluator, miscalibrated_predictions):
    """C2: fit isotonic calibration on the VALIDATION split only."""
    y_val, y_prob_val = miscalibrated_predictions

    calibrator = evaluator.fit_calibrator(y_val, y_prob_val, method="isotonic")

    assert hasattr(calibrator, "predict")


def test_fit_calibrator_rejects_unknown_method(evaluator, miscalibrated_predictions):
    y_val, y_prob_val = miscalibrated_predictions

    with pytest.raises(ValueError):
        evaluator.fit_calibrator(y_val, y_prob_val, method="not_a_real_method")


def test_apply_calibration_returns_probabilities_in_unit_range(evaluator, miscalibrated_predictions):
    y_val, y_prob_val = miscalibrated_predictions
    calibrator = evaluator.fit_calibrator(y_val, y_prob_val)

    calibrated = evaluator.apply_calibration(calibrator, y_prob_val)

    assert calibrated.shape == y_prob_val.shape
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


def test_compute_brier_score_matches_sklearn(evaluator, mock_predictions):
    y_true, y_prob = mock_predictions
    expected = brier_score_loss(y_true, y_prob)

    brier = evaluator.compute_brier_score(y_true, y_prob)

    assert isinstance(brier, float)
    assert brier == pytest.approx(expected)


def test_calibration_fit_on_val_reduces_brier_score_on_held_out_val(evaluator, miscalibrated_predictions):
    """The whole point of C2: isotonic-calibrated probabilities must be
    better calibrated (lower Brier score) than the raw scores, evaluated
    on the same split used to fit the calibrator (in-sample lower bound;
    the trainer applies the fitted calibrator to test separately)."""
    y_val, y_prob_val = miscalibrated_predictions
    calibrator = evaluator.fit_calibrator(y_val, y_prob_val)
    y_prob_val_cal = evaluator.apply_calibration(calibrator, y_prob_val)

    brier_before = evaluator.compute_brier_score(y_val, y_prob_val)
    brier_after = evaluator.compute_brier_score(y_val, y_prob_val_cal)

    assert brier_after < brier_before


def test_calibrator_fit_on_val_generalizes_to_test_split(evaluator):
    """C2 must not leak: the calibrator is fit on val and only *applied*
    (never refit) to test. Simulate val/test drawn from the same
    generative process and confirm applying the val-fit calibrator still
    improves test calibration."""
    rng = np.random.default_rng(7)

    def sample(n_per_class, seed_offset):
        r = np.random.default_rng(7 + seed_offset)
        y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
        p = np.concatenate([
            r.uniform(0.55, 0.75, n_per_class),
            r.uniform(0.75, 0.95, n_per_class),
        ])
        return y, p

    y_val, p_val = sample(200, 1)
    y_test, p_test = sample(200, 2)

    calibrator = evaluator.fit_calibrator(y_val, p_val)
    p_test_cal = evaluator.apply_calibration(calibrator, p_test)

    brier_before = evaluator.compute_brier_score(y_test, p_test)
    brier_after = evaluator.compute_brier_score(y_test, p_test_cal)

    assert brier_after < brier_before


def test_plot_reliability_curve_creates_file(evaluator, mock_predictions, tmp_path):
    y_true, y_prob = mock_predictions
    save_path = tmp_path / "reliability.png"

    evaluator.plot_reliability_curve(y_true, y_prob, str(save_path))

    assert save_path.exists()


def test_plot_reliability_curve_with_calibrated_overlay(evaluator, miscalibrated_predictions, tmp_path):
    """Reliability curve must be able to overlay before/after calibration
    on one figure so the improvement is visible."""
    y_val, y_prob_val = miscalibrated_predictions
    calibrator = evaluator.fit_calibrator(y_val, y_prob_val)
    y_prob_val_cal = evaluator.apply_calibration(calibrator, y_prob_val)
    save_path = tmp_path / "reliability_overlay.png"

    evaluator.plot_reliability_curve(
        y_val, y_prob_val, str(save_path), y_prob_calibrated=y_prob_val_cal
    )

    assert save_path.exists()


def test_compute_metrics_at_threshold(evaluator, mock_predictions):
    y_true, y_prob = mock_predictions

    # Threshold 0.5 gives:
    # Pred = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # TP=1, FP=1, FN=1, TN=7
    metrics = evaluator.compute_metrics_at_threshold(y_true, y_prob, threshold=0.5)

    assert metrics["TP"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert metrics["TN"] == 7
    assert metrics["precision"] == 0.5  # 1 / (1 + 1)
    assert metrics["recall"] == 0.5     # 1 / (1 + 1)
    assert metrics["accuracy"] == 0.8   # 8 / 10


# ── C5: per-slice metrics ───────────────────────────────────────────────

@pytest.fixture
def sliced_predictions():
    """20 samples split across two slices ('A', 'B'), each internally
    identical in shape to `mock_predictions` (2 positives / 8 negatives per
    slice) so per-slice metrics can be checked against known closed-form
    values while still exercising a >1-slice groupby."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0] * 2)
    y_prob = np.array([0.9, 0.4, 0.8, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.01] * 2)
    slice_labels = np.array(["A"] * 10 + ["B"] * 10)
    return y_true, y_prob, slice_labels


def test_compute_slice_metrics_returns_one_row_per_slice(evaluator, sliced_predictions):
    y_true, y_prob, slice_labels = sliced_predictions

    table = evaluator.compute_slice_metrics(y_true, y_prob, slice_labels, threshold=0.5)

    assert set(table["slice"]) == {"A", "B"}
    assert len(table) == 2


def test_compute_slice_metrics_matches_compute_metrics_at_threshold_per_slice(evaluator, sliced_predictions):
    """Each slice here is a verbatim copy of `mock_predictions`, so its row
    must match `compute_metrics_at_threshold` computed directly on that
    slice's rows — the slice table must not be silently pooling or
    mis-aligning rows across slices."""
    y_true, y_prob, slice_labels = sliced_predictions
    expected = evaluator.compute_metrics_at_threshold(
        y_true[slice_labels == "A"], y_prob[slice_labels == "A"], threshold=0.5
    )

    table = evaluator.compute_slice_metrics(y_true, y_prob, slice_labels, threshold=0.5)
    row_a = table[table["slice"] == "A"].iloc[0]

    assert row_a["count"] == 10
    assert row_a["TP"] == expected["TP"]
    assert row_a["FP"] == expected["FP"]
    assert row_a["FN"] == expected["FN"]
    assert row_a["precision"] == pytest.approx(expected["precision"])
    assert row_a["recall"] == pytest.approx(expected["recall"])
    assert row_a["f1"] == pytest.approx(expected["f1"])
    assert row_a["fraud_rate"] == pytest.approx(0.2)  # 2/10 positives


def test_compute_slice_metrics_flags_small_slices_instead_of_dropping_them(evaluator):
    """A slice thinner than `min_slice_size` still gets a row (so nothing
    silently vanishes from the report) but is flagged `reliable=False`
    rather than reported as if it had the same statistical weight as a
    well-populated slice."""
    y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.2])
    slice_labels = np.array(["big"] * 10 + ["tiny"] * 2)

    table = evaluator.compute_slice_metrics(
        y_true, y_prob, slice_labels, threshold=0.5, min_slice_size=30
    )

    tiny_row = table[table["slice"] == "tiny"].iloc[0]
    big_row = table[table["slice"] == "big"].iloc[0]
    assert tiny_row["reliable"] == False
    assert big_row["reliable"] == False  # both are below 30 in this tiny fixture
    assert tiny_row["count"] == 2


def test_compute_slice_metrics_handles_single_class_slice_pr_auc(evaluator):
    """A slice with zero positives can't have a PR-AUC (undefined); the
    method must report NaN there rather than raising, so one thin slice
    doesn't crash the whole table."""
    y_true = np.array([0, 0, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.15, 0.7, 0.8, 0.05])
    slice_labels = np.array(["no_fraud"] * 3 + ["mixed"] * 3)

    table = evaluator.compute_slice_metrics(y_true, y_prob, slice_labels, threshold=0.5)

    no_fraud_row = table[table["slice"] == "no_fraud"].iloc[0]
    assert np.isnan(no_fraud_row["pr_auc"])


def test_compute_slice_metrics_returns_empty_table_for_empty_input(evaluator):
    """Guards against a bare pd.DataFrame([]) lacking a 'count' column,
    which previously raised KeyError from .sort_values('count') on a
    zero-row slice_labels array instead of returning a clean empty table."""
    table = evaluator.compute_slice_metrics(
        np.array([]), np.array([]), np.array([]), threshold=0.5
    )

    assert len(table) == 0
    assert "count" in table.columns
    assert "slice" in table.columns


def test_bucket_hour_of_day_groups_into_named_windows(evaluator):
    hours = np.array([0, 5, 6, 11, 12, 17, 18, 22, 23])

    buckets = evaluator.bucket_hour_of_day(hours)

    # night: [0,6), morning: [6,12), afternoon: [12,18), evening: [18,24)
    assert list(buckets) == [
        "night", "night", "morning", "morning",
        "afternoon", "afternoon", "evening", "evening", "evening",
    ]


def test_bucket_card_tenure_quantile_buckets_are_ordered(evaluator):
    tx_counts = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55])

    buckets = evaluator.bucket_card_tenure(tx_counts, n_buckets=4)

    assert len(buckets) == len(tx_counts)
    # Lowest tx_count must land in the lowest-tenure bucket, highest in the highest.
    assert buckets[0] == "new"
    assert buckets[-1] == "established"
