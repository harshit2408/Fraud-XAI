"""
tests/unit/test_model_ensemble.py

Covers src/models/ensemble.py — the generalized N-model simplex weight
search added 2026-08-20 (docs/IMPLEMENTATION_PLAN.md Phase F-Audit) when
LightGBM was proposed as a 3rd ensemble input alongside XGBoost+TFT. No test
file existed for the ensemble module before this (mle-reviewer finding).
"""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.models.ensemble import (
    bootstrap_weight_ci,
    blend,
    grid_search_simplex_weights,
    pairwise_diagnostics,
)


# ── blend ──────────────────────────────────────────────────────────────────


def test_blend_computes_weighted_sum_two_models():
    prob_dict = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}

    result = blend(prob_dict, {"a": 0.3, "b": 0.7})

    np.testing.assert_allclose(result, [0.3, 0.7])


def test_blend_computes_weighted_sum_three_models():
    prob_dict = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0]),
    }

    result = blend(prob_dict, {"a": 0.2, "b": 0.3, "c": 0.5})

    np.testing.assert_allclose(result, [0.2, 0.3, 0.5])


def test_blend_raises_on_key_mismatch():
    prob_dict = {"a": np.array([1.0]), "b": np.array([0.0])}

    with pytest.raises(ValueError, match="do not match"):
        blend(prob_dict, {"a": 0.5, "c": 0.5})


def test_blend_raises_on_empty_prob_dict():
    with pytest.raises(ValueError, match="at least one model"):
        blend({}, {})


# ── grid_search_simplex_weights ──────────────────────────────────────────


def _synthetic_labels_and_scores(seed: int = 0, n: int = 2000):
    """A strongly informative (but not perfectly separating) score and a
    score uncorrelated with the label, for a known-optimal-weight test.

    Deliberately continuous rather than `perfect = y` exactly: a hard 0/1
    score blended with noise bounded to [0, 1] creates a wide plateau of
    tied-optimal weights (any w >= 0.5 gives perfect separation), which
    would make "the search found the true optimum" untestable — any weight
    in that plateau is equally correct, so asserting a specific one is
    asserting an artifact of the earlier test's construction, not of the
    search. Continuous, unbounded scores degrade PR-AUC smoothly as noise
    is mixed in, giving a much sharper true optimum near w=1.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    informative = y * 3.0 + rng.normal(0.0, 1.0, size=n)
    noise = rng.normal(0.0, 1.0, size=n)
    return y, informative, noise


def test_grid_search_two_models_assigns_near_full_weight_to_informative_model():
    y, informative, noise = _synthetic_labels_and_scores()
    informative_only_score = average_precision_score(y, informative)

    weights, score = grid_search_simplex_weights(
        y, {"informative": informative, "noise": noise}, step=0.02, refine_step=0.002
    )

    assert weights["informative"] > 0.9
    assert weights["noise"] < 0.1
    assert weights["informative"] + weights["noise"] == pytest.approx(1.0, abs=1e-6)
    # The search must do at least as well as the informative model alone —
    # it degenerately can, by putting ~all weight on it.
    assert score >= informative_only_score - 1e-6


def test_grid_search_three_models_ignores_noise_third_model():
    y, informative, noise = _synthetic_labels_and_scores()
    rng = np.random.default_rng(1)
    second_noise = rng.normal(0.0, 1.0, size=len(y))
    informative_only_score = average_precision_score(y, informative)

    weights, score = grid_search_simplex_weights(
        y,
        {"informative": informative, "noise": noise, "noise2": second_noise},
        step=0.05,
        refine_step=0.01,
    )

    assert weights["informative"] > 0.8
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert score >= informative_only_score - 1e-6


def test_grid_search_single_model_returns_full_weight():
    y, informative, _ = _synthetic_labels_and_scores()
    expected_score = average_precision_score(y, informative)

    weights, score = grid_search_simplex_weights(y, {"only": informative})

    assert weights == {"only": 1.0}
    assert score == pytest.approx(expected_score, abs=1e-9)


def test_grid_search_weights_always_sum_to_one():
    y, informative, noise = _synthetic_labels_and_scores(seed=7)
    rng = np.random.default_rng(3)
    mixed = 0.5 * informative + 0.5 * rng.normal(0.0, 1.0, size=len(y))

    weights, _ = grid_search_simplex_weights(
        y,
        {"informative": informative, "noise": noise, "mixed": mixed},
        step=0.05,
        refine_step=0.01,
    )

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(w >= -1e-9 for w in weights.values())


# ── bootstrap_weight_ci ──────────────────────────────────────────────────


def test_bootstrap_weight_ci_output_shape_and_normalization():
    y, perfect, noise = _synthetic_labels_and_scores(seed=2, n=500)

    samples = bootstrap_weight_ci(
        y, {"perfect": perfect, "noise": noise}, step=0.1, n_boot=10, seed=42
    )

    assert set(samples.keys()) == {"perfect", "noise"}
    assert samples["perfect"].shape == (10,)
    assert samples["noise"].shape == (10,)
    # Every bootstrap resample's weights must still sum to 1.
    totals = samples["perfect"] + samples["noise"]
    np.testing.assert_allclose(totals, np.ones(10), atol=1e-6)


def test_bootstrap_weight_ci_is_deterministic_given_seed():
    y, perfect, noise = _synthetic_labels_and_scores(seed=2, n=500)

    samples_a = bootstrap_weight_ci(
        y, {"perfect": perfect, "noise": noise}, step=0.1, n_boot=10, seed=99
    )
    samples_b = bootstrap_weight_ci(
        y, {"perfect": perfect, "noise": noise}, step=0.1, n_boot=10, seed=99
    )

    np.testing.assert_array_equal(samples_a["perfect"], samples_b["perfect"])


def test_bootstrap_weight_ci_recovers_high_weight_for_perfect_predictor():
    y, perfect, noise = _synthetic_labels_and_scores(seed=5, n=1000)

    samples = bootstrap_weight_ci(
        y, {"perfect": perfect, "noise": noise}, step=0.1, n_boot=20, seed=1
    )

    # A clearly-superior predictor should win in (almost) every resample.
    assert np.median(samples["perfect"]) > 0.8


# ── pairwise_diagnostics ──────────────────────────────────────────────────


def test_pairwise_diagnostics_identical_arrays_correlate_perfectly():
    prob_dict = {"a": np.array([0.1, 0.5, 0.9, 0.3]), "b": np.array([0.1, 0.5, 0.9, 0.3])}

    diag = pairwise_diagnostics(prob_dict)

    assert diag["corr_a_b"] == pytest.approx(1.0, abs=1e-9)


def test_pairwise_diagnostics_returns_every_pair_for_three_models():
    prob_dict = {
        "a": np.array([0.1, 0.2, 0.3, 0.4]),
        "b": np.array([0.4, 0.3, 0.2, 0.1]),
        "c": np.array([0.2, 0.5, 0.1, 0.6]),
    }

    diag = pairwise_diagnostics(prob_dict)

    assert set(diag.keys()) == {"corr_a_b", "corr_a_c", "corr_b_c"}
    assert diag["corr_a_b"] == pytest.approx(-1.0, abs=1e-9)
