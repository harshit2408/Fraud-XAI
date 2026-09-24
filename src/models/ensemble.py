"""
src/models/ensemble.py

N-model probability ensemble via constrained (simplex) weight search.

Single source of truth for `scripts/run_ensemble_eval.py`'s weight-search
logic (2026-08-20, docs/IMPLEMENTATION_PLAN.md Phase F-Audit). Previously
this module held a `ModelEnsemble` class doing conceptually the same thing
but hardcoded to exactly two named weights (`w_xgb`/`w_tft`) via grid
search, and was dead code — `run_ensemble_eval.py` never called it,
reimplementing a 2-model `scipy.optimize.minimize_scalar` search inline
instead, and no test file existed for either path. Two divergent, drifting
ensemble implementations. This rewrite (prompted by proposing LightGBM as a
3rd ensemble input, reviewed by mle-reviewer) generalizes to N models, adds
the review's required diagnostics, and is the only implementation left.

mle-reviewer's review (2026-08-20) is why this module looks the way it
does, specifically:
  - Grid search, not gradient-based (SLSQP): PR-AUC as a function of blend
    weights is piecewise (driven by rank swaps at each candidate weight),
    not smooth, so finite-difference gradients are noisy and prone to local
    optima. A coarse-to-fine grid is cheap here (each candidate point costs
    one weighted sum + one `average_precision_score` call over
    already-computed probability arrays) and gives an interpretable,
    loggable sweep.
  - `pairwise_diagnostics`: XGBoost and LightGBM are both GBDTs trained on
    the identical feature set with the identical imbalance strategy
    (`scale_pos_weight`), unlike TFT which is architecturally decorrelated
    (sequential, windowed per-card history). High correlation between two
    inputs means any ensemble gain is more likely variance reduction than a
    new decorrelated signal — this should be checked before trusting a
    validation PR-AUC lift, not assumed away.
  - `bootstrap_weight_ci`: a single point-optimized weight on a validation
    set already reused for per-model calibration and threshold selection
    can chase sampling noise, especially when two inputs are correlated
    (the objective surface goes nearly flat along that axis). Resampling
    gives a distribution to inspect instead of trusting one number.
"""

import logging
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)


def blend(prob_dict: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Weighted sum of per-model probability arrays.

    Weights are used as given, not renormalized — callers that want a
    convex combination (the only case this codebase actually uses) must
    ensure their weights already sum to 1; `grid_search_simplex_weights`
    and `bootstrap_weight_ci` both guarantee that.

    Raises:
        ValueError: `weights` and `prob_dict` don't have identical key sets,
            or `prob_dict` is empty.
    """
    if not prob_dict:
        raise ValueError("prob_dict must contain at least one model's probabilities.")
    if set(prob_dict.keys()) != set(weights.keys()):
        raise ValueError(
            f"weights keys {sorted(weights.keys())} do not match "
            f"prob_dict keys {sorted(prob_dict.keys())}"
        )
    out = None
    for name, prob in prob_dict.items():
        term = weights[name] * np.asarray(prob, dtype=float)
        out = term if out is None else out + term
    return out


def _simplex_grid(model_names: List[str], axes: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
    """Enumerate weight combinations, one free axis per model but the last
    (whose weight is fixed by the simplex constraint: all weights sum to
    1), from each model's candidate axis values in `axes`. Points where the
    implied last weight would be negative are dropped.
    """
    free_names = model_names[:-1]
    last_name = model_names[-1]
    points = []
    for combo in product(*(axes[name] for name in free_names)):
        remainder = 1.0 - sum(combo)
        if remainder < -1e-9:
            continue
        point = dict(zip(free_names, combo))
        point[last_name] = max(0.0, remainder)
        points.append(point)
    return points


def _best_over(
    y_true: np.ndarray, prob_dict: Dict[str, np.ndarray], points: List[Dict[str, float]]
) -> Tuple[Dict[str, float], float]:
    best_w, best_score = None, -1.0
    for point in points:
        score = average_precision_score(y_true, blend(prob_dict, point))
        if score > best_score:
            best_score, best_w = score, point
    return best_w, best_score


def grid_search_simplex_weights(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    step: float = 0.02,
    refine_step: float = 0.002,
) -> Tuple[Dict[str, float], float]:
    """
    Coarse-to-fine grid search over the probability simplex, maximizing
    PR-AUC (average precision) on `(y_true, blend(prob_dict, weights))`.

    Works for any number of models (1+); the 2-model case reproduces a
    standard 1-D sweep. Pass 1 covers the full simplex at `step`
    granularity; pass 2 re-grids the ±`step` neighborhood of pass 1's
    winner at `refine_step` granularity. Cheap because every evaluated
    point is just a weighted sum over already-computed arrays.

    Returns:
        (best_weights, best_pr_auc) — `best_weights` sums to 1.0 (up to
        floating point) with one key per model in `prob_dict`.
    """
    y_true = np.asarray(y_true)
    model_names = list(prob_dict.keys())

    if len(model_names) == 1:
        only = model_names[0]
        weights = {only: 1.0}
        return weights, float(average_precision_score(y_true, blend(prob_dict, weights)))

    n_steps = max(1, round(1.0 / step))
    coarse_axes = {name: np.linspace(0.0, 1.0, n_steps + 1) for name in model_names}
    coarse_w, coarse_score = _best_over(y_true, prob_dict, _simplex_grid(model_names, coarse_axes))

    n_fine = max(1, round((step * 2) / refine_step))
    fine_axes = {
        name: np.linspace(max(0.0, w - step), min(1.0, w + step), n_fine + 1)
        for name, w in coarse_w.items()
    }
    fine_points = _simplex_grid(model_names, fine_axes)
    refined_w, refined_score = (
        _best_over(y_true, prob_dict, fine_points) if fine_points else (coarse_w, coarse_score)
    )

    return (refined_w, refined_score) if refined_score >= coarse_score else (coarse_w, coarse_score)


def bootstrap_weight_ci(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    step: float = 0.05,
    n_boot: int = 50,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Bootstrap-resample `(y_true, prob_dict)` with replacement `n_boot`
    times, re-run the grid search on each resample, and collect the
    resulting per-model weight distribution.

    A single point-optimized weight on a fixed validation set can chase
    sampling noise, particularly when two inputs are correlated (the PR-AUC
    surface goes nearly flat along that axis, so the point optimum is
    unstable). This gives a distribution: a wide interval, or one that
    straddles zero, for a given model's weight means "the optimizer's
    choice for this model isn't robust to which validation rows happened to
    be sampled" — evidence to shrink that model's weight toward zero rather
    than trust the single point estimate, not that the model has no value.

    Uses `step` for both the coarse and refine grid (no sub-refinement) to
    keep the O(n_boot × grid_size) cost bounded — this is a stability
    diagnostic, not the production weight search.

    Returns:
        Dict mapping each model name in `prob_dict` to a `(n_boot,)` array
        of its weight in that resample's optimum.
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    model_names = list(prob_dict.keys())
    weight_samples = {name: np.empty(n_boot) for name in model_names}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        prob_boot = {name: np.asarray(prob)[idx] for name, prob in prob_dict.items()}
        w, _ = grid_search_simplex_weights(y_boot, prob_boot, step=step, refine_step=step)
        for name in model_names:
            weight_samples[name][b] = w[name]

    return weight_samples


def pairwise_diagnostics(prob_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Pearson correlation between every pair of models' probability arrays.

    A cheap pre-registered diversity check, run before trusting a
    multi-model blend's validation PR-AUC: two inputs with near-identical
    scores add variance-reduction at best, not a new decorrelated signal,
    however good the blend looks on paper. Compare against a known-good
    pair (e.g. XGBoost/TFT here) to judge whether a candidate addition
    (e.g. LightGBM) is likely to behave the same way.
    """
    names = list(prob_dict.keys())
    out: Dict[str, float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            corr = float(np.corrcoef(np.asarray(prob_dict[a]), np.asarray(prob_dict[b]))[0, 1])
            out[f"corr_{a}_{b}"] = corr
    return out
