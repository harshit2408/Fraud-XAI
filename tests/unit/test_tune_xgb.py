"""
tests/unit/test_tune_xgb.py

Phase D2 (docs/IMPLEMENTATION_PLAN.md, MEDIUM finding L238-248):
  (a) the Optuna study used the default (unseeded) TPESampler, so the search
      path was not reproducible run-to-run;
  (b) MedianPruner(n_warmup_steps=10) was configured but objective() never
      called trial.report()/trial.should_prune(), so pruning was a
      structural no-op that gave a false impression compute was being saved.

This module asserts both are actually fixed:
  1. `build_sampler(seed)` used across two independent Optuna studies with
     the same seed and search space yields *identical* trial trajectories
     (docs/IMPLEMENTATION_PLAN.md's stated acceptance: "Repeat study yields
     identical trials").
  2. `XGBPruningCallback` genuinely reports per-round and prunes — proven at
     the unit level (a fake trial) and end-to-end (a real, small XGBoost +
     Optuna run where pruning demonstrably fires).

Run: pytest tests/unit/test_tune_xgb.py -v
"""

from unittest.mock import MagicMock

import numpy as np
import optuna
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score

from src.training.tune_xgb import XGBPruningCallback, build_sampler


# ── D2(a): seeded sampler reproducibility ──────────────────────────────────


def _toy_objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10.0, 10.0)
    y = trial.suggest_int("y", 0, 20)
    return -((x - 3.0) ** 2 + (y - 7) ** 2)


def _run_toy_study(seed: int, n_trials: int = 12) -> list[dict]:
    study = optuna.create_study(direction="maximize", sampler=build_sampler(seed))
    study.optimize(_toy_objective, n_trials=n_trials)
    return [t.params for t in study.trials]


def test_build_sampler_returns_a_seeded_tpe_sampler():
    sampler = build_sampler(42)
    assert isinstance(sampler, optuna.samplers.TPESampler)


def test_two_studies_with_same_seed_yield_identical_trials():
    """The core D2 acceptance criterion: repeating a study with the same
    seed must produce the exact same sequence of suggested trial params."""
    first_run = _run_toy_study(seed=42)
    second_run = _run_toy_study(seed=42)
    assert first_run == second_run


def test_studies_with_different_seeds_diverge():
    """Sanity check on the above: the seed must actually be doing something,
    not just being accepted and ignored."""
    run_a = _run_toy_study(seed=1)
    run_b = _run_toy_study(seed=2)
    assert run_a != run_b


# ── D2(b): per-round pruning via XGBPruningCallback ────────────────────────


def test_pruning_callback_reports_score_each_round():
    trial = MagicMock()
    trial.should_prune.return_value = False
    callback = XGBPruningCallback(trial, eval_set_name="validation_0", metric_name="aucpr")

    evals_log = {"validation_0": {"aucpr": [0.1, 0.2, 0.31]}}
    result = callback.after_iteration(model=None, epoch=2, evals_log=evals_log)

    trial.report.assert_called_once_with(0.31, 2)
    assert result is False  # False == "keep training"


def test_pruning_callback_raises_trial_pruned_when_should_prune():
    trial = MagicMock()
    trial.should_prune.return_value = True
    callback = XGBPruningCallback(trial, eval_set_name="validation_0", metric_name="aucpr")

    evals_log = {"validation_0": {"aucpr": [0.1]}}
    with pytest.raises(optuna.TrialPruned):
        callback.after_iteration(model=None, epoch=0, evals_log=evals_log)

    trial.report.assert_called_once_with(0.1, 0)


def test_pruning_callback_raises_clear_error_on_unknown_eval_set():
    trial = MagicMock()
    callback = XGBPruningCallback(trial, eval_set_name="validation_0", metric_name="aucpr")

    with pytest.raises(KeyError, match="validation_0"):
        callback.after_iteration(model=None, epoch=0, evals_log={"train": {"aucpr": [0.5]}})


@pytest.fixture(scope="module")
def toy_classification_data():
    X, y = make_classification(
        n_samples=600, n_features=10, weights=[0.85, 0.15], random_state=0
    )
    return X[:450], y[:450], X[450:], y[450:]


def test_pruning_fires_in_a_real_optuna_plus_xgboost_study(toy_classification_data):
    """End-to-end regression: with the callback wired in and an aggressive
    pruner, at least one trial must actually be pruned. Before Phase D2 this
    was structurally impossible — trial.report() was never called, so
    trial.should_prune() could never return True."""
    X_train, y_train, X_val, y_val = toy_classification_data

    def objective(trial: optuna.Trial) -> float:
        callback = XGBPruningCallback(trial, eval_set_name="validation_0", metric_name="aucpr")
        model = xgb.XGBClassifier(
            n_estimators=40,
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=0.3,
            eval_metric="aucpr",
            tree_method="hist",
            early_stopping_rounds=10,
            callbacks=[callback],
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, proba)

    study = optuna.create_study(
        direction="maximize",
        sampler=build_sampler(42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=1),
    )
    study.optimize(objective, n_trials=8)

    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    assert len(pruned) > 0, "MedianPruner never fired — pruning is still a no-op"
