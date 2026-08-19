"""
tests/unit/test_tuned_params_artifact.py

Phase D3 (docs/IMPLEMENTATION_PLAN.md, HIGH finding L204-219): tune_xgb.py's
winning hyperparameters had no provenance record — the shipped config.yaml
diverged from the documented Optuna result, and there was no run ID or
dataset hash linking either to an actual experiment. tune_xgb.py used to end
by telling a human to "Update config/config.yaml with these parameters"
(a manual, unrecorded promotion step).

This module asserts:
  1. tune_xgb.py writes a versioned, run-ID-tagged YAML artifact
     (config/tuned/xgb_<run_id>.yaml) containing the winning params plus
     enough metadata to trace it back to the MLflow run.
  2. train_xgb.py can read that file "by reference" via load_tuned_params()
     and fails fast (not silently) on a missing or malformed file.

Run: pytest tests/unit/test_tuned_params_artifact.py -v
"""

import copy
from pathlib import Path
from typing import Any, Dict

import optuna
import pytest
import yaml

from src.config import Settings, config_hash
from src.training.train_xgb import load_tuned_params
from src.training.tune_xgb import build_tuned_params_document, write_tuned_params


# ── tune_xgb.py: building and writing the artifact ─────────────────────────


def test_build_tuned_params_document_shape():
    doc = build_tuned_params_document(
        run_id="abc123",
        best_params={"n_estimators": 897, "learning_rate": 0.0893},
        best_value=0.6012,
        seed=42,
        n_trials=30,
    )
    assert doc["mlflow_run_id"] == "abc123"
    assert doc["random_seed"] == 42
    assert doc["n_trials"] == 30
    assert doc["best_value"] == 0.6012
    assert doc["params"] == {"n_estimators": 897, "learning_rate": 0.0893}
    assert "created_at" in doc  # timestamp present, not asserting exact value


def _toy_study(seed: int = 42, n_trials: int = 3) -> optuna.Study:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: -(t.suggest_float("learning_rate", 0.01, 0.2) - 0.09) ** 2, n_trials=n_trials)
    return study


def test_write_tuned_params_creates_run_id_tagged_yaml_file(tmp_path: Path):
    study = _toy_study()
    path = write_tuned_params(tmp_path, run_id="run-xyz", study=study, seed=42, n_trials=3)

    assert path == tmp_path / "xgb_run-xyz.yaml"
    assert path.exists()


def test_write_tuned_params_file_is_valid_yaml_with_expected_keys(tmp_path: Path):
    study = _toy_study()
    path = write_tuned_params(tmp_path, run_id="run-xyz", study=study, seed=42, n_trials=3)

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["mlflow_run_id"] == "run-xyz"
    assert loaded["random_seed"] == 42
    assert loaded["n_trials"] == 3
    assert loaded["best_value"] == study.best_value
    assert loaded["params"] == study.best_params


def test_write_tuned_params_creates_directory_if_missing(tmp_path: Path):
    nested_dir = tmp_path / "config" / "tuned"
    assert not nested_dir.exists()

    study = _toy_study()
    write_tuned_params(nested_dir, run_id="run-1", study=study, seed=1, n_trials=3)

    assert nested_dir.exists()


def test_write_tuned_params_file_has_a_provenance_header_comment(tmp_path: Path):
    study = _toy_study()
    path = write_tuned_params(tmp_path, run_id="run-xyz", study=study, seed=42, n_trials=3)

    raw = path.read_text(encoding="utf-8")
    assert raw.startswith("#")
    assert "run-xyz" in raw.splitlines()[0] + raw.splitlines()[1]


# ── train_xgb.py: reading the artifact by reference ─────────────────────────


def test_load_tuned_params_reads_a_valid_file(tmp_path: Path):
    study = _toy_study()
    path = write_tuned_params(tmp_path, run_id="run-1", study=study, seed=42, n_trials=3)

    document = load_tuned_params(path)
    assert document["params"] == study.best_params
    assert document["mlflow_run_id"] == "run-1"


def test_load_tuned_params_missing_file_raises_file_not_found_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_tuned_params(tmp_path / "does_not_exist.yaml")


def test_load_tuned_params_file_without_params_key_raises_value_error(tmp_path: Path):
    bad_file = tmp_path / "malformed.yaml"
    bad_file.write_text(yaml.safe_dump({"mlflow_run_id": "x", "best_value": 0.5}), encoding="utf-8")

    with pytest.raises(ValueError, match="params"):
        load_tuned_params(bad_file)


def test_load_tuned_params_non_dict_document_raises_value_error(tmp_path: Path):
    """A YAML file that parses but whose root isn't a mapping at all (e.g. a
    bare list) must fail the same fail-fast ValueError path as a dict
    missing 'params' — not an unhandled TypeError/AttributeError from
    `"params" not in document`/`document["params"]` further downstream."""
    bad_file = tmp_path / "not_a_mapping.yaml"
    bad_file.write_text(yaml.safe_dump(["n_estimators", 897]), encoding="utf-8")

    with pytest.raises(ValueError, match="params"):
        load_tuned_params(bad_file)


def test_round_trip_write_then_load_preserves_params(tmp_path: Path):
    """The exact write -> read cycle train_xgb.py --tuned-params exercises."""
    study = _toy_study(seed=7, n_trials=5)
    path = write_tuned_params(tmp_path, run_id="round-trip", study=study, seed=7, n_trials=5)

    document = load_tuned_params(path)
    assert document["params"] == study.best_params


# ── D4 regression (mle-reviewer HIGH finding) ───────────────────────────────


def test_config_hash_reflects_tuned_params_merge(tmp_path: Path, config: Dict[str, Any]):
    """A manifest's config_hash must change when --tuned-params overrides
    xgboost hyperparameters — otherwise two models trained from genuinely
    different effective hyperparameters get byte-identical config_hash
    values in their manifest sidecars, defeating D4's "rollback target is
    identifiable without retraining" acceptance criterion.

    Replicates train_xgb.py main()'s exact merge-then-revalidate pattern:
    config["model"]["xgboost"].update(tuned_document["params"]) followed by
    Settings.model_validate(config) — NOT hashing the pre-merge `settings`
    object, which was the bug this test guards against.
    """
    base_settings = Settings.model_validate(config)
    base_hash = config_hash(base_settings)

    # A tuned value guaranteed to differ from config.yaml's shipped value.
    tuned_params = {"n_estimators": config["model"]["xgboost"]["n_estimators"] + 1}
    document = build_tuned_params_document(
        run_id="run-1", best_params=tuned_params, best_value=0.6, seed=42, n_trials=1
    )
    tuned_path = tmp_path / "xgb_run-1.yaml"
    tuned_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    loaded = load_tuned_params(tuned_path)
    merged = copy.deepcopy(config)
    merged["model"]["xgboost"].update(loaded["params"])

    effective_settings = Settings.model_validate(merged)
    effective_hash = config_hash(effective_settings)

    assert effective_hash != base_hash


def test_config_hash_unchanged_when_tuned_params_equal_existing_config(
    tmp_path: Path, config: Dict[str, Any]
):
    """Sanity check on the above: re-validating must not spuriously change
    the hash when the tuned params happen to equal the existing config —
    the hash tracks content, not "was a merge performed"."""
    base_settings = Settings.model_validate(config)
    base_hash = config_hash(base_settings)

    tuned_params = {"n_estimators": config["model"]["xgboost"]["n_estimators"]}
    document = build_tuned_params_document(
        run_id="run-1", best_params=tuned_params, best_value=0.6, seed=42, n_trials=1
    )
    tuned_path = tmp_path / "xgb_run-1.yaml"
    tuned_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    loaded = load_tuned_params(tuned_path)
    merged = copy.deepcopy(config)
    merged["model"]["xgboost"].update(loaded["params"])

    effective_settings = Settings.model_validate(merged)
    assert config_hash(effective_settings) == base_hash


def test_config_hash_rejects_malformed_tuned_params_with_clear_validation_error(
    tmp_path: Path, config: Dict[str, Any]
):
    """The MEDIUM finding bundled with the HIGH fix: re-validating the
    merged config means a malformed tuned-params file (wrong type) now
    fails fast with a pydantic ValidationError naming the field, instead of
    a late, unclear XGBoost/sklearn TypeError deep inside build_model()."""
    from pydantic import ValidationError

    merged = copy.deepcopy(config)
    merged["model"]["xgboost"]["learning_rate"] = "not-a-float"

    with pytest.raises(ValidationError, match="learning_rate"):
        Settings.model_validate(merged)
