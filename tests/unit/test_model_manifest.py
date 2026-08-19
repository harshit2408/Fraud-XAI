"""
tests/unit/test_model_manifest.py

Phase D4 (docs/IMPLEMENTATION_PLAN.md, architecture note #5): "No model
versioning or rollback path. Artifacts are flat overwritten files with no
manifest linking them to config, dataset, git SHA, or metrics." This module
covers src/training/manifest.py (the manifest writer used by train_xgb.py
and train_tft.py) and src/config.py's config_hash().

Covers:
  - compute_file_sha256 / compute_dataset_hash: deterministic, content- and
    name-sensitive, order-independent, fail fast on a missing file.
  - resolve_git_sha: real subprocess behavior against a temp git repo in
    all three states this project has actually been in — no repo, a repo
    with no commits (git init but nothing committed — this project's exact
    state right after Phase D4's git-init decision), and a repo with a
    commit — plus a mocked "git binary missing" path.
  - config_hash (src/config.py): deterministic, sensitive to any field change.
  - build_manifest / write_manifest: end-to-end assembly and the sidecar
    naming convention (`<stem>.manifest.json` next to the model artifact).

Run: pytest tests/unit/test_model_manifest.py -v
"""

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest

from src.config import Settings, config_hash
from src.training.manifest import (
    ModelManifest,
    build_manifest,
    compute_dataset_hash,
    compute_file_sha256,
    manifest_path_for,
    resolve_git_sha,
    write_manifest,
)


# ── compute_file_sha256 / compute_dataset_hash ──────────────────────────────


def test_compute_file_sha256_matches_hashlib_reference(tmp_path: Path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"some file content" * 1000)

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert compute_file_sha256(path) == expected


def test_compute_file_sha256_changes_with_content(tmp_path: Path):
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    path_a.write_bytes(b"content A")
    path_b.write_bytes(b"content B")

    assert compute_file_sha256(path_a) != compute_file_sha256(path_b)


def test_compute_dataset_hash_is_order_independent(tmp_path: Path):
    (tmp_path / "train.parquet").write_bytes(b"train-bytes")
    (tmp_path / "val.parquet").write_bytes(b"val-bytes")

    forward = compute_dataset_hash(tmp_path, ["train.parquet", "val.parquet"])
    backward = compute_dataset_hash(tmp_path, ["val.parquet", "train.parquet"])
    assert forward == backward


def test_compute_dataset_hash_changes_when_content_changes(tmp_path: Path):
    (tmp_path / "train.parquet").write_bytes(b"train-bytes-v1")
    before = compute_dataset_hash(tmp_path, ["train.parquet"])

    (tmp_path / "train.parquet").write_bytes(b"train-bytes-v2")
    after = compute_dataset_hash(tmp_path, ["train.parquet"])

    assert before != after


def test_compute_dataset_hash_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        compute_dataset_hash(tmp_path, ["does_not_exist.parquet"])


# ── resolve_git_sha ──────────────────────────────────────────────────────────


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_resolve_git_sha_returns_none_outside_a_repo(tmp_path: Path):
    assert resolve_git_sha(cwd=tmp_path) is None


def test_resolve_git_sha_returns_none_when_repo_has_no_commits(tmp_path: Path):
    """This project's exact state immediately after `git init` with nothing
    committed yet — HEAD does not resolve. Must degrade, not raise."""
    _git("init", cwd=tmp_path)
    assert resolve_git_sha(cwd=tmp_path) is None


def test_resolve_git_sha_returns_sha_after_a_commit(tmp_path: Path):
    _git("init", cwd=tmp_path)
    _git("commit", "--allow-empty", "-m", "test commit", cwd=tmp_path)

    sha = resolve_git_sha(cwd=tmp_path)
    assert sha is not None
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_resolve_git_sha_handles_missing_git_binary(tmp_path: Path, monkeypatch):
    def _raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise_file_not_found)
    assert resolve_git_sha(cwd=tmp_path) is None


# ── config_hash (src/config.py) ──────────────────────────────────────────────


def test_config_hash_is_deterministic(config: Dict[str, Any]):
    settings = Settings.model_validate(config)
    assert config_hash(settings) == config_hash(settings)


def test_config_hash_matches_for_separately_validated_identical_configs(config: Dict[str, Any]):
    settings_a = Settings.model_validate(config)
    settings_b = Settings.model_validate(config)
    assert config_hash(settings_a) == config_hash(settings_b)


def test_config_hash_changes_when_a_field_changes(config: Dict[str, Any]):
    import copy

    settings_a = Settings.model_validate(config)

    modified = copy.deepcopy(config)
    modified["model"]["xgboost"]["n_estimators"] += 1
    settings_b = Settings.model_validate(modified)

    assert config_hash(settings_a) != config_hash(settings_b)


def test_config_hash_is_a_64_char_hex_digest(config: Dict[str, Any]):
    settings = Settings.model_validate(config)
    digest = config_hash(settings)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ── build_manifest / write_manifest ──────────────────────────────────────────


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "processed"
    d.mkdir()
    for name in ["train_features.parquet", "train_labels.parquet"]:
        (d / name).write_bytes(f"content of {name}".encode("utf-8"))
    return d


def test_build_manifest_populates_all_fields(config: Dict[str, Any], dataset_dir: Path):
    settings = Settings.model_validate(config)

    manifest = build_manifest(
        model_type="xgboost",
        mlflow_run_id="run-123",
        settings=settings,
        dataset_dir=dataset_dir,
        dataset_files=["train_features.parquet", "train_labels.parquet"],
        random_seed=42,
        metrics={"pr_auc_test": 0.56},
        git_sha=None,  # explicit — this test doesn't care about git state
    )

    assert isinstance(manifest, ModelManifest)
    assert manifest.model_type == "xgboost"
    assert manifest.mlflow_run_id == "run-123"
    assert manifest.config_hash == config_hash(settings)
    assert manifest.git_sha is None
    assert manifest.dataset_files == sorted(["train_features.parquet", "train_labels.parquet"])
    assert manifest.random_seed == 42
    assert manifest.metrics == {"pr_auc_test": 0.56}
    assert manifest.created_at  # non-empty timestamp string


def test_build_manifest_uses_supplied_git_sha_without_calling_git(
    config: Dict[str, Any], dataset_dir: Path, monkeypatch
):
    """When git_sha is passed explicitly, resolve_git_sha must not be invoked."""
    settings = Settings.model_validate(config)

    def _fail(*args, **kwargs):
        raise AssertionError("resolve_git_sha should not have been called")

    monkeypatch.setattr("src.training.manifest.resolve_git_sha", _fail)

    manifest = build_manifest(
        model_type="tft",
        mlflow_run_id=None,
        settings=settings,
        dataset_dir=dataset_dir,
        dataset_files=["train_features.parquet", "train_labels.parquet"],
        random_seed=42,
        metrics={},
        git_sha="deadbeef" * 5,
    )
    assert manifest.git_sha == "deadbeef" * 5


def test_manifest_path_for_naming_convention():
    assert manifest_path_for("models/xgb_model.pkl") == Path("models/xgb_model.manifest.json")
    assert manifest_path_for("models/tft_model.ckpt") == Path("models/tft_model.manifest.json")


def test_write_manifest_creates_valid_json_sidecar(
    config: Dict[str, Any], dataset_dir: Path, tmp_path: Path
):
    settings = Settings.model_validate(config)
    manifest = build_manifest(
        model_type="xgboost",
        mlflow_run_id="run-abc",
        settings=settings,
        dataset_dir=dataset_dir,
        dataset_files=["train_features.parquet", "train_labels.parquet"],
        random_seed=42,
        metrics={"pr_auc_test": 0.56},
        git_sha=None,
    )

    model_path = tmp_path / "models" / "xgb_model.pkl"
    written_path = write_manifest(model_path, manifest)

    assert written_path == tmp_path / "models" / "xgb_model.manifest.json"
    assert written_path.exists()

    reloaded = ModelManifest.model_validate_json(written_path.read_text(encoding="utf-8"))
    assert reloaded == manifest
