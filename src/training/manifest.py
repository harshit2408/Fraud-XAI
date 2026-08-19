"""
src/training/manifest.py

Model manifest: reproducibility & provenance record (Phase D4,
docs/IMPLEMENTATION_PLAN.md).

MEDIUM finding: model artifacts (models/xgb_model.pkl, models/tft_model.ckpt)
were flat overwritten files with no manifest linking them to the config,
dataset, git commit, or metrics that produced them — "no model versioning or
rollback path" (architecture note #5). This module writes a small JSON
sidecar next to every trained artifact so a specific saved model can always
be traced back to the exact run that produced it, without retraining.

Used by src/training/train_xgb.py and src/training/train_tft.py.
"""

import hashlib
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict

from src.utils.checksums import compute_file_sha256  # noqa: F401 — re-exported;
# tests/unit/test_model_manifest.py and other call sites import
# compute_file_sha256 from this module. Phase D6 centralized the hashing
# implementation in src/utils/checksums.py (shared with the artifact
# checksum manifests XGBTrainer/TFTTrainer/FeatureEngineer now write), but
# this module keeps re-exporting it so existing import sites don't need to
# change.

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = "1.0"


class ModelManifest(BaseModel):
    """Provenance record written next to every trained model artifact.

    Every field here answers one rollback question: "given only this file,
    can I find the exact code, config, and data that produced the artifact
    sitting beside it?" `git_sha` is the one field that can legitimately be
    absent (see `resolve_git_sha()`) — everything else is always populated.
    """

    # protected_namespaces=() — same reasoning as src/config.py's
    # _StrictModel: `model_type` is a legitimate field name here, not an
    # accidental collision with pydantic's own "model_*" internals.
    model_config = ConfigDict(protected_namespaces=())

    schema_version: str = MANIFEST_SCHEMA_VERSION
    model_type: str
    mlflow_run_id: Optional[str]
    config_hash: str
    git_sha: Optional[str]
    dataset_hash: str
    dataset_files: List[str]
    random_seed: int
    metrics: Dict[str, float]
    created_at: str


def compute_dataset_hash(processed_dir: Union[str, Path], filenames: Sequence[str]) -> str:
    """
    Combined sha256 over the listed processed-data files.

    Filenames are sorted before hashing so the result doesn't depend on the
    order the caller passed them in — only on which files and what content.
    Each file's name is folded into the hash alongside its content hash, so
    swapping two same-content files under different names still changes the
    combined hash (name is part of what "the dataset" means here).

    Raises:
        FileNotFoundError: any listed file is missing from `processed_dir` —
            a manifest must never silently describe a dataset it didn't
            actually observe.
    """
    processed_dir = Path(processed_dir)
    combined = hashlib.sha256()
    for name in sorted(filenames):
        path = processed_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot compute dataset hash: {path} does not exist. "
                "The manifest must reflect the data the model actually trained on."
            )
        combined.update(name.encode("utf-8"))
        combined.update(compute_file_sha256(path).encode("utf-8"))
    return combined.hexdigest()


def resolve_git_sha(cwd: Optional[Union[str, Path]] = None) -> Optional[str]:
    """
    Resolve the current commit SHA via `git rev-parse HEAD`.

    Returns None (with a loud warning) rather than raising when:
      - `git` is not installed,
      - `cwd` is not inside a git repository, or
      - the repository has no commits yet (a fresh `git init` with nothing
        committed — HEAD does not resolve).
    A model manifest must always be written even when provenance is
    incomplete; a missing git_sha degrades traceability, it must never block
    training or artifact saving.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(
            "⚠️  git_sha unavailable — could not run `git rev-parse HEAD` (%s). "
            "Model manifest will record git_sha: null.",
            exc,
        )
        return None

    if result.returncode != 0:
        logger.warning(
            "⚠️  git_sha unavailable — not a git repository, or the repository has "
            "no commits yet (%s). Model manifest will record git_sha: null; this "
            "artifact cannot be traced back to a source commit.",
            result.stderr.strip() or "git rev-parse HEAD failed",
        )
        return None

    return result.stdout.strip()


def build_manifest(
    *,
    model_type: str,
    mlflow_run_id: Optional[str],
    settings: Any,
    dataset_dir: Union[str, Path],
    dataset_files: Sequence[str],
    random_seed: int,
    metrics: Dict[str, float],
    git_sha: Optional[str] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> ModelManifest:
    """Assemble a `ModelManifest` for a just-trained model.

    Args:
        model_type: short identifier, e.g. "xgboost" or "tft".
        mlflow_run_id: the active MLflow run's ID, so the manifest and the
            MLflow run cross-reference each other.
        settings: the validated `src.config.Settings` instance used for this
            run — hashed via `src.config.config_hash`.
        dataset_dir: directory containing the processed parquet files that
            were actually read for this run.
        dataset_files: filenames (relative to `dataset_dir`) that make up
            "the dataset" for hashing purposes — typically all six
            train/val/test features+labels files.
        random_seed: the seed applied via `set_seed()` for this run.
        metrics: flat name -> float metrics to embed (e.g. pr_auc_test).
        git_sha: pass an already-resolved SHA to avoid re-invoking git; if
            omitted, resolved here via `resolve_git_sha(cwd=repo_root)`.
        repo_root: working directory for git SHA resolution when `git_sha`
            is not supplied. Defaults to the current process's cwd.
    """
    # Imported here (not at module top) to avoid a circular import: src.config
    # does not import src.training, but keeping this local makes the
    # dependency direction explicit and cheap to re-check.
    from src.config import config_hash as compute_config_hash

    resolved_git_sha = git_sha if git_sha is not None else resolve_git_sha(cwd=repo_root)

    return ModelManifest(
        model_type=model_type,
        mlflow_run_id=mlflow_run_id,
        config_hash=compute_config_hash(settings),
        git_sha=resolved_git_sha,
        dataset_hash=compute_dataset_hash(dataset_dir, dataset_files),
        dataset_files=sorted(dataset_files),
        random_seed=random_seed,
        metrics=metrics,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def manifest_path_for(model_path: Union[str, Path]) -> Path:
    """The manifest sidecar path for a given model artifact path.

    `models/xgb_model.pkl` -> `models/xgb_model.manifest.json`
    `models/tft_model.ckpt` -> `models/tft_model.manifest.json`
    """
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}.manifest.json")


def write_manifest(model_path: Union[str, Path], manifest: ModelManifest) -> Path:
    """Write `manifest` as JSON next to `model_path`. Returns the manifest's path."""
    path = manifest_path_for(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"Model manifest written to {path}")
    return path
