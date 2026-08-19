"""
src/utils/checksums.py

SHA-256 checksum helpers for artifact integrity verification (Phase D6,
docs/IMPLEMENTATION_PLAN.md).

Background: torch.load(..., weights_only=False) and raw pickle.load() both
execute arbitrary code on deserialization. D6 replaces every such call on
the model-serving path (XGBoost, TFT, feature transformers) with a safe
native/joblib format plus a checksum manifest that MUST be verified before
any bytes are handed to a deserializer. This module defines that checksum
manifest format exactly once so every artifact loader shares the same
algorithm and the same fail-closed behavior.

Shared by src/training/manifest.py (dataset/config hashing, pre-dates D6)
and every D6 artifact loader (XGBTrainer, TFTTrainer, FeatureEngineer).
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Mapping, Union

logger = logging.getLogger(__name__)

_HASH_CHUNK_SIZE = 1 << 20  # 1 MiB — stream large files rather than load in memory


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Stream a file through sha256 in fixed-size chunks (avoids loading a
    multi-hundred-MB artifact fully into memory just to hash it)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(
    checksum_path: Union[str, Path], files: Mapping[str, Union[str, Path]]
) -> Dict[str, str]:
    """
    Hash each path in `files` and write a {name: sha256} JSON manifest to
    `checksum_path`.

    `files` keys are logical artifact names (e.g. "model", "metadata") used
    again at verify time — they need not match the files' own basenames.

    Returns the computed {name: sha256} mapping.
    """
    checksums = {name: compute_file_sha256(path) for name, path in files.items()}
    checksum_path = Path(checksum_path)
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    return checksums


def read_checksum_manifest(checksum_path: Union[str, Path]) -> Dict[str, str]:
    """
    Parse a checksum manifest into its {name: sha256} mapping without
    verifying anything.

    Exists so callers can decide which of several *candidate* artifact
    files are actually required (i.e. present in the manifest) before
    building the `files` argument passed to `verify_checksums` — filtering
    candidates by filesystem existence instead would let a required file
    that has gone missing from disk be silently skipped rather than
    rejected. See FeatureEngineer.load_transformers for the motivating case
    (some artifact files, like the fitted PCA, are legitimately optional —
    absent from the manifest entirely when never fit — while others are
    always written by save_transformers and must never be missing).

    Raises:
        FileNotFoundError: `checksum_path` does not exist.
    """
    checksum_path = Path(checksum_path)
    if not checksum_path.exists():
        raise FileNotFoundError(
            f"Checksum manifest not found: {checksum_path}. Refusing to load "
            "an artifact whose integrity cannot be verified."
        )
    return json.loads(checksum_path.read_text(encoding="utf-8"))


def verify_checksums(
    checksum_path: Union[str, Path], files: Mapping[str, Union[str, Path]]
) -> None:
    """
    Recompute sha256 for each path in `files` and compare against the
    manifest recorded at `checksum_path`.

    This is the control that makes it safe to joblib.load()/load_model() an
    artifact on the fraud-api container's startup path: a corrupted or
    tampered file is rejected here, before any deserializer ever sees its
    bytes. Callers MUST call this before deserializing — a failed check must
    never be caught and silently ignored.

    NOTE: this only checks the files it is given. A caller that pre-filters
    `files` down to whatever happens to exist on disk defeats the "missing
    file" guard below — build `files` from the manifest's own keys (see
    `read_checksum_manifest`) when some candidates are optional, not from
    `Path.exists()`.

    Raises:
        FileNotFoundError: `checksum_path` does not exist, or one of the
            files in `files` does not exist — an artifact whose integrity
            cannot even be checked must never be trusted.
        ValueError: a file has no entry in the manifest, or its recomputed
            hash does not match the recorded one.
    """
    recorded = read_checksum_manifest(checksum_path)

    for name, path in files.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact file missing: {path} (checksum entry '{name}')."
            )
        if name not in recorded:
            raise ValueError(
                f"Checksum manifest {checksum_path} has no entry for '{name}' "
                f"({path}) — refusing to load an unverified artifact file."
            )
        actual = compute_file_sha256(path)
        if actual != recorded[name]:
            raise ValueError(
                f"Checksum mismatch for '{name}' ({path}): expected "
                f"{recorded[name]}, got {actual}. The artifact may be "
                "corrupted or tampered with — refusing to load it."
            )
