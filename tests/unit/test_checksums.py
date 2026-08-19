"""
tests/unit/test_checksums.py

TDD for src/utils/checksums.py (Phase D6, docs/IMPLEMENTATION_PLAN.md) — the
shared checksum-manifest helper every D6 artifact loader (XGBTrainer,
TFTTrainer, FeatureEngineer) relies on to refuse deserializing a corrupted
or tampered artifact.
"""

import json
from pathlib import Path

import pytest

from src.utils.checksums import compute_file_sha256, verify_checksums, write_checksums


def test_compute_file_sha256_matches_hashlib_reference(tmp_path):
    import hashlib

    path = tmp_path / "a.bin"
    path.write_bytes(b"hello world")
    assert compute_file_sha256(path) == hashlib.sha256(b"hello world").hexdigest()


def test_compute_file_sha256_changes_with_content(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert compute_file_sha256(a) != compute_file_sha256(b)


def test_write_checksums_creates_manifest_with_correct_hashes(tmp_path):
    model_file = tmp_path / "model.bin"
    meta_file = tmp_path / "meta.bin"
    model_file.write_bytes(b"model-bytes")
    meta_file.write_bytes(b"meta-bytes")

    checksum_path = tmp_path / "artifact.checksums.json"
    result = write_checksums(checksum_path, {"model": model_file, "metadata": meta_file})

    assert checksum_path.exists()
    recorded = json.loads(checksum_path.read_text())
    assert recorded == result
    assert recorded["model"] == compute_file_sha256(model_file)
    assert recorded["metadata"] == compute_file_sha256(meta_file)


def test_verify_checksums_passes_for_untampered_files(tmp_path):
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"model-bytes")
    checksum_path = tmp_path / "artifact.checksums.json"
    write_checksums(checksum_path, {"model": model_file})

    # Must not raise.
    verify_checksums(checksum_path, {"model": model_file})


def test_verify_checksums_raises_on_tampered_file(tmp_path):
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"model-bytes")
    checksum_path = tmp_path / "artifact.checksums.json"
    write_checksums(checksum_path, {"model": model_file})

    # Simulate corruption/tampering after the checksum was recorded.
    model_file.write_bytes(b"a-different-payload-entirely")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksums(checksum_path, {"model": model_file})


def test_verify_checksums_raises_on_missing_manifest(tmp_path):
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"model-bytes")

    with pytest.raises(FileNotFoundError):
        verify_checksums(tmp_path / "does_not_exist.checksums.json", {"model": model_file})


def test_verify_checksums_raises_on_missing_artifact_file(tmp_path):
    checksum_path = tmp_path / "artifact.checksums.json"
    write_checksums(checksum_path, {})  # empty manifest is fine to write

    with pytest.raises(FileNotFoundError):
        verify_checksums(checksum_path, {"model": tmp_path / "never_written.bin"})


def test_verify_checksums_raises_when_manifest_missing_entry(tmp_path):
    model_file = tmp_path / "model.bin"
    extra_file = tmp_path / "extra.bin"
    model_file.write_bytes(b"model-bytes")
    extra_file.write_bytes(b"extra-bytes")

    checksum_path = tmp_path / "artifact.checksums.json"
    write_checksums(checksum_path, {"model": model_file})  # no "extra" entry

    with pytest.raises(ValueError, match="no entry"):
        verify_checksums(checksum_path, {"model": model_file, "extra": extra_file})
