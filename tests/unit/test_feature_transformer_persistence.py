"""
tests/unit/test_feature_transformer_persistence.py

TDD for Phase D6 (docs/IMPLEMENTATION_PLAN.md) — FeatureEngineer.
save_transformers/load_transformers must persist via joblib with a sha256
checksum manifest instead of raw pickle.dump/pickle.load, and load_transformers
must verify checksums before deserializing anything.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_engineering import FeatureEngineer


@pytest.fixture()
def fitted_feature_engineer() -> FeatureEngineer:
    """A FeatureEngineer with every persisted piece of state populated, via
    the same public fit=True call sequence run_pipeline uses."""
    n = 60
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "TransactionAmt": rng.uniform(1.0, 500.0, n),
            "isFraud": (rng.random(n) < 0.2).astype(int),
            "card1": rng.integers(1000, 1010, n),
            "card2": rng.integers(100, 110, n),
            "ProductCD": rng.choice(["W", "H", "C"], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", np.nan], n),
            **{f"V{i}": rng.uniform(-1, 1, n) for i in range(1, 6)},
        }
    )

    fe = FeatureEngineer()
    df = fe.create_null_count_features(df)  # must run before imputation/PCA
    df = fe.handle_missing_values(df, fit=True)
    df = fe.encode_categoricals(df, fit=True)
    df = fe.create_card_hash_features(df, fit=True)
    df = fe.create_target_encoding(df, fit=True, update_state=True)
    fe.reduce_v_features(df, fit=True)
    return fe


def test_load_transformers_restores_identical_state(fitted_feature_engineer, tmp_path):
    fe = fitted_feature_engineer
    save_dir = tmp_path / "transformers"
    fe.save_transformers(str(save_dir))

    loaded = FeatureEngineer()
    loaded.load_transformers(str(save_dir))

    assert loaded._label_encoders.keys() == fe._label_encoders.keys()
    for col, le in fe._label_encoders.items():
        np.testing.assert_array_equal(loaded._label_encoders[col].classes_, le.classes_)
    assert loaded._freq_encoders == fe._freq_encoders
    assert loaded._num_fill_values == fe._num_fill_values
    assert loaded._card_hash_freq == fe._card_hash_freq
    assert loaded._global_target_mean == pytest.approx(fe._global_target_mean)
    assert loaded._target_enc_state.keys() == fe._target_enc_state.keys()
    assert loaded._null_ratio_denominator == fe._null_ratio_denominator

    assert loaded._pca is not None
    np.testing.assert_array_almost_equal(loaded._pca.components_, fe._pca.components_)


def test_save_transformers_writes_no_raw_pickle_files(fitted_feature_engineer, tmp_path):
    """Phase D6: every persisted transformer file must be joblib, not a
    bare pickle.dump — and a checksum manifest must exist alongside them."""
    save_dir = tmp_path / "transformers"
    fitted_feature_engineer.save_transformers(str(save_dir))

    written = {p.name for p in save_dir.iterdir()}
    assert not any(name.endswith(".pkl") for name in written), (
        f"Found raw .pkl file(s) in {written} — expected .joblib + a checksums manifest"
    )
    assert any(name.endswith(".joblib") for name in written)
    assert "checksums.json" in written


def test_load_transformers_raises_on_tampered_file(fitted_feature_engineer, tmp_path):
    """Phase D6: a corrupted/tampered transformer file must be rejected
    before any deserializer sees its bytes, never silently loaded."""
    save_dir = tmp_path / "transformers"
    fitted_feature_engineer.save_transformers(str(save_dir))

    label_encoders_path = next(save_dir.glob("label_encoders.joblib"))
    with open(label_encoders_path, "ab") as f:
        f.write(b"tampered-bytes")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        FeatureEngineer().load_transformers(str(save_dir))


def test_load_transformers_raises_when_a_required_file_is_deleted(fitted_feature_engineer, tmp_path):
    """Regression test (review finding): load_transformers used to filter
    candidate files down to whatever existed on disk BEFORE calling
    verify_checksums — so a required artifact (label_encoders.joblib is
    unconditionally written by save_transformers, unlike the optional
    pca.joblib) that goes missing while checksums.json still references it
    was silently skipped (self._label_encoders left as {}, no error) rather
    than rejected. Deleting it here must raise, not silently no-op."""
    save_dir = tmp_path / "transformers"
    fitted_feature_engineer.save_transformers(str(save_dir))

    (save_dir / "label_encoders.joblib").unlink()

    with pytest.raises(FileNotFoundError, match="label_encoders"):
        FeatureEngineer().load_transformers(str(save_dir))


def test_load_transformers_raises_on_missing_checksum_manifest(fitted_feature_engineer, tmp_path):
    """A directory with transformer files but no checksum manifest (e.g. an
    artifact from before Phase D6) must never be silently trusted."""
    save_dir = tmp_path / "transformers"
    fitted_feature_engineer.save_transformers(str(save_dir))

    (save_dir / "checksums.json").unlink()

    with pytest.raises(FileNotFoundError):
        FeatureEngineer().load_transformers(str(save_dir))
