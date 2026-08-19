"""
tests/unit/test_tft_artifact_persistence.py

TDD for Phase C4 — freeze the validation-selected threshold and the
validation-fitted calibrator into the TFT checkpoint, mirroring the same
contract already added to XGBTrainer: serving must read threshold/calibrator
from the artifact, never recompute/refit either.

File: docs/IMPLEMENTATION_PLAN.md Phase C4, src/training/train_tft.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_tft import TFTTrainer, _artifact_paths


def _minimal_config() -> dict:
    return {
        "project": {"random_seed": 42},
        "data": {"sequence_length": 3},
        "model": {"tft": {"device": "cpu"}},
        "imbalance": {"sampling_strategy": "none", "loss_function": "bce"},
    }


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "card1": [1] * n,
        "TransactionAmt": np.arange(n, dtype=float),
    })


def test_predict_raises_without_frozen_threshold():
    trainer = TFTTrainer(_minimal_config())
    trainer.build_model(num_numeric_features=1, num_static_features=0)

    with pytest.raises(ValueError):
        trainer.predict(_frame(4))


def test_set_threshold_and_predict_applies_frozen_threshold():
    trainer = TFTTrainer(_minimal_config())
    trainer.build_model(num_numeric_features=1, num_static_features=0)
    trainer.model.eval()
    trainer.set_threshold(0.5)

    X = _frame(4)
    y_prob = trainer.predict_proba(X)
    y_pred = trainer.predict(X)

    np.testing.assert_array_equal(y_pred, (y_prob >= 0.5).astype(int))


def test_predict_proba_calibrated_raises_without_calibrator():
    trainer = TFTTrainer(_minimal_config())
    trainer.build_model(num_numeric_features=1, num_static_features=0)

    with pytest.raises(ValueError):
        trainer.predict_proba_calibrated(_frame(4))


def test_threshold_and_calibrator_persist_through_save_load(tmp_path):
    trainer = TFTTrainer(_minimal_config())
    trainer.build_model(num_numeric_features=1, num_static_features=0)
    trainer.model.eval()

    X = _frame(4)
    y_prob = trainer.predict_proba(X)
    y_dummy = np.array([0, 0, 1, 1])
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(y_prob, y_dummy)

    trainer.set_threshold(0.3721)
    trainer.set_calibrator(calibrator)

    ckpt_path = tmp_path / "tft_model.ckpt"
    trainer.save(str(ckpt_path))

    paths = _artifact_paths(ckpt_path)
    assert paths["weights"].exists()
    assert paths["metadata"].exists()
    assert paths["checksums"].exists()
    # A literal ".ckpt" file must never be written — Phase D6 replaces the
    # single weights_only=False checkpoint with a tensors-only weights file
    # plus a separate joblib metadata sidecar.
    assert not ckpt_path.exists()

    loaded = TFTTrainer.load(str(ckpt_path))

    assert loaded.threshold == pytest.approx(0.3721)
    loaded.model.eval()
    y_prob_loaded = loaded.predict_proba(X)
    np.testing.assert_array_almost_equal(y_prob, y_prob_loaded)

    y_prob_cal_expected = calibrator.predict(y_prob)
    y_prob_cal_loaded = loaded.predict_proba_calibrated(X)
    np.testing.assert_array_almost_equal(y_prob_cal_expected, y_prob_cal_loaded)

    # Loaded trainer can predict without ever recomputing the threshold.
    loaded.predict(X)


def test_saved_artifact_paths_are_all_real_files_mlflow_can_log(tmp_path):
    """Regression test (review finding): train_tft.py's main() used to call
    mlflow.log_artifact(model_path) where model_path is the literal stem
    passed to save() — post-D6 save() never writes that literal path, so
    mlflow.log_artifact would raise FileNotFoundError immediately after
    every successful training run. main() was fixed to iterate
    _artifact_paths(path).values() instead; this guards that every one of
    those derived paths is a real, existing file after save()."""
    trainer = TFTTrainer(_minimal_config())
    trainer.build_model(num_numeric_features=1, num_static_features=0)
    trainer.model.eval()

    ckpt_path = tmp_path / "tft_model.ckpt"
    trainer.save(str(ckpt_path))

    assert not ckpt_path.exists()
    for artifact_path in _artifact_paths(ckpt_path).values():
        assert artifact_path.is_file(), (
            f"{artifact_path} does not exist — mlflow.log_artifact(str(artifact_path)) would raise"
        )


def test_weights_file_loads_under_weights_only_true(tmp_path):
    """Phase D6: the weights file must contain nothing but the model's
    state_dict — loadable with torch.load(weights_only=True), the strict
    unpickler that refuses arbitrary classes. This is the actual guarantee
    D6 exists to provide; asserting it directly (not just that
    TFTTrainer.load succeeds) prevents a future edit from smuggling a
    non-tensor object back into the weights file."""
    trainer = TFTTrainer(_minimal_config())
    trainer.build_model(num_numeric_features=1, num_static_features=0)
    trainer.model.eval()

    ckpt_path = tmp_path / "tft_model.ckpt"
    trainer.save(str(ckpt_path))
    paths = _artifact_paths(ckpt_path)

    state_dict = torch.load(paths["weights"], map_location="cpu", weights_only=True)
    assert set(state_dict.keys()) == set(trainer.model.state_dict().keys())


def test_load_raises_on_tampered_weights_file(tmp_path):
    """Phase D6: a corrupted/tampered checkpoint must be rejected before
    any deserializer sees its bytes, never silently loaded."""
    trainer = TFTTrainer(_minimal_config())
    trainer.build_model(num_numeric_features=1, num_static_features=0)
    trainer.model.eval()
    trainer.set_threshold(0.5)

    ckpt_path = tmp_path / "tft_model.ckpt"
    trainer.save(str(ckpt_path))
    paths = _artifact_paths(ckpt_path)

    with open(paths["weights"], "ab") as f:
        f.write(b"tampered-bytes")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        TFTTrainer.load(str(ckpt_path))


def test_load_forces_cpu_even_when_config_requests_auto_and_cuda_available(tmp_path, monkeypatch):
    """Phase D7: TFTTrainer.load() must force CPU regardless of what the
    checkpoint's config says and regardless of GPU availability — the
    serving host is not guaranteed to have a GPU."""
    trainer = TFTTrainer(_minimal_config())
    trainer.build_model(num_numeric_features=1, num_static_features=0)
    trainer.model.eval()

    ckpt_path = tmp_path / "tft_model.ckpt"
    trainer.save(str(ckpt_path))

    auto_cuda_config = _minimal_config()
    auto_cuda_config["model"]["tft"]["device"] = "auto"
    monkeypatch.setattr("src.device.torch.cuda.is_available", lambda: True)

    loaded = TFTTrainer.load(str(ckpt_path), config=auto_cuda_config)

    assert loaded.device.type == "cpu"
