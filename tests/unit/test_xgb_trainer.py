import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from src.training.run_logging import RunLogger
from src.training.train_xgb import XGBTrainer, _artifact_paths, resolve_scale_pos_weight


# ─── PRD Phase 9 P9-3: scale_pos_weight config-override resolution ──────────


def test_resolve_scale_pos_weight_uses_positive_config_value():
    cfg = {"model": {"xgboost": {"scale_pos_weight": 1}}}
    assert resolve_scale_pos_weight(cfg, neg_count=27_000, pos_count=1_000) == 1.0


def test_resolve_scale_pos_weight_falls_back_to_ratio_when_absent():
    cfg = {"model": {"xgboost": {}}}
    assert resolve_scale_pos_weight(cfg, neg_count=27_000, pos_count=1_000) == 27.0


@pytest.mark.parametrize("bad", [0, -5, None, False, True])
def test_resolve_scale_pos_weight_ignores_non_positive_or_bool_config(bad):
    cfg = {"model": {"xgboost": {"scale_pos_weight": bad}}}
    # bool is a subclass of int — True must NOT be read as spw=1.
    assert resolve_scale_pos_weight(cfg, neg_count=20, pos_count=2) == 10.0


def test_resolve_scale_pos_weight_reads_the_named_model_key():
    cfg = {"model": {"lightgbm": {"scale_pos_weight": 3}, "xgboost": {"scale_pos_weight": 9}}}
    assert resolve_scale_pos_weight(cfg, 20, 2, model_key="lightgbm") == 3.0
    assert resolve_scale_pos_weight(cfg, 20, 2, model_key="xgboost") == 9.0


def test_resolve_scale_pos_weight_rejects_zero_positives():
    with pytest.raises(ValueError, match="pos_count must be > 0"):
        resolve_scale_pos_weight({"model": {}}, neg_count=10, pos_count=0)


@pytest.fixture
def mock_config():
    return {
        "model": {
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "eval_metric": "aucpr",
                "early_stopping_rounds": 5,
                # Phase D7: pinned explicitly so these tests train/predict on
                # CPU deterministically regardless of what GPU hardware (if
                # any) happens to be present on the machine running pytest —
                # build_model() now resolves a missing key to "auto", which
                # would otherwise pick "cuda" on a CUDA-equipped host.
                "device": "cpu",
            }
        }
    }


@pytest.fixture
def mock_data():
    np.random.seed(42)
    # 100 rows, 5 features
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)])
    # Target: somewhat dependent on feat_0
    y = (X["feat_0"] > 0.5).astype(int)
    
    # Split
    X_train, y_train = X.iloc[:80], y.iloc[:80]
    X_val, y_val = X.iloc[80:], y.iloc[80:]
    return X_train, y_train, X_val, y_val


def test_train_with_run_logger_writes_tensorboard_events_and_checkpoints(
    mock_config, mock_data, tmp_path
):
    """Closes part of finding F2's fix (mid-training visibility) for
    XGBoost: passing run_logger= to the real train() must produce an actual
    TensorBoard event file and at least one periodic checkpoint, not just
    console log lines. n_estimators is bumped to 60 here (default
    checkpoint cadence is every 50 rounds) so the default callback
    configuration — the one main() actually uses — has a chance to fire at
    least once, without early stopping cutting the run short first."""
    config = {
        "model": {
            "xgboost": {
                **mock_config["model"]["xgboost"],
                "n_estimators": 60,
                "early_stopping_rounds": 60,  # disable early stopping for this probe
            }
        }
    }
    trainer = XGBTrainer(config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)

    run_logger = RunLogger(
        run_type="xgboost", run_name="test",
        base_dir=tmp_path / "runs", checkpoint_base_dir=tmp_path / "checkpoints",
    )
    try:
        trainer.train(X_train, y_train, X_val, y_val, run_logger=run_logger)
    finally:
        run_logger.close()

    event_files = list(run_logger.log_dir.glob("events.out.tfevents.*"))
    assert event_files, f"No TensorBoard event file written to {run_logger.log_dir}"

    checkpoint_files = list(run_logger.checkpoint_dir.glob("checkpoint_step_*.ubj"))
    assert checkpoint_files, f"No checkpoint written to {run_logger.checkpoint_dir}"


def test_model_build_with_params(mock_config):
    trainer = XGBTrainer(mock_config)
    model = trainer.build_model(scale_pos_weight=10.0)
    
    assert isinstance(model, XGBClassifier)
    assert model.n_estimators == 10
    assert model.max_depth == 3
    assert model.learning_rate == 0.1
    assert model.scale_pos_weight == 10.0
    assert model.eval_metric == "aucpr"
    assert model.early_stopping_rounds == 5


def test_model_train_predict(mock_config, mock_data):
    trainer = XGBTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Predict
    y_prob = trainer.predict_proba(X_val)
    
    assert y_prob.shape == (20,)
    assert (y_prob >= 0).all() and (y_prob <= 1).all()


def test_model_save_load(mock_config, mock_data, tmp_path):
    # Train
    trainer = XGBTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    y_prob_original = trainer.predict_proba(X_val)

    # Save
    model_path = tmp_path / "model.pkl"
    trainer.save(str(model_path))

    paths = _artifact_paths(model_path)
    assert paths["model"].exists()
    assert paths["metadata"].exists()
    assert paths["checksums"].exists()

    # Load
    loaded_trainer = XGBTrainer.load(str(model_path))
    assert loaded_trainer.model is not None
    assert loaded_trainer.feature_names == list(X_val.columns)

    # Predict again
    y_prob_loaded = loaded_trainer.predict_proba(X_val)

    np.testing.assert_array_almost_equal(y_prob_original, y_prob_loaded)


def test_model_save_does_not_pickle_the_booster(mock_config, mock_data, tmp_path):
    """Phase D6: the XGBoost model itself must be persisted via
    save_model() (JSON/UBJ) — never pickle — so loading an artifact can
    never execute arbitrary code as a side effect of deserializing it."""
    trainer = XGBTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    model_path = tmp_path / "model.pkl"
    trainer.save(str(model_path))

    # A literal ".pkl" file must never be written — the whole point of D6
    # is that nothing on this path is a pickle anymore.
    assert not model_path.exists()

    paths = _artifact_paths(model_path)
    # XGBoost's UBJ/JSON native format is not a pickle stream: it must not
    # start with pickle's protocol-2+ opcode, and must be independently
    # loadable via xgb.Booster().load_model() with no custom unpickler.
    raw = paths["model"].read_bytes()
    assert not raw.startswith(pickle.PROTO)
    booster = xgb.Booster()
    booster.load_model(str(paths["model"]))  # must not raise


def test_saved_artifact_paths_are_all_real_files_mlflow_can_log(mock_config, mock_data, tmp_path):
    """Regression test (review finding, mle-reviewer/security-reviewer/
    python-reviewer independently caught this): train_xgb.py's main() used
    to call mlflow.log_artifact(model_path) where model_path is the literal
    stem passed to save() — post-D6 save() never writes that literal path,
    so mlflow.log_artifact would raise FileNotFoundError immediately after
    every successful training run. main() was fixed to iterate
    _artifact_paths(path).values() instead; this guards that every one of
    those derived paths is a real, existing file after save(), and that the
    literal stem itself is not."""
    trainer = XGBTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    model_path = tmp_path / "model.pkl"
    trainer.save(str(model_path))

    assert not model_path.exists()
    for artifact_path in _artifact_paths(model_path).values():
        assert artifact_path.is_file(), (
            f"{artifact_path} does not exist — mlflow.log_artifact(str(artifact_path)) would raise"
        )


def test_load_raises_on_tampered_model_file(mock_config, mock_data, tmp_path):
    """Phase D6: a corrupted/tampered artifact must be rejected before any
    deserializer sees its bytes, never silently loaded."""
    trainer = XGBTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    model_path = tmp_path / "model.pkl"
    trainer.save(str(model_path))

    paths = _artifact_paths(model_path)
    with open(paths["metadata"], "ab") as f:
        f.write(b"tampered-bytes")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        XGBTrainer.load(str(model_path))


def test_predict_raises_without_frozen_threshold(mock_config, mock_data):
    """Phase C4: predict() must not silently invent a threshold."""
    trainer = XGBTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    with pytest.raises(ValueError):
        trainer.predict(X_val)


def test_set_threshold_and_predict_applies_frozen_threshold(mock_config, mock_data):
    trainer = XGBTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    trainer.set_threshold(0.5)
    y_prob = trainer.predict_proba(X_val)
    y_pred = trainer.predict(X_val)

    np.testing.assert_array_equal(y_pred, (y_prob >= 0.5).astype(int))


def test_threshold_persists_through_save_load(mock_config, mock_data, tmp_path):
    """Phase C4: serving must read the frozen threshold, never recompute it."""
    trainer = XGBTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)
    trainer.set_threshold(0.3721)

    model_path = tmp_path / "model.pkl"
    trainer.save(str(model_path))
    loaded_trainer = XGBTrainer.load(str(model_path))

    assert loaded_trainer.threshold == pytest.approx(0.3721)
    # Loaded trainer can predict without ever calling find_optimal_threshold again.
    loaded_trainer.predict(X_val)


def test_calibrator_persists_through_save_load(mock_config, mock_data, tmp_path):
    """Phase C4 (mle-reviewer follow-up from C2): the fitted calibrator must
    ship with the artifact, not be discarded at the end of training."""
    trainer = XGBTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    y_prob_val = trainer.predict_proba(X_val)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(y_prob_val, y_val)
    trainer.set_calibrator(calibrator)

    model_path = tmp_path / "model.pkl"
    trainer.save(str(model_path))
    loaded_trainer = XGBTrainer.load(str(model_path))

    assert loaded_trainer.calibrator is not None
    y_prob_cal_original = calibrator.predict(y_prob_val)
    y_prob_cal_loaded = loaded_trainer.predict_proba_calibrated(X_val)
    np.testing.assert_array_almost_equal(y_prob_cal_original, y_prob_cal_loaded)


def test_load_forces_cpu_even_if_artifact_trained_on_cuda(mock_config, mock_data, tmp_path):
    """Phase D7: serving/inference must run on CPU regardless of the device
    the artifact was trained on — the fraud-api container has no GPU.

    Trains on CPU (portable across CI hosts), then simulates a GPU-trained
    artifact by overwriting the booster's device param before save, so this
    asserts load()'s force_cpu wiring specifically, not merely that
    training happened to already be on CPU.
    """
    trainer = XGBTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    # Simulate an artifact that was trained with device="cuda".
    trainer.model.set_params(device="cuda")

    model_path = tmp_path / "model.pkl"
    trainer.save(str(model_path))
    loaded_trainer = XGBTrainer.load(str(model_path))

    assert loaded_trainer.model.get_params()["device"] == "cpu"


def test_predict_proba_calibrated_raises_without_calibrator(mock_config, mock_data):
    trainer = XGBTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    with pytest.raises(ValueError):
        trainer.predict_proba_calibrated(X_val)
