import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

from src.training.run_logging import RunLogger
from src.training.train_lgbm import LGBMTrainer


@pytest.fixture
def mock_config():
    return {
        "model": {
            "lightgbm": {
                "n_estimators": 10,
                "max_depth": 3,
                "num_leaves": 15,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_data_in_leaf": 1,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "early_stopping_rounds": 5,
            }
        }
    }


@pytest.fixture
def mock_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y = (X["feat_0"] > 0.5).astype(int)

    X_train, y_train = X.iloc[:80], y.iloc[:80]
    X_val, y_val = X.iloc[80:], y.iloc[80:]
    return X_train, y_train, X_val, y_val


def test_train_with_run_logger_writes_tensorboard_events_and_checkpoints(
    mock_config, mock_data, tmp_path
):
    """Closes finding F2's mid-training-visibility gap for LightGBM:
    passing run_logger= to the real train() must produce an actual
    TensorBoard event file and at least one periodic checkpoint. n_estimators
    is bumped to 100 (default checkpoint cadence is every 100 rounds, see
    _make_tb_checkpoint_callback) with early stopping disabled so the
    default callback configuration — the one main() actually uses — has a
    chance to fire at least once."""
    config = {
        "model": {
            "lightgbm": {
                **mock_config["model"]["lightgbm"],
                "n_estimators": 100,
                "early_stopping_rounds": 100,  # disable early stopping for this probe
            }
        }
    }
    trainer = LGBMTrainer(config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)

    run_logger = RunLogger(
        run_type="lightgbm", run_name="test",
        base_dir=tmp_path / "runs", checkpoint_base_dir=tmp_path / "checkpoints",
    )
    try:
        trainer.train(X_train, y_train, X_val, y_val, run_logger=run_logger)
    finally:
        run_logger.close()

    event_files = list(run_logger.log_dir.glob("events.out.tfevents.*"))
    assert event_files, f"No TensorBoard event file written to {run_logger.log_dir}"

    checkpoint_files = list(run_logger.checkpoint_dir.glob("checkpoint_step_*.txt"))
    assert checkpoint_files, f"No checkpoint written to {run_logger.checkpoint_dir}"


def test_model_build_with_params(mock_config):
    trainer = LGBMTrainer(mock_config)
    model = trainer.build_model(scale_pos_weight=10.0)

    assert isinstance(model, LGBMClassifier)
    assert model.n_estimators == 10
    assert model.max_depth == 3
    assert model.scale_pos_weight == 10.0


def test_model_train_predict(mock_config, mock_data):
    trainer = LGBMTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data

    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    y_prob = trainer.predict_proba(X_val)

    assert y_prob.shape == (20,)
    assert (y_prob >= 0).all() and (y_prob <= 1).all()


def test_predict_raises_without_frozen_threshold(mock_config, mock_data):
    """Phase C4 parity with XGBTrainer/TFTTrainer: predict() must not
    silently invent a threshold."""
    trainer = LGBMTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    with pytest.raises(ValueError):
        trainer.predict(X_val)


def test_set_threshold_and_predict_applies_frozen_threshold(mock_config, mock_data):
    trainer = LGBMTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    trainer.set_threshold(0.5)
    y_prob = trainer.predict_proba(X_val)
    y_pred = trainer.predict(X_val)

    np.testing.assert_array_equal(y_pred, (y_prob >= 0.5).astype(int))


def test_threshold_persists_through_save_load(mock_config, mock_data, tmp_path):
    trainer = LGBMTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)
    trainer.set_threshold(0.3721)

    model_path = tmp_path / "model.pkl"
    trainer.save(str(model_path))
    loaded_trainer = LGBMTrainer.load(str(model_path))

    assert loaded_trainer.threshold == pytest.approx(0.3721)
    loaded_trainer.predict(X_val)


def test_calibrator_persists_through_save_load(mock_config, mock_data, tmp_path):
    """Phase C4 (mle-reviewer follow-up from C2): LGBMTrainer previously had
    no calibrator field at all — this is the parity gap the reviewer
    flagged as HIGH ("fitted calibrator isn't persisted")."""
    trainer = LGBMTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    y_prob_val = trainer.predict_proba(X_val)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(y_prob_val, y_val)
    trainer.set_calibrator(calibrator)

    model_path = tmp_path / "model.pkl"
    trainer.save(str(model_path))
    loaded_trainer = LGBMTrainer.load(str(model_path))

    assert loaded_trainer.calibrator is not None
    y_prob_cal_original = calibrator.predict(y_prob_val)
    y_prob_cal_loaded = loaded_trainer.predict_proba_calibrated(X_val)
    np.testing.assert_array_almost_equal(y_prob_cal_original, y_prob_cal_loaded)


def test_predict_proba_calibrated_raises_without_calibrator(mock_config, mock_data):
    trainer = LGBMTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    with pytest.raises(ValueError):
        trainer.predict_proba_calibrated(X_val)


def test_load_warns_on_pre_c4_artifact_missing_threshold_and_calibrator(mock_config, mock_data, tmp_path, caplog):
    """An artifact saved before this fix (no threshold/calibrator keys) must
    still load, but load() should warn rather than silently proceed, same
    contract as XGBTrainer.load()/TFTTrainer.load()."""
    trainer = LGBMTrainer(mock_config)
    X_train, y_train, X_val, y_val = mock_data
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train, y_train, X_val, y_val)

    model_path = tmp_path / "model.pkl"
    trainer.save(str(model_path))  # no threshold/calibrator ever set

    with caplog.at_level("WARNING"):
        loaded_trainer = LGBMTrainer.load(str(model_path))

    assert loaded_trainer.threshold is None
    assert loaded_trainer.calibrator is None
    assert any("pre-Phase-C4" in record.message for record in caplog.records)
