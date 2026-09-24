"""
scripts/export_test_probabilities.py

Export the deployed ensemble's blended test-set probabilities (PRD Phase 8).

`notebooks/05_business_impact.ipynb` needs one vector of scores and one vector
of labels to sweep thresholds against. Producing them inside the notebook would
mean loading XGBoost, LightGBM *and* the TFT (torch) on every execution — slow,
memory-hungry, and impossible while the docker stack holds the paging file. So
the heavy work happens here, once, and the notebook reads a small `.npz`.

**The blend is read from `models/ensemble.json`, not recomputed.** That artifact
is what serving loads (ADR-001 §3.5), so the business report describes the
weights and threshold actually deployed rather than re-deriving a fresh optimum
nobody is running. Per-model probabilities go through each trainer's frozen
calibrator, matching `ensemble.json`'s declared `per_model_calibrated` space —
blending raw scores from three model families would put the report on a
different scale than production.

Output: `reports/ensemble_test_probabilities.npz`
    y_true     int8    (n,)  ground-truth labels for the test split
    y_prob     float64 (n,)  blended calibrated fraud probability
    threshold  float64 ()    the deployed operating threshold

Usage:
    python scripts/export_test_probabilities.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings  # noqa: E402
from src.serving.ensemble_spec import load_ensemble_spec  # noqa: E402
from src.training.train_lgbm import LGBMTrainer  # noqa: E402
from src.training.train_tft import TFTTrainer  # noqa: E402
from src.training.train_xgb import XGBTrainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "reports" / "ensemble_test_probabilities.npz"
# PRD Phase 9 (9.1 onward): every step re-derives the operating threshold on
# the VALIDATION split before reporting frozen test metrics — same discipline
# as the trainers (Phase C1). The val blend is exported here alongside test so
# the Phase 9 threshold re-derivation reads it rather than re-scoring three
# model families itself.
VAL_OUTPUT_PATH = PROJECT_ROOT / "reports" / "ensemble_val_probabilities.npz"

# Must match SequenceBuilder's grouping in TFTTrainer (train_tft.py:316).
_SEQUENCE_GROUP_COL = "card1"


def _calibrated(trainer, name: str, *args, **kwargs) -> np.ndarray:
    """Per-model calibrated probabilities, or a hard failure.

    Unlike `run_ensemble_eval.py`'s equivalent, this does NOT fall back to raw
    scores: `models/ensemble.json` declares `probability_space:
    per_model_calibrated`, so a missing calibrator means the artifact on disk
    does not match the spec the weights were fitted under. Silently blending
    raw scores would put the business report on the wrong scale.
    """
    if trainer.calibrator is None:
        raise RuntimeError(
            f"{name} has no frozen calibrator, but models/ensemble.json declares "
            f"probability_space=per_model_calibrated. Re-train {name} (Phase C4) "
            f"before exporting probabilities for the business report."
        )
    return trainer.predict_proba_calibrated(*args, **kwargs)


def _sequence_history(
    X_train: pd.DataFrame, X_val: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """The tail of train+val the TFT needs to fill test's first sequences.

    The TFT looks back `sequence_length` transactions **per card** (Phase B6),
    so only the last `sequence_length - 1` rows of each card can ever enter a
    test-row window. Passing the full 590k-row concatenation instead is what an
    earlier version did, and pandas' consolidating copy of it exhausts memory on
    a 16 GB box. Trimming per card is exact, not an approximation: the rows
    dropped are unreachable from any test sequence.
    """
    # `TFTTrainer` constructs its SequenceBuilder with a hardcoded
    # group_col="card1" (src/training/train_tft.py:316); `sequence_length`
    # lives under the `data` block.
    group_col = _SEQUENCE_GROUP_COL
    seq_len = int(config.get("data", {}).get("sequence_length", 10))
    combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    if group_col not in combined.columns or seq_len <= 1:
        return combined
    tail = combined.groupby(group_col, sort=False).tail(seq_len - 1)
    logger.info(
        "Sequence history trimmed to %d rows (last %d per %s) from %d",
        len(tail),
        seq_len - 1,
        group_col,
        len(combined),
    )
    return tail.reset_index(drop=True)


def export() -> Path:
    config = load_settings("config/config.yaml").model_dump()
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]

    spec = load_ensemble_spec(PROJECT_ROOT / "models" / "ensemble.json")
    mode = spec.mode()
    logger.info(
        "Deployed blend: weights=%s threshold=%.6f", mode.weights, mode.threshold
    )

    logger.info("Loading splits...")
    X_train = pd.read_parquet(processed_dir / "train_features.parquet")
    X_val = pd.read_parquet(processed_dir / "val_features.parquet")
    X_test = pd.read_parquet(processed_dir / "test_features.parquet")
    y_val = pd.read_parquet(processed_dir / "val_labels.parquet").squeeze()
    y_test = pd.read_parquet(processed_dir / "test_labels.parquet").squeeze()

    serving = config.get("serving", {})
    logger.info("Loading models...")
    xgb = XGBTrainer.load(serving.get("model_path", "models/xgb_model.pkl"))
    lgbm = LGBMTrainer.load(serving.get("lgbm_model_path", "models/lgbm_model.pkl"))
    tft = TFTTrainer.load(
        serving.get("tft_model_path", "models/tft_model.pt"), config=config
    )

    def _blend(per_model: dict) -> np.ndarray:
        blended = np.zeros(len(next(iter(per_model.values()))), dtype=np.float64)
        for name, weight in mode.weights.items():
            blended += weight * np.asarray(per_model[name], dtype=np.float64)
        return blended

    # The TFT needs the last sequence_length-1 rows per card as history for
    # the first windows of each split. For val that history is train's tail;
    # for test it is (train+val)'s tail.
    logger.info("Scoring the validation split...")
    val_history = _sequence_history(X_train, X_train.iloc[:0], config)
    val_prob = _blend({
        "xgb": _calibrated(xgb, "XGBoost", X_val),
        "lgbm": _calibrated(lgbm, "LightGBM", X_val),
        "tft": _calibrated(tft, "TFT", X_val, history_X=val_history),
    })
    _write_npz(VAL_OUTPUT_PATH, y_val, val_prob, mode.threshold, "validation")

    logger.info("Scoring the test split...")
    history = _sequence_history(X_train, X_val, config)
    test_prob = _blend({
        "xgb": _calibrated(xgb, "XGBoost", X_test),
        "lgbm": _calibrated(lgbm, "LightGBM", X_test),
        "tft": _calibrated(tft, "TFT", X_test, history_X=history),
    })
    _write_npz(OUTPUT_PATH, y_test, test_prob, mode.threshold, "test")
    return OUTPUT_PATH


def _write_npz(
    path: Path, y_true_series: pd.Series, y_prob: np.ndarray, threshold: float, split: str
) -> None:
    y_true = np.asarray(y_true_series.values, dtype=np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, y_true=y_true, y_prob=y_prob, threshold=np.float64(threshold)
    )
    logger.info(
        "Wrote %s (%s): %d rows, %d fraud, %d flagged at threshold %.6f",
        path,
        split,
        len(y_true),
        int(y_true.sum()),
        int((y_prob >= threshold).sum()),
        threshold,
    )


if __name__ == "__main__":
    export()
