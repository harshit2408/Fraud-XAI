"""
Standalone resume of ADR-006 Arm C trial 14, which crashed with a CUDA
"bad allocation" error at epoch 26 (val PR-AUC had just hit 0.5505 at
epoch 25, clearing Gate 1's 0.5502 threshold, before the crash discarded
the in-study Optuna record). Runs in an isolated process, config taken
directly from the trial's logged hyperparameters, resuming from the
surviving checkpoint (reports/gnn_hpo_checkpoints/trial_014.ckpt.pt,
epoch 26, best_val_pr_auc=0.5505, epochs_no_improve=1).

Not an Optuna trial — this is a one-off confirmatory continuation to see
whether the config's val curve genuinely plateaus above or below 0.5502,
independent of the study's bookkeeping.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings  # noqa: E402
from src.training.train_gnn import GNNTrainer  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("reports/logs/gnn_trial14_resume.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

SEED = 42
CHECKPOINT_PATH = "reports/gnn_hpo_checkpoints/trial_014.ckpt.pt"

# Trial 14's exact sampled params, from reports/logs/gnn_hpo_20260920_191348.log
TRIAL_14_PARAMS = {
    "input_dropout": 0.15,
    "hidden_dims": [128, 64, 32],
    "residual": True,
    "l2_normalize": False,
    "aggr": "mean",
    "num_neighbors": [15, 10, 5],
    "edge_spec_set": "addr_card",
}


def main() -> None:
    settings = load_settings("config/config.yaml")
    config = settings.model_dump()

    g = config["model"]["gnn"]
    g["hidden_dims"] = TRIAL_14_PARAMS["hidden_dims"]
    g["residual"] = TRIAL_14_PARAMS["residual"]
    g["l2_normalize"] = TRIAL_14_PARAMS["l2_normalize"]
    g["aggr"] = TRIAL_14_PARAMS["aggr"]
    g["num_neighbors"] = TRIAL_14_PARAMS["num_neighbors"]
    g["input_dropout"] = TRIAL_14_PARAMS["input_dropout"]
    g["edge_spec_set"] = TRIAL_14_PARAMS["edge_spec_set"]
    # Extended budget: give the still-rising curve real headroom past the
    # original 60-epoch/patience-10 search-mode limit now that this is a
    # single confirmatory continuation, not one of many trials.
    g["max_epochs"] = 100
    g["patience"] = 15

    processed_dir = Path(config["data"]["processed_dir"])
    X_train = pd.read_parquet(processed_dir / "train_features.parquet")
    y_train = pd.read_parquet(processed_dir / "train_labels.parquet").squeeze()
    X_val = pd.read_parquet(processed_dir / "val_features.parquet")
    y_val = pd.read_parquet(processed_dir / "val_labels.parquet").squeeze()
    X_test = pd.read_parquet(processed_dir / "test_features.parquet")
    y_test_placeholder = pd.Series([0] * len(X_test))

    # pos_weight_scale=0.5055 from the trial log, applied on top of the
    # empirical train neg/pos ratio, same as run_gnn_hpo.py's objective.
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    empirical_ratio = (neg / pos) if pos > 0 else 1.0
    g["pos_weight"] = empirical_ratio * 0.5055221171236024

    logger.info("Resuming trial 14 from checkpoint: %s", CHECKPOINT_PATH)
    logger.info("Config: %s", {k: v for k, v in g.items() if k != "pos_weight"})

    set_seed(SEED)
    trainer = GNNTrainer(config)
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        X_test=X_test, y_test=y_test_placeholder,
        checkpoint_path=CHECKPOINT_PATH,
    )

    best_val = max(history["val_pr_auc"]) if history["val_pr_auc"] else 0.0
    logger.info("=" * 60)
    logger.info("Trial 14 resume complete. Best val PR-AUC: %.4f", best_val)
    logger.info("GATE 1 (>= 0.5502): %s", "PASS" if best_val >= 0.5502 else "FAIL")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
