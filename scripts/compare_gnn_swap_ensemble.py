"""
scripts/compare_gnn_swap_ensemble.py

Exploratory diagnostic, OUTSIDE ADR-006's pre-registered gate sequence.
ADR-006 closed Rejected on 2026-09-23 (Gate 2 failed: GNN standalone test
PR-AUC 0.4630 < 0.5552). Gate 3 (4-way ensemble min-lift) is conditioned on
Gate 2 passing and was never triggered. This script does not reopen or
amend ADR-006's status regardless of its result; it exists to answer a
narrower, user-asked question Gate 3 wasn't designed for: does an ensemble
built from {GNN, XGBoost, LightGBM} (a SWAP of TFT for GNN, not a 4-way add)
outperform the deployed {XGBoost, TFT, LightGBM} 3-way ensemble, read once
on test.

Both ensembles' blend weights are found via VALIDATION PR-AUC weight search
(same simplex-grid method as run_ensemble_eval.py), then each candidate's
weights are applied to TEST probabilities exactly once. This keeps the
model-selection step (weight choice) off the held-out set, consistent with
the project's standing rule (apply_min_lift_gate's 2026-09-09 fix) even
though this script itself is not a gate.

Usage:
    python scripts/compare_gnn_swap_ensemble.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings  # noqa: E402
from src.evaluation.evaluator import ModelEvaluator  # noqa: E402
from src.models.ensemble import (  # noqa: E402
    blend,
    grid_search_simplex_weights,
    pairwise_diagnostics,
)
from src.training.train_gnn import GNNTrainer  # noqa: E402
from src.training.train_lgbm import LGBMTrainer  # noqa: E402
from src.training.train_tft import TFTTrainer  # noqa: E402
from src.training.train_xgb import XGBTrainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

_SEQUENCE_GROUP_COL = "card1"


def _calibrated(trainer, name: str, *args, **kwargs) -> np.ndarray:
    if trainer.calibrator is None:
        raise RuntimeError(f"{name} artifact has no frozen calibrator.")
    return np.asarray(trainer.predict_proba_calibrated(*args, **kwargs), dtype=float)


def _sequence_history(
    X_train: pd.DataFrame, X_prev: pd.DataFrame, config: dict
) -> pd.DataFrame:
    seq_len = int(config.get("data", {}).get("sequence_length", 10))
    combined = pd.concat([X_train, X_prev], axis=0, ignore_index=True)
    if _SEQUENCE_GROUP_COL not in combined.columns or seq_len <= 1:
        return combined
    return (
        combined.groupby(_SEQUENCE_GROUP_COL, sort=False)
        .tail(seq_len - 1)
        .reset_index(drop=True)
    )


def main() -> None:
    config = load_settings("config/config.yaml").model_dump()
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    ev = ModelEvaluator()

    logger.info("Loading splits...")
    X_train = pd.read_parquet(processed_dir / "train_features.parquet")
    X_val = pd.read_parquet(processed_dir / "val_features.parquet")
    y_val = pd.read_parquet(processed_dir / "val_labels.parquet").squeeze().to_numpy()
    X_test = pd.read_parquet(processed_dir / "test_features.parquet")
    y_test = pd.read_parquet(processed_dir / "test_labels.parquet").squeeze().to_numpy()

    serving = config.get("serving", {})
    logger.info("Loading models (xgb, tft, lgbm, gnn)...")
    xgb = XGBTrainer.load(serving.get("model_path", "models/xgb_model.pkl"))
    tft = TFTTrainer.load(
        serving.get("tft_model_path", "models/tft_model.pt"), config=config
    )
    lgbm = LGBMTrainer.load(serving.get("lgbm_model_path", "models/lgbm_model.pkl"))
    gnn = GNNTrainer.load("models/gnn_model", config=config)

    logger.info("Scoring validation split...")
    val_hist = _sequence_history(X_train, X_train.iloc[:0], config)
    val_prob = {
        "xgb": _calibrated(xgb, "XGBoost", X_val),
        "tft": _calibrated(tft, "TFT", X_val, history_X=val_hist),
        "lgbm": _calibrated(lgbm, "LightGBM", X_val),
        "gnn": _calibrated(gnn, "GNN", X_val),
    }

    logger.info("Scoring test split...")
    test_hist = _sequence_history(X_train, X_val, config)
    test_prob = {
        "xgb": _calibrated(xgb, "XGBoost", X_test),
        "tft": _calibrated(tft, "TFT", X_test, history_X=test_hist),
        "lgbm": _calibrated(lgbm, "LightGBM", X_test),
        "gnn": _calibrated(gnn, "GNN", X_test),
    }

    corr = pairwise_diagnostics(val_prob)
    logger.info("Pairwise val-probability correlation (all 4 models): %s", corr)

    ens_cfg = config.get("ensemble", {})
    step = float(ens_cfg.get("grid_step", 0.02))
    refine = float(ens_cfg.get("refine_step", 0.002))

    # Deployed candidate: xgb + tft + lgbm (unchanged from production).
    deployed_val = {k: val_prob[k] for k in ("xgb", "tft", "lgbm")}
    deployed_test = {k: test_prob[k] for k in ("xgb", "tft", "lgbm")}
    w_deployed, val_pr_auc_deployed = grid_search_simplex_weights(
        y_val, deployed_val, step, refine
    )
    test_pr_auc_deployed = ev.compute_pr_auc(y_test, blend(deployed_test, w_deployed))

    # Swap candidate: xgb + lgbm + gnn (TFT replaced by GNN, not added to it).
    swap_val = {k: val_prob[k] for k in ("xgb", "lgbm", "gnn")}
    swap_test = {k: test_prob[k] for k in ("xgb", "lgbm", "gnn")}
    w_swap, val_pr_auc_swap = grid_search_simplex_weights(y_val, swap_val, step, refine)
    test_pr_auc_swap = ev.compute_pr_auc(y_test, blend(swap_test, w_swap))

    # No-TFT candidate: xgb + lgbm only (TFT dropped, nothing added in its
    # place). Not the same as models/ensemble.json's existing "no_tft"
    # fallback mode, which is XGB-alone (weight 1.0) for serving resilience
    # when TFT scoring fails at inference time — this is a from-scratch
    # weight search over {xgb, lgbm} to see whether TFT's ~0.008 deployed
    # weight is actively helping, hurting, or irrelevant to the blend.
    no_tft_val = {k: val_prob[k] for k in ("xgb", "lgbm")}
    no_tft_test = {k: test_prob[k] for k in ("xgb", "lgbm")}
    w_no_tft, val_pr_auc_no_tft = grid_search_simplex_weights(
        y_val, no_tft_val, step, refine
    )
    test_pr_auc_no_tft = ev.compute_pr_auc(y_test, blend(no_tft_test, w_no_tft))

    logger.info("=" * 70)
    logger.info("Exploratory diagnostic — NOT an ADR-006 gate, does not amend its status")
    logger.info("-" * 70)
    logger.info(
        "Deployed 3-way (xgb+tft+lgbm)  weights=%s  val=%.4f  test=%.4f",
        {k: round(v, 4) for k, v in w_deployed.items()},
        val_pr_auc_deployed,
        test_pr_auc_deployed,
    )
    logger.info(
        "Swap candidate (xgb+lgbm+gnn)  weights=%s  val=%.4f  test=%.4f",
        {k: round(v, 4) for k, v in w_swap.items()},
        val_pr_auc_swap,
        test_pr_auc_swap,
    )
    logger.info(
        "No-TFT candidate (xgb+lgbm)    weights=%s  val=%.4f  test=%.4f",
        {k: round(v, 4) for k, v in w_no_tft.items()},
        val_pr_auc_no_tft,
        test_pr_auc_no_tft,
    )
    logger.info(
        "Test PR-AUC delta (swap - deployed): %.4f",
        test_pr_auc_swap - test_pr_auc_deployed,
    )
    logger.info(
        "Test PR-AUC delta (no_tft - deployed): %.4f",
        test_pr_auc_no_tft - test_pr_auc_deployed,
    )
    logger.info(
        "Val PR-AUC delta (no_tft - deployed): %.4f",
        val_pr_auc_no_tft - val_pr_auc_deployed,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
