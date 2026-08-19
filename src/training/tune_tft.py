"""
src/training/tune_tft.py

Extensive hyperparameter tuning for the Temporal Fusion Transformer using Optuna.
Implements data subsampling (by card ID) and early pruning to make tuning feasible.
Uses SQLite storage for pausing/resuming tuning over long periods.
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import mlflow
import optuna
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.data.sequence_builder import SequenceBuilder
from src.training.train_tft import TFTTrainer
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def subset_by_card(X: pd.DataFrame, y: pd.Series, fraction: float, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Subsets the dataset by randomly sampling a fraction of unique card1 IDs."""
    if fraction >= 1.0:
        return X, y
    
    unique_cards = X["card1"].unique()
    sampled_cards = pd.Series(unique_cards).sample(frac=fraction, random_state=seed)
    
    mask = X["card1"].isin(sampled_cards)
    logger.info(f"Subsampled {fraction*100:.0f}% of cards: {len(sampled_cards):,} / {len(unique_cards):,} cards")
    logger.info(f"  Rows reduced from {len(X):,} to {mask.sum():,}")
    
    return X[mask].copy(), y[mask].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extensive TFT Hyperparameter Tuning")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of cards to use for tuning (e.g., 0.2)")
    parser.add_argument("--db", type=str, default="sqlite:///tft_tuning.db", help="Optuna storage DB")
    parser.add_argument("--study_name", type=str, default="tft_extensive_tuning")
    args = parser.parse_args()

    config = load_settings(args.config).model_dump()
    seed = set_seed(config.get("project", {}).get("random_seed", 42))
    data_cfg = config["data"]
    processed_dir = Path(data_cfg["processed_dir"])

    logger.info("Loading parquet data...")
    X_train = pd.read_parquet(processed_dir / "train_features.parquet")
    y_train = pd.read_parquet(processed_dir / "train_labels.parquet").squeeze()
    X_val = pd.read_parquet(processed_dir / "val_features.parquet")
    y_val = pd.read_parquet(processed_dir / "val_labels.parquet").squeeze()

    # Subsample to speed up tuning
    if args.sample_fraction < 1.0:
        logger.info("Subsampling training set...")
        X_train, y_train = subset_by_card(X_train, y_train, args.sample_fraction, seed=seed)
        logger.info("Subsampling validation set...")
        X_val, y_val = subset_by_card(X_val, y_val, args.sample_fraction, seed=seed)

    # Pre-calculate feature dimensions to avoid repeating it
    seq_builder = SequenceBuilder(sequence_length=config["data"].get("sequence_length", 10))
    trial_seq = seq_builder.build_sequences(X_train.head(100), y_train.head(100))
    num_numeric = trial_seq["sequences"].shape[2]
    num_static = trial_seq["static"].shape[1]
    del trial_seq
    gc.collect()

    def objective(trial: optuna.Trial) -> float:
        # Define hyperparameter search space
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4])
        num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 2)
        hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [8, 16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.3, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256])

        # Prepare config for this trial. load_settings() is lru_cache'd, so
        # this re-validates nothing; model_dump() still returns a fresh dict
        # each call, which is required since it's mutated per-trial below.
        trial_config = load_settings(args.config).model_dump()
        trial_config["model"]["tft"]["hidden_size"] = hidden_size
        trial_config["model"]["tft"]["attention_head_size"] = attention_head_size
        trial_config["model"]["tft"]["num_lstm_layers"] = num_lstm_layers
        trial_config["model"]["tft"]["hidden_continuous_size"] = hidden_continuous_size
        trial_config["model"]["tft"]["dropout"] = dropout
        trial_config["model"]["tft"]["learning_rate"] = learning_rate
        trial_config["model"]["tft"]["batch_size"] = batch_size
        
        # Keep epochs moderate for tuning
        trial_config["model"]["tft"]["max_epochs"] = 15
        trial_config["model"]["tft"]["patience"] = 4

        trainer = TFTTrainer(trial_config)
        trainer.build_model(num_numeric, num_static)
        
        try:
            history = trainer.train(X_train, y_train, X_val, y_val, trial=trial)
            best_val_pr_auc = max(history["val_pr_auc"])
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial failed due to error: {e}")
            raise optuna.TrialPruned()
            
        return best_val_pr_auc

    logger.info(f"Connecting to Optuna DB: {args.db}")
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.db,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3, n_startup_trials=5),
    )
    
    logger.info(f"Starting Optuna study for {args.trials} trials...")
    
    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "fraud_detection"))
    
    with mlflow.start_run(run_name="tft_extensive_tuning"):
        mlflow.log_param("random_seed", seed)
        study.optimize(objective, n_trials=args.trials, catch=(Exception,))
        
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_tft_pr_auc", study.best_value)

    logger.info("="*60)
    logger.info("Tuning complete/paused!")
    logger.info(f"Best trial PR-AUC: {study.best_value:.4f}")
    logger.info("Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info("Update config/config.yaml with these parameters to reproduce.")
    logger.info("="*60)


if __name__ == "__main__":
    main()
