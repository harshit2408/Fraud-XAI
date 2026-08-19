import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.training.train_tft import TFTTrainer
from src.evaluation.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model():
    config = load_settings("config/config.yaml").model_dump()
    processed_dir = Path(config["data"]["processed_dir"])
    
    logger.info("Loading parquet data...")
    X_train = pd.read_parquet(processed_dir / "train_features.parquet")
    y_train = pd.read_parquet(processed_dir / "train_labels.parquet").squeeze()
    X_val = pd.read_parquet(processed_dir / "val_features.parquet")
    y_val = pd.read_parquet(processed_dir / "val_labels.parquet").squeeze()
    X_test = pd.read_parquet(processed_dir / "test_features.parquet")
    y_test = pd.read_parquet(processed_dir / "test_labels.parquet").squeeze()

    model_path = config.get("serving", {}).get("tft_model_path", "models/tft_model.pt")
    logger.info(f"Loading TFT model from {model_path}...")
    trainer = TFTTrainer.load(model_path, config=config)

    evaluator = ModelEvaluator()

    logger.info("Evaluating Train...")
    y_prob_train = trainer.predict_proba(X_train)
    pr_auc_train = evaluator.compute_pr_auc(y_train.values, y_prob_train)

    logger.info("Evaluating Val...")
    y_prob_val = trainer.predict_proba(X_val, history_X=X_train)
    pr_auc_val = evaluator.compute_pr_auc(y_val.values, y_prob_val)

    logger.info("Evaluating Test...")
    y_prob_test = trainer.predict_proba(
        X_test, history_X=pd.concat([X_train, X_val], axis=0, ignore_index=True)
    )
    pr_auc_test = evaluator.compute_pr_auc(y_test.values, y_prob_test)

    logger.info("=" * 60)
    logger.info("TFT CORRECTED OVERFITTING ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"  Train PR-AUC:  {pr_auc_train:.4f}")
    logger.info(f"  Val PR-AUC:    {pr_auc_val:.4f}")
    logger.info(f"  Test PR-AUC:   {pr_auc_test:.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    evaluate_model()
