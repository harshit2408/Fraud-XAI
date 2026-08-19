"""
src/data/preprocess.py

End-to-end preprocessing pipeline orchestrator.

Runs the full Phase 1 pipeline:
  1. Load raw CSVs + merge (DataLoader)
  2. Validate schema
  3. Null-count features over the raw frame (before PCA erases V missingness)
  4. Assign the time-based 70/10/20 train/val/test split — BEFORE any transformer
     is fitted, so no fit ever observes a val or test row
  5. Fit V-feature PCA on the train rows only, transform every row
  6. Sort temporally, then derive the strictly backward-looking features
     (temporal, amount, email, D/C columns, card aggregates, velocity, address,
     device, interactions) over the full ordered frame
  7. Materialise the three splits, then fit the remaining stateful transformers
     (card hash, target encoding, imputation, categorical encoding) on train and
     apply them to val/test with fit=False
  8. Save parquet outputs to data/processed/
  9. Serialize fitted transformers to data/processed/transformers/

Leakage contract (see docs/IMPLEMENTATION_PLAN.md Phase A): every fitted
transformer observes train rows only. Features derived in step 6 run over the
full frame deliberately — they read strictly past rows relative to each
transaction, which is exactly what is available in production, and computing
them per split would instead destroy card history at each split boundary.

Usage:
    python src/data/preprocess.py
    python src/data/preprocess.py --config config/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.data.data_loader import DataLoader
from src.data.data_splitter import time_based_split_3way
from src.data.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Positional row key added after load so split membership survives the row
# reordering that sorting and the card-aggregate features perform.
SPLIT_ID_COL = "_split_row_id"

TARGET_COL = "isFraud"

# Splits must be processed in this order: target-encoding state carries
# forward train → val → test, so val's fit must run after train's and
# before test's. A tuple (not a set) makes that order load-bearing and
# explicit rather than implied by dict/loop iteration.
SPLIT_ORDER: Tuple[str, ...] = ("train", "val", "test")

# Splits whose entity history is folded into carried state for the next
# split to consume. Test is deliberately excluded — nothing consumes its
# state within this pipeline run.
SPLITS_THAT_UPDATE_STATE = frozenset({"val"})


def _assign_split_ids(
    df: pd.DataFrame, temporal_col: str, train_ratio: float, val_ratio: float
) -> Dict[str, np.ndarray]:
    """
    Decide train/val/test membership before any transformer is fitted.

    The split runs against a three-column projection rather than the full
    frame: boundary placement only needs the temporal column and the target,
    and projecting avoids copying the ~434-column raw frame just to learn
    where the cut points are. Membership is returned as SPLIT_ID_COL values so
    it can be reapplied after later steps reorder rows.
    """
    projection = df[[temporal_col, TARGET_COL, SPLIT_ID_COL]]

    id_train, id_val, id_test, _, _, _ = time_based_split_3way(
        df=projection,
        temporal_col=temporal_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    split_ids = {
        "train": id_train[SPLIT_ID_COL].to_numpy(),
        "val": id_val[SPLIT_ID_COL].to_numpy(),
        "test": id_test[SPLIT_ID_COL].to_numpy(),
    }

    # time_based_split_3way is an external dependency; if it ever changed to
    # drop or duplicate a boundary row, np.isin downstream would silently
    # lose or double-count rows instead of raising. Fail loudly here instead.
    all_ids = np.concatenate([split_ids["train"], split_ids["val"], split_ids["test"]])
    if len(all_ids) != len(df):
        raise ValueError(
            f"Split assignment lost or gained rows: got {len(all_ids)} ids "
            f"across train/val/test but the frame has {len(df)} rows."
        )
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError(
            "Split assignment produced duplicate row ids across train/val/test — "
            "the same row would be fitted and evaluated on."
        )

    return split_ids


def _derive_causal_features(fe: FeatureEngineer, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the feature groups that hold no fitted state.

    Every transform here is either row-wise or strictly backward-looking
    (cumulative counts and time deltas built from shift/cumsum), so running
    them across the whole ordered frame reads only a transaction's own past.
    Splitting first would zero out each card's history at the split boundary
    and create a train/serve mismatch that does not exist in production.
    """
    steps = [
        ("temporal features", fe.create_temporal_features),
        ("amount features", fe.create_amount_features),
        ("email domain features", fe.create_email_features),
        ("D-column summary features", fe.create_d_column_features),
        ("C-column summary features", fe.create_c_column_features),
        ("card aggregates", fe.create_card_aggregates),
        ("velocity features", fe.create_velocity_features),
        ("address features", fe.create_address_features),
        ("device features", fe.create_device_features),
        ("interactions", fe.create_interactions),
    ]

    for description, step in steps:
        logger.info(f"Creating {description}...")
        df = step(df)

    return df


def _apply_stateful_transforms(
    fe: FeatureEngineer, df: pd.DataFrame, split_name: str, fit: bool
) -> pd.DataFrame:
    """
    Apply the transformers that carry fitted state to one split.

    fit=True is passed for train only. For val and test the same fitted
    encoders, prior, PCA and fill values are reused unchanged, which is the
    property the Phase A leakage tests assert.

    Order matters: imputation fills categorical NaNs with the MISSING sentinel
    that encode_categoricals expects, and target encoding must read the raw
    category strings before they are replaced with codes.
    """
    logger.info(f"[{split_name}] card hash features (fit={fit})...")
    df = fe.create_card_hash_features(df, fit=fit)

    # val folds its history into the carried state so test continues from
    # train+val, reproducing what a single pass over the full frame would give.
    update_state = split_name in SPLITS_THAT_UPDATE_STATE
    logger.info(f"[{split_name}] expanding target encodings (fit={fit})...")
    df = fe.create_target_encoding(df, fit=fit, update_state=update_state)

    logger.info(f"[{split_name}] missing value handling (fit={fit})...")
    df = fe.handle_missing_values(df, fit=fit)

    logger.info(f"[{split_name}] categorical encoding (fit={fit})...")
    df = fe.encode_categoricals(df, fit=fit)

    return df


def run_pipeline(config: dict) -> None:
    """
    Execute the full preprocessing pipeline.

    All paths are read from config — nothing is hardcoded.
    """
    data_cfg = config["data"]
    processed_dir = Path(data_cfg["processed_dir"])
    transformer_dir = processed_dir / "transformers"
    processed_dir.mkdir(parents=True, exist_ok=True)
    transformer_dir.mkdir(parents=True, exist_ok=True)

    temporal_col = data_cfg["temporal_col"]
    train_ratio = data_cfg.get("train_split_ratio", 0.70)
    val_ratio = data_cfg.get("val_split_ratio", 0.10)

    # ── Step 1: Load raw data ────────────────────────────────────────────────
    loader = DataLoader(config)
    df = loader.load_raw()
    loader.validate_schema(df)

    fe = FeatureEngineer()

    # ── Step 2: Null-count features over the RAW frame ───────────────────────
    # Must precede PCA, which converts V-column NaNs to 0.0 and would otherwise
    # hide all V missingness from the count.
    logger.info("Creating null-count meta features (raw frame)...")
    df = fe.create_null_count_features(df)
    df[SPLIT_ID_COL] = np.arange(len(df), dtype=np.int64)

    # ── Step 3: Decide the split BEFORE fitting anything ─────────────────────
    logger.info("Assigning train/val/test membership (time-based, no shuffle)...")
    split_ids = _assign_split_ids(df, temporal_col, train_ratio, val_ratio)
    train_mask = np.isin(df[SPLIT_ID_COL].to_numpy(), split_ids["train"])

    # ── Step 4: V-feature PCA — fitted on train rows, applied to all ─────────
    # Running the reduction before the sort keeps the memory optimisation:
    # 434 → ~125 columns means sort_values allocates ~220 MB, not ~764 MB.
    n_pca = config["features"].get("v_features_pca_components", 30)
    logger.info(f"Reducing V-features to {n_pca} PCA components (fit on train rows only)...")
    df = fe.reduce_v_features(df, fit=True, n_components=n_pca, fit_rows=train_mask)
    logger.info(f"DataFrame after PCA reduction: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── Step 5: Sort temporally, then derive stateless/causal features ───────
    df = loader.sort_temporal(df)
    df = _derive_causal_features(fe, df)

    # Drop columns that must not enter the model — EXCEPT TransactionDT, which
    # the splits still need. It is removed from X_* at the end.
    drop_cols = [
        c for c in config["features"].get("drop_cols", [])
        if c in df.columns and c != temporal_col
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"Dropped columns: {drop_cols}")

    # ── Step 6: Materialise splits, fit on train, transform val/test ─────────
    row_ids = df[SPLIT_ID_COL].to_numpy()
    # Boolean masks preserve the frame's temporal ordering within each split,
    # which the expanding target encoding depends on.
    split_frames = {
        name: df[np.isin(row_ids, ids)].reset_index(drop=True)
        for name, ids in split_ids.items()
    }
    del df

    # Sequential and order-dependent: train fits the transformers, and the
    # target encoding carries entity history forward per SPLIT_ORDER. Iterating
    # a named tuple (rather than a dict or set) makes that order load-bearing
    # and visible, instead of resting on incidental iteration order.
    transformed = {}
    for name in SPLIT_ORDER:
        transformed[name] = _apply_stateful_transforms(
            fe, split_frames.pop(name), name, fit=(name == "train")
        )

    non_feature_cols = [SPLIT_ID_COL, temporal_col, TARGET_COL]
    features = {
        name: frame.drop(columns=[c for c in non_feature_cols if c in frame.columns])
        for name, frame in transformed.items()
    }
    labels = {name: frame[TARGET_COL] for name, frame in transformed.items()}

    X_train, X_val, X_test = features["train"], features["val"], features["test"]
    y_train, y_val, y_test = labels["train"], labels["val"], labels["test"]

    # ── Step 7: Save parquet outputs ──────────────────────────────────────────
    logger.info(f"Saving processed data to {processed_dir}/")

    X_train.to_parquet(processed_dir / "train_features.parquet", index=False)
    X_val.to_parquet(processed_dir / "val_features.parquet", index=False)
    X_test.to_parquet(processed_dir / "test_features.parquet", index=False)
    y_train.to_frame().to_parquet(processed_dir / "train_labels.parquet", index=False)
    y_val.to_frame().to_parquet(processed_dir / "val_labels.parquet", index=False)
    y_test.to_frame().to_parquet(processed_dir / "test_labels.parquet", index=False)

    logger.info("Parquet files saved:")
    for fname in ["train_features.parquet", "val_features.parquet", "test_features.parquet",
                  "train_labels.parquet", "val_labels.parquet", "test_labels.parquet"]:
        fpath = processed_dir / fname
        size_mb = fpath.stat().st_size / (1024 * 1024)
        logger.info(f"  {fname}: {size_mb:.1f}MB")

    # ── Step 8: Save fitted transformers ─────────────────────────────────────
    fe.save_transformers(str(transformer_dir))
    logger.info(f"Transformers saved to {transformer_dir}/")

    logger.info("✓ Preprocessing pipeline complete.")
    logger.info(f"  Train features: {X_train.shape}")
    logger.info(f"  Val features:   {X_val.shape}")
    logger.info(f"  Test features:  {X_test.shape}")
    logger.info(f"  Feature columns: {len(X_train.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fraud detection preprocessing pipeline.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    logger.info(f"Loading config from: {args.config}")
    config = load_settings(args.config).model_dump()
    run_pipeline(config)


if __name__ == "__main__":
    main()
