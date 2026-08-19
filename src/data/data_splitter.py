"""
src/data/data_splitter.py

Time-based train/validation/test split for temporal financial datasets.

CRITICAL: Never use random shuffling. Always sort by temporal_col first.
This prevents data leakage from future fraud patterns into the training set.

Usage:
    X_train, X_test, y_train, y_test = time_based_split(df, "TransactionDT", 0.80)
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split_3way(df, "TransactionDT", 0.70, 0.10)
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def time_based_split(
    df: pd.DataFrame,
    temporal_col: str,
    ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sort by temporal_col, take first `ratio` fraction as train, remainder as test.

    MUST NOT shuffle. Preserves temporal ordering to prevent leakage.

    Args:
        df: Full merged and feature-engineered DataFrame containing target column.
        temporal_col: Column name used for temporal ordering (e.g. 'TransactionDT').
        ratio: Fraction of rows for training set (e.g. 0.80 for 80/20 split).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) where:
            - X_* are DataFrames of all feature columns (target excluded)
            - y_* are Series of the target column ('isFraud')

    Raises:
        ValueError: If temporal_col not in df or ratio not in (0, 1).
    """
    if temporal_col not in df.columns:
        raise ValueError(
            f"temporal_col '{temporal_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")

    target_col = "isFraud"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Sort by temporal column — CRITICAL: no shuffle
    df_sorted = df.sort_values(temporal_col, ascending=True).reset_index(drop=True)

    split_idx = int(len(df_sorted) * ratio)

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    # Log split statistics
    train_fraud_rate = y_train.mean() * 100
    test_fraud_rate = y_test.mean() * 100

    logger.info(
        f"Time-based split: train={len(X_train):,} rows "
        f"({train_fraud_rate:.2f}% fraud), "
        f"test={len(X_test):,} rows ({test_fraud_rate:.2f}% fraud)"
    )
    logger.info(
        f"Temporal boundary: train ends at DT={train_df[temporal_col].max():.0f}, "
        f"test starts at DT={test_df[temporal_col].min():.0f}"
    )

    return X_train, X_test, y_train, y_test


def time_based_split_3way(
    df: pd.DataFrame,
    temporal_col: str,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Sort by temporal_col, split into train / validation / test.

    The validation set sits between train and test temporally.
    Used for early stopping without leaking test performance.

    MUST NOT shuffle. Preserves temporal ordering to prevent leakage.

    Args:
        df: Full merged and feature-engineered DataFrame containing target column.
        temporal_col: Column name used for temporal ordering (e.g. 'TransactionDT').
        train_ratio: Fraction of rows for training set (e.g. 0.70).
        val_ratio: Fraction of rows for validation set (e.g. 0.10).
            Remaining (1 - train_ratio - val_ratio) becomes test set.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).

    Raises:
        ValueError: If ratios are invalid.
    """
    if temporal_col not in df.columns:
        raise ValueError(
            f"temporal_col '{temporal_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    test_ratio = 1.0 - train_ratio - val_ratio
    if not (0.0 < train_ratio < 1.0) or not (0.0 < val_ratio < 1.0) or test_ratio <= 0:
        raise ValueError(
            f"Invalid ratios: train={train_ratio}, val={val_ratio}, "
            f"test={test_ratio:.2f}. All must be positive and sum to 1.0."
        )

    target_col = "isFraud"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Sort by temporal column — CRITICAL: no shuffle
    df_sorted = df.sort_values(temporal_col, ascending=True).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    X_train = train_df.drop(columns=[target_col])
    X_val = val_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    y_test = test_df[target_col]

    # Log split statistics
    for name, X_s, y_s in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        fraud_rate = y_s.mean() * 100
        logger.info(f"  {name}: {len(X_s):,} rows ({fraud_rate:.2f}% fraud)")

    logger.info(
        f"Temporal boundaries: "
        f"train ends DT={train_df[temporal_col].max():.0f}, "
        f"val ends DT={val_df[temporal_col].max():.0f}, "
        f"test starts DT={test_df[temporal_col].min():.0f}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
