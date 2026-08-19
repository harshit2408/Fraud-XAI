"""
src/data/data_loader.py

Loads and validates the raw IEEE-CIS Fraud Detection dataset.

Responsibilities:
  1. Load train_transaction.csv + train_identity.csv from data/raw/
  2. Merge on TransactionID (left join — ~75% of transactions have no identity data)
  3. Validate schema (required columns, target binary, no future leakage)
  4. Sort by TransactionDT — CRITICAL prerequisite for time-based split

The identity join is a left join by design: it is expected that ~75% of
transactions have no corresponding identity record. NaN from missing
identity rows is handled in feature_engineering.py.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum required columns that must exist in the merged DataFrame
REQUIRED_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "isFraud",
    "card1",
    "ProductCD",
]

# Columns that would constitute data leakage if present at inference time
LEAKAGE_COLUMNS: list[str] = []  # None in IEEE-CIS, but check is here for safety


class DataLoader:
    """
    Loads, merges, validates, and temporally sorts the raw IEEE-CIS dataset.

    Args:
        config: Loaded config.yaml dict. Reads paths from config['data'].
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        data_cfg = config["data"]
        self.raw_dir = Path(data_cfg["raw_dir"])
        self.train_file = data_cfg["train_file"]
        self.identity_file = data_cfg["identity_file"]
        self.target_col = data_cfg["target_col"]
        self.temporal_col = data_cfg["temporal_col"]

    def load_raw(self) -> pd.DataFrame:
        """
        Load and merge train_transaction.csv + train_identity.csv.

        Merge strategy: LEFT JOIN on TransactionID.
        ~75% of transactions have no identity record — NaN is expected and correct.

        Returns:
            Merged DataFrame with all transaction + identity columns.

        Raises:
            FileNotFoundError: If CSV files are not present in data/raw/.
        """
        tx_path = self.raw_dir / self.train_file
        id_path = self.raw_dir / self.identity_file

        if not tx_path.exists():
            raise FileNotFoundError(
                f"Transaction file not found: {tx_path}. "
                "Run: python src/data/download_data.py"
            )
        if not id_path.exists():
            raise FileNotFoundError(
                f"Identity file not found: {id_path}. "
                "Run: python src/data/download_data.py"
            )

        logger.info(f"Loading transaction data from {tx_path}")
        try:
            import pyarrow as pa
            import pyarrow.csv as pcsv

            # Cast V-features to float32 in Arrow schema before converting to pandas.
            # PyArrow uses columnar memory layout — never allocates a 2D numpy array,
            # so it avoids the ~1.6 GiB consolidation OOM that pandas' C parser triggers.
            v_col_types = {f"V{i}": pa.float32() for i in range(1, 340)}
            convert_opts = pcsv.ConvertOptions(column_types=v_col_types)
            read_opts = pcsv.ReadOptions(use_threads=True)

            arrow_table = pcsv.read_csv(
                str(tx_path),
                read_options=read_opts,
                convert_options=convert_opts,
            )
            df_tx = arrow_table.to_pandas(self_destruct=True)  # free arrow table ASAP
            del arrow_table

        except Exception as e:
            logger.warning(f"PyArrow CSV read failed ({e}), falling back to pandas chunked read")
            # Fallback: read in chunks and concat — slower but always works
            chunks = []
            v_dtypes = {f"V{i}": "float32" for i in range(1, 340)}
            for chunk in pd.read_csv(tx_path, dtype=v_dtypes, chunksize=100_000):
                chunks.append(chunk)
            df_tx = pd.concat(chunks, ignore_index=True)
            del chunks

        logger.info(f"Transaction data: {df_tx.shape[0]:,} rows × {df_tx.shape[1]} cols")

        logger.info(f"Loading identity data from {id_path}")
        df_id = pd.read_csv(id_path, low_memory=False)


        logger.info(f"Identity data: {df_id.shape[0]:,} rows × {df_id.shape[1]} cols")

        # Left join — identity data is optional per transaction
        df = df_tx.merge(df_id, on="TransactionID", how="left")
        identity_match_rate = df_id.shape[0] / df_tx.shape[0] * 100
        logger.info(
            f"Merged dataset: {df.shape[0]:,} rows × {df.shape[1]} cols "
            f"(identity match rate: {identity_match_rate:.1f}%)"
        )

        # Log fraud rate
        fraud_rate = df[self.target_col].mean() * 100
        logger.info(f"Fraud rate in full dataset: {fraud_rate:.2f}%")

        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        """
        Assert expected columns exist, target col is binary, no leakage columns.

        Args:
            df: Merged DataFrame to validate.

        Raises:
            AssertionError: If any validation check fails.
        """
        # Check required columns
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        assert not missing_cols, (
            f"Required columns missing from dataset: {missing_cols}"
        )

        # Check target is binary
        unique_targets = set(df[self.target_col].dropna().unique())
        assert unique_targets.issubset({0, 1}), (
            f"Target column '{self.target_col}' must be binary (0/1), "
            f"found: {unique_targets}"
        )

        # Check no obvious leakage columns
        leakage_found = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        assert not leakage_found, (
            f"Potential data leakage columns found: {leakage_found}"
        )

        logger.info("Schema validation passed ✓")

    def sort_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort DataFrame by TransactionDT ascending.

        CRITICAL: Must be called before time_based_split().
        Sorting ensures that the first 80% of rows are the earliest transactions.

        Args:
            df: Merged DataFrame.

        Returns:
            Temporally sorted DataFrame (index reset).
        """
        if self.temporal_col not in df.columns:
            raise ValueError(
                f"Temporal column '{self.temporal_col}' not found. "
                "Cannot sort for time-based split."
            )

        df_sorted = df.sort_values(self.temporal_col, ascending=True, ignore_index=True)
        logger.info(
            f"Sorted by {self.temporal_col}: "
            f"range [{df_sorted[self.temporal_col].min():.0f}, "
            f"{df_sorted[self.temporal_col].max():.0f}]"
        )
        return df_sorted
