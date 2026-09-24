"""
src/data/sequence_builder.py

Converts flat transaction rows into temporal sequences for the TFT model.

For each transaction, we look back N transactions on the same card (card1)
to provide historical context. This is the core data engineering step that
enables sequence-aware fraud detection.

CRITICAL: All lookback is strictly causal — a transaction's sequence only
contains transactions that occurred BEFORE it in time. No look-ahead leakage.

Usage:
    builder = SequenceBuilder(sequence_length=10)
    dataset = builder.create_dataset(df_features, df_labels)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """
    Builds temporal sequences from flat transaction data for the TFT model.

    Groups transactions by card1 (proxy for card number), creates sliding
    windows of length N, and produces PyTorch-compatible datasets.

    Args:
        sequence_length: Number of historical transactions to include per sequence.
        group_col: Column that identifies "same account" (card1 is proxy for card number).
        temporal_col: Column used for temporal ordering.
    """

    # Static columns that are label-encoded multi-class categoricals (Phase B5).
    # Binary flags like P_email_is_free/R_email_is_free stay continuous — an
    # embedding table adds no value for a 2-valued feature.
    CATEGORICAL_STATIC_COLS = {"ProductCD", "card4", "card6", "DeviceType"}

    def __init__(
        self,
        sequence_length: int = 10,
        group_col: str = "card1",
        temporal_col: str = "TransactionDT",
    ) -> None:
        self.sequence_length = sequence_length
        self.group_col = group_col
        self.temporal_col = temporal_col

        # Feature column lists — set during build
        self._numeric_features: List[str] = []
        self._static_features: List[str] = []
        self._feature_dim: int = 0
        self._static_dim: int = 0

        # Phase B4 — scaler is opt-in (fit_scaler=True) and persists across
        # calls so val/test/inference transform with the train-fitted scaler.
        self.scaler: Optional[QuantileTransformer] = None

        # Phase B5 — cardinalities of static categorical columns, computed
        # once (on the first build that sees them) and reused thereafter so
        # embedding table sizes stay consistent across train/val/test.
        self._static_cardinalities: Optional[Dict[str, int]] = None

    @property
    def feature_dim(self) -> int:
        """Dimension of time-varying numeric features."""
        return self._feature_dim

    @property
    def static_dim(self) -> int:
        """Dimension of static (card-level) features."""
        return self._static_dim

    @property
    def numeric_features(self) -> List[str]:
        """List of time-varying numeric feature names."""
        return self._numeric_features

    @property
    def static_features(self) -> List[str]:
        """List of static feature names."""
        return self._static_features

    @property
    def static_categorical_indices(self) -> List[int]:
        """Indices into `static_features` that are label-encoded categoricals."""
        return [
            i for i, c in enumerate(self._static_features)
            if c in self.CATEGORICAL_STATIC_COLS
        ]

    @property
    def static_cardinalities(self) -> List[int]:
        """Cardinality (max code + 1) for each column in `static_categorical_indices`."""
        if not self._static_cardinalities:
            return []
        return [
            self._static_cardinalities[self._static_features[i]]
            for i in self.static_categorical_indices
        ]

    def fit_scaler_on(self, X: pd.DataFrame) -> None:
        """Fit the QuantileTransformer on `X` alone and store it, without
        building sequences or touching any other split.

        2026-09-09 (4-agent ML review, ecc:code-reviewer): `build_sequences`
        fits `self.scaler` on whatever frame it is given. Phase B6's
        `_build_sequences_for_splits` (train_tft.py) concatenates train+val
        into one frame *for sequence continuity* — a val card's first
        transaction correctly retains real train history — but that combined
        frame was also what `fit_scaler=True` saw, so the scaler's quantile
        boundaries were fit on train+val rather than train alone. Val's
        distribution (temporally later) leaked into the transform val is
        then scored through, inflating val PR-AUC and biasing every
        selection that reads it (early stopping, the ensemble weight
        search). Call this on the train-only frame first, then build the
        train+val combined sequences with `fit_scaler=False` so the already
        -fitted scaler is reused (transform-only) for both splits.
        """
        numeric_cols, _ = self._identify_features(X)
        numeric_data = X[numeric_cols].values.astype(np.float32)
        n_quantiles = min(1000, max(len(numeric_data), 10))
        self.scaler = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            random_state=0,
        )
        self.scaler.fit(numeric_data)

    def _identify_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify time-varying numeric and static categorical features.

        Static features are card-level attributes that don't change within
        a card's transaction sequence (card4, card6, ProductCD).

        Time-varying features are everything else (amount, temporal, PCA, etc.)

        Returns:
            Tuple of (time_varying_feature_names, static_feature_names)
        """
        # Static categoricals: attributes of the card/account, not the transaction
        static_candidates = [
            "ProductCD", "card4", "card6",
            "P_email_is_free", "R_email_is_free",
            "DeviceType",
        ]
        static_cols = [c for c in static_candidates if c in df.columns]

        # Columns to exclude from time-varying features.
        # CRITICAL: "__target__" is the internal column build_sequences() adds
        # when `y` is provided — without excluding it here, the label itself
        # was silently fed into the model as a numeric feature whenever `y`
        # was passed (a real label-leakage bug, distinct from the literal
        # "isFraud" column name which is never actually present at this point
        # since it's dropped from X upstream).
        exclude_cols = {
            "isFraud", "__target__", self.group_col,
            # These are IDs or raw strings, not features
            "TransactionID", "TransactionDT",
        }
        exclude_cols.update(static_cols)

        # PRD Phase 9 P9-2: the UID (card1_addr1_D1n) aggregate features are
        # added for the two GBDTs (XGBoost, LightGBM) only. The TFT's sequence
        # grouping is already a per-card1 client mechanism (PRD §10 9.2: "a
        # separate, already-present mechanism — do not assume it needs the
        # richer UID grouping before measuring"), and its calibrator is frozen
        # against the pre-P9-2 feature width. Excluding uid_* here keeps the
        # TFT's input dimension stable so its existing artifact stays valid
        # while the GBDTs pick the features up.
        uid_cols = [c for c in df.columns if c.startswith("uid_")]

        # PRD Phase 9 P9-4: same treatment for the multi-window RFM/velocity
        # aggregates and the dist1 gradient features. The PRD scopes 9.4 to an
        # "XGBoost + LightGBM retrain", and the TFT already models per-card
        # temporal structure through its own sequence encoder — feeding it
        # hand-rolled trailing-window counts over the same history would be
        # redundant, and widening its input would invalidate the frozen
        # calibrator for no measured gain.
        p9_4_cols = [
            c
            for c in df.columns
            if c.startswith("tx_count_")
            and c.endswith("_per_card")
            and c != "tx_count_per_card"
        ] + [
            c
            for c in (
                "amt_24h_mean_per_card",
                "amt_24h_vs_card_mean_ratio",
                "dist1_log",
                "dist1_high",
            )
            if c in df.columns
        ]

        # All remaining numeric columns are time-varying features
        numeric_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and c not in uid_cols
            and c not in p9_4_cols
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8, np.float16]
        ]

        logger.info(
            f"Feature identification: {len(numeric_cols)} time-varying, "
            f"{len(static_cols)} static"
        )

        return numeric_cols, static_cols

    def build_sequences(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        temporal_values: Optional[pd.Series] = None,
        fit_scaler: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Build temporal sequences from flat features, optionally with labels.

        For each transaction i on card c:
        - Look back sequence_length transactions on the same card
        - If fewer exist, pad with zeros (left-padding)
        - The target is the label of the LAST (most recent) transaction

        Phase B7: `y` is optional. Labels are not available at inference time
        in production, so sequence construction must not require them.
        `build_sequences(X)` returns sequences/static/mask/indices with
        `targets=None`; use `attach_targets()` separately to attach labels
        for training/evaluation.

        Args:
            X: Feature DataFrame (no target column). Must be temporally sorted.
            y: Optional target Series (isFraud). Omit at inference time.
            temporal_values: Optional TransactionDT values for temporal sorting
                            within groups. If None, assumes X is already sorted.
            fit_scaler: If True, fit a QuantileTransformer on this call's
                numeric features (Phase B4) and store it for reuse on later
                calls (e.g. train=True, then val/test/inference=False). If
                False and a scaler was already fitted, it is reused
                (transform-only) — never refit on val/test data.

        Returns:
            Dict with keys:
                'sequences': np.ndarray of shape (N, seq_len, feature_dim) — time-varying features
                'static': np.ndarray of shape (N, static_dim) — static features per sequence
                'targets': np.ndarray of shape (N,), or None if `y` was not provided
                'mask': np.ndarray of shape (N, seq_len) — 1 for real data, 0 for padding
                'group_ids': np.ndarray of shape (N,) — card1 values for each sequence
                'original_indices': np.ndarray of shape (N,) — row position in `X`
        """
        df = X.copy()
        has_labels = y is not None
        if has_labels:
            df["__target__"] = y.values

        if temporal_values is not None:
            df["__temporal__"] = temporal_values.values
            df = df.sort_values("__temporal__").reset_index(drop=True)

        numeric_cols, static_cols = self._identify_features(df)
        self._numeric_features = numeric_cols
        self._static_features = static_cols
        self._feature_dim = len(numeric_cols)
        self._static_dim = len(static_cols)

        logger.info(
            f"Building sequences: seq_len={self.sequence_length}, "
            f"feature_dim={self._feature_dim}, static_dim={self._static_dim}, "
            f"total rows={len(df):,}"
        )

        # Pre-extract arrays for speed
        numeric_data = df[numeric_cols].values.astype(np.float32)
        static_data = df[static_cols].values.astype(np.float32) if static_cols else np.zeros((len(df), 0), dtype=np.float32)
        targets_data = df["__target__"].values.astype(np.float32) if has_labels else None
        group_data = df[self.group_col].values

        # Phase B4 — fit (train only) or reuse a QuantileTransformer so raw
        # dollar amounts and -999 imputation sentinels don't dominate the
        # gradient signal relative to already-unit-scale PCA components.
        if fit_scaler:
            n_quantiles = min(1000, max(len(numeric_data), 10))
            self.scaler = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                random_state=0,
            )
            numeric_data = self.scaler.fit_transform(numeric_data).astype(np.float32)
        elif self.scaler is not None:
            numeric_data = self.scaler.transform(numeric_data).astype(np.float32)

        # Phase B5 — record static categorical cardinalities once (from the
        # first call that establishes them, i.e. train) and reuse afterwards
        # so embedding table sizes stay fixed across train/val/test.
        categorical_cols_present = [c for c in static_cols if c in self.CATEGORICAL_STATIC_COLS]
        if categorical_cols_present and self._static_cardinalities is None:
            self._static_cardinalities = {
                col: int(df[col].max()) + 1 for col in categorical_cols_present
            }

        # Build sequences per group using vectorized approach
        sequences_list = []
        static_list = []
        targets_list = []
        masks_list = []
        group_ids_list = []
        original_indices_list = []

        seq_len = self.sequence_length

        # Group indices for efficient lookback
        # Process each card group
        group_indices = {}
        for idx, g in enumerate(group_data):
            if g not in group_indices:
                group_indices[g] = []
            group_indices[g].append(idx)

        n_groups = len(group_indices)
        total_sequences = 0

        for group_id, indices in group_indices.items():
            n_tx = len(indices)

            for i in range(n_tx):
                # Current transaction is the prediction target
                target_idx = indices[i]

                # Look back up to seq_len transactions (including current)
                start = max(0, i - seq_len + 1)
                history_indices = indices[start: i + 1]
                actual_len = len(history_indices)

                # Create padded sequence
                seq = np.zeros((seq_len, self._feature_dim), dtype=np.float32)
                mask = np.zeros(seq_len, dtype=np.float32)

                # Right-align: most recent transactions at the end
                pad_len = seq_len - actual_len
                seq[pad_len:] = numeric_data[history_indices]
                mask[pad_len:] = 1.0

                sequences_list.append(seq)
                static_list.append(static_data[target_idx])
                if has_labels:
                    targets_list.append(targets_data[target_idx])
                masks_list.append(mask)
                group_ids_list.append(group_id)
                original_indices_list.append(target_idx)
                total_sequences += 1

        logger.info(
            f"Built {total_sequences:,} sequences from {n_groups:,} card groups"
        )

        result = {
            "sequences": np.array(sequences_list, dtype=np.float32),
            "static": np.array(static_list, dtype=np.float32),
            "targets": np.array(targets_list, dtype=np.float32) if has_labels else None,
            "mask": np.array(masks_list, dtype=np.float32),
            "group_ids": np.array(group_ids_list),
            "original_indices": np.array(original_indices_list, dtype=int),
        }

        if has_labels:
            fraud_rate = result["targets"].mean() * 100
            logger.info(
                f"Sequence fraud rate: {fraud_rate:.2f}% "
                f"({int(result['targets'].sum()):,} fraud / {total_sequences:,} total)"
            )

        return result

    def attach_targets(
        self,
        seq_data: Dict[str, np.ndarray],
        y: pd.Series,
    ) -> Dict[str, np.ndarray]:
        """
        Attach labels to sequence data built via `build_sequences(X)` (no `y`).

        Phase B7: labels are matched to sequences via `original_indices`, not
        positional order — sequences are built per card group, so the Nth
        sequence does not generally correspond to the Nth row of `y`.

        Args:
            seq_data: Output of `build_sequences(X)` (targets=None).
            y: Target Series aligned with the original `X` passed to
                `build_sequences` (same row order, same length).

        Returns:
            A new dict (input is not mutated) with `targets` populated.
        """
        y_values = y.values if hasattr(y, "values") else np.asarray(y)
        targets = y_values[seq_data["original_indices"]].astype(np.float32)
        return {**seq_data, "targets": targets}

    def build_sequences_efficient(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_sequences: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Memory-efficient sequence builder that processes data in chunks.

        For very large datasets, this avoids holding all sequences in memory
        simultaneously by processing groups in batches.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            max_sequences: Maximum number of sequences to build (for debugging).

        Returns:
            Same dict format as build_sequences().
        """
        df = X.copy()
        df["__target__"] = y.values

        numeric_cols, static_cols = self._identify_features(df)
        self._numeric_features = numeric_cols
        self._static_features = static_cols
        self._feature_dim = len(numeric_cols)
        self._static_dim = len(static_cols)

        numeric_data = df[numeric_cols].values.astype(np.float32)
        static_data = df[static_cols].values.astype(np.float32) if static_cols else np.zeros((len(df), 0), dtype=np.float32)
        targets_data = df["__target__"].values.astype(np.float32)
        group_data = df[self.group_col].values

        seq_len = self.sequence_length

        # Pre-compute total count for pre-allocation
        total_count = len(df)
        if max_sequences is not None:
            total_count = min(total_count, max_sequences)

        # Pre-allocate output arrays
        sequences = np.zeros((total_count, seq_len, self._feature_dim), dtype=np.float32)
        statics = np.zeros((total_count, self._static_dim), dtype=np.float32)
        targets = np.zeros(total_count, dtype=np.float32)
        masks = np.zeros((total_count, seq_len), dtype=np.float32)
        original_indices = np.zeros(total_count, dtype=int)

        # Build group index map
        group_indices = {}
        for idx, g in enumerate(group_data):
            if g not in group_indices:
                group_indices[g] = []
            group_indices[g].append(idx)

        seq_idx = 0
        for group_id, indices in group_indices.items():
            n_tx = len(indices)
            for i in range(n_tx):
                if max_sequences is not None and seq_idx >= max_sequences:
                    break

                target_idx = indices[i]
                start = max(0, i - seq_len + 1)
                history_indices = indices[start: i + 1]
                actual_len = len(history_indices)
                pad_len = seq_len - actual_len

                sequences[seq_idx, pad_len:] = numeric_data[history_indices]
                masks[seq_idx, pad_len:] = 1.0
                statics[seq_idx] = static_data[target_idx]
                targets[seq_idx] = targets_data[target_idx]
                original_indices[seq_idx] = target_idx
                seq_idx += 1

            if max_sequences is not None and seq_idx >= max_sequences:
                break

        # Trim if we didn't fill all pre-allocated space
        if seq_idx < total_count:
            sequences = sequences[:seq_idx]
            statics = statics[:seq_idx]
            targets = targets[:seq_idx]
            masks = masks[:seq_idx]
            original_indices = original_indices[:seq_idx]

        logger.info(f"Built {seq_idx:,} sequences (efficient mode)")
        return {
            "sequences": sequences,
            "static": statics,
            "targets": targets,
            "mask": masks,
            "group_ids": np.array([]),  # Not tracked in efficient mode
            "original_indices": original_indices,
        }
