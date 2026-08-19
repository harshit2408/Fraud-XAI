"""
src/data/feature_engineering.py

Feature engineering pipeline for the IEEE-CIS Fraud Detection dataset.

Each feature group is a separate method for:
  - Independent testability (unit tests per method)
  - Clean ablation studies (skip/swap individual feature groups)
  - Inference reuse (same transformations applied at serving time)

CRITICAL RULES:
  - fit=True: fit transformers on training data
  - fit=False: transform only using pre-fitted transformers (test/inference)
  - Never fit on test data — prevents data leakage
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from src.utils.checksums import read_checksum_manifest, verify_checksums, write_checksums

logger = logging.getLogger(__name__)

# V-feature columns in the IEEE-CIS dataset
V_FEATURE_COLS = [f"V{i}" for i in range(1, 340)]

# Categorical columns per config
CATEGORICAL_COLS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30",
    "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo", "browser_family", "device_brand"
]

# D-columns (time-delta features in IEEE-CIS)
D_COLS = [f"D{i}" for i in range(1, 16)]

# C-columns (count features in IEEE-CIS)
C_COLS = [f"C{i}" for i in range(1, 15)]

# Free email domains — frequently used in fraud
FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "mail.com", "icloud.com", "ymail.com", "live.com", "msn.com",
    "protonmail.com", "gmx.com", "yahoo.com.mx", "att.net",
    "comcast.net", "cox.net", "sbcglobal.net", "verizon.net",
}

# Entity columns that receive expanding target encoding
TARGET_ENCODING_COLS = [
    "card1", "card2", "addr1", "P_emaildomain", "R_emaildomain", "device_brand",
]

# Smoothing weight for the target-encoding prior: an entity needs this many
# observations before its own history outweighs the global fraud rate.
TARGET_ENCODING_PRIOR_WEIGHT = 10.0

# Stand-in for a missing entity key in the carried target-encoding state. NaN
# cannot serve as its own key there: nan != nan and each frame's groupby emits
# a fresh NaN object, so successive splits would accumulate onto separate
# entries instead of one. A string cannot collide with a real key because every
# entity column is either numeric or a lowercase domain/brand token.
MISSING_ENTITY_KEY = "__MISSING_ENTITY__"

# Fallback fraud rate used before a train-only prior has been fitted
DEFAULT_GLOBAL_TARGET_MEAN = 0.035

# Rows per IncrementalPCA batch: 10k × 339 cols × 4B ≈ 13 MB per batch
PCA_BATCH_SIZE = 10_000

# Columns per chunk when counting nulls, to bound the transient boolean mask
NULL_COUNT_CHUNK_SIZE = 50


def _partial_fit_batches(
    positions: np.ndarray, batch_size: int, min_batch: int
) -> Iterator[np.ndarray]:
    """
    Split *positions* into batches of at most *batch_size*, guaranteeing every
    yielded batch — including the last — has at least *min_batch* elements.

    IncrementalPCA.partial_fit raises when a batch holds fewer samples than
    n_components, so a short trailing batch is folded into its predecessor
    rather than emitted alone. Looking one batch ahead (instead of only
    checking the final remainder up front) keeps this correct even when
    min_batch is a large fraction of batch_size, where a single fold-in step
    would otherwise still leave the merged batch too small.

    Callers must ensure len(positions) >= min_batch; the caller-side
    `actual_components = min(n_components, ..., len(fit_positions))` in
    reduce_v_features guarantees this for the PCA use case.
    """
    n = len(positions)
    if n == 0:
        raise ValueError("positions is empty — cannot partial_fit an empty batch.")

    # If min_batch exceeds batch_size, batch_size cannot be honoured without
    # violating min_batch — every batch must grow to at least min_batch, not
    # just the last one. Growing the step size keeps a single loop correct
    # for both regimes instead of special-casing min_batch > batch_size.
    step = max(batch_size, min_batch)

    start = 0
    while start < n:
        remaining = n - start
        if remaining <= step or remaining - step < min_batch:
            # Either this is naturally the last batch, or splitting it further
            # would leave a trailing remainder below min_batch — take it whole.
            yield positions[start:n]
            return
        yield positions[start : start + step]
        start += step


class FeatureEngineer:
    """
    Stateful feature engineering pipeline.

    Holds fitted transformers (PCA, encoders, imputers) so they can be
    serialized and reloaded for inference-time use without re-fitting.
    """

    def __init__(self) -> None:
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._freq_encoders: Dict[str, Dict[str, float]] = {}
        self._pca: Optional[PCA] = None
        self._num_fill_values: Dict[str, float] = {}
        self._cat_fill_value: str = "MISSING"
        self._card_hash_freq: Dict[str, float] = {}
        self._global_target_mean: float = DEFAULT_GLOBAL_TARGET_MEAN
        # Per-entity (cumulative fraud sum, observation count) carried between
        # splits so val/test encodings continue from train history instead of
        # restarting at the prior.
        self._target_enc_state: Dict[str, Dict[Any, Tuple[float, float]]] = {}
        # Frozen at first use so null_ratio means the same thing in every split
        # and at inference, regardless of how many columns exist at call time.
        self._null_ratio_denominator: Optional[int] = None

    # ─── 1. Temporal Features ────────────────────────────────────────────────

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert TransactionDT (seconds offset) to cyclical time features.

        TransactionDT is seconds since a reference point — not Unix time.
        We extract the periodic structure via sin/cos encoding, which allows
        the model to understand that hour 23 and hour 0 are adjacent.

        Produces:
            - hour_of_day (0–23): raw hour extracted from TransactionDT mod 86400
            - day_of_week (0–6): day extracted from TransactionDT // 86400 mod 7
            - hour_sin, hour_cos: cyclical hour encoding
            - day_sin, day_cos: cyclical day-of-week encoding
        """
        df = df.copy()
        seconds_in_day = 86400

        # Extract hour and day from the second-offset column
        df["hour_of_day"] = (df["TransactionDT"] % seconds_in_day // 3600).astype(int)
        df["day_of_week"] = (df["TransactionDT"] // seconds_in_day % 7).astype(int)

        # Cyclical encoding — preserves periodicity for tree and linear models
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        logger.debug("Created temporal features: hour_of_day, day_of_week, cyclical sin/cos")
        return df

    # ─── 2. Amount Features ──────────────────────────────────────────────────

    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive informative features from raw TransactionAmt.

        - amount_log: log1p transform to reduce skew (large transactions dominate)
        - amount_cents: fractional cents (transactions ending in .00 vs .99 differ
          behaviorally — round amounts are more common in fraud)
        - amount_is_round: 1 if amount has no decimal part (fraud signal)
        - amount_decimal_len: number of significant decimal digits
        """
        df = df.copy()
        df["amount_log"] = np.log1p(df["TransactionAmt"])
        df["amount_cents"] = df["TransactionAmt"] % 1.0

        # Round-amount detection: fraudulent transactions often use round numbers
        df["amount_is_round"] = (df["TransactionAmt"] % 1.0 == 0).astype(np.int8)

        # Decimal digit count: 100.00 → 0, 99.99 → 2, 49.995 → 3
        def _decimal_len(x):
            s = f"{x:.10f}".rstrip("0")
            if "." in s:
                return len(s.split(".")[1])
            return 0

        df["amount_decimal_len"] = df["TransactionAmt"].apply(_decimal_len).astype(np.int8)

        logger.debug("Created amount features: amount_log, amount_cents, amount_is_round, amount_decimal_len")
        return df

    # ─── 3. Card-Level Aggregates ────────────────────────────────────────────

    def create_card_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute card-level rolling/expanding statistics using card1 as proxy.
        Fully vectorized to guarantee index alignment and high performance.
        """
        df = df.copy()

        if "TransactionDT" in df.columns:
            df = df.sort_values("TransactionDT", ascending=True).reset_index(drop=True)

        df["tx_count_per_card"] = df.groupby("card1").cumcount()
        
        # Cumulative sum excluding current row
        df["tx_sum_per_card"] = df.groupby("card1")["TransactionAmt"].cumsum() - df["TransactionAmt"]
        
        # Mean excluding current row
        df["mean_amount_per_card"] = np.where(
            df["tx_count_per_card"] > 0,
            df["tx_sum_per_card"] / df["tx_count_per_card"],
            0.0
        )
        
        # Max amount per card (using shift to exclude current row)
        shifted_amt = df.groupby("card1")["TransactionAmt"].shift(1)
        df["max_amount_per_card"] = df.assign(_shifted=shifted_amt).groupby("card1")["_shifted"].cummax().fillna(0.0)
        
        # Standard deviation via rolling variance formula
        shifted_amt_sq = (df["TransactionAmt"] ** 2).groupby(df["card1"]).shift(1)
        sum_sq = df.assign(_shifted_sq=shifted_amt_sq).groupby("card1")["_shifted_sq"].cumsum().fillna(0.0)
        
        var = np.where(
            df["tx_count_per_card"] > 1,
            (sum_sq - (df["tx_sum_per_card"] ** 2) / df["tx_count_per_card"]) / (df["tx_count_per_card"] - 1),
            0.0
        )
        df["std_amount_per_card"] = np.sqrt(np.maximum(var, 0.0))

        # Anomaly signals
        df["amount_vs_mean_ratio"] = np.where(
            df["mean_amount_per_card"] > 0,
            df["TransactionAmt"] / df["mean_amount_per_card"],
            1.0
        )
        
        df["amount_zscore_per_card"] = np.where(
            df["std_amount_per_card"] > 0,
            (df["TransactionAmt"] - df["mean_amount_per_card"]) / (df["std_amount_per_card"] + 1e-6),
            0.0
        )

        logger.debug("Created card aggregate features (vectorized)")
        return df

    # ─── 3.5 Cross-Interactions ───────────────────────────────────────────────

    def create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create explicit non-linear interaction features for the tree-based model.
        - hour_of_day * amount_log
        - card-level amount deviation * hour
        """
        df = df.copy()
        
        if "hour_of_day" in df.columns and "amount_log" in df.columns:
            df["interaction_hour_amount"] = df["hour_of_day"] * df["amount_log"]
            
        if "hour_of_day" in df.columns and "amount_zscore_per_card" in df.columns:
            df["interaction_hour_zscore"] = df["hour_of_day"] * df["amount_zscore_per_card"]
            
        logger.debug("Created interaction features")
        return df

    # ─── 4. Email Domain Features ────────────────────────────────────────────

    def create_email_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract fraud-predictive signals from email domain columns.

        - email_match: whether P_emaildomain matches R_emaildomain
          (mismatched domains are a fraud signal)
        - P_email_is_free / R_email_is_free: whether domain is a free provider
          (gmail, yahoo, etc. — more common in fraud)
        - email_both_present: whether both email fields are filled
          (missing R_emaildomain is common in fraud)
        """
        df = df.copy()

        p_email = df.get("P_emaildomain")
        r_email = df.get("R_emaildomain")

        if p_email is not None and r_email is not None:
            # Domain match: same domain for payer and receiver suggests legitimacy
            df["email_match"] = (
                p_email.fillna("__NONE__") == r_email.fillna("__NONE__")
            ).astype(np.int8)

            # Both present
            df["email_both_present"] = (
                p_email.notna() & r_email.notna()
            ).astype(np.int8)

        if p_email is not None:
            df["P_email_is_free"] = (
                p_email.fillna("").str.lower().isin(FREE_EMAIL_DOMAINS)
            ).astype(np.int8)

        if r_email is not None:
            df["R_email_is_free"] = (
                r_email.fillna("").str.lower().isin(FREE_EMAIL_DOMAINS)
            ).astype(np.int8)

        logger.debug("Created email domain features")
        return df

    # ─── 5. D-Column (Time Delta) Summary Features ──────────────────────────

    def create_d_column_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        D1–D15 are time-delta features in IEEE-CIS (exact meaning is masked).
        Their missingness pattern is itself highly predictive of fraud.

        Produces:
            - D_null_count: count of null D-columns per row
            - D_mean: mean of non-null D values per row
            - D_std: std of non-null D values per row
            - D_max: max of non-null D values per row
        """
        df = df.copy()
        d_cols = [c for c in D_COLS if c in df.columns]

        if d_cols:
            d_data = df[d_cols]
            df["D_null_count"] = d_data.isnull().sum(axis=1).astype(np.int8)
            df["D_mean"] = d_data.mean(axis=1, skipna=True)
            df["D_std"] = d_data.std(axis=1, skipna=True)
            df["D_max"] = d_data.max(axis=1, skipna=True)
            # Fill remaining NaN (all-null rows)
            for col in ["D_mean", "D_std", "D_max"]:
                df[col] = df[col].fillna(-1.0)

        logger.debug("Created D-column summary features")
        return df

    # ─── 6. C-Column (Count) Summary Features ───────────────────────────────

    def create_c_column_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        C1–C14 are count features in IEEE-CIS (exact meaning masked).
        Aggregating them provides a signal about transaction complexity.

        Produces:
            - C_sum: sum of all C-columns per row
            - C_max: max of C-columns per row
            - C_nonzero_count: number of non-zero C values per row
        """
        df = df.copy()
        c_cols = [c for c in C_COLS if c in df.columns]

        if c_cols:
            c_data = df[c_cols].fillna(0)
            df["C_sum"] = c_data.sum(axis=1)
            df["C_max"] = c_data.max(axis=1)
            df["C_nonzero_count"] = (c_data != 0).sum(axis=1).astype(np.int8)

        logger.debug("Created C-column summary features")
        return df

    # ─── 7. Card Combination Identity ────────────────────────────────────────

    def create_card_hash_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create a pseudo-unique card identifier by hashing a combination of
        card-related columns. Then frequency-encode this hash.

        Hash of (card1, card2, card3, card5, addr1) creates a proxy for the
        unique card+address combination. Frequency encoding reveals how often
        each combination appears — rare combos are more suspicious.
        """
        df = df.copy()
        hash_cols = ["card1", "card2", "card3", "card5", "addr1"]
        present_cols = [c for c in hash_cols if c in df.columns]

        if len(present_cols) >= 2:
            # Create string hash of the card combination
            combined = df[present_cols].fillna(-999).astype(str).agg("_".join, axis=1)
            card_hash = combined.apply(
                lambda x: hashlib.md5(x.encode()).hexdigest()[:8]
            )

            if fit:
                freq_map = card_hash.value_counts(normalize=True).to_dict()
                self._card_hash_freq = freq_map

            df["card_hash_freq"] = card_hash.map(self._card_hash_freq).fillna(0.0)
            logger.debug(f"Created card_hash_freq from {len(present_cols)} card columns")
        else:
            logger.warning("Insufficient card columns for hash feature")

        return df

    # ─── 8. Transaction Velocity Features ────────────────────────────────────

    def create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transaction velocity: how quickly transactions happen per card.

        - time_since_last_tx: seconds since previous transaction on same card
          (very short gaps are suspicious — rapid-fire fraud)
        - tx_velocity_1h: count of card's transactions in the last 3600 seconds
          (approximated via cumcount in a rolling window)

        CRITICAL: Requires temporally sorted data. Uses shift(1) to avoid leakage.
        """
        df = df.copy()

        if "TransactionDT" not in df.columns or "card1" not in df.columns:
            logger.warning("TransactionDT or card1 missing — skipping velocity features")
            return df

        # Ensure temporal sort
        df = df.sort_values("TransactionDT", ascending=True).reset_index(drop=True)

        # Time since last transaction on same card
        df["time_since_last_tx"] = df.groupby("card1")["TransactionDT"].diff()
        df["time_since_last_tx"] = df["time_since_last_tx"].fillna(-1.0)

        # Log-transform of time delta (very right-skewed)
        df["time_since_last_tx_log"] = np.where(
            df["time_since_last_tx"] > 0,
            np.log1p(df["time_since_last_tx"]),
            -1.0
        )

        logger.debug("Created velocity features: time_since_last_tx, time_since_last_tx_log")
        return df

    # ─── 9. Null-Count Meta Feature ──────────────────────────────────────────

    def create_null_count_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Count of null values per row across all columns.

        In IEEE-CIS, missingness patterns are strongly correlated with fraud.
        Fraudulent transactions often have more missing identity/device fields.

        Must run on the RAW frame — before imputation and before
        reduce_v_features, which converts V-column NaNs to 0.0 and would
        otherwise erase V-missingness from the count entirely.

        The null_ratio denominator is frozen on the first call so the feature
        carries the same meaning across train/val/test and at inference,
        instead of shifting with the column count at the call site.
        """
        df = df.copy()

        # Counted in column chunks: df.isnull() over all columns at once
        # materializes a full-size boolean frame (~256 MB on the raw dataset).
        null_count = pd.Series(0, index=df.index, dtype=np.int32)
        columns = list(df.columns)
        for start in range(0, len(columns), NULL_COUNT_CHUNK_SIZE):
            chunk = columns[start : start + NULL_COUNT_CHUNK_SIZE]
            null_count += df[chunk].isnull().sum(axis=1).astype(np.int32)

        if self._null_ratio_denominator is None:
            self._null_ratio_denominator = max(len(columns), 1)

        df["null_count_total"] = null_count.astype(np.int16)
        df["null_ratio"] = null_count / self._null_ratio_denominator

        logger.debug(
            f"Created null-count meta features over {len(columns)} columns "
            f"(null_ratio denominator={self._null_ratio_denominator})"
        )
        return df

    # ─── 10. Address Signal Features ─────────────────────────────────────────

    def create_address_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract fraud signals from address-related columns.

        - addr_present: whether addr1 is not null (missing address is fraud signal)
        - dist_present: whether dist1 is not null
        - addr1_card1_count: frequency of (addr1, card1) combo
        """
        df = df.copy()

        if "addr1" in df.columns:
            df["addr_present"] = df["addr1"].notna().astype(np.int8)
        if "dist1" in df.columns:
            df["dist_present"] = df["dist1"].notna().astype(np.int8)

        logger.debug("Created address signal features")
        return df

    # ─── 4. Device and Browser Parsing (id_31, DeviceInfo, id_33) ────────────

    def create_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract meaningful OS, browser, and device characteristics from raw identity strings.
        """
        df = df.copy()

        # Parse id_31 (Browser string)
        if "id_31" in df.columns:
            # Lowercase for robust matching
            id_31 = df["id_31"].astype(str).str.lower()
            df["browser_family"] = "other"
            df.loc[id_31.str.contains("chrome|crios"), "browser_family"] = "chrome"
            df.loc[id_31.str.contains("safari") & ~id_31.str.contains("chrome|crios|edge"), "browser_family"] = "safari"
            df.loc[id_31.str.contains("firefox"), "browser_family"] = "firefox"
            df.loc[id_31.str.contains("edge|ie|rv:11"), "browser_family"] = "edge_ie"
            df.loc[id_31.str.contains("samsung"), "browser_family"] = "samsung"
            
            # Outdated/rare browsers often correlate with fraud/bots
            df["browser_is_mobile"] = id_31.str.contains("mobile|android|ios").astype(int)

        # Parse DeviceInfo
        if "DeviceInfo" in df.columns:
            dev = df["DeviceInfo"].astype(str).str.lower()
            df["device_brand"] = "other"
            df.loc[dev.str.contains("windows"), "device_brand"] = "windows"
            df.loc[dev.str.contains("ios|mac"), "device_brand"] = "apple"
            df.loc[dev.str.contains("sm-|samsung"), "device_brand"] = "samsung"
            df.loc[dev.str.contains("huawei|ale-|rv:"), "device_brand"] = "huawei"
            df.loc[dev.str.contains("moto"), "device_brand"] = "motorola"
            df.loc[dev.str.contains("lg-"), "device_brand"] = "lg"

        # Parse id_33 (Screen Resolution)
        if "id_33" in df.columns:
            # Format is typically "1920x1080"
            res = df["id_33"].astype(str).str.split("x", expand=True)
            if res.shape[1] == 2:
                df["screen_width"] = pd.to_numeric(res[0], errors="coerce").fillna(0)
                df["screen_height"] = pd.to_numeric(res[1], errors="coerce").fillna(0)
                # Aspect ratio is a strong indicator of emulators
                df["screen_aspect_ratio"] = np.where(
                    df["screen_height"] > 0, 
                    df["screen_width"] / df["screen_height"], 
                    0
                )
            else:
                df["screen_width"] = 0
                df["screen_height"] = 0
                df["screen_aspect_ratio"] = 0

        logger.debug("Created device and browser parsed features")
        return df

    # ─── 5. Expanding Target Encoding ──────────────────────────────────────────

    def create_target_encoding(
        self, df: pd.DataFrame, fit: bool = True, update_state: bool = False
    ) -> pd.DataFrame:
        """
        Expanding-window target encoding for high-cardinality entity columns.

        Two independent leakage guards:
          - Within a frame, the current row's own label is excluded
            (cumsum minus the row, cumcount is exclusive by construction), so a
            transaction can never encode itself.
          - Across splits, the global prior is estimated once with fit=True on
            the train slice. fit=False reuses that prior and never re-estimates
            it, so val/test labels cannot influence the encoding.

        Rows are assumed to be in temporal order; the caller is responsible for
        sorting, and for calling train → val → test in that order.

        Args:
            df: Temporally ordered frame containing the target column.
            fit: True on the train slice — estimates the prior and initialises
                per-entity state from this frame alone. False on val/test —
                reuses the fitted prior and seeds counters from carried state.
            update_state: Fold this frame's entity totals into the carried
                state so a later fit=False call sees this history. Implied by
                fit=True. Leave False at inference, where transforms must not
                mutate fitted state.

        Returns:
            A new DataFrame with one `<col>_target_enc` column per entity column.
        """
        df = df.copy()

        target_col = "isFraud"
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found. Skipping target encoding.")
            return df

        if fit:
            self._global_target_mean = float(df[target_col].mean())
            logger.debug(
                f"Fitted target-encoding prior on {len(df):,} rows: "
                f"{self._global_target_mean:.6f}"
            )

        prior = self._global_target_mean
        present_cols = [c for c in TARGET_ENCODING_COLS if c in df.columns]
        next_state: Dict[str, Dict[Any, Tuple[float, float]]] = {}

        for col in present_cols:
            # fit=True starts from an empty history; fit=False continues from
            # whatever train (and optionally val) already accumulated.
            carried = {} if fit else self._target_enc_state.get(col, {})
            carried_sum, carried_count = self._carried_totals(df[col], carried)

            # dropna=False keeps a missing entity key as its own group, matching
            # _accumulate_entity_totals. Under the default (dropna=True) those
            # rows get a NaN cumcount, which makes the cum_count == 0 prior
            # fallback unreachable and emits a NaN feature — and target encoding
            # runs before imputation, so raw NaN keys do reach this point.
            grouped = df.groupby(col, dropna=False)[target_col]
            cum_sum = grouped.cumsum() - df[target_col] + carried_sum
            cum_count = grouped.cumcount() + carried_count

            encoded = (cum_sum + TARGET_ENCODING_PRIOR_WEIGHT * prior) / (
                cum_count + TARGET_ENCODING_PRIOR_WEIGHT
            )
            # No history for this entity yet — fall back to the global prior
            encoded = np.where(cum_count == 0, prior, encoded)

            df[f"{col}_target_enc"] = encoded

            if fit or update_state:
                next_state[col] = self._accumulate_entity_totals(
                    df[col], df[target_col], carried
                )

        if next_state:
            self._target_enc_state = {**self._target_enc_state, **next_state}

        logger.debug(
            f"Created expanding target encoding (fit={fit}) for: {present_cols}"
        )
        return df

    @staticmethod
    def _canonical_entity_keys(keys: pd.Series) -> pd.Series:
        """
        Replace missing entity keys with MISSING_ENTITY_KEY.

        The in-frame groupby uses dropna=False, so missing keys form one real
        expanding group whose history has to survive in the carried state. Left
        as NaN it cannot: dict lookup falls back to equality and nan != nan, so
        each split writes a *new* missing-key entry rather than accumulating
        onto the previous one — losing the history, and eventually making the
        state a non-unique index that Series.map refuses to align against.
        """
        missing = keys.isna()
        if not missing.any():
            return keys
        return keys.astype(object).where(~missing, MISSING_ENTITY_KEY)

    @staticmethod
    def _carried_totals(
        keys: pd.Series, carried: Dict[Any, Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-row (sum, count) history carried in from previously seen splits."""
        if not carried:
            zeros = np.zeros(len(keys), dtype=float)
            return zeros, zeros.copy()

        keys = FeatureEngineer._canonical_entity_keys(keys)
        sum_map = {k: v[0] for k, v in carried.items()}
        count_map = {k: v[1] for k, v in carried.items()}
        return (
            keys.map(sum_map).fillna(0.0).to_numpy(dtype=float),
            keys.map(count_map).fillna(0.0).to_numpy(dtype=float),
        )

    @staticmethod
    def _accumulate_entity_totals(
        keys: pd.Series,
        target: pd.Series,
        carried: Dict[Any, Tuple[float, float]],
    ) -> Dict[Any, Tuple[float, float]]:
        """
        Return a new state dict: *carried* plus this frame's entity totals.

        Keys are canonicalised first so the missing-key group lands on the same
        state entry every split, matching the NaN-group behaviour of the
        cumsum/cumcount call in create_target_encoding. sort=False because
        canonicalisation makes the key set mixed-type (numeric entities plus
        the string sentinel), which is unorderable; group order is irrelevant
        to a summed total.
        """
        keys = FeatureEngineer._canonical_entity_keys(keys)
        totals = target.groupby(keys, dropna=False, sort=False).agg(["sum", "count"])
        updated = dict(carried)
        for key, row in totals.iterrows():
            prev_sum, prev_count = updated.get(key, (0.0, 0.0))
            updated[key] = (prev_sum + float(row["sum"]), prev_count + float(row["count"]))
        return updated

    # ─── 6. Label Encoding & PCA ──────────────────────────────────────────────

    def encode_categoricals(
        self, df: pd.DataFrame, fit: bool = True, cat_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Encode categorical columns. Strategy:
          - High-cardinality (P_emaildomain, R_emaildomain, card1): frequency encoding
            (replace category with its frequency in training set)
          - Low-cardinality (ProductCD, card4, card6, M1-M9): label encoding

        fit=True: fit encoders from df (training data).
        fit=False: apply pre-fitted encoders (test/inference — no new state).

        NaN values must already be filled with 'MISSING' before calling this method.
        """
        df = df.copy()
        high_card_cols = ["P_emaildomain", "R_emaildomain", "id_31", "id_33", "DeviceInfo"]
        low_card_cols = [
            c for c in (cat_cols or CATEGORICAL_COLS)
            if c in df.columns and c not in high_card_cols
        ]

        # Frequency encoding for high-cardinality columns
        for col in high_card_cols:
            if col not in df.columns:
                continue
            if fit:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                self._freq_encoders[col] = freq_map
            freq_map = self._freq_encoders.get(col, {})
            df[col] = df[col].map(freq_map).fillna(0.0)

        # Label encoding for low-cardinality columns
        for col in low_card_cols:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str)
            if fit:
                le = LabelEncoder()
                # Reserve the sentinel even when training data has no NaNs,
                # otherwise the unseen-label path below maps to a class the
                # encoder has never seen and transform() raises at inference.
                le.fit(np.append(df[col].to_numpy(dtype=object), self._cat_fill_value))
                self._label_encoders[col] = le
            le = self._label_encoders.get(col)
            if le is not None:
                # Handle unseen labels at inference time
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "MISSING")
                df[col] = le.transform(df[col])
            else:
                logger.warning(f"No fitted encoder found for '{col}' — skipping.")

        logger.debug(f"Encoded categoricals (fit={fit})")
        return df

    # ─── 12. V-Feature PCA Reduction ─────────────────────────────────────────

    def reduce_v_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        n_components: int = 30,
        fit_rows: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        PCA on V1–V339 columns to reduce dimensionality from ~339 to 30.

        Uses IncrementalPCA with row batches to avoid allocating a 764 MB
        numpy array (339 V-cols × 590,540 rows × 4 bytes float32).
        Each batch processes PCA_BATCH_SIZE rows, requiring only ~13 MB per batch.

        Fit and transform are scoped independently so the reduction can run on
        the whole frame — which keeps the pre-sort memory optimisation — while
        the components themselves are estimated from train rows alone.

        Args:
            df: Frame containing the V-columns.
            fit: True to estimate components, False to apply the fitted PCA.
            n_components: Upper bound on retained components.
            fit_rows: Row selector (boolean mask or integer positions) limiting
                which rows the fit observes. None fits on every row. Ignored
                when fit=False. Every row is transformed either way.

        Raises:
            RuntimeError: fit=False before any fit.
            ValueError: fit_rows selects no rows.
        """
        import gc
        from sklearn.decomposition import IncrementalPCA

        df = df.copy()
        v_cols = [c for c in V_FEATURE_COLS if c in df.columns]

        if not v_cols:
            logger.warning("No V-feature columns found — skipping PCA reduction.")
            return df

        n_rows = len(df)

        if fit:
            fit_positions = self._resolve_fit_positions(fit_rows, n_rows)
            actual_components = min(n_components, len(v_cols), len(fit_positions))
            logger.info(
                f"IncrementalPCA fit: {len(v_cols)} V-cols → {actual_components} components, "
                f"fitted on {len(fit_positions):,} of {n_rows:,} rows "
                f"in batches of {PCA_BATCH_SIZE:,}"
            )
            self._pca = IncrementalPCA(n_components=actual_components)
            for batch_positions in _partial_fit_batches(
                fit_positions, PCA_BATCH_SIZE, actual_components
            ):
                # Row-first slice keeps each batch small (batch rows × 339)
                batch = (
                    df.iloc[batch_positions][v_cols]
                    .to_numpy(dtype=np.float32, na_value=0.0)
                )
                self._pca.partial_fit(batch)
                del batch
                gc.collect()

            variance_retained = self._pca.explained_variance_ratio_.sum()
            logger.info(f"PCA fit complete. Variance retained: {variance_retained:.1%}")
        else:
            if self._pca is None:
                raise RuntimeError(
                    "PCA not fitted. Call reduce_v_features(fit=True) on training data first."
                )

        # Transform every row, including those excluded from the fit
        reduced_chunks: list = []
        for start in range(0, n_rows, PCA_BATCH_SIZE):
            batch = (
                df.iloc[start : start + PCA_BATCH_SIZE][v_cols]
                .to_numpy(dtype=np.float32, na_value=0.0)
            )
            reduced_chunks.append(self._pca.transform(batch))
            del batch
            gc.collect()

        v_reduced = np.concatenate(reduced_chunks, axis=0)
        del reduced_chunks

        # Replace V-columns with PCA components
        df = df.drop(columns=v_cols)
        pca_cols = [f"pca_v_{i}" for i in range(v_reduced.shape[1])]
        pca_df = pd.DataFrame(v_reduced, columns=pca_cols, index=df.index)
        df = pd.concat([df, pca_df], axis=1)
        del v_reduced

        logger.info(f"V-feature PCA complete: {len(v_cols)} cols → {len(pca_cols)} PCA components")
        return df

    @staticmethod
    def _resolve_fit_positions(
        fit_rows: Optional[np.ndarray], n_rows: int
    ) -> np.ndarray:
        """Normalise a boolean mask or integer selector into row positions."""
        if fit_rows is None:
            return np.arange(n_rows)

        positions = np.asarray(fit_rows)
        if positions.dtype == bool:
            if len(positions) != n_rows:
                raise ValueError(
                    f"fit_rows mask has {len(positions)} entries but the frame "
                    f"has {n_rows} rows."
                )
            positions = np.flatnonzero(positions)

        if positions.size == 0:
            raise ValueError("fit_rows selects no rows — PCA cannot be fitted.")
        return positions


    # ─── 13. Missing Value Handling ──────────────────────────────────────────

    def handle_missing_values(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Impute missing values with a strategy that signals missingness to tree models.

        - Numerical NaN → -999 (sentinel value; XGBoost handles well; signals absence)
        - Categorical NaN → 'MISSING' string (before encoding; becomes its own category)

        fit=True: compute fill values from training data.
        fit=False: apply pre-computed fill values (no re-fitting on test data).
        """
        df = df.copy()

        # Identify column types
        cat_cols = [c for c in df.select_dtypes(include=["object"]).columns]
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

        # Categorical: fill with 'MISSING' sentinel string
        for col in cat_cols:
            df[col] = df[col].fillna(self._cat_fill_value)

        # Numerical: fill with -999 sentinel
        if fit:
            # Store fill values from training data (for reproducibility)
            self._num_fill_values = {col: -999.0 for col in num_cols}

        for col in num_cols:
            fill_val = self._num_fill_values.get(col, -999.0)
            df[col] = df[col].fillna(fill_val)

        remaining_nan = df.isna().sum().sum()
        if remaining_nan > 0:
            logger.warning(f"{remaining_nan} NaN values remain after imputation!")
        else:
            logger.debug("All NaN values imputed successfully.")

        return df

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save_transformers(self, path: str) -> None:
        """
        Persist all fitted transformers to disk (Phase D6, joblib + sha256
        checksum manifest — no raw pickle.dump).

        Saves: label_encoders, freq_encoders, PCA, num_fill_values,
        card_hash_freq, and the derived feature state (target-encoding prior
        and per-entity counters, null_ratio denominator).
        These must be loaded at inference time to apply identical transformations.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        files: Dict[str, Path] = {
            "label_encoders": save_path / "label_encoders.joblib",
            "freq_encoders": save_path / "freq_encoders.joblib",
            "imputer": save_path / "imputer.joblib",
            "card_hash_freq": save_path / "card_hash_freq.joblib",
            "feature_state": save_path / "feature_state.joblib",
        }
        joblib.dump(self._label_encoders, files["label_encoders"])
        joblib.dump(self._freq_encoders, files["freq_encoders"])
        joblib.dump(self._num_fill_values, files["imputer"])
        joblib.dump(self._card_hash_freq, files["card_hash_freq"])
        joblib.dump(self._feature_state(), files["feature_state"])

        if self._pca is not None:
            files["pca"] = save_path / "pca.joblib"
            joblib.dump(self._pca, files["pca"])

        write_checksums(save_path / "checksums.json", files)

        logger.info(f"Transformers saved to {save_path}/")

    def _feature_state(self) -> Dict[str, Any]:
        """Fitted state that is not itself a transformer object."""
        return {
            "global_target_mean": self._global_target_mean,
            "target_enc_state": self._target_enc_state,
            "null_ratio_denominator": self._null_ratio_denominator,
        }

    def load_transformers(self, path: str) -> None:
        """
        Load pre-fitted transformers from disk for inference-time use
        (Phase D6, joblib + sha256 checksum manifest — no raw pickle.load).

        Must be called before using fit=False on any transformation method.

        Raises:
            FileNotFoundError: `path` has no checksums.json manifest — a
                directory of transformer files (or an empty/nonexistent
                directory) whose integrity cannot be verified must never be
                silently trusted. This is a deliberate change from the
                pre-D6 behavior, which silently no-op'd on missing files.
            ValueError: a file's recomputed sha256 does not match the
                manifest (corrupted or tampered artifact).
        """
        load_path = Path(path)

        file_candidates: Dict[str, Path] = {
            "label_encoders": load_path / "label_encoders.joblib",
            "freq_encoders": load_path / "freq_encoders.joblib",
            "pca": load_path / "pca.joblib",
            "imputer": load_path / "imputer.joblib",
            "card_hash_freq": load_path / "card_hash_freq.joblib",
            "feature_state": load_path / "feature_state.joblib",
        }
        # Filter by the checksum manifest's own recorded keys, NOT by
        # filesystem existence. "pca" is legitimately optional — absent
        # from the manifest entirely when PCA was never fit — but every
        # other candidate is unconditionally written by save_transformers,
        # so if one of those goes missing from disk it must still appear
        # here and let verify_checksums's FileNotFoundError fire below,
        # rather than being silently dropped and leaving that piece of
        # state un-restored with no error at all.
        recorded = read_checksum_manifest(load_path / "checksums.json")
        files = {name: p for name, p in file_candidates.items() if name in recorded}

        verify_checksums(load_path / "checksums.json", files)

        if "label_encoders" in files:
            self._label_encoders = joblib.load(files["label_encoders"])

        if "freq_encoders" in files:
            self._freq_encoders = joblib.load(files["freq_encoders"])

        if "pca" in files:
            self._pca = joblib.load(files["pca"])

        if "imputer" in files:
            self._num_fill_values = joblib.load(files["imputer"])

        if "card_hash_freq" in files:
            self._card_hash_freq = joblib.load(files["card_hash_freq"])

        if "feature_state" in files:
            state = joblib.load(files["feature_state"])
            self._global_target_mean = state.get(
                "global_target_mean", DEFAULT_GLOBAL_TARGET_MEAN
            )
            self._target_enc_state = state.get("target_enc_state", {})
            self._null_ratio_denominator = state.get("null_ratio_denominator")

        logger.info(f"Transformers loaded from {load_path}/")
