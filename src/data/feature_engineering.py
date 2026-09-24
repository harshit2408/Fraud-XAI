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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from src.utils.checksums import (
    read_checksum_manifest,
    verify_checksums,
    write_checksums,
)

logger = logging.getLogger(__name__)

# V-feature columns in the IEEE-CIS dataset
V_FEATURE_COLS = [f"V{i}" for i in range(1, 340)]

# Categorical columns per config
CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
    "browser_family",
    "device_brand",
]

# D-columns (time-delta features in IEEE-CIS)
D_COLS = [f"D{i}" for i in range(1, 16)]

# C-columns (count features in IEEE-CIS)
C_COLS = [f"C{i}" for i in range(1, 15)]

# Free email domains — frequently used in fraud
FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "aol.com",
    "mail.com",
    "icloud.com",
    "ymail.com",
    "live.com",
    "msn.com",
    "protonmail.com",
    "gmx.com",
    "yahoo.com.mx",
    "att.net",
    "comcast.net",
    "cox.net",
    "sbcglobal.net",
    "verizon.net",
}

# Entity columns that receive expanding target encoding
TARGET_ENCODING_COLS = [
    "card1",
    "card2",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "device_brand",
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

# Scalars retained per card by CardAggregateState; the width
# _carried_card_totals reshapes its per-row lookup into.
CARD_STATE_SCALAR_COUNT = 4

# Rows per IncrementalPCA batch: 10k × 339 cols × 4B ≈ 13 MB per batch
PCA_BATCH_SIZE = 10_000

# Columns per chunk when counting nulls, to bound the transient boolean mask
NULL_COUNT_CHUNK_SIZE = 50

# PRD Phase 9 (P9-4): fixed trailing windows for the per-card velocity counts,
# in seconds. 10 minutes / 1 hour / 24 hours per Whitrow and Bahnsen — short
# windows catch rapid-fire bursts, the 24h window carries a day's rhythm.
WINDOW_10MIN = 600.0
WINDOW_1H = 3600.0
WINDOW_24H = 86400.0

ROLLING_WINDOW_COUNTS = {
    "tx_count_10min_per_card": WINDOW_10MIN,
    "tx_count_1h_per_card": WINDOW_1H,
    "tx_count_24h_per_card": WINDOW_24H,
}

ROLLING_WINDOW_FEATURES = list(ROLLING_WINDOW_COUNTS) + [
    "amt_24h_mean_per_card",
    "amt_24h_vs_card_mean_ratio",
]

# PRD Phase 9 (P9-4): `dist1` is non-null for ProductCD == "W" ONLY (verified
# against data/raw/train_transaction.csv on 2026-09-08: 54.2% populated within
# W, 0% for C/H/R/S). Its apparent presence signal is a ProductCD mix effect
# and vanishes within W (1.996% fraud present vs 2.092% absent); the real
# signal is a value gradient, fraud roughly doubling in the top quintile
# (2.95% above dist1 > 36 vs a flat ~1.6-1.8% below).
DIST1_PRODUCT = "W"
DIST1_HIGH_QUANTILE = 0.8
DIST1_MISSING_FILL = -1.0


@dataclass(frozen=True)
class CardAggregateState:
    """Per-`card1` sufficient statistics for the expanding card aggregates.

    docs/adr/ADR-002-realtime-feature-state.md §2.3: every feature
    `create_card_aggregates` and `create_velocity_features` emit
    (`tx_count/sum/mean/max/std_amount_per_card`, `amount_vs_mean_ratio`,
    `amount_zscore_per_card`, `time_since_last_tx`) is a pure function of
    these five scalars over a card's strictly-prior transactions. Retaining
    them is therefore enough to reproduce the batch values EXACTLY at
    serving time, in O(1) space per card — which is what makes ADR-002's
    chosen "online per-entity store" option viable at all.

    Frozen, and `observe` returns a new instance, so a snapshot handed to
    the serving state store cannot be mutated underneath it.
    """

    n: int = 0
    sum_amt: float = 0.0
    sum_amt_sq: float = 0.0
    max_amt: float = 0.0
    last_dt: float = float("nan")

    def observe(self, amount: float, dt: float) -> "CardAggregateState":
        """Fold one transaction in, returning a new state (never in-place)."""
        return replace(
            self,
            n=self.n + 1,
            sum_amt=self.sum_amt + amount,
            sum_amt_sq=self.sum_amt_sq + amount * amount,
            max_amt=amount if self.n == 0 else max(self.max_amt, amount),
            last_dt=dt,
        )


@dataclass(frozen=True)
class CardWindowState:
    """Per-`card1` trailing-window history for the P9-4 velocity features.

    ADR-003. `CardAggregateState` above is a *sufficient statistic* for the
    expanding aggregates — five scalars reproduce them exactly. The trailing
    windows are not reconstructible that way: a count over `[t - span, t)`
    needs the individual timestamps, so this carries them.

    That retires ADR-002 §2.3's constant-space-per-card property for this
    feature class, which is the trade ADR-003 records. Space is bounded by the
    longest window instead: entries older than `WINDOW_24H` are dropped on
    every write, so a card holds only its last day of activity (measured on
    the training data: mean 21 entries, p99 359, max 654).

    Frozen, and `observe` returns a new instance, so a snapshot handed to the
    serving state store cannot be mutated underneath it.
    """

    # (TransactionDT, TransactionAmt) pairs, ascending by dt. A tuple rather
    # than a deque so the whole state stays immutable and cheap to snapshot.
    entries: Tuple[Tuple[float, float], ...] = ()

    def observe(self, amount: float, dt: float) -> "CardWindowState":
        """Fold one transaction in and drop whatever has aged out.

        A backdated transaction (dt before the newest entry) is REJECTED
        rather than inserted in place: the ascending order is what makes the
        trailing counts correct, and batch never sees out-of-order rows
        because it sorts globally first. ADR-002 §5.3 already clamps the
        analogous `time_since_last_tx` case and counts it at the call site.
        """
        if self.entries and dt < self.entries[-1][0]:
            return self
        kept = self.trim(dt) + ((dt, float(amount)),)
        return replace(self, entries=kept)

    def trim(self, now: float) -> Tuple[Tuple[float, float], ...]:
        """Entries still inside the longest trailing window as of `now`.

        The boundary matches the batch `searchsorted(..., side="left")`
        exactly: an entry precisely `WINDOW_24H` old is still INSIDE.
        """
        cutoff = now - WINDOW_24H
        return tuple(e for e in self.entries if e[0] >= cutoff)


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
        # The exact V-column list the PCA was fitted on, in order. A serving
        # request carries only the V-columns that were non-null for that one
        # transaction, so `fit=False` must reindex to this list (missing cols
        # → 0.0, the same na_value the fit used) before calling transform,
        # or sklearn rejects the row on n_features. None until a fit or a
        # load restores it; the transform path then falls back to the
        # V_FEATURE_COLS ∩ df.columns behaviour for pre-existing artifacts.
        self._pca_fit_cols: Optional[List[str]] = None
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
        # The raw column set present at the first (training) call to
        # create_null_count_features, frozen and persisted. A serving request
        # carries only its non-null raw columns, so `null_count_total` /
        # `null_ratio` would undercount by every column the client omitted;
        # the serving transform reindexes the frame to this list (absent
        # columns → NaN) before null-counting so the feature means the same
        # thing it did in training. None for artifacts saved before this.
        self._null_count_cols: Optional[List[str]] = None
        # Per-card1 sufficient statistics for the expanding card aggregates,
        # carried between splits and persisted for serving (ADR-002 §2.3/§6).
        self._card_agg_state: Dict[Any, CardAggregateState] = {}
        # ADR-003: per-card trailing-24h (dt, amount) history backing the P9-4
        # rolling-window features. Unlike the five scalars above this is not
        # constant space, but it is bounded by one day of a card's activity.
        self._card_window_state: Dict[Any, CardWindowState] = {}
        # PRD Phase 9 (P9-2): train-fitted per-UID aggregate lookup maps, where
        # UID = card1_addr1_D1n (the Kaggle client identifier for this exact
        # dataset). Static statistics computed once on train and mapped onto
        # val/test/serving — an unseen UID (a new card, or a known card on a
        # new day) falls back to the global value, which is the intended
        # "no client history yet" signal. Same fit/apply shape as
        # `_card_hash_freq`; NOT an expanding window, so no ADR-002 per-entity
        # state is needed. `UID` itself never enters the model.
        self._uid_maps: Dict[str, Dict[Any, float]] = {}
        self._uid_global: Dict[str, float] = {}
        # PRD Phase 9 (P9-4): top-quintile `dist1` cut, fitted on train W rows
        # only and reused unchanged everywhere else. None until a fit or a load
        # provides it, in which case `dist1_high` degrades to all-zero rather
        # than inventing a split point from the frame in hand.
        self._dist1_high_threshold: Optional[float] = None

    @property
    def card_agg_state(self) -> Dict[Any, "CardAggregateState"]:
        """Read-only snapshot of the carried per-card accumulators.

        A copy, so callers (e.g. the serving state store's seed) cannot alias
        and mutate the engineer's internals.
        """
        return dict(self._card_agg_state)

    @property
    def card_window_state(self) -> Dict[Any, "CardWindowState"]:
        """Read-only snapshot of the carried per-card trailing windows (ADR-003).

        A copy, for the same reason `card_agg_state` is: the serving state
        store's seed must not alias the engineer's internals.
        """
        return dict(self._card_window_state)

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

        logger.debug(
            "Created temporal features: hour_of_day, day_of_week, cyclical sin/cos"
        )
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

        df["amount_decimal_len"] = (
            df["TransactionAmt"].apply(_decimal_len).astype(np.int8)
        )

        logger.debug(
            "Created amount features: amount_log, amount_cents, amount_is_round, amount_decimal_len"
        )
        return df

    # ─── 3. Card-Level Aggregates ────────────────────────────────────────────

    def create_card_aggregates(
        self,
        df: pd.DataFrame,
        update_state: bool = False,
        card_state: Optional[Dict[Any, "CardAggregateState"]] = None,
    ) -> pd.DataFrame:
        """
        Compute card-level rolling/expanding statistics using card1 as proxy.
        Fully vectorized to guarantee index alignment and high performance.

        Carried state (ADR-002 §2.3): when `self._card_agg_state` holds history
        for a card — seeded from a previous frame or restored by
        `load_transformers` — this frame's expanding totals CONTINUE from it
        rather than restarting at zero. Because the five retained scalars are
        sufficient statistics for every feature below, a frame transformed with
        carried state is bit-identical to the same rows transformed as part of
        one long batch pass. With no carried state the behaviour is unchanged.

        Args:
            df: Frame to transform. Sorted temporally here if `TransactionDT`
                is present, matching the batch pipeline.
            update_state: Fold this frame's transactions into the carried
                accumulators AFTER computing features, so a later call sees
                this history. Leave False at inference: the read-then-write
                ordering that keeps a transaction out of its own z-score is
                the caller's to sequence (ADR-002 §5.2), and transforms must
                not mutate fitted state.
            card_state: Carried accumulators to read INSTEAD of
                `self._card_agg_state`. Serving passes the one card it is
                scoring, so a request never mutates — or even reads — shared
                instance state. Without this the serving path would have to
                swap the attribute in place, which races across FastAPI's sync
                threadpool: two concurrent requests for different cards would
                clobber each other's scoped state and silently produce wrong
                aggregates. `None` means "use the instance's own state", which
                is what the batch pipeline wants.

        Raises:
            ValueError: `update_state` and `card_state` are combined —
                `update_state` writes to `self._card_agg_state`, which is not
                the mapping that was read, so the result would be incoherent.
        """
        if update_state and card_state is not None:
            raise ValueError(
                "create_card_aggregates: update_state=True cannot be combined "
                "with an explicit card_state — the write target "
                "(self._card_agg_state) would differ from the mapping read. "
                "Call update_card_aggregate_state() explicitly instead."
            )
        df = df.copy()

        if "TransactionDT" in df.columns:
            df = df.sort_values("TransactionDT", ascending=True).reset_index(drop=True)

        carried_n, carried_sum, carried_sum_sq, carried_max = self._carried_card_totals(
            df["card1"], self._card_agg_state if card_state is None else card_state
        )

        # cumcount/cumsum are exclusive of the current row by construction;
        # adding the carried totals extends that same expanding window
        # backwards over history this frame does not contain.
        df["tx_count_per_card"] = df.groupby("card1").cumcount() + carried_n

        df["tx_sum_per_card"] = (
            df.groupby("card1")["TransactionAmt"].cumsum()
            - df["TransactionAmt"]
            + carried_sum
        )

        # Mean excluding current row
        df["mean_amount_per_card"] = np.where(
            df["tx_count_per_card"] > 0,
            df["tx_sum_per_card"] / df["tx_count_per_card"],
            0.0,
        )

        # Max amount per card (using shift to exclude current row). np.fmax
        # ignores NaN, so a card's first row in this frame takes the carried
        # max when there is one and falls through to the 0.0 fill when there
        # is not — matching the batch first-transaction value.
        shifted_amt = df.groupby("card1")["TransactionAmt"].shift(1)
        exclusive_max = (
            df.assign(_shifted=shifted_amt).groupby("card1")["_shifted"].cummax()
        )
        df["max_amount_per_card"] = np.fmax(
            exclusive_max.to_numpy(dtype=float), carried_max
        )
        df["max_amount_per_card"] = np.nan_to_num(df["max_amount_per_card"], nan=0.0)

        # Standard deviation via rolling variance formula
        shifted_amt_sq = (df["TransactionAmt"] ** 2).groupby(df["card1"]).shift(1)
        sum_sq = (
            df.assign(_shifted_sq=shifted_amt_sq)
            .groupby("card1")["_shifted_sq"]
            .cumsum()
            .fillna(0.0)
            + carried_sum_sq
        )

        var = np.where(
            df["tx_count_per_card"] > 1,
            (sum_sq - (df["tx_sum_per_card"] ** 2) / df["tx_count_per_card"])
            / (df["tx_count_per_card"] - 1),
            0.0,
        )
        df["std_amount_per_card"] = np.sqrt(np.maximum(var, 0.0))

        # Anomaly signals
        df["amount_vs_mean_ratio"] = np.where(
            df["mean_amount_per_card"] > 0,
            df["TransactionAmt"] / df["mean_amount_per_card"],
            1.0,
        )

        df["amount_zscore_per_card"] = np.where(
            df["std_amount_per_card"] > 0,
            (df["TransactionAmt"] - df["mean_amount_per_card"])
            / (df["std_amount_per_card"] + 1e-6),
            0.0,
        )

        if update_state:
            self.update_card_aggregate_state(df)

        logger.debug("Created card aggregate features (vectorized)")
        return df

    @staticmethod
    def _carried_card_totals(
        keys: pd.Series, state: Dict[Any, "CardAggregateState"]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-row (n, sum, sum_sq, max) history carried in for each row's card.

        `max` is returned as NaN where a card has no history, so `np.fmax` can
        treat "no carried max" as "no constraint" rather than as a real 0.0.

        One `map` pass, not four: the lookup is a Python-level callable per row,
        so four separate passes cost four times the interpreter overhead over
        the full training frame.
        """
        n = len(keys)
        if not state:
            zeros = np.zeros(n, dtype=float)
            return zeros, zeros.copy(), zeros.copy(), np.full(n, np.nan)

        empty = (0.0, 0.0, 0.0, np.nan)
        rows = keys.map(
            lambda k: (
                (state[k].n, state[k].sum_amt, state[k].sum_amt_sq, state[k].max_amt)
                if k in state
                else empty
            )
        )
        totals = np.asarray(list(rows), dtype=float).reshape(n, CARD_STATE_SCALAR_COUNT)
        return totals[:, 0], totals[:, 1], totals[:, 2], totals[:, 3]

    def update_card_aggregate_state(self, df: pd.DataFrame) -> None:
        """Fold a frame's transactions into the carried per-card accumulators.

        Separated from `create_card_aggregates` so the batch pipeline can
        derive every causal feature from one consistent pre-frame snapshot and
        advance the state exactly once at the end — otherwise
        `create_velocity_features`, which reads `last_dt` from this same state,
        would see a `last_dt` already advanced to this frame's own maximum and
        compute every card's first gap against the wrong reference.
        """
        if "card1" not in df.columns or "TransactionAmt" not in df.columns:
            return

        has_dt = "TransactionDT" in df.columns
        ordered = df.sort_values("TransactionDT") if has_dt else df

        state = dict(self._card_agg_state)
        for card, amount, dt in zip(
            ordered["card1"].to_numpy(),
            ordered["TransactionAmt"].to_numpy(dtype=float),
            (
                ordered["TransactionDT"].to_numpy(dtype=float)
                if has_dt
                else np.full(len(ordered), np.nan)
            ),
        ):
            state[card] = state.get(card, CardAggregateState()).observe(amount, dt)
        self._card_agg_state = state

        # ADR-003: advance the trailing-window history in the same pass, so the
        # two states never disagree about what a card has done. Skipped without
        # timestamps, since a trailing window is meaningless then.
        if not has_dt:
            return
        window = dict(self._card_window_state)
        for card, amount, dt in zip(
            ordered["card1"].to_numpy(),
            ordered["TransactionAmt"].to_numpy(dtype=float),
            ordered["TransactionDT"].to_numpy(dtype=float),
        ):
            window[card] = window.get(card, CardWindowState()).observe(amount, dt)
        self._card_window_state = window

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
            df["interaction_hour_zscore"] = (
                df["hour_of_day"] * df["amount_zscore_per_card"]
            )

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
            df["email_both_present"] = (p_email.notna() & r_email.notna()).astype(
                np.int8
            )

        # .astype(str) before .str: a single-row serving frame whose email
        # column is all-NaN infers a float dtype, and the .str accessor raises
        # "Can only use .str accessor with string values" on it. Batch frames
        # are wide enough that this never surfaced there. Casting first is a
        # no-op for real string columns, so the batch values are unchanged.
        if p_email is not None:
            df["P_email_is_free"] = (
                p_email.fillna("").astype(str).str.lower().isin(FREE_EMAIL_DOMAINS)
            ).astype(np.int8)

        if r_email is not None:
            df["R_email_is_free"] = (
                r_email.fillna("").astype(str).str.lower().isin(FREE_EMAIL_DOMAINS)
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

    def create_card_hash_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
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
            logger.debug(
                f"Created card_hash_freq from {len(present_cols)} card columns"
            )
        else:
            logger.warning("Insufficient card columns for hash feature")

        return df

    # ─── 7.5 UID client-identifier aggregates (PRD Phase 9, P9-2) ────────────

    # UID = card1 + "_" + addr1 + "_" + floor(day - D1), day = TransactionDT/86400.
    # The Kaggle 1st-place client identifier for IEEE-CIS. `D1n = floor(day - D1)`
    # is (approximately) the account's first-seen day, so the triple keys a
    # card+address+account-origin cohort. Aggregates on it, NOT the id itself,
    # per the PRD (feeding the id overfits to an identifier absent at serving
    # time for a new card).
    # Minimum train rows a UID must have before its amount/D1 aggregates are
    # kept — below this the "cohort" is one or two transactions and the stat
    # is self-referential (uid_amt_mean == the row's own amount, std == 0),
    # a property of this train partition's UID multiplicity, not a signal
    # (mle-reviewer P9-2 M2). Small UIDs fall through to the global fallback;
    # `uid_count` is emitted so the tree can still condition on cohort size.
    _UID_MIN_COUNT = 3
    _UID_MISSING = "na"

    def _compute_uid(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Build the UID string series, or None if a required column is absent.

        `card1` is non-null throughout IEEE-CIS; `addr1` and `D1` are not.
        Their missing values get DISTINCT tokens (`addr1` NaN → "naA", `D1`
        NaN → "naD") rather than a shared sentinel, so a card missing only
        its address is not merged with the fully-missing rows into one
        spuriously frequent pseudo-UID (mle-reviewer P9-2 M3).
        """
        needed = ("card1", "addr1", "D1", "TransactionDT")
        if any(c not in df.columns for c in needed):
            logger.warning(
                "create_uid_features: missing one of %s — skipping UID aggregates.",
                needed,
            )
            return None

        day = df["TransactionDT"].astype(float) / 86400.0
        d1 = pd.to_numeric(df["D1"], errors="coerce")
        d1n = np.floor(day - d1)
        d1n_str = np.where(d1n.isna(), "naD", d1n.fillna(0).astype("int64").astype(str))

        card1_str = (
            pd.to_numeric(df["card1"], errors="coerce")
            .fillna(-1)
            .astype("int64")
            .astype(str)
        )
        addr1 = pd.to_numeric(df["addr1"], errors="coerce")
        addr1_str = np.where(
            addr1.isna(), "naA", addr1.fillna(0).astype("int64").astype(str)
        )

        return pd.Series(
            card1_str.to_numpy() + "_" + addr1_str + "_" + d1n_str, index=df.index
        )

    def create_uid_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Per-UID aggregate features (PRD Phase 9, P9-2).

        UID = ``card1_addr1_D1n`` (the Kaggle client identifier). Emits, never
        the UID itself:

          - ``uid_count``: number of train rows on this UID (0 if unseen). The
            explicit cohort-size feature — lets the tree condition on "seen
            once" directly instead of inferring it from ``uid_amt_std == 0``.
          - ``uid_freq``: ``uid_count`` normalised by the train row count.
            Unseen → 0.0.
          - ``uid_amt_mean`` / ``uid_amt_std`` / ``uid_d1_mean``:
            `TransactionAmt` / `D1` statistics over the UID's train rows —
            **only kept for UIDs with at least ``_UID_MIN_COUNT`` train
            rows**. Smaller cohorts (and unseen UIDs) get the global train
            fallback, so a self-referential one-row "mean" never reaches the
            model (mle-reviewer P9-2 M2).
          - ``uid_amt_ratio``: this row's `TransactionAmt` ÷ its
            ``uid_amt_mean`` **only when the UID is a real seen cohort
            (≥ ``_UID_MIN_COUNT``)**; otherwise **1.0** (neutral, matching the
            per-card ``amount_vs_mean_ratio`` no-history convention). It is not
            a rescaled amount for the unseen-dominant case.

        fit=True fits the maps on `df` (train). fit=False maps the pre-fitted
        values onto val/test/serving — same fit/apply contract as
        `create_card_hash_features`, no cross-row dependency, safe for a
        single-row serving frame.

        Self-inclusion: for a kept UID (≥ ``_UID_MIN_COUNT`` rows) a train
        row's own amount is one of ≥ 3 in its aggregate, so the stat is not
        purely self-referential; val/test/serving rows are never in the
        train-fitted map. This is the same shape as `card_hash_freq`, with a
        count floor the Kaggle full-data aggregates did not need (their UID
        was coarser; the ``D1n`` term here makes UIDs near-per-transaction).
        """
        df = df.copy()
        uid = self._compute_uid(df)
        if uid is None:
            return df

        amt = pd.to_numeric(df.get("TransactionAmt"), errors="coerce")
        d1 = pd.to_numeric(df.get("D1"), errors="coerce")

        if fit:
            n = len(uid)
            counts = uid.value_counts()
            big = counts[counts >= self._UID_MIN_COUNT].index
            amt_big = amt[uid.isin(big)]
            d1_big = d1[uid.isin(big)]
            uid_big = uid[uid.isin(big)]
            self._uid_maps = {
                "uid_count": counts.to_dict(),
                "uid_amt_mean": amt_big.groupby(uid_big).mean().dropna().to_dict(),
                "uid_amt_std": amt_big.groupby(uid_big).std(ddof=0).dropna().to_dict(),
                "uid_d1_mean": d1_big.groupby(uid_big).mean().dropna().to_dict(),
            }
            self._uid_global = {
                "train_rows": int(n),
                "uid_amt_mean": float(amt.mean()) if n else 0.0,
                "uid_amt_std": float(amt.std(ddof=0)) if n else 0.0,
                "uid_d1_mean": float(d1.mean()) if d1.notna().any() else -1.0,
            }
            logger.info(
                "create_uid_features fit: %d distinct UIDs over %d train rows "
                "(%d with >= %d rows kept for amount/D1 aggregates)",
                len(counts),
                n,
                len(big),
                self._UID_MIN_COUNT,
            )

        maps = self._uid_maps
        train_rows = max(1, int(self._uid_global.get("train_rows", 1)))
        uid_count = uid.map(maps.get("uid_count", {})).fillna(0.0)
        df["uid_count"] = uid_count
        df["uid_freq"] = uid_count / train_rows
        df["uid_amt_mean"] = uid.map(maps.get("uid_amt_mean", {})).fillna(
            self._uid_global.get("uid_amt_mean", 0.0)
        )
        df["uid_amt_std"] = uid.map(maps.get("uid_amt_std", {})).fillna(
            self._uid_global.get("uid_amt_std", 0.0)
        )
        df["uid_d1_mean"] = uid.map(maps.get("uid_d1_mean", {})).fillna(
            self._uid_global.get("uid_d1_mean", -1.0)
        )
        # Ratio only for a real seen cohort; 1.0 (neutral) otherwise.
        seen_cohort = uid_count.to_numpy() >= self._UID_MIN_COUNT
        df["uid_amt_ratio"] = np.where(
            seen_cohort & (df["uid_amt_mean"].to_numpy() > 0),
            amt.to_numpy(dtype=float) / df["uid_amt_mean"].to_numpy(),
            1.0,
        )

        logger.debug("Created UID aggregate features (fit=%s)", fit)
        return df

    # ─── 8. Transaction Velocity Features ────────────────────────────────────

    def create_velocity_features(
        self,
        df: pd.DataFrame,
        card_state: Optional[Dict[Any, "CardAggregateState"]] = None,
        card_window: Optional[Dict[Any, "CardWindowState"]] = None,
    ) -> pd.DataFrame:
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
            logger.warning(
                "TransactionDT or card1 missing — skipping velocity features"
            )
            return df

        # Ensure temporal sort
        df = df.sort_values("TransactionDT", ascending=True).reset_index(drop=True)

        # Time since last transaction on same card. A card's first row in
        # this frame has no in-frame predecessor (diff is NaN); ADR-002 §2.3
        # carries that predecessor's timestamp in the per-card accumulator, so
        # fall back to it before the no-history sentinel. Cards with neither
        # keep the batch's -1.0, which is what a first transaction gets in
        # training too (ADR-002 §5.4).
        df["time_since_last_tx"] = df.groupby("card1")["TransactionDT"].diff()

        carried_last_dt = self._carried_last_dt(
            df["card1"], self._card_agg_state if card_state is None else card_state
        )
        gap_from_carried = df["TransactionDT"].to_numpy(dtype=float) - carried_last_dt
        df["time_since_last_tx"] = df["time_since_last_tx"].fillna(
            pd.Series(gap_from_carried, index=df.index)
        )
        df["time_since_last_tx"] = df["time_since_last_tx"].fillna(-1.0)

        # Log-transform of time delta (very right-skewed)
        df["time_since_last_tx_log"] = np.where(
            df["time_since_last_tx"] > 0, np.log1p(df["time_since_last_tx"]), -1.0
        )

        # PRD Phase 9 (P9-4): fixed trailing-window counts per card. Distinct
        # from `tx_count_per_card`, which is an EXPANDING window from the card's
        # first seen transaction: a card with 200 transactions over six months
        # and one with 20 in the last hour can share an expanding count while
        # only the second reads as a burst.
        df = self._add_rolling_window_features(
            df, self._card_window_state if card_window is None else card_window
        )

        logger.debug(
            "Created velocity features: time_since_last_tx, time_since_last_tx_log, %s",
            ", ".join(ROLLING_WINDOW_FEATURES),
        )
        return df

    @staticmethod
    def _add_rolling_window_features(
        df: pd.DataFrame,
        window_state: Optional[Dict[Any, "CardWindowState"]] = None,
    ) -> pd.DataFrame:
        """
        Fixed trailing-window transaction counts and amount deviation per card.

        For each row, counts the card's STRICTLY PRIOR transactions whose
        timestamp falls in `[t - span, t)`. Computed with a per-card
        `searchsorted` over that card's own sorted timestamps: O(n log n)
        rather than the O(n^2) a naive per-row scan costs, and exact rather
        than the approximation an index-based `rolling` window gives when
        transactions are unevenly spaced in time.

        The current row is excluded by construction — the window's right edge
        is the row's own position in its card's history, never position + 1.

        Serving note (ADR-002): unlike the expanding aggregates, these are NOT
        reproducible from the five O(1) scalars in `CardAggregateState` — a
        trailing window needs the timestamps themselves. Measured on the raw
        training data (2026-09-08) a 24h window holds a mean of 21 prior
        transactions per row, p99 359, max 654, so the existing 10-slot
        sequence ring buffer (ADR-002 §2.5) is far too small to back these.
        Serving therefore needs a per-card timestamp/amount deque bounded by
        the longest span, which is an ADR-002 amendment rather than a free ride
        on state that already exists.
        """
        if "TransactionDT" not in df.columns or "card1" not in df.columns:
            return df

        n = len(df)
        counts = {name: np.zeros(n, dtype=np.int32) for name in ROLLING_WINDOW_COUNTS}
        amt_24h_mean = np.zeros(n, dtype=float)
        has_amt = "TransactionAmt" in df.columns

        dt_all = df["TransactionDT"].to_numpy(dtype=float)
        amt_all = (
            df["TransactionAmt"].to_numpy(dtype=float)
            if has_amt
            else np.zeros(n, dtype=float)
        )

        for card, positions in df.groupby("card1", sort=False).indices.items():
            # `positions` is ascending and the frame is sorted by TransactionDT
            # above, so this card's timestamps are sorted too.
            own_dt = dt_all[positions]
            own_amt = amt_all[positions]

            # ADR-003: prepend whatever this card carries from earlier frames
            # (or, at serving time, from the state store) so a trailing window
            # spans history this frame does not contain. Same role the carried
            # scalars play for the expanding aggregates in
            # create_card_aggregates. Empty for the batch pipeline's first
            # pass, which reduces this to the plain within-frame computation.
            carried = (
                window_state.get(card, CardWindowState()).entries
                if window_state
                else ()
            )
            if carried:
                carried_dt = np.fromiter(
                    (e[0] for e in carried), dtype=float, count=len(carried)
                )
                carried_amt = np.fromiter(
                    (e[1] for e in carried), dtype=float, count=len(carried)
                )
                dt = np.concatenate((carried_dt, own_dt))
                amt = np.concatenate((carried_amt, own_amt))
            else:
                dt = own_dt
                amt = own_amt

            offset = len(dt) - len(own_dt)
            # Positions of this frame's own rows within the combined history.
            idx = np.arange(offset, len(dt))

            # Index of the first transaction still inside each window. side is
            # "left" so a transaction exactly `span` seconds old counts as
            # inside — the boundary is half-open, [t - span, t).
            for name, span in ROLLING_WINDOW_COUNTS.items():
                left = np.searchsorted(dt, own_dt - span, side="left")
                counts[name][positions] = idx - left

            if has_amt:
                # Prefix sums make each window's mean a single subtraction.
                prefix = np.concatenate(([0.0], np.cumsum(amt)))
                left24 = np.searchsorted(dt, own_dt - WINDOW_24H, side="left")
                window_n = idx - left24
                window_sum = prefix[idx] - prefix[left24]
                amt_24h_mean[positions] = np.divide(
                    window_sum,
                    window_n,
                    out=np.zeros_like(window_sum),
                    where=window_n > 0,
                )

        for name, values in counts.items():
            df[name] = values

        if has_amt:
            df["amt_24h_mean_per_card"] = amt_24h_mean
            # Short-window vs long-window comparison: how far the card's recent
            # normal sits from its lifetime normal. `mean_amount_per_card` is
            # the expanding mean from create_card_aggregates; once a card has
            # enough history that mean cannot express a recent spike, which is
            # what this ratio recovers. Neutral 1.0 wherever either side is
            # absent, matching the `amount_vs_mean_ratio` convention.
            long_mean = (
                df["mean_amount_per_card"].to_numpy(dtype=float)
                if "mean_amount_per_card" in df.columns
                else np.zeros(n, dtype=float)
            )
            safe_long = np.where(long_mean > 0, long_mean, 1.0)
            df["amt_24h_vs_card_mean_ratio"] = np.where(
                (long_mean > 0) & (amt_24h_mean > 0),
                amt_24h_mean / safe_long,
                1.0,
            )

        return df

    @staticmethod
    def _carried_last_dt(
        keys: pd.Series, state: Dict[Any, "CardAggregateState"]
    ) -> np.ndarray:
        """Per-row last-seen `TransactionDT` for each row's card, NaN when the
        card has no carried history (so the -1.0 sentinel fill still applies).
        """
        if not state:
            return np.full(len(keys), np.nan)
        return keys.map(lambda k: state[k].last_dt if k in state else np.nan).to_numpy(
            dtype=float
        )

    # ─── 8b. dist1 Signal Features (PRD Phase 9, P9-4) ───────────────────────

    def create_dist_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        `dist1` value-gradient features, scoped to `ProductCD == "W"`.

        Verified against `data/raw/train_transaction.csv` on 2026-09-08:
        `dist1` is populated for `ProductCD == "W"` ONLY (54.2% within W, 0%
        for C/H/R/S). Its apparent overall presence-to-fraud correlation is
        confounded by `ProductCD` — product C alone runs 11.7% fraud and never
        carries `dist1` — and the correlation vanishes once measured inside W
        (1.996% fraud when present vs 2.092% absent). So there is deliberately
        no `dist1_present` feature here: the presence signal is a product-mix
        artifact, not a per-product one.

        What does survive within W is a value gradient: fraud is flat at
        ~1.6-1.8% across the bottom three quintiles and roughly doubles in the
        top one (2.95% above `dist1 > 36`). Hence two features:

        - `dist1_log`  — `log1p(dist1)`, the gradient itself (heavily skewed:
          the W range runs 0 to 10,286).
        - `dist1_high` — top-quintile flag, cut fitted on TRAIN ONLY.

        Non-W rows take `DIST1_MISSING_FILL` / 0 rather than a value derived
        from a null, since there is nothing to condition on there. This is
        additive to the existing `dist_present` from
        `create_null_count_features`, not a replacement.

        Args:
            df: Frame to transform.
            fit: Fit the top-quintile cut on this frame (train only). With
                `fit=False` the stored cut is reused unchanged — re-fitting per
                split would leak the holdout's own distribution into the flag.
        """
        if "dist1" not in df.columns or "ProductCD" not in df.columns:
            logger.warning(
                "dist1 or ProductCD missing — skipping dist1 features "
                "(dist1 is a ProductCD=='W'-only column)"
            )
            return df

        df = df.copy()
        is_w = df["ProductCD"] == DIST1_PRODUCT
        dist1 = pd.to_numeric(df["dist1"], errors="coerce")
        # Only W rows can contribute a value; a non-null dist1 on a non-W row
        # would be off-contract, and is ignored rather than silently trusted.
        scoped = dist1.where(is_w)

        if fit:
            observed = scoped.dropna()
            self._dist1_high_threshold = (
                float(observed.quantile(DIST1_HIGH_QUANTILE))
                if not observed.empty
                else None
            )
            logger.info(
                "Fitted dist1 top-quintile cut on %d W rows: %s",
                len(observed),
                self._dist1_high_threshold,
            )

        df["dist1_log"] = np.where(
            scoped.notna(), np.log1p(scoped.fillna(0.0)), DIST1_MISSING_FILL
        )

        cut = self._dist1_high_threshold
        df["dist1_high"] = (
            np.where(scoped.notna() & (scoped >= cut), 1, 0).astype(np.int8)
            if cut is not None
            else np.zeros(len(df), dtype=np.int8)
        )

        logger.debug(
            "Created dist1 features (cut=%s, W rows=%d)", cut, int(is_w.sum())
        )
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
        if self._null_count_cols is None:
            # Exclude the two columns this method itself appends, plus the
            # split-id helper preprocess.py adds — none are part of the raw
            # schema a serving request would carry.
            self._null_count_cols = [
                c
                for c in columns
                if c not in ("null_count_total", "null_ratio", "_split_row_id")
            ]

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
            df.loc[
                id_31.str.contains("safari") & ~id_31.str.contains("chrome|crios|edge"),
                "browser_family",
            ] = "safari"
            df.loc[id_31.str.contains("firefox"), "browser_family"] = "firefox"
            df.loc[id_31.str.contains("edge|ie|rv:11"), "browser_family"] = "edge_ie"
            df.loc[id_31.str.contains("samsung"), "browser_family"] = "samsung"

            # Outdated/rare browsers often correlate with fraud/bots
            df["browser_is_mobile"] = id_31.str.contains("mobile|android|ios").astype(
                int
            )

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
                    df["screen_height"] > 0, df["screen_width"] / df["screen_height"], 0
                )
            else:
                df["screen_width"] = 0
                df["screen_height"] = 0
                df["screen_aspect_ratio"] = 0

        logger.debug("Created device and browser parsed features")
        return df

    # ─── 5. Expanding Target Encoding ──────────────────────────────────────────

    def create_target_encoding(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        update_state: bool = False,
        time_col: Optional[str] = None,
        label_lag_seconds: float = 0.0,
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

        Label latency (finding F6, 2026-08-19 metrics audit): the two guards
        above prevent LABEL leakage but not LATENCY leakage — a served
        transaction cannot actually benefit from same-window history the way
        this same-window expanding encoding implicitly assumes, because
        confirmed fraud labels are not available instantly in production.
        When `label_lag_seconds > 0`, a row's cumulative sum/count include
        only OTHER rows in the same frame whose own `time_col` is at least
        `label_lag_seconds` earlier — i.e. old enough to plausibly be
        labeled by the time this row is scored. `label_lag_seconds=0.0`
        (the default) preserves the original same-window behavior exactly,
        so every existing call site is unaffected unless it opts in.

        Carried state folded in from an earlier split is NOT re-lagged here:
        this pipeline's splits are weeks-to-months apart (see
        `_assign_split_ids`'s time-based split), which safely exceeds any
        realistic `label_lag_seconds`, so treating carried totals as fully
        known introduces no additional leakage.

        Args:
            df: Temporally ordered frame containing the target column.
            fit: True on the train slice — estimates the prior and initialises
                per-entity state from this frame alone. False on val/test —
                reuses the fitted prior and seeds counters from carried state.
            update_state: Fold this frame's entity totals into the carried
                state so a later fit=False call sees this history. Implied by
                fit=True. Leave False at inference, where transforms must not
                mutate fitted state.
            time_col: Column holding each row's own timestamp (e.g.
                "TransactionDT"). Required (non-None) when
                `label_lag_seconds > 0`; unused otherwise.
            label_lag_seconds: Minimum time gap, in the same units as
                `time_col`, that must separate a contributing row's own
                timestamp from the row being encoded. 0.0 (default) disables
                lag-adjustment.

        Returns:
            A new DataFrame with one `<col>_target_enc` column per entity column.

        Raises:
            ValueError: `label_lag_seconds > 0` but `time_col` is None, or
                `time_col` is given but not present in `df`.
        """
        df = df.copy()

        target_col = "isFraud"
        if target_col not in df.columns:
            logger.warning(
                f"Target column '{target_col}' not found. Skipping target encoding."
            )
            return df

        if label_lag_seconds > 0:
            if time_col is None:
                raise ValueError(
                    "create_target_encoding: label_lag_seconds > 0 requires "
                    "time_col to be set — cannot lag-adjust the encoding "
                    "without each row's own timestamp."
                )
            if time_col not in df.columns:
                raise ValueError(
                    f"create_target_encoding: time_col={time_col!r} not found in df."
                )

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

            if label_lag_seconds > 0:
                cum_sum, cum_count = self._lagged_expanding_totals(
                    df, col, target_col, time_col, label_lag_seconds
                )
                cum_sum = cum_sum + carried_sum
                cum_count = cum_count + carried_count
            else:
                # dropna=False keeps a missing entity key as its own group,
                # matching _accumulate_entity_totals. Under the default
                # (dropna=True) those rows get a NaN cumcount, which makes
                # the cum_count == 0 prior fallback unreachable and emits a
                # NaN feature — and target encoding runs before imputation,
                # so raw NaN keys do reach this point.
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
            f"Created expanding target encoding (fit={fit}, "
            f"label_lag_seconds={label_lag_seconds}) for: {present_cols}"
        )
        return df

    @staticmethod
    def _lagged_expanding_totals(
        df: pd.DataFrame,
        col: str,
        target_col: str,
        time_col: str,
        lag_seconds: float,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Per-entity expanding (sum, count) of `target_col`, counting only
        rows whose OWN `time_col` is at least `lag_seconds` earlier than the
        row being scored (see `create_target_encoding`'s label-latency note).

        Requires `df` sorted ascending by `time_col` — the whole
        preprocessing pipeline sorts temporally before this runs
        (`DataLoader.sort_temporal`), and each entity's rows inherit that
        same order. Because a group's rows are time-sorted, the set of
        "eligible" (old enough) prior rows for row i is always a *prefix*
        of that group's rows, found via `np.searchsorted` in
        O(log group_size) rather than a per-row scan — this is what keeps
        the lag-aware path from being asymptotically worse than the plain
        cumsum path it replaces.
        """
        out_sum = pd.Series(0.0, index=df.index)
        out_count = pd.Series(0.0, index=df.index)

        for _, group in df.groupby(col, dropna=False, sort=False):
            idx = group.index
            times = group[time_col].to_numpy()
            labels = group[target_col].to_numpy(dtype=float)

            # searchsorted(..., side="right") on ascending `times` returns,
            # for each cutoff, the count of entries <= that cutoff — i.e.
            # exactly the number of eligible prior rows. Since
            # cutoff = own_time - lag < own_time for lag > 0, a row's own
            # position (value == own_time) can never be counted, so
            # self-exclusion holds automatically, same as the unlagged path.
            cutoffs = times - lag_seconds
            eligible_count = np.searchsorted(times, cutoffs, side="right")
            cum_labels = np.concatenate(([0.0], np.cumsum(labels)))

            out_sum.loc[idx] = cum_labels[eligible_count]
            out_count.loc[idx] = eligible_count.astype(float)

        return out_sum, out_count

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
            updated[key] = (
                prev_sum + float(row["sum"]),
                prev_count + float(row["count"]),
            )
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
        high_card_cols = [
            "P_emaildomain",
            "R_emaildomain",
            "id_31",
            "id_33",
            "DeviceInfo",
        ]
        low_card_cols = [
            c
            for c in (cat_cols or CATEGORICAL_COLS)
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

        if fit:
            v_cols = [c for c in V_FEATURE_COLS if c in df.columns]
        else:
            if self._pca is None:
                raise RuntimeError(
                    "PCA not fitted. Call reduce_v_features(fit=True) on training data first."
                )
            # Serving path: a single-transaction request only carries the
            # V-columns that were non-null for that row, so the frame is
            # reindexed to the EXACT list the PCA was fitted on (missing
            # columns added as NaN, then zero-filled below by na_value=0.0 —
            # identical to the fit). Falls back to the historical
            # intersection behaviour for artifacts saved before this list
            # was persisted.
            if self._pca_fit_cols is not None:
                v_cols = list(self._pca_fit_cols)
            else:
                # Artifact predates pca_fit_cols. The batch fit used
                # `V_FEATURE_COLS ∩ df.columns` on a frame that had every
                # V-column, so when the fitted PCA's input width equals the
                # full V range, that intersection was exactly V_FEATURE_COLS.
                n_in = getattr(self._pca, "n_features_in_", None)
                if n_in == len(V_FEATURE_COLS):
                    v_cols = list(V_FEATURE_COLS)
                else:
                    v_cols = [c for c in V_FEATURE_COLS if c in df.columns]
            missing = [c for c in v_cols if c not in df.columns]
            if missing:
                df = df.reindex(columns=[*df.columns, *missing])

        if not v_cols:
            logger.warning("No V-feature columns found — skipping PCA reduction.")
            return df

        n_rows = len(df)

        if fit:
            self._pca_fit_cols = list(v_cols)
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
                batch = df.iloc[batch_positions][v_cols].to_numpy(
                    dtype=np.float32, na_value=0.0
                )
                self._pca.partial_fit(batch)
                del batch
                gc.collect()

            variance_retained = self._pca.explained_variance_ratio_.sum()
            logger.info(f"PCA fit complete. Variance retained: {variance_retained:.1%}")

        # Transform every row, including those excluded from the fit
        reduced_chunks: list = []
        for start in range(0, n_rows, PCA_BATCH_SIZE):
            batch = df.iloc[start : start + PCA_BATCH_SIZE][v_cols].to_numpy(
                dtype=np.float32, na_value=0.0
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

        logger.info(
            f"V-feature PCA complete: {len(v_cols)} cols → {len(pca_cols)} PCA components"
        )
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

    def handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
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
        and per-entity counters, null_ratio denominator, and the per-card
        aggregate accumulators ADR-002 §6 requires for serving).
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
            # ADR-002 §6: without these, every card cold-starts at serving
            # time and nine features silently lose their history.
            "card_agg_state": self._card_agg_state,
            # The V-column list the PCA was fitted on, so a serving request
            # carrying only its non-null V-columns can be reindexed to the
            # fitted schema before transform (see reduce_v_features).
            "pca_fit_cols": self._pca_fit_cols,
            # The raw column set null-counting saw in training, so a serving
            # request can be reindexed to it before create_null_count_features
            # and produce the same null_count_total / null_ratio.
            "null_count_cols": self._null_count_cols,
            # PRD Phase 9 P9-2: train-fitted per-UID aggregate maps + their
            # global fallbacks, so serving reproduces uid_* exactly.
            "uid_maps": self._uid_maps,
            "uid_global": self._uid_global,
            # PRD Phase 9 P9-4: the train-fitted dist1 top-quintile cut. Without
            # it serving would re-derive a different `dist1_high` than training
            # used, or silently emit all-zero.
            "dist1_high_threshold": self._dist1_high_threshold,
            # ADR-003 §4.5: without this a card cold-starts its trailing window
            # at zero while its expanding aggregates continue from the snapshot
            # above — an inconsistent pairing (tx_count_per_card in the
            # thousands beside tx_count_24h_per_card == 0) that the training
            # distribution never contains, and so worse than a clean start.
            "card_window_state": self._card_window_state,
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
            # Absent in artifacts written before ADR-002. Loading must still
            # succeed — the consequence is cold-start card aggregates, which
            # is a state the model saw in training, not a corrupt artifact.
            self._card_agg_state = state.get("card_agg_state", {})
            # Absent in artifacts written before this list was persisted;
            # reduce_v_features(fit=False) then falls back to the historical
            # V_FEATURE_COLS ∩ df.columns behaviour.
            self._pca_fit_cols = state.get("pca_fit_cols")
            self._null_count_cols = state.get("null_count_cols")
            # Absent in artifacts written before PRD Phase 9 P9-2 — the
            # create_uid_features transform then emits its unseen-UID
            # fallbacks for every row, which is a defined (if uninformative)
            # state, not a corrupt artifact.
            self._uid_maps = state.get("uid_maps", {})
            self._uid_global = state.get("uid_global", {})
            # None for artifacts saved before P9-4; `dist1_high` then stays
            # all-zero rather than being re-fit against serving traffic.
            self._dist1_high_threshold = state.get("dist1_high_threshold")
            # Empty for artifacts saved before P9-4/ADR-003; the windows then
            # cold-start, which is correct-but-stale rather than wrong.
            self._card_window_state = state.get("card_window_state", {})
            if not self._card_agg_state:
                logger.warning(
                    "Loaded transformers carry no per-card aggregate state "
                    "(artifact predates ADR-002, or the batch run did not "
                    "record it). Every card will cold-start: tx_count_per_card=0, "
                    "time_since_last_tx=-1.0, etc. Re-run preprocessing to "
                    "restore real card history."
                )

        logger.info(f"Transformers loaded from {load_path}/")
