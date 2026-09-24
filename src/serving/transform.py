"""
src/serving/transform.py

Serving-side feature construction (ADR-001 §4.5 step 2, ADR-002 §5.2).

The single most common way a fraud model degrades silently in production is
train/serve skew introduced by a hand-written copy of the training
preprocessing. This module therefore does NOT reimplement any feature: it
calls the very same `FeatureEngineer` methods `src/data/preprocess.py` calls,
in the same order, with `fit=False` and `update_state=False`. Task E4's
equivalence test (`tests/integration/test_train_serve_equivalence.py`) exists
to keep it that way.

What is serving-specific is only:
  - seeding the per-card expanding aggregates from the online store rather
    than from a full ordered frame (ADR-002 §2.3), and
  - the read-then-write ordering: the caller updates the store only AFTER
    scoring, so a transaction never contributes to its own aggregates
    (ADR-002 §5.2).
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.feature_engineering import (
    CardAggregateState,
    CardWindowState,
    FeatureEngineer,
)
from src.serving.feature_state import FeatureStateStore

logger = logging.getLogger(__name__)

# The target column never arrives at inference — a served transaction's label
# is precisely what the model is being asked to predict. `create_target_encoding`
# skips (with a warning) when the column is absent, so a neutral placeholder is
# inserted for the transform and dropped again before the vector is returned.
# It can never influence the encoding: `update_state=False` means this frame's
# labels are not accumulated, and the current row is excluded from its own
# expanding total by construction.
_TARGET_COL = "isFraud"


class ServingFeatureTransformer:
    """Turns one validated transaction into the model's feature vector.

    Args:
        feature_engineer: A `FeatureEngineer` restored via `load_transformers`.
            Its fitted state is read, never mutated — carried card history is
            passed per call instead, which is what makes this safe to share
            across concurrent requests.
        state_store: Online per-card accumulators (ADR-002 §5.1).
        feature_names: The exact column list the models were trained on,
            taken from the XGBoost artifact's persisted `feature_names`. The
            returned frame is reindexed to this, so column ORDER is part of
            the contract rather than an accident of dict iteration.
        label_lag_seconds: Must match the value the batch run used
            (`features.target_encoding_label_lag_days` × 86400). A single-row
            serving frame has no in-frame history for the lag to exclude, so
            this changes nothing today — it is passed so the value cannot
            silently diverge from training if serving ever batches requests.
    """

    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        state_store: FeatureStateStore,
        feature_names: Optional[List[str]] = None,
        label_lag_seconds: float = 0.0,
    ) -> None:
        self.feature_engineer = feature_engineer
        self.state_store = state_store
        self.feature_names = list(feature_names) if feature_names else None
        self.label_lag_seconds = label_lag_seconds

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Apply the full training transform chain to a single-row frame.

        Raises:
            ValueError: `raw` does not hold exactly one row, or a column the
                trained models require is missing after transformation. The
                loud failure is deliberate — a silently zero-filled feature
                would produce a plausible-looking but wrong score.
        """
        if len(raw) != 1:
            raise ValueError(
                f"ServingFeatureTransformer.transform expects exactly one "
                f"transaction, got {len(raw)} rows."
            )

        fe = self.feature_engineer
        df = raw.copy()

        # The carried accumulators for just this card, read from the online
        # store and passed DOWN as an argument. Deliberately not swapped onto
        # the shared FeatureEngineer: sync FastAPI routes run in a threadpool,
        # so mutating shared instance state would let two concurrent requests
        # clobber each other's scoped history and silently score one card
        # against another's aggregates.
        card_id = df.iloc[0].get("card1")
        card_state = self._scoped_card_state(card_id)
        card_window = self._scoped_card_window(card_id)

        df = self._restore_raw_schema(fe, df)
        df = self._coerce_numeric_dtypes(fe, df)

        placeholder_target = _TARGET_COL not in df.columns
        if placeholder_target:
            df[_TARGET_COL] = 0

        # Order mirrors run_pipeline: null counts over the raw frame, then
        # PCA, then the causal groups, then the stateful ones.
        df = fe.create_null_count_features(df)
        df = fe.reduce_v_features(df, fit=False)
        df = self._derive_causal(fe, df, card_state, card_window)
        df = fe.create_card_hash_features(df, fit=False)
        # PRD Phase 9 P9-2: per-UID aggregates from train-fitted maps. Row-wise,
        # no cross-row or carried state — an unseen UID (new card, or known
        # card on a new day) gets the global fallbacks the training fit
        # recorded, matching what the model saw for cold UIDs in train.
        df = fe.create_uid_features(df, fit=False)
        # PRD Phase 9 P9-4: dist1 gradient features from the train-fitted
        # top-quintile cut. Row-wise and stateless; a non-W row (or one with no
        # dist1) takes the same neutral fill training gave it.
        df = fe.create_dist_features(df, fit=False)
        df = fe.create_target_encoding(
            df,
            fit=False,
            update_state=False,
            time_col="TransactionDT" if "TransactionDT" in df.columns else None,
            label_lag_seconds=self.label_lag_seconds,
        )
        df = fe.handle_missing_values(df, fit=False)
        df = fe.encode_categoricals(df, fit=False)

        if placeholder_target:
            df = df.drop(columns=[_TARGET_COL])

        return self._align(df)

    @staticmethod
    def _restore_raw_schema(fe: FeatureEngineer, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex the request to the raw column set training null-counted over.

        A serving request only carries the raw columns that were non-null for
        that one transaction (the producer's `_json_safe` drops NaN keys, and
        the request schema is `extra="allow"` with four required fields). But
        `create_null_count_features` counts nulls across *every* column
        present, so a request missing 200 null columns would undercount
        `null_count_total` by 200 and shift `null_ratio` — train/serve skew in
        two features the model uses.

        The fix: add back every column training saw, as NaN, in the original
        order, before null-counting. Columns the request carries that were not
        in the training schema (a client sending something novel) are left in
        place — they still get counted, which matches how batch would have
        treated a genuinely new column, and `_align` drops non-features later.

        No-op for artifacts saved before `null_count_cols` was persisted; the
        pre-existing (skewed) behaviour is unchanged for those until a
        preprocessing re-run.
        """
        expected = getattr(fe, "_null_count_cols", None)
        if not expected:
            return df
        missing = [c for c in expected if c not in df.columns]
        if not missing:
            return df
        ordered = [*expected, *[c for c in df.columns if c not in expected]]
        return df.reindex(columns=ordered)

    @staticmethod
    def _coerce_numeric_dtypes(fe: FeatureEngineer, df: pd.DataFrame) -> pd.DataFrame:
        """Restore the dtypes the batch pipeline inferred, before transforming.

        A single-row request whose column is null (`dist2`, most `id_*`) leaves
        pandas with an `object` dtype, because there is no non-null value to
        infer from. `handle_missing_values` then classifies that column as
        CATEGORICAL and fills the "MISSING" string, where the 590k-row batch
        frame — which always had some numeric value — classified it as numeric
        and filled -999.0. The model is then handed a string column and
        XGBoost rejects the frame outright.

        The fitted imputer's `_num_fill_values` is the authoritative record of
        which columns were numeric at fit time, so it is used as the contract.
        This is train/serve skew that only a one-row frame can expose, which is
        precisely why it survived until the full path was exercised end to end.
        """
        numeric_at_fit = getattr(fe, "_num_fill_values", {}) or {}
        if not numeric_at_fit:
            return df

        to_coerce = [
            col
            for col in df.columns
            if col in numeric_at_fit and df[col].dtype == object
        ]
        if not to_coerce:
            return df

        df = df.copy()
        for col in to_coerce:
            original = df[col]
            converted = pd.to_numeric(original, errors="coerce")
            # errors="coerce" turns a garbage string into NaN, which
            # handle_missing_values would then impute to -999.0 — a plausible
            # looking feature built from unusable input. Only genuinely-null
            # cells may become NaN here; anything else is a bad request.
            corrupted = converted.isna() & original.notna()
            if corrupted.any():
                raise ValueError(
                    f"Column '{col}' was numeric at training time but this "
                    f"request sent a non-numeric value "
                    f"({original[corrupted].iloc[0]!r}) — refusing to score "
                    "rather than silently imputing it."
                )
            df[col] = converted
        logger.debug("Coerced %d all-null column(s) back to numeric", len(to_coerce))
        return df

    def _scoped_card_state(self, card_id: Any) -> Dict[Any, CardAggregateState]:
        """This card's carried accumulators, as the mapping FeatureEngineer reads.

        An unseen card yields an empty mapping, which reproduces exactly the
        values batch emits for a card's first transaction (ADR-002 §5.4).
        """
        if card_id is None or pd.isna(card_id):
            return {}
        state = self.state_store.snapshot(card_id)
        # n == 0 is exactly the zero state, whose last_dt is NaN; both mean
        # "no history", which is the batch first-transaction path.
        return {card_id: state} if state.n else {}

    def _scoped_card_window(self, card_id: Any) -> Dict[Any, "CardWindowState"]:
        """This card's trailing-24h history, scoped exactly like the scalars.

        ADR-003. An unseen card (or one whose window has fully aged out) yields
        an empty mapping, which is what batch computes for a card with no
        transactions inside the window.
        """
        if card_id is None or pd.isna(card_id):
            return {}
        window = self.state_store.window(card_id)
        return {card_id: window} if window.entries else {}

    @staticmethod
    def _derive_causal(
        fe: FeatureEngineer,
        df: pd.DataFrame,
        card_state: Dict[Any, CardAggregateState],
        card_window: Optional[Dict[Any, "CardWindowState"]] = None,
    ) -> pd.DataFrame:
        """The stateless/causal groups, in `_derive_causal_features`' order.

        Deliberately NOT importing `preprocess._derive_causal_features`: that
        function also advances the per-card accumulators at the end, which is
        precisely what serving must not do on the read path.

        The two history-dependent groups take `card_state` explicitly so this
        whole path is read-only with respect to the shared engineer.
        """
        df = fe.create_temporal_features(df)
        df = fe.create_amount_features(df)
        df = fe.create_email_features(df)
        df = fe.create_d_column_features(df)
        df = fe.create_c_column_features(df)
        df = fe.create_card_aggregates(df, card_state=card_state)
        df = fe.create_velocity_features(
            df, card_state=card_state, card_window=card_window or {}
        )
        df = fe.create_address_features(df)
        df = fe.create_device_features(df)
        df = fe.create_interactions(df)
        return df

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop non-feature columns and reindex to the trained column list."""
        drop = [
            c
            for c in ("TransactionID", "TransactionDT", "_split_row_id")
            if c in df.columns
        ]
        if drop:
            df = df.drop(columns=drop)

        if self.feature_names is None:
            return df.reset_index(drop=True)

        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in transformed input: {missing[:10]}"
                f"{' …' if len(missing) > 10 else ''}. The serving transform "
                "produced a vector the trained model cannot consume — refusing "
                "to score rather than filling defaults."
            )
        return df[self.feature_names].reset_index(drop=True)
