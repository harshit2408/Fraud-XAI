"""
tests/integration/test_train_serve_equivalence.py

Phase E task **E4**: "Train/serve equivalence test: same transaction through
batch and serving paths yields identical features" — done when "byte-identical
feature vectors".

This is the test docs/adr/ADR-002-realtime-feature-state.md was written to make
possible. §2.3's finding — that the card-history features are a pure function
of five retained scalars — is what turns equivalence from "close enough within
a tolerance" into an exact-equality assertion. Engineered features are asserted
bit-for-bit apart from two narrow, documented float-associativity carve-outs
(see "What byte-identical can and cannot mean" below). If a feature outside
those two classes ever needs a tolerance, ADR-002's central claim is wrong and
the online store design needs revisiting.

The comparison is deliberately made against the REAL batch orchestrator
(`src.data.preprocess._derive_causal_features`), not a re-implementation of it
inside the test: a test that reimplements the pipeline it is checking proves
only that the test agrees with itself.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_engineering import FeatureEngineer
from src.data.preprocess import _derive_causal_features
from src.serving.feature_state import InMemoryFeatureStateStore
from src.serving.transform import ServingFeatureTransformer

pytestmark = pytest.mark.integration


# ── Synthetic raw frame ──────────────────────────────────────────────────────
# Mirrors the IEEE-CIS columns the causal feature groups actually read. Kept
# small and seeded so the test is hermetic and fast; the shape of the data
# matters here, not its volume.

def _raw_frame(n: int = 120, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionID": [f"tx{i:05d}" for i in range(n)],
            "TransactionDT": np.sort(rng.integers(0, 2_000_000, n)).astype(float),
            "TransactionAmt": rng.uniform(1.0, 900.0, n).round(2),
            "isFraud": (rng.random(n) < 0.15).astype(int),
            "card1": rng.integers(1000, 1006, n),
            "card2": rng.integers(100, 106, n),
            "card3": rng.integers(140, 152, n),
            "card5": rng.integers(100, 240, n),
            "addr1": rng.integers(200, 340, n),
            "dist1": rng.choice([np.nan, 1.0, 7.0, 33.0], n),
            "ProductCD": rng.choice(["W", "H", "C", "R"], n),
            "card4": rng.choice(["visa", "mastercard"], n),
            "card6": rng.choice(["debit", "credit"], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", np.nan], n),
            "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", np.nan], n),
            "DeviceType": rng.choice(["desktop", "mobile", np.nan], n),
            "DeviceInfo": rng.choice(["Windows", "iOS Device", "SM-G930V", np.nan], n),
            "id_31": rng.choice(["chrome 62.0", "safari 11.0", "mobile safari", np.nan], n),
            "id_33": rng.choice(["1920x1080", "2208x1242", np.nan], n),
            **{f"D{i}": rng.choice([np.nan, 10.0, 50.0, 200.0], n) for i in range(1, 16)},
            **{f"C{i}": rng.integers(0, 6, n).astype(float) for i in range(1, 15)},
            **{f"V{i}": rng.normal(0, 1, n) for i in range(1, 12)},
        }
    )


# ── The two paths under comparison ───────────────────────────────────────────

def _batch_causal(df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureEngineer]:
    """The production batch path: exactly the orchestration steps
    `src/data/preprocess.py:run_pipeline` runs, on one ordered frame —
    null counts over the raw frame, PCA fitted before the temporal sort,
    then the causal feature groups."""
    fe = FeatureEngineer()
    out = fe.create_null_count_features(df)
    out = fe.reduce_v_features(out, fit=True, n_components=5)
    out = out.sort_values("TransactionDT").reset_index(drop=True)
    out = _derive_causal_features(fe, out)
    return out, fe


# config/config.yaml's production value (finding F6). Applied on BOTH paths:
# without it, the batch frame's own rows count as instantly-labelled history,
# which is the same-window leak F6 removed and which a served transaction can
# never have. Omitting it here would make the test demand that serving
# reproduce a leak.
LABEL_LAG_SECONDS = 30 * 86400


def _fit_stateful(fe: FeatureEngineer, batch_out: pd.DataFrame) -> pd.DataFrame:
    """Fit the stateful transformers the way `_apply_stateful_transforms` does."""
    out = fe.create_card_hash_features(batch_out, fit=True)
    out = fe.create_target_encoding(
        out, fit=True, update_state=True,
        time_col="TransactionDT", label_lag_seconds=LABEL_LAG_SECONDS,
    )
    out = fe.handle_missing_values(out, fit=True)
    out = fe.encode_categoricals(out, fit=True)
    return out


def _transform_stateful(fe: FeatureEngineer, out: pd.DataFrame) -> pd.DataFrame:
    """The `fit=False` counterpart of `_fit_stateful` — what val/test and
    serving both get."""
    out = fe.create_card_hash_features(out, fit=False)
    out = fe.create_target_encoding(
        out, fit=False, update_state=False,
        time_col="TransactionDT", label_lag_seconds=LABEL_LAG_SECONDS,
    )
    out = fe.handle_missing_values(out, fit=False)
    out = fe.encode_categoricals(out, fit=False)
    return out


@pytest.fixture()
def paths(tmp_path):
    """A fitted batch run, its serving-side counterpart, and the held-out rows
    the two paths are compared on.

    The batch reference is produced by ONE `FeatureEngineer` that fits on the
    history rows and then transforms the live rows with `fit=False` — exactly
    what `_apply_stateful_transforms` does for val/test. Fitting a *second*
    engineer on history+live and comparing against that would compare two
    different fitted models: the frequency and label encoders would be
    estimated over different row counts, so `P_emaildomain` and friends would
    differ for reasons that have nothing to do with train/serve skew. The
    fitted transformers are the constant here; the online card state is the
    variable under test.
    """
    raw = _raw_frame()
    history, live = raw.iloc[:100].copy(), raw.iloc[100:].copy().reset_index(drop=True)

    # ── Batch: fit on history, then transform `live` as a following split ──
    fe = FeatureEngineer()
    hist_out = fe.create_null_count_features(history)
    hist_out = fe.reduce_v_features(hist_out, fit=True, n_components=5)
    hist_out = hist_out.sort_values("TransactionDT").reset_index(drop=True)
    hist_out = _derive_causal_features(fe, hist_out)
    _fit_stateful(fe, hist_out)

    live_out = fe.create_null_count_features(live)
    live_out = fe.reduce_v_features(live_out, fit=False)
    live_out = live_out.sort_values("TransactionDT").reset_index(drop=True)
    live_out = _derive_causal_features(fe, live_out)
    batch_features = _transform_stateful(fe, live_out)
    # run_pipeline drops the identifier/temporal/target columns from X_* before
    # the parquet write; the serving vector must match that column set exactly.
    batch_features = batch_features.drop(
        columns=[c for c in ("TransactionID", "TransactionDT", "isFraud") if c in batch_features.columns]
    )

    # `_derive_causal_features` advanced the accumulators over `live` too;
    # persist the state as of the END of history, which is what a serving
    # instance restores before the live traffic arrives.
    fe._card_agg_state, fe._card_window_state = _state_after(history)
    transformer_dir = tmp_path / "transformers"
    fe.save_transformers(str(transformer_dir))

    # ── Serving: restore from the artifact, score `live` row by row ──
    serving_fe = FeatureEngineer()
    serving_fe.load_transformers(str(transformer_dir))
    store = InMemoryFeatureStateStore(
        card_state=serving_fe._card_agg_state,
        card_window=serving_fe._card_window_state,
    )
    transformer = ServingFeatureTransformer(
        feature_engineer=serving_fe,
        state_store=store,
        feature_names=list(batch_features.columns),
    )

    return {
        "live": live.sort_values("TransactionDT").reset_index(drop=True),
        "batch": batch_features.reset_index(drop=True),
        "transformer": transformer,
        "store": store,
    }


def _state_after(history: pd.DataFrame) -> tuple[dict, dict]:
    """Per-card accumulators AND trailing-24h windows as of the end of
    `history`, the way a batch run records them for serving
    (`preprocess.run_pipeline`).

    Both must be replayed: restoring the expanding scalars while leaving the
    windows empty would pair a card's full lifetime history with a zero
    trailing count, a combination batch never produces for an active card
    (ADR-003 §4.5)."""
    recorder = FeatureEngineer()
    recorder.update_card_aggregate_state(
        history.sort_values("TransactionDT").reset_index(drop=True)
    )
    return recorder._card_agg_state, recorder._card_window_state


# ── What "byte-identical" can and cannot mean ────────────────────────────────
# Every engineered feature is reproduced by the serving path from the retained
# accumulators, and the vast majority match bit-for-bit. Two narrow classes
# differ in the last ulp for reasons that are properties of floating-point
# arithmetic, NOT of train/serve skew. Both are asserted at a tolerance far
# tighter than any real skew could hide in, and both are named explicitly so a
# genuine regression still fails loudly.
#
# 1. `pca_v_*` — PCA.transform is a float32 matmul. BLAS sums the dot products
#    in a different order for a 20-row block (batch) than for a 1-row block
#    (serving), and float addition is not associative. Reproducible with stock
#    scikit-learn on random data, no project code involved (~2.4e-07, float32).
#
# 2. Card-history sums (`tx_sum_per_card` and the mean/std/zscore derived from
#    them) — batch computes one continuous `cumsum` over a card's whole
#    history; serving computes `carried_subtotal + cumsum(this_frame)`. Those
#    regroup the same additions differently, so they can differ by one float64
#    ulp (~2e-16 relative) once a card's running total grows large. This is
#    intrinsic to carrying a subtotal at all: bit-identity would require
#    replaying every card's full history on each request, i.e. the
#    recompute-per-request option ADR-002 §4 Option D rejected on cost.
#    Documented in ADR-002 §2.3.
#
# What this test still guarantees: no feature differs for any reason other than
# these two, and none differs by more than float noise. A dropped join, a stale
# encoder, a missed state update or a read/write inversion all move features by
# orders of magnitude more than this and are caught.
PCA_PREFIX = "pca_v_"
PCA_FLOAT32_TOL = 1e-6

# Features derived from a carried float subtotal (see note 2 above).
CARRIED_SUM_FEATURES = frozenset(
    {
        "tx_sum_per_card",
        "mean_amount_per_card",
        "std_amount_per_card",
        "amount_vs_mean_ratio",
        "amount_zscore_per_card",
        "interaction_hour_zscore",
        # ADR-003: the trailing-24h mean is a windowed sum over a carried
        # deque, where batch computes it as a prefix-sum difference. Same
        # float-associativity class as the entries above, and strictly
        # smaller error — the window spans at most a day (<= ~650 terms),
        # where the carried subtotals span a card's whole lifetime. The three
        # tx_count_* features are integer-valued and stay bit-exact.
        "amt_24h_mean_per_card",
        "amt_24h_vs_card_mean_ratio",
    }
)
CARRIED_SUM_TOL = 1e-12


def _split_columns(columns) -> tuple[list, list, list]:
    """(bit-exact, pca-tolerance, carried-sum-tolerance) column partition."""
    pca = [c for c in columns if str(c).startswith(PCA_PREFIX)]
    carried = [c for c in columns if c in CARRIED_SUM_FEATURES]
    exact = [c for c in columns if c not in set(pca) | set(carried)]
    return exact, pca, carried


def _assert_equivalent(served, expected, context: str) -> None:
    exact, pca, carried = _split_columns(expected.columns)
    assert exact, "expected some exactly-reproducible features to compare"
    for col in exact:
        np.testing.assert_array_equal(
            served[col].to_numpy(),
            expected[col].to_numpy(),
            err_msg=f"feature '{col}' {context}",
        )
    for col in pca:
        np.testing.assert_allclose(
            served[col].to_numpy(), expected[col].to_numpy(),
            rtol=PCA_FLOAT32_TOL, atol=PCA_FLOAT32_TOL,
            err_msg=f"PCA feature '{col}' {context} by more than float32 noise",
        )
    for col in carried:
        np.testing.assert_allclose(
            served[col].to_numpy(), expected[col].to_numpy(),
            rtol=CARRIED_SUM_TOL, atol=CARRIED_SUM_TOL,
            err_msg=f"carried-sum feature '{col}' {context} by more than one-ulp noise",
        )


# ── E4's actual assertions ───────────────────────────────────────────────────

class TestFeatureVectorEquivalence:
    def test_single_transaction_matches_batch_exactly(self, paths):
        """One transaction, both paths, identical vector."""
        row = paths["live"].iloc[[0]].reset_index(drop=True)
        served = paths["transformer"].transform(row)
        expected = paths["batch"].iloc[[0]].reset_index(drop=True)

        assert list(served.columns) == list(expected.columns), "column order diverged"
        _assert_equivalent(served, expected, "differs between batch and serving paths")

    def test_a_stream_of_transactions_matches_batch_exactly(self, paths):
        """The interesting case: each served transaction updates the store, so
        later rows must pick up the history earlier rows created — reproducing
        the batch pass's expanding window one row at a time (ADR-002 §5.2)."""
        transformer, store = paths["transformer"], paths["store"]
        live, expected = paths["live"], paths["batch"]

        served_rows = []
        for i in range(len(live)):
            row = live.iloc[[i]].reset_index(drop=True)
            served_rows.append(transformer.transform(row))
            # Read-then-write: the update happens only after scoring.
            store.observe(
                card_id=row.loc[0, "card1"],
                amount=float(row.loc[0, "TransactionAmt"]),
                dt=float(row.loc[0, "TransactionDT"]),
                transaction_id=str(row.loc[0, "TransactionID"]),
            )

        served = pd.concat(served_rows, ignore_index=True)

        assert list(served.columns) == list(expected.columns)
        _assert_equivalent(served, expected, "diverged over the stream")

    def test_float_columns_are_bitwise_identical(self, paths):
        """`assert_array_equal` on floats already compares bit patterns for
        finite values; this states the E4 criterion explicitly so a future
        relaxation to `assert_allclose` is a visible change, not a silent one."""
        row = paths["live"].iloc[[0]].reset_index(drop=True)
        served = paths["transformer"].transform(row)
        expected = paths["batch"].iloc[[0]].reset_index(drop=True)

        exact, _, _ = _split_columns(expected.columns)
        float_cols = [c for c in exact if expected[c].dtype.kind == "f"]
        assert float_cols, "expected some float features to compare"
        for col in float_cols:
            assert served[col].to_numpy().tobytes() == expected[col].to_numpy().tobytes(), (
                f"'{col}' is not byte-identical between batch and serving"
            )


class TestServingTransformDoesNotMutateFittedState:
    def test_transform_leaves_the_engineer_untouched(self, paths):
        """A transform that quietly advanced fitted state would make the
        service's behaviour depend on how many requests preceded it."""
        transformer = paths["transformer"]
        fe = transformer.feature_engineer

        before_target_enc = {k: dict(v) for k, v in fe._target_enc_state.items()}
        before_cards = dict(fe._card_agg_state)
        before_prior = fe._global_target_mean

        for i in range(5):
            transformer.transform(paths["live"].iloc[[i]].reset_index(drop=True))

        assert fe._global_target_mean == before_prior
        assert fe._card_agg_state == before_cards
        assert {k: dict(v) for k, v in fe._target_enc_state.items()} == before_target_enc


class TestOrderingContract:
    def test_writing_before_scoring_would_corrupt_the_vector(self, paths):
        """Guards ADR-002 §5.2's correctness condition. If the store is updated
        BEFORE the transform, the transaction contributes to its own expanding
        aggregates — the exact leak the batch pipeline's exclusive cumsum
        prevents. This test asserts the two orderings genuinely differ, so the
        read-then-write rule is load-bearing rather than decorative."""
        transformer, store = paths["transformer"], paths["store"]
        row = paths["live"].iloc[[0]].reset_index(drop=True)
        card = row.loc[0, "card1"]

        correct = transformer.transform(row)
        store.observe(
            card_id=card,
            amount=float(row.loc[0, "TransactionAmt"]),
            dt=float(row.loc[0, "TransactionDT"]),
            transaction_id="dup-guard-1",
        )
        wrong_order = transformer.transform(row)

        assert correct.loc[0, "tx_count_per_card"] != wrong_order.loc[0, "tx_count_per_card"]


class TestConcurrentTransformsAreIsolated:
    """Regression test for a CRITICAL race found in review.

    `predict` is a SYNC FastAPI route, so Starlette runs it in a threadpool and
    every request shares one `FeatureEngineer`. An earlier version of
    `ServingFeatureTransformer.transform` swapped the engineer's
    `_card_agg_state` attribute in place for the duration of the call and
    restored it in a `finally`. That is a read-modify-write on shared mutable
    state from multiple threads: two concurrent requests for different cards
    could clobber each other's scoped history, so one card would be scored
    against another card's aggregates — silently, with no exception.

    The fix passes carried state down as an argument instead. This test pins
    that behaviour: transforms running concurrently must produce exactly the
    values they produce when run alone.
    """

    def test_concurrent_transforms_match_their_serial_results(self, paths):
        from concurrent.futures import ThreadPoolExecutor

        transformer = paths["transformer"]
        rows = [
            paths["live"].iloc[[i]].reset_index(drop=True)
            for i in range(min(12, len(paths["live"])))
        ]

        serial = [transformer.transform(r) for r in rows]

        with ThreadPoolExecutor(max_workers=8) as pool:
            concurrent = list(pool.map(transformer.transform, rows))

        for i, (want, got) in enumerate(zip(serial, concurrent)):
            for col in want.columns:
                np.testing.assert_array_equal(
                    got[col].to_numpy(),
                    want[col].to_numpy(),
                    err_msg=(
                        f"row {i} feature '{col}' differs under concurrency — "
                        "requests are contaminating each other's card state"
                    ),
                )

    def test_transform_does_not_mutate_shared_engineer_state(self, paths):
        """The structural guarantee behind the test above."""
        transformer = paths["transformer"]
        before = dict(transformer.feature_engineer._card_agg_state)
        transformer.transform(paths["live"].iloc[[0]].reset_index(drop=True))
        assert transformer.feature_engineer._card_agg_state == before

