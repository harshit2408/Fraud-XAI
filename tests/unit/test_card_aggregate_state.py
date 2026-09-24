"""
tests/unit/test_card_aggregate_state.py

TDD for Phase E task E3 (docs/IMPLEMENTATION_PLAN.md), per the code change
docs/adr/ADR-002-realtime-feature-state.md §6 forces:

`create_card_aggregates` computes per-card expanding statistics vectorised and
currently keeps NOTHING afterwards. ADR-002 §2.3 established that every one of
those features is a pure function of five scalars per `card1` —
`(n, sum_amt, sum_amt_sq, max_amt, last_dt)` — so serving can reproduce the
batch values exactly. That requires the batch run to (a) retain those
accumulators and (b) persist them through save_transformers/load_transformers,
which is what these tests pin down.

The accumulators are *sufficient statistics*: these tests assert the online
reconstruction equals the batch output exactly, not approximately.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_engineering import CardAggregateState, FeatureEngineer


def _frame(n: int = 40, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionDT": np.sort(rng.integers(0, 500_000, n)).astype(float),
            "TransactionAmt": rng.uniform(1.0, 500.0, n).round(2),
            "card1": rng.integers(1000, 1004, n),
        }
    )


class TestAccumulatorsAreRetained:
    def test_create_card_aggregates_records_state_per_card(self):
        df = _frame()
        fe = FeatureEngineer()
        fe.create_card_aggregates(df, update_state=True)

        assert fe._card_agg_state, "create_card_aggregates retained no accumulator state"
        assert set(fe._card_agg_state) == set(df["card1"].unique())

    def test_accumulators_match_a_direct_recomputation(self):
        df = _frame()
        fe = FeatureEngineer()
        fe.create_card_aggregates(df, update_state=True)

        for card, group in df.groupby("card1"):
            state = fe._card_agg_state[card]
            amounts = group["TransactionAmt"].to_numpy(dtype=float)
            assert state.n == len(amounts)
            assert state.sum_amt == pytest.approx(amounts.sum())
            assert state.sum_amt_sq == pytest.approx((amounts**2).sum())
            assert state.max_amt == pytest.approx(amounts.max())
            assert state.last_dt == pytest.approx(group["TransactionDT"].max())

    def test_state_accumulates_across_successive_frames(self):
        """Batch runs card aggregates over one ordered frame, but serving
        continues from training history — so a second call must extend the
        accumulators, not reset them."""
        first, second = _frame(20, seed=1), _frame(20, seed=2)
        # Force overlap on one card so accumulation is actually exercised.
        second["card1"] = first["card1"].to_numpy()
        second["TransactionDT"] = second["TransactionDT"] + first["TransactionDT"].max()

        fe = FeatureEngineer()
        fe.create_card_aggregates(first, update_state=True)
        fe.create_card_aggregates(second, update_state=True)

        combined = pd.concat([first, second], ignore_index=True)
        for card, group in combined.groupby("card1"):
            state = fe._card_agg_state[card]
            assert state.n == len(group)
            assert state.sum_amt == pytest.approx(group["TransactionAmt"].sum())

    def test_update_state_false_leaves_state_untouched(self):
        """Inference transforms must not mutate fitted state — the same
        contract create_target_encoding's update_state flag already carries."""
        df = _frame()
        fe = FeatureEngineer()
        fe.create_card_aggregates(df, update_state=True)
        before = dict(fe._card_agg_state)

        fe.create_card_aggregates(_frame(10, seed=99), update_state=False)

        assert fe._card_agg_state == before


class TestAccumulatorsAreSufficientStatistics:
    """ADR-002 §2.3's core claim: five scalars reproduce the batch features
    EXACTLY. If these fail, the online store cannot be made equivalent and
    ADR-002's chosen option is invalid."""

    def test_carried_state_reproduces_batch_features_for_a_following_frame(self):
        full = _frame(60, seed=5).sort_values("TransactionDT").reset_index(drop=True)
        head, tail = full.iloc[:40].copy(), full.iloc[40:].copy().reset_index(drop=True)

        # Batch: one pass over the whole frame.
        batch = FeatureEngineer().create_card_aggregates(full)
        expected = batch.iloc[40:].reset_index(drop=True)

        # Split: fit on head, carry state, transform tail.
        fe = FeatureEngineer()
        fe.create_card_aggregates(head, update_state=True)
        actual = fe.create_card_aggregates(tail, update_state=False)

        for col in [
            "tx_count_per_card",
            "tx_sum_per_card",
            "mean_amount_per_card",
            "max_amount_per_card",
            "std_amount_per_card",
            "amount_vs_mean_ratio",
            "amount_zscore_per_card",
        ]:
            np.testing.assert_allclose(
                actual[col].to_numpy(), expected[col].to_numpy(), rtol=1e-9, atol=1e-9,
                err_msg=f"{col} diverged between batch and carried-state paths",
            )

    def test_unseen_card_falls_back_to_the_batch_first_transaction_values(self):
        """ADR-002 §5.4: cold start is the normal path, producing exactly the
        values batch emits for a card's first transaction."""
        fe = FeatureEngineer()
        fe.create_card_aggregates(_frame(30, seed=3), update_state=True)

        new_card = pd.DataFrame(
            {"TransactionDT": [9_999_999.0], "TransactionAmt": [42.0], "card1": [999_999]}
        )
        out = fe.create_card_aggregates(new_card, update_state=False)

        assert out.loc[0, "tx_count_per_card"] == 0
        assert out.loc[0, "tx_sum_per_card"] == pytest.approx(0.0)
        assert out.loc[0, "mean_amount_per_card"] == pytest.approx(0.0)
        assert out.loc[0, "max_amount_per_card"] == pytest.approx(0.0)
        assert out.loc[0, "std_amount_per_card"] == pytest.approx(0.0)
        assert out.loc[0, "amount_vs_mean_ratio"] == pytest.approx(1.0)
        assert out.loc[0, "amount_zscore_per_card"] == pytest.approx(0.0)


class TestVelocityUsesCarriedState:
    def test_time_since_last_tx_continues_from_carried_state(self):
        head = pd.DataFrame(
            {"TransactionDT": [100.0, 200.0], "TransactionAmt": [10.0, 20.0], "card1": [7, 7]}
        )
        tail = pd.DataFrame(
            {"TransactionDT": [350.0], "TransactionAmt": [30.0], "card1": [7]}
        )

        fe = FeatureEngineer()
        fe.create_card_aggregates(head, update_state=True)
        out = fe.create_velocity_features(tail)

        assert out.loc[0, "time_since_last_tx"] == pytest.approx(150.0)

    def test_unseen_card_gets_the_no_history_sentinel(self):
        fe = FeatureEngineer()
        out = fe.create_velocity_features(
            pd.DataFrame({"TransactionDT": [5.0], "TransactionAmt": [1.0], "card1": [42]})
        )
        assert out.loc[0, "time_since_last_tx"] == pytest.approx(-1.0)
        assert out.loc[0, "time_since_last_tx_log"] == pytest.approx(-1.0)


class TestPersistence:
    def test_accumulators_survive_a_save_load_round_trip(self, tmp_path):
        fe = FeatureEngineer()
        fe.create_card_aggregates(_frame(), update_state=True)
        fe.save_transformers(str(tmp_path / "t"))

        loaded = FeatureEngineer()
        loaded.load_transformers(str(tmp_path / "t"))

        assert loaded._card_agg_state == fe._card_agg_state

    def test_artifact_predating_adr_002_loads_with_empty_state(self, tmp_path):
        """The shipped data/processed/transformers/ was written before this
        change. Loading it must succeed (every card simply cold-starts), not
        raise — the operator decision recorded for E3."""
        import joblib

        from src.utils.checksums import write_checksums

        fe = FeatureEngineer()
        fe.create_card_aggregates(_frame(), update_state=True)
        save_dir = tmp_path / "t"
        fe.save_transformers(str(save_dir))

        # Simulate the old artifact: feature_state.joblib without the new key.
        state = joblib.load(save_dir / "feature_state.joblib")
        state.pop("card_agg_state", None)
        joblib.dump(state, save_dir / "feature_state.joblib")
        manifest = {
            name: save_dir / f"{name}.joblib"
            for name in [
                "label_encoders",
                "freq_encoders",
                "imputer",
                "card_hash_freq",
                "feature_state",
            ]
        }
        write_checksums(save_dir / "checksums.json", manifest)

        loaded = FeatureEngineer()
        loaded.load_transformers(str(save_dir))
        assert loaded._card_agg_state == {}


class TestCardAggregateStateValueSemantics:
    def test_state_is_immutable(self):
        """Per common/coding-style.md: updates return a new object rather than
        mutating in place, so a snapshot handed to the serving store cannot be
        changed underneath it."""
        state = CardAggregateState(n=1, sum_amt=10.0, sum_amt_sq=100.0, max_amt=10.0, last_dt=5.0)
        with pytest.raises((AttributeError, TypeError)):
            state.n = 2  # type: ignore[misc]

    def test_observe_returns_a_new_extended_state(self):
        state = CardAggregateState(n=1, sum_amt=10.0, sum_amt_sq=100.0, max_amt=10.0, last_dt=5.0)
        nxt = state.observe(amount=20.0, dt=9.0)

        assert (state.n, state.sum_amt) == (1, 10.0)
        assert nxt.n == 2
        assert nxt.sum_amt == pytest.approx(30.0)
        assert nxt.sum_amt_sq == pytest.approx(500.0)
        assert nxt.max_amt == pytest.approx(20.0)
        assert nxt.last_dt == pytest.approx(9.0)
