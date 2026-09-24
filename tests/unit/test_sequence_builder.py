"""
tests/unit/test_sequence_builder.py

Unit tests for the SequenceBuilder module.
Tests sequence creation, padding, and no look-ahead leakage.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sequence_builder import SequenceBuilder


@pytest.fixture
def sample_data():
    """Create a small sample dataset with known structure."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "card1": [1, 1, 1, 1, 1, 2, 2, 2, 3, 3] * 5,
        "TransactionAmt": np.random.uniform(10, 1000, n),
        "amount_log": np.log1p(np.random.uniform(10, 1000, n)),
        "hour_sin": np.sin(np.random.uniform(0, 2 * np.pi, n)),
        "hour_cos": np.cos(np.random.uniform(0, 2 * np.pi, n)),
        "pca_v_0": np.random.randn(n),
        "pca_v_1": np.random.randn(n),
        "ProductCD": np.random.choice([0, 1, 2], n),
        "card4": np.random.choice([0, 1, 2, 3], n),
        "card6": np.random.choice([0, 1], n),
        "isFraud": np.random.choice([0, 1], n, p=[0.95, 0.05]),
    })
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]
    return X, y


class TestSequenceBuilder:
    def test_init(self):
        """Test SequenceBuilder initialization."""
        sb = SequenceBuilder(sequence_length=5, group_col="card1")
        assert sb.sequence_length == 5
        assert sb.group_col == "card1"

    def test_build_sequences_shape(self, sample_data):
        """Test that output shapes are correct."""
        X, y = sample_data
        sb = SequenceBuilder(sequence_length=5)
        result = sb.build_sequences(X, y)

        assert "sequences" in result
        assert "static" in result
        assert "targets" in result
        assert "mask" in result

        # Total sequences == total transactions
        assert len(result["targets"]) == len(X)

        # Sequence shape: (N, seq_len, feature_dim)
        assert result["sequences"].shape[0] == len(X)
        assert result["sequences"].shape[1] == 5  # sequence_length

        # Targets are binary
        assert set(np.unique(result["targets"])).issubset({0.0, 1.0})

    def test_build_sequences_padding(self, sample_data):
        """Test that short sequences are left-padded with zeros."""
        X, y = sample_data
        sb = SequenceBuilder(sequence_length=5)
        result = sb.build_sequences(X, y)

        # First transaction of each card should have padding (only 1 real timestep)
        # The mask should have at most 1 non-zero entry for the first tx of a card
        masks = result["mask"]
        
        # At least some sequences should have padding
        has_padding = (masks.sum(axis=1) < 5).any()
        assert has_padding, "Expected some sequences to have padding"

    def test_mask_values(self, sample_data):
        """Test that mask values are 0 (padding) or 1 (real)."""
        X, y = sample_data
        sb = SequenceBuilder(sequence_length=5)
        result = sb.build_sequences(X, y)

        assert set(np.unique(result["mask"])).issubset({0.0, 1.0})

    def test_sequence_length_1(self, sample_data):
        """Test with sequence_length=1 (no history)."""
        X, y = sample_data
        sb = SequenceBuilder(sequence_length=1)
        result = sb.build_sequences(X, y)

        assert result["sequences"].shape[1] == 1
        # All masks should be 1 (no padding needed)
        assert (result["mask"] == 1.0).all()

    def test_feature_dimensions(self, sample_data):
        """Test that feature and static dimensions are set correctly."""
        X, y = sample_data
        sb = SequenceBuilder(sequence_length=5)
        result = sb.build_sequences(X, y)

        assert sb.feature_dim > 0
        assert result["sequences"].shape[2] == sb.feature_dim

    def test_no_nan_in_output(self, sample_data):
        """Test that output arrays contain no NaN values."""
        X, y = sample_data
        sb = SequenceBuilder(sequence_length=5)
        result = sb.build_sequences(X, y)

        assert not np.isnan(result["sequences"]).any(), "NaN found in sequences"
        assert not np.isnan(result["static"]).any(), "NaN found in static features"
        assert not np.isnan(result["targets"]).any(), "NaN found in targets"
        assert not np.isnan(result["mask"]).any(), "NaN found in mask"

    def test_target_column_never_leaks_into_numeric_features(self, sample_data):
        """Regression guard: build_sequences() internally adds a '__target__'
        column to merge X and y for grouping. It must never be picked up as
        a time-varying numeric feature — doing so would leak the label
        itself into the model's inputs whenever `y` is passed."""
        X, y = sample_data
        sb = SequenceBuilder(sequence_length=5)
        sb.build_sequences(X, y)
        assert "__target__" not in sb.numeric_features

        # Feature count/shape must be identical whether or not `y` is passed —
        # a mismatch would mean the target leaked into the feature set.
        sb_no_y = SequenceBuilder(sequence_length=5)
        result_no_y = sb_no_y.build_sequences(X)
        assert sb_no_y.feature_dim == sb.feature_dim

    def test_targets_match_input(self, sample_data):
        """Test that targets correspond to the original labels."""
        X, y = sample_data
        sb = SequenceBuilder(sequence_length=5)
        result = sb.build_sequences(X, y)

        # Total number of targets should match input
        assert len(result["targets"]) == len(y)

    def test_different_sequence_lengths(self, sample_data):
        """Test with various sequence lengths."""
        X, y = sample_data
        for seq_len in [1, 3, 5, 10, 20]:
            sb = SequenceBuilder(sequence_length=seq_len)
            result = sb.build_sequences(X, y)
            assert result["sequences"].shape[1] == seq_len

    def test_single_card_group(self):
        """Test with all transactions belonging to one card."""
        n = 20
        X = pd.DataFrame({
            "card1": [1] * n,
            "TransactionAmt": np.random.uniform(10, 100, n),
            "amount_log": np.log1p(np.random.uniform(10, 100, n)),
            "hour_sin": np.random.randn(n),
        })
        y = pd.Series([0] * 18 + [1, 1])

        sb = SequenceBuilder(sequence_length=5)
        result = sb.build_sequences(X, y)

        assert len(result["targets"]) == n
        # Last transactions should have full sequences (mask all 1)
        assert result["mask"][-1].sum() == 5.0

    def test_efficient_builder(self, sample_data):
        """Test the memory-efficient builder produces valid output."""
        X, y = sample_data
        sb = SequenceBuilder(sequence_length=5)
        result = sb.build_sequences_efficient(X, y, max_sequences=20)

        assert len(result["targets"]) == 20
        assert result["sequences"].shape[1] == 5


# ── Phase B7 TDD — decouple labels from sequence construction ──────────────────
#
# HIGH finding (docs/IMPLEMENTATION_PLAN.md): TFTTrainer.predict_proba(X, y)
# threaded labels into build_sequences even at inference time — labels are
# not available in production. build_sequences(X) must work without y, and a
# separate attach_targets(seq_data, y) attaches labels only where needed
# (training/evaluation).


class TestLabelDecoupling:
    def test_build_sequences_without_y_has_no_targets(self, sample_data):
        X, y = sample_data
        sb = SequenceBuilder(sequence_length=5)
        result = sb.build_sequences(X)

        assert result["targets"] is None
        assert len(result["sequences"]) == len(X)
        assert len(result["mask"]) == len(X)
        assert "original_indices" in result

    def test_build_sequences_without_y_still_produces_valid_shapes(self, sample_data):
        X, y = sample_data
        sb = SequenceBuilder(sequence_length=5)
        result = sb.build_sequences(X)

        assert result["sequences"].shape == (len(X), 5, sb.feature_dim)
        assert not np.isnan(result["sequences"]).any()

    def test_attach_targets_matches_build_with_y(self, sample_data):
        """attach_targets(build_sequences(X), y) must equal build_sequences(X, y)."""
        X, y = sample_data

        sb_with_y = SequenceBuilder(sequence_length=5)
        with_y = sb_with_y.build_sequences(X, y)

        sb_without_y = SequenceBuilder(sequence_length=5)
        without_y = sb_without_y.build_sequences(X)
        attached = sb_without_y.attach_targets(without_y, y)

        np.testing.assert_array_equal(attached["targets"], with_y["targets"])
        np.testing.assert_array_equal(
            attached["original_indices"], with_y["original_indices"]
        )

    def test_attach_targets_uses_original_indices_not_positional_order(self):
        """Sequences are built per card group, not in original row order —
        attach_targets must map by original_indices, not by position."""
        n = 12
        X = pd.DataFrame({
            "card1": [1, 2] * (n // 2),
            "TransactionAmt": np.arange(n, dtype=float),
        })
        y = pd.Series(np.arange(n) % 2)  # alternating 0/1, distinct per row

        sb = SequenceBuilder(sequence_length=3)
        seq_data = sb.build_sequences(X)
        attached = sb.attach_targets(seq_data, y)

        expected = y.values[seq_data["original_indices"]]
        np.testing.assert_array_equal(attached["targets"], expected)


# ── Phase B4 TDD — feature scaling ──────────────────────────────────────────────
#
# HIGH finding: no scaling anywhere — raw dollar amounts, PCA components, and
# -999 imputation sentinels are fed into the TFT in the same input vector.
# Fit a scaler (QuantileTransformer, robust to -999 sentinels) on train only;
# persist it and apply consistently to val/test/inference.


def _sentinel_frame(n: int = 200, seed: int = 0) -> tuple:
    rng = np.random.RandomState(seed)
    vals = rng.normal(loc=500.0, scale=200.0, size=n)  # dollar-scale amounts
    vals[:20] = -999.0  # imputation sentinel
    X = pd.DataFrame({
        "card1": np.arange(n) % 5,
        "TransactionAmt": vals,
        "pca_v_0": rng.randn(n),  # already unit-scale
    })
    y = pd.Series(np.zeros(n))
    return X, y


class TestFeatureScaling:
    def test_fit_scaler_produces_unit_scale_features(self):
        X, y = _sentinel_frame()
        sb = SequenceBuilder(sequence_length=3)
        result = sb.build_sequences(X, y, fit_scaler=True)

        assert sb.scaler is not None
        real_values = result["sequences"][result["mask"] == 1.0]
        # QuantileTransformer output is bounded/roughly unit-scale, unlike raw
        # dollar amounts (hundreds) or -999 sentinels.
        assert np.abs(real_values).max() < 10

    def test_scaler_not_refit_on_subsequent_calls(self):
        X, y = _sentinel_frame()
        sb = SequenceBuilder(sequence_length=3)
        sb.build_sequences(X, y, fit_scaler=True)
        fitted_scaler = sb.scaler

        sb.build_sequences(X, y, fit_scaler=False)
        assert sb.scaler is fitted_scaler, "scaler must not be refit when fit_scaler=False"

    def test_transform_reuses_existing_scaler_without_explicit_refit_flag(self):
        """Once fitted, later calls transform using the stored scaler even if
        fit_scaler isn't passed again (opt-in scaling, not opt-in every call)."""
        X, y = _sentinel_frame()
        sb = SequenceBuilder(sequence_length=3)
        fitted_result = sb.build_sequences(X, y, fit_scaler=True)
        reused_result = sb.build_sequences(X, y)  # fit_scaler defaults to False

        np.testing.assert_allclose(
            fitted_result["sequences"], reused_result["sequences"], atol=1e-6
        )

    def test_sentinel_values_are_not_extreme_after_scaling(self):
        """The -999 sentinel must not dominate the input scale after fitting."""
        X, y = _sentinel_frame()
        sb = SequenceBuilder(sequence_length=3)
        result = sb.build_sequences(X, y, fit_scaler=True)

        assert result["sequences"].min() > -10, (
            "raw -999 sentinel leaked through scaling unchanged"
        )

    def test_default_behavior_unchanged_when_scaler_not_requested(self, sample_data):
        """fit_scaler defaults to False and scaler stays None — the common
        unscaled path used by all pre-existing tests must be untouched."""
        X, y = sample_data
        sb = SequenceBuilder(sequence_length=5)
        result = sb.build_sequences(X, y)  # no fit_scaler kwarg
        assert sb.scaler is None
        assert not np.isnan(result["sequences"]).any()


class TestFitScalerOn:
    """2026-09-09 (4-agent ML review, ecc:code-reviewer): `fit_scaler_on`
    lets a caller fit the QuantileTransformer on a train-only frame before
    building sequences over a larger (e.g. train+val) combined frame, so the
    scaler's quantile boundaries never see val/test rows. Regression guard
    for the leak in train_tft.py's `_build_sequences_for_splits` caller,
    reproducing the proof used to find it: the fitted scaler's quantile
    range must not reach a value that exists only outside the fit frame."""

    def test_scaler_is_fit_only_on_the_given_frame(self):
        train = pd.DataFrame({
            "card1": [1] * 10,
            "TransactionAmt": np.arange(1.0, 11.0),  # 1..10
        })
        val = pd.DataFrame({
            "card1": [1] * 10,
            "TransactionAmt": np.arange(1000.0, 1010.0),  # 1000..1009, disjoint
        })

        sb = SequenceBuilder(sequence_length=3)
        sb.fit_scaler_on(train)

        assert sb.scaler is not None
        fitted_max = sb.scaler.quantiles_[-1, 0]
        assert fitted_max <= 10.0, (
            f"scaler.quantiles_ max is {fitted_max}, which reaches into val's "
            "range (1000-1009) — fit_scaler_on must only see the train frame"
        )
        assert sb.scaler.n_quantiles == 10, (
            "n_quantiles should derive from the 10 train rows only, not "
            "train+val combined"
        )

    def test_combined_build_reuses_the_pre_fitted_train_only_scaler(self):
        """The intended calling pattern: fit_scaler_on(train), then build
        sequences over train+val with fit_scaler=False. The stored scaler
        (and therefore its quantile range) must stay the train-only one."""
        train = pd.DataFrame({
            "card1": [1] * 10,
            "TransactionAmt": np.arange(1.0, 11.0),
        })
        val = pd.DataFrame({
            "card1": [1] * 10,
            "TransactionAmt": np.arange(1000.0, 1010.0),
        })
        combined = pd.concat([train, val], axis=0, ignore_index=True)
        combined_y = pd.Series([0] * 20)

        sb = SequenceBuilder(sequence_length=3)
        sb.fit_scaler_on(train)
        train_only_max = sb.scaler.quantiles_[-1, 0]

        sb.build_sequences(combined, combined_y, fit_scaler=False)

        assert sb.scaler.quantiles_[-1, 0] == train_only_max, (
            "build_sequences(fit_scaler=False) must not refit the scaler "
            "even when the frame it transforms includes val rows"
        )
