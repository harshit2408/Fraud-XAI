"""
tests/unit/test_imbalance_config.py

TDD for Phase B1/B2 — `resolve_imbalance_config()` in `src/training/train_tft.py`.

Written FIRST, against code that does not exist yet (RED). Proves:
  B1: `imbalance.sampling_strategy` / `imbalance.loss_function` are validated and
      fail fast on unknown values; focal loss is reachable (previously dead code
      because "smote" != "focal_loss" silently fell through to WeightedBCELoss).
  B2: exactly one imbalance-correction mechanism is ever active — when the
      WeightedRandomSampler is in use, WeightedBCELoss's count-based pos_weight
      must NOT also be applied on top of it (that was the ~27x * ~27x = ~729x bug).

Run: pytest tests/unit/test_imbalance_config.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.losses import FocalLoss, WeightedBCELoss
from src.training.train_tft import resolve_imbalance_config


# 27x is the imbalance ratio cited in the HIGH finding (neg/pos ~= 27 for IEEE-CIS).
NEG_COUNT = 27_000
POS_COUNT = 1_000


class TestFailFastOnUnknownValues:
    def test_unknown_sampling_strategy_raises(self):
        with pytest.raises(ValueError, match="sampling_strategy"):
            resolve_imbalance_config(
                {"sampling_strategy": "bogus", "loss_function": "focal_loss"},
                pos_count=POS_COUNT,
                neg_count=NEG_COUNT,
            )

    def test_unknown_loss_function_raises(self):
        with pytest.raises(ValueError, match="loss_function"):
            resolve_imbalance_config(
                {"sampling_strategy": "none", "loss_function": "bogus"},
                pos_count=POS_COUNT,
                neg_count=NEG_COUNT,
            )

    def test_missing_keys_raise_rather_than_silently_defaulting(self):
        """The old code's `.get("strategy", "focal_loss")` silently swallowed a
        misconfigured/missing key. The new config must fail loudly instead."""
        with pytest.raises(ValueError):
            resolve_imbalance_config({}, pos_count=POS_COUNT, neg_count=NEG_COUNT)

    def test_smote_sampling_strategy_is_explicitly_unsupported(self):
        """SMOTE has no defined semantics for sequence data in this trainer;
        it must fail loudly rather than silently doing nothing."""
        with pytest.raises(NotImplementedError, match="smote"):
            resolve_imbalance_config(
                {"sampling_strategy": "smote", "loss_function": "focal_loss"},
                pos_count=POS_COUNT,
                neg_count=NEG_COUNT,
            )


class TestFocalLossIsReachable:
    def test_focal_loss_selected_returns_focal_loss_instance(self):
        _, criterion = resolve_imbalance_config(
            {
                "sampling_strategy": "none",
                "loss_function": "focal_loss",
                "focal_loss_gamma": 2.0,
                "focal_loss_alpha": 0.25,
            },
            pos_count=POS_COUNT,
            neg_count=NEG_COUNT,
        )
        assert isinstance(criterion, FocalLoss)
        assert criterion.gamma == 2.0
        assert criterion.alpha == 0.25

    def test_focal_loss_reachable_even_with_legacy_smote_style_typo_absent(self):
        """Regression guard for the exact HIGH finding: a config key collision
        that made 'strategy: smote' silently select WeightedBCELoss instead of
        FocalLoss. With the split keys, requesting focal_loss must always work."""
        _, criterion = resolve_imbalance_config(
            {"sampling_strategy": "oversample", "loss_function": "focal_loss"},
            pos_count=POS_COUNT,
            neg_count=NEG_COUNT,
        )
        assert isinstance(criterion, FocalLoss)


class TestNoDoubleCorrection:
    def test_weighted_bce_without_sampler_uses_full_count_based_pos_weight(self):
        """No sampler active -> the loss is the only correction mechanism, so it
        must carry the full neg/pos ratio (~27x)."""
        use_sampler, criterion = resolve_imbalance_config(
            {"sampling_strategy": "none", "loss_function": "weighted_bce"},
            pos_count=POS_COUNT,
            neg_count=NEG_COUNT,
        )
        assert use_sampler is False
        assert isinstance(criterion, WeightedBCELoss)
        expected = NEG_COUNT / POS_COUNT
        assert criterion.pos_weight == pytest.approx(expected, rel=1e-6)
        assert criterion.pos_weight == pytest.approx(27.0, rel=0.1)

    def test_oversample_with_weighted_bce_does_not_double_correct(self):
        """This is the exact HIGH-finding regression: sampler (~27x effective
        resampling) + count-based pos_weight (~27x) previously compounded to
        ~729x. With one mechanism only, pos_weight collapses to 1.0 when the
        sampler is active."""
        use_sampler, criterion = resolve_imbalance_config(
            {"sampling_strategy": "oversample", "loss_function": "weighted_bce"},
            pos_count=POS_COUNT,
            neg_count=NEG_COUNT,
        )
        assert use_sampler is True
        assert isinstance(criterion, WeightedBCELoss)
        assert criterion.pos_weight == pytest.approx(1.0)

        double_corrected = (NEG_COUNT / POS_COUNT) ** 2
        assert criterion.pos_weight < double_corrected / 10

    def test_oversample_flag_reflects_sampling_strategy(self):
        use_sampler_none, _ = resolve_imbalance_config(
            {"sampling_strategy": "none", "loss_function": "bce"},
            pos_count=POS_COUNT,
            neg_count=NEG_COUNT,
        )
        use_sampler_oversample, _ = resolve_imbalance_config(
            {"sampling_strategy": "oversample", "loss_function": "bce"},
            pos_count=POS_COUNT,
            neg_count=NEG_COUNT,
        )
        assert use_sampler_none is False
        assert use_sampler_oversample is True

    def test_plain_bce_is_unweighted_regardless_of_sampler(self):
        _, criterion_no_sampler = resolve_imbalance_config(
            {"sampling_strategy": "none", "loss_function": "bce"},
            pos_count=POS_COUNT,
            neg_count=NEG_COUNT,
        )
        _, criterion_with_sampler = resolve_imbalance_config(
            {"sampling_strategy": "oversample", "loss_function": "bce"},
            pos_count=POS_COUNT,
            neg_count=NEG_COUNT,
        )
        assert criterion_no_sampler.pos_weight == pytest.approx(1.0)
        assert criterion_with_sampler.pos_weight == pytest.approx(1.0)

    def test_focal_loss_alpha_is_independent_of_sampler_and_counts(self):
        """FocalLoss's alpha is a fixed hyperparameter, not derived from the
        imbalance ratio, so it cannot compound multiplicatively with the
        sampler the way count-based pos_weight did."""
        _, criterion_no_sampler = resolve_imbalance_config(
            {
                "sampling_strategy": "none",
                "loss_function": "focal_loss",
                "focal_loss_alpha": 0.25,
            },
            pos_count=POS_COUNT,
            neg_count=NEG_COUNT,
        )
        _, criterion_with_sampler = resolve_imbalance_config(
            {
                "sampling_strategy": "oversample",
                "loss_function": "focal_loss",
                "focal_loss_alpha": 0.25,
            },
            pos_count=POS_COUNT,
            neg_count=NEG_COUNT,
        )
        assert criterion_no_sampler.alpha == criterion_with_sampler.alpha == 0.25

    def test_zero_positive_count_does_not_divide_by_zero(self):
        """Edge case: guard the max(pos_count, 1) fallback."""
        _, criterion = resolve_imbalance_config(
            {"sampling_strategy": "none", "loss_function": "weighted_bce"},
            pos_count=0,
            neg_count=NEG_COUNT,
        )
        assert np.isfinite(criterion.pos_weight)
