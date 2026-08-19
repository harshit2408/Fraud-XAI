"""
tests/unit/test_seed.py

TDD for Phase B3 — `src/utils/seed.py`.

Written FIRST, against code that does not exist yet (RED). Proves:
  1. set_seed() makes Python's `random`, NumPy, and PyTorch RNG streams
     reproducible across two independent calls with the same seed.
  2. Different seeds diverge (guards against a vacuous "always equal" test).
  3. set_seed() seeds model weight initialization deterministically — the
     closest unit-testable proxy for "two consecutive runs produce identical
     val PR-AUC" without running a full training job.
  4. seed_worker() is safe to use as a DataLoader worker_init_fn.

Run: pytest tests/unit/test_seed.py -v
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import seed_worker, set_seed


class TestSetSeed:
    def test_returns_the_seed_that_was_applied(self):
        assert set_seed(42) == 42
        assert set_seed(123) == 123

    def test_sets_pythonhashseed_env_var(self):
        set_seed(7)
        assert os.environ["PYTHONHASHSEED"] == "7"

    def test_python_random_is_reproducible(self):
        set_seed(42)
        first = [random.random() for _ in range(5)]
        set_seed(42)
        second = [random.random() for _ in range(5)]
        assert first == second

    def test_numpy_random_is_reproducible(self):
        set_seed(42)
        first = np.random.rand(10)
        set_seed(42)
        second = np.random.rand(10)
        np.testing.assert_array_equal(first, second)

    def test_numpy_default_rng_legacy_global_state_is_reproducible(self):
        """np.random.seed affects the legacy global RNG used throughout the codebase
        (e.g. np.random.choice calls in training scripts)."""
        set_seed(99)
        first = np.random.randint(0, 1_000_000, size=20)
        set_seed(99)
        second = np.random.randint(0, 1_000_000, size=20)
        np.testing.assert_array_equal(first, second)

    def test_torch_random_is_reproducible(self):
        set_seed(42)
        first = torch.randn(10)
        set_seed(42)
        second = torch.randn(10)
        assert torch.equal(first, second)

    def test_different_seeds_diverge(self):
        """Guard against a vacuous always-equal implementation."""
        set_seed(1)
        a = torch.randn(50)
        set_seed(2)
        b = torch.randn(50)
        assert not torch.equal(a, b)

    def test_model_weight_initialization_is_reproducible(self):
        """Proxy for 'two consecutive runs produce identical val PR-AUC':
        with the same seed, model initialization (weights, dropout masks are
        not exercised here, but init is) must be bit-identical.
        """
        from src.models.tft_model import TemporalFusionTransformer

        set_seed(42)
        model_a = TemporalFusionTransformer(
            num_numeric_features=10, num_static_features=3, hidden_size=16
        )
        set_seed(42)
        model_b = TemporalFusionTransformer(
            num_numeric_features=10, num_static_features=3, hidden_size=16
        )

        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        assert state_a.keys() == state_b.keys()
        for key in state_a:
            assert torch.equal(state_a[key], state_b[key]), (
                f"Parameter '{key}' differs between two seeded initializations"
            )

    def test_model_weight_initialization_diverges_across_seeds(self):
        from src.models.tft_model import TemporalFusionTransformer

        set_seed(1)
        model_a = TemporalFusionTransformer(
            num_numeric_features=10, num_static_features=3, hidden_size=16
        )
        set_seed(2)
        model_b = TemporalFusionTransformer(
            num_numeric_features=10, num_static_features=3, hidden_size=16
        )

        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        any_different = any(
            not torch.equal(state_a[key], state_b[key]) for key in state_a
        )
        assert any_different, "Different seeds produced identical model weights"

    def test_cuda_manual_seed_all_called_when_available(self, monkeypatch):
        """When CUDA is reported available, set_seed must seed it too."""
        calls = {}

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "manual_seed_all", lambda s: calls.setdefault("seed", s)
        )

        set_seed(55)

        assert calls.get("seed") == 55

    def test_deterministic_cuda_flag_is_opt_in(self, monkeypatch):
        """deterministic_cuda=True should call torch.use_deterministic_algorithms."""
        calls = []
        monkeypatch.setattr(
            torch, "use_deterministic_algorithms", lambda flag: calls.append(flag)
        )

        set_seed(42, deterministic_cuda=True)

        assert calls == [True]

    def test_deterministic_cuda_not_forced_by_default(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            torch, "use_deterministic_algorithms", lambda flag: calls.append(flag)
        )

        set_seed(42)

        assert calls == []


class TestSeedWorker:
    def test_seed_worker_does_not_raise(self):
        torch.manual_seed(0)
        seed_worker(0)  # must not raise

    def test_seed_worker_is_deterministic_given_torch_initial_seed(self):
        torch.manual_seed(123)
        seed_worker(0)
        first = np.random.rand(5)

        torch.manual_seed(123)
        seed_worker(0)
        second = np.random.rand(5)

        np.testing.assert_array_equal(first, second)
