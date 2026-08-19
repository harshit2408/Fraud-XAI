"""
src/utils/seed.py

Deterministic seeding utility for reproducible training runs.

HIGH finding (docs/IMPLEMENTATION_PLAN.md, Phase B3): no training entry point
called torch.manual_seed / np.random.seed / random.seed anywhere, so two runs
of `make train` produced different TFT models and different reported metrics.
This module centralizes seeding so every entry point calls one function.
"""

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a required dependency in this project
    _TORCH_AVAILABLE = False


def set_seed(seed: int, deterministic_cuda: bool = False) -> int:
    """
    Seed every RNG source used across the training pipeline.

    Covers Python's `random`, NumPy's global RNG, and (when installed) PyTorch's
    CPU and CUDA generators. Also sets `PYTHONHASHSEED` so hash-dependent
    iteration order (e.g. over sets/dicts in some code paths) is stable across
    process restarts — note this only takes effect for *new* interpreter
    processes, not retroactively within the current one.

    Args:
        seed: Seed value to apply everywhere.
        deterministic_cuda: If True, also calls `torch.use_deterministic_algorithms(True)`
            and disables cuDNN benchmarking/enables cuDNN determinism. Opt-in because
            it can noticeably slow down training and is only needed for bit-for-bit
            reproducibility on CUDA, not for CPU or same-run-to-run determinism.

    Returns:
        The seed that was applied, so callers can log it (e.g. to MLflow) in one line.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    logger.info(f"Random seed set to {seed} (deterministic_cuda={deterministic_cuda})")
    return seed


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - required DataLoader signature
    """
    `worker_init_fn` for `torch.utils.data.DataLoader`.

    Seeds each worker process's NumPy/`random` state from PyTorch's per-worker
    initial seed, so multi-worker shuffling and any worker-side augmentation is
    reproducible given a fixed base seed. Safe to pass even when `num_workers=0`
    (PyTorch simply never calls it in that case).
    """
    if not _TORCH_AVAILABLE:
        return
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
