"""
src/device.py

Shared device auto-resolution helper (Phase D7, docs/IMPLEMENTATION_PLAN.md).

MEDIUM finding: `device: "cuda"` was hardcoded in tune_xgb.py and shipped in
config.yaml unconditionally, so XGBoost raised on any CPU-only host (the
fraud-api container has no GPU). train_tft.py's `_get_device()` already
resolved "auto" -> cuda-if-available-else-cpu correctly; this module extracts
that logic into one place every trainer (and the tuners) can share, instead of
each reimplementing — or hardcoding — it.
"""

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

# "auto" resolves to cuda-if-available-else-cpu (see resolve_device() below);
# "cpu"/"cuda" are explicit literals. Owned here, not in src/config.py, since
# this module is the lower-level primitive the Settings model's device
# fields are validated against.
DeviceLiteral = Literal["auto", "cpu", "cuda"]

_VALID_LITERAL_DEVICES = frozenset({"cpu", "cuda"})


def resolve_device(device: DeviceLiteral, *, force_cpu: bool = False) -> Literal["cpu", "cuda"]:
    """
    Resolve a configured device string to a concrete "cpu" or "cuda" value.

    Args:
        device: "auto" (pick CUDA if available, else CPU), or an explicit
            "cpu" / "cuda" literal.
        force_cpu: If True, always returns "cpu" regardless of `device` or
            hardware availability. Use this at inference/serving load time,
            where the artifact may have been trained on a GPU host but the
            serving container is not guaranteed to have one.

    Returns:
        "cpu" or "cuda".

    Raises:
        ValueError: `device` is not "auto", "cpu", or "cuda".
    """
    if force_cpu:
        return "cpu"

    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Resolved device 'auto' -> '{resolved}'")
        return resolved

    if device not in _VALID_LITERAL_DEVICES:
        raise ValueError(
            f"Unsupported device {device!r}; expected 'auto', 'cpu', or 'cuda'"
        )

    return device
