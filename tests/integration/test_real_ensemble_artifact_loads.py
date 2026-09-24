"""
tests/integration/test_real_ensemble_artifact_loads.py

A single smoke test that the REAL trained artifacts under `models/` load
through `ModelRegistry` without the registry's fail-closed guards firing.

The unit tests in `test_serving_registry.py` all use stub trainers and
synthetic manifests, so a genuine mixed-vintage `dataset_hash` — e.g. two of
the three models retrained on a new feature set and the third not — is
invisible to CI (this was mle-reviewer finding C1 / L5 on PRD Phase 9 P9-2).
This test closes that gap. It is skipped when the artifacts are absent (a
fresh checkout, or CI without a training run), so it never blocks a machine
that has not trained.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.integration

_MODELS = PROJECT_ROOT / "models"
_REQUIRED = [
    _MODELS / "ensemble.json",
    _MODELS / "ensemble.checksums.json",
    _MODELS / "xgb_model.ubj",
    _MODELS / "tft_model.weights.pt",
]


@pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED),
    reason="real models/ artifacts not present (no local training run)",
)
def test_real_ensemble_artifact_loads_through_registry():
    from src.config import load_settings
    from src.serving.registry import ModelRegistry

    config = load_settings("config/config.yaml").model_dump()
    registry = ModelRegistry(config=config)

    # Raises ValueError on: mixed dataset_hash, spec fitted on another dataset,
    # uncalibrated model, missing per-mode threshold, raw probability space.
    loaded = registry.load()

    spec = loaded.ensemble
    # The deployed spec must register a full mode and the no_tft fallback,
    # each with its own in-range threshold (ADR-001 §3.3).
    assert "full" in spec.modes
    assert "no_tft" in spec.modes, (
        "no_tft fallback missing — InferenceService cannot degrade past a TFT "
        "failure (mle-reviewer P9-2 H2)."
    )
    for name in ("full", "no_tft"):
        assert 0.0 <= spec.mode(name).threshold <= 1.0

    # Every model the spec can call for must have actually loaded.
    for model_name in spec.models_required():
        assert model_name in loaded.trainers
