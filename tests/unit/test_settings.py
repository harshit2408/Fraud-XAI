"""
Phase D1/D7 TDD — tests/unit/test_settings.py

Written FIRST per the mandatory TDD workflow (RED before src/config.py exists).

Covers docs/IMPLEMENTATION_PLAN.md:
  D1 (MEDIUM finding, ~L384-393): `load_config` was copy-pasted verbatim across
      six modules with zero schema validation, so a missing or mistyped config
      key surfaced as a KeyError deep inside a long-running training job. This
      file asserts the replacement — a validated pydantic `Settings` model —
      fails fast at load time instead, with a clear ValidationError.
  D7 (MEDIUM finding, ~L396-406): `device: "cuda"` was hardcoded in tune_xgb.py
      and config.yaml, which raises on a CPU-only host. This file asserts a
      shared `resolve_device()` helper (modeled on train_tft.py's pre-existing
      "auto" mode) resolves "auto" -> cuda-if-available-else-cpu, validates
      literal values, and can force CPU for inference regardless of config.

Run: pytest tests/unit/test_settings.py -v
"""

import copy
from typing import Any, Dict
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import Settings, load_settings
from src.device import resolve_device


# ── Settings model: valid config round-trips ──────────────────────────────


def test_real_config_yaml_validates_cleanly(config: Dict[str, Any]) -> None:
    """The shipped config/config.yaml must satisfy the schema as-is."""
    settings = Settings.model_validate(config)
    assert settings.project.random_seed == 42


def test_settings_model_dump_is_dict_compatible_with_existing_call_sites(
    config: Dict[str, Any],
) -> None:
    """
    model_dump() must reproduce the same nested-dict shape the six call sites
    already consume via config["data"]["processed_dir"]-style access, so D1
    can replace the loader without rewriting every downstream dict access.
    """
    settings = Settings.model_validate(config)
    dumped = settings.model_dump()

    assert dumped["data"]["processed_dir"] == config["data"]["processed_dir"]
    assert dumped["model"]["xgboost"]["n_estimators"] == config["model"]["xgboost"]["n_estimators"]
    assert dumped["model"]["tft"]["device"] == config["model"]["tft"]["device"]
    assert dumped["imbalance"]["sampling_strategy"] == config["imbalance"]["sampling_strategy"]


def test_load_settings_reads_real_config_file(project_root) -> None:
    """load_settings() end-to-end against the real file on disk."""
    settings = load_settings(str(project_root / "config" / "config.yaml"))
    assert isinstance(settings, Settings)
    assert settings.data.target_col == "isFraud"


def test_load_settings_missing_file_raises_clear_error(tmp_path) -> None:
    """A missing config path must fail fast with a clear, specific error —
    not an opaque FileNotFoundError from deep inside yaml/open()."""
    missing_path = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="does_not_exist.yaml"):
        load_settings(str(missing_path))


def test_load_settings_is_cached(project_root) -> None:
    """Repeated loads of the same path must not re-parse+re-validate the file."""
    path = str(project_root / "config" / "config.yaml")
    first = load_settings(path)
    second = load_settings(path)
    assert first is second


def test_settings_instance_is_frozen(config: Dict[str, Any]) -> None:
    """The cached Settings singleton must be immutable: since load_settings()
    hands the same object to every caller, an accidental direct mutation
    must raise instead of silently corrupting it for everyone else."""
    settings = Settings.model_validate(config)
    with pytest.raises(ValidationError):
        settings.project.random_seed = 1


# ── Settings model: fail-fast validation (the core D1 ask) ────────────────


def test_missing_required_key_raises_validation_error(config: Dict[str, Any]) -> None:
    """A missing key must fail at startup with a pydantic ValidationError,
    not a KeyError deep inside a training run."""
    broken = copy.deepcopy(config)
    del broken["project"]["random_seed"]

    with pytest.raises(ValidationError, match="random_seed"):
        Settings.model_validate(broken)


def test_missing_top_level_section_raises_validation_error(config: Dict[str, Any]) -> None:
    broken = copy.deepcopy(config)
    del broken["kafka"]

    with pytest.raises(ValidationError, match="kafka"):
        Settings.model_validate(broken)


def test_wrong_type_raises_validation_error(config: Dict[str, Any]) -> None:
    """A value that cannot be coerced to the declared type must fail fast."""
    broken = copy.deepcopy(config)
    broken["model"]["xgboost"]["learning_rate"] = "very fast"  # not float-coercible

    with pytest.raises(ValidationError, match="learning_rate"):
        Settings.model_validate(broken)


def test_wrong_type_for_random_seed_raises_validation_error(config: Dict[str, Any]) -> None:
    broken = copy.deepcopy(config)
    broken["project"]["random_seed"] = "forty-two"

    with pytest.raises(ValidationError, match="random_seed"):
        Settings.model_validate(broken)


def test_mistyped_key_name_raises_validation_error(config: Dict[str, Any]) -> None:
    """A typo'd key (e.g. `randome_seed`) must be rejected rather than
    silently ignored, which is what a plain dict would do."""
    broken = copy.deepcopy(config)
    broken["project"]["randome_seed"] = broken["project"].pop("random_seed")

    with pytest.raises(ValidationError):
        Settings.model_validate(broken)


@pytest.mark.parametrize("bad_device", ["gpu", "GPU", "tpu", "cuda:0", "", 1])
def test_bad_xgboost_device_value_raises_validation_error(
    config: Dict[str, Any], bad_device: Any
) -> None:
    broken = copy.deepcopy(config)
    broken["model"]["xgboost"]["device"] = bad_device

    with pytest.raises(ValidationError, match="device"):
        Settings.model_validate(broken)


@pytest.mark.parametrize("bad_device", ["gpu", "GPU", "tpu", "cuda:0", "", 1])
def test_bad_tft_device_value_raises_validation_error(
    config: Dict[str, Any], bad_device: Any
) -> None:
    broken = copy.deepcopy(config)
    broken["model"]["tft"]["device"] = bad_device

    with pytest.raises(ValidationError, match="device"):
        Settings.model_validate(broken)


@pytest.mark.parametrize("good_device", ["auto", "cpu", "cuda"])
def test_valid_device_values_accepted(config: Dict[str, Any], good_device: str) -> None:
    ok = copy.deepcopy(config)
    ok["model"]["xgboost"]["device"] = good_device
    ok["model"]["tft"]["device"] = good_device

    settings = Settings.model_validate(ok)
    assert settings.model.xgboost.device == good_device
    assert settings.model.tft.device == good_device


def test_bad_imbalance_sampling_strategy_raises_validation_error(config: Dict[str, Any]) -> None:
    broken = copy.deepcopy(config)
    broken["imbalance"]["sampling_strategy"] = "not_a_real_strategy"

    with pytest.raises(ValidationError):
        Settings.model_validate(broken)


# ── D7: shared device auto-resolution helper ───────────────────────────────


def test_resolve_device_auto_picks_cuda_when_available() -> None:
    with patch("src.device.torch.cuda.is_available", return_value=True):
        assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_picks_cpu_when_cuda_unavailable() -> None:
    with patch("src.device.torch.cuda.is_available", return_value=False):
        assert resolve_device("auto") == "cpu"


def test_resolve_device_passes_through_explicit_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_passes_through_explicit_cuda() -> None:
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="gpu"):
        resolve_device("gpu")


def test_resolve_device_force_cpu_overrides_auto_even_when_cuda_available() -> None:
    """Inference/serving hosts have no GPU guarantee — force_cpu must win
    regardless of what the config says or what hardware is detected."""
    with patch("src.device.torch.cuda.is_available", return_value=True):
        assert resolve_device("auto", force_cpu=True) == "cpu"


def test_resolve_device_force_cpu_overrides_explicit_cuda() -> None:
    assert resolve_device("cuda", force_cpu=True) == "cpu"
