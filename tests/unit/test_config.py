"""
Phase 0 TDD — test_config.py

Written FIRST per test_driven_dev.md skill.
All 5 tests must pass before Phase 0 is marked done.

Run: pytest tests/unit/test_config.py -v
"""

import os
import pathlib

import pytest
import yaml


CONFIG_PATH = pathlib.Path(__file__).parents[2] / "config" / "config.yaml"

REQUIRED_TOP_LEVEL_KEYS = [
    "project",
    "data",
    "features",
    "model",
    "imbalance",
    "thresholds",
    "mlflow",
    "serving",
    "kafka",
    "monitoring",
]


@pytest.fixture(scope="module")
def config() -> dict:
    """Load config.yaml once for all tests in this module."""
    assert CONFIG_PATH.exists(), f"config.yaml not found at {CONFIG_PATH}"
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def test_config_loads_without_error(config: dict) -> None:
    """config.yaml must parse cleanly into a non-empty dict."""
    assert isinstance(config, dict), "Config must be a dict"
    assert len(config) > 0, "Config must not be empty"


def test_required_keys_present(config: dict) -> None:
    """All top-level section keys must be present."""
    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in config, f"Missing required top-level key: '{key}'"


def test_random_seed_is_integer(config: dict) -> None:
    """project.random_seed must be an integer (42 per PRD)."""
    seed = config["project"]["random_seed"]
    assert isinstance(seed, int), f"random_seed must be int, got {type(seed)}"
    assert seed == 42, f"Expected random_seed=42, got {seed}"


def test_no_hardcoded_absolute_paths(config: dict) -> None:
    """data.raw_dir and data.processed_dir must be relative paths, not absolute."""
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]

    assert not os.path.isabs(raw_dir), (
        f"data.raw_dir must be a relative path, got absolute: '{raw_dir}'"
    )
    assert not os.path.isabs(processed_dir), (
        f"data.processed_dir must be a relative path, got absolute: '{processed_dir}'"
    )


def test_kafka_topics_defined(config: dict) -> None:
    """kafka.input_topic and kafka.output_topic must be non-empty strings."""
    input_topic = config["kafka"]["input_topic"]
    output_topic = config["kafka"]["output_topic"]

    assert isinstance(input_topic, str) and len(input_topic) > 0, (
        "kafka.input_topic must be a non-empty string"
    )
    assert isinstance(output_topic, str) and len(output_topic) > 0, (
        "kafka.output_topic must be a non-empty string"
    )
    assert input_topic != output_topic, (
        "kafka.input_topic and kafka.output_topic must be different topics"
    )


# ── Phase B1 TDD — imbalance.sampling_strategy / imbalance.loss_function ──────
#
# Written FIRST against the shipped config, which still has the single
# `imbalance.strategy: "smote"` key (RED). Proves the CRITICAL/HIGH finding:
# a config key collision made `strategy: "smote"` silently fall through to
# WeightedBCELoss because train_tft.py compared it against "focal_loss".
# The fix splits the single ambiguous key into two independently validated
# ones so each concern (resampling vs. loss function) fails fast on typos.

VALID_SAMPLING_STRATEGIES = {"none", "oversample", "smote"}
VALID_LOSS_FUNCTIONS = {"focal_loss", "weighted_bce", "bce"}


def test_imbalance_strategy_key_no_longer_exists(config: dict) -> None:
    """The old ambiguous `imbalance.strategy` key must be removed, not left
    dangling alongside the new keys where it could silently be read by
    forgotten call sites."""
    assert "strategy" not in config["imbalance"], (
        "config.imbalance.strategy must be replaced by sampling_strategy + "
        "loss_function, not left in place"
    )


def test_imbalance_sampling_strategy_is_valid(config: dict) -> None:
    sampling_strategy = config["imbalance"].get("sampling_strategy")
    assert sampling_strategy is not None, (
        "config.imbalance.sampling_strategy must be set"
    )
    assert sampling_strategy in VALID_SAMPLING_STRATEGIES, (
        f"config.imbalance.sampling_strategy={sampling_strategy!r} must be one "
        f"of {sorted(VALID_SAMPLING_STRATEGIES)}"
    )


def test_imbalance_loss_function_is_valid(config: dict) -> None:
    loss_function = config["imbalance"].get("loss_function")
    assert loss_function is not None, "config.imbalance.loss_function must be set"
    assert loss_function in VALID_LOSS_FUNCTIONS, (
        f"config.imbalance.loss_function={loss_function!r} must be one of "
        f"{sorted(VALID_LOSS_FUNCTIONS)}"
    )


def test_imbalance_shipped_config_does_not_double_correct(config: dict) -> None:
    """Shipped default must not combine sampler-based oversampling with a
    count-based weighted-BCE loss — that specific combination is the
    ~27x * ~27x double-correction bug this phase fixes."""
    imb_cfg = config["imbalance"]
    if imb_cfg.get("sampling_strategy") == "oversample":
        assert imb_cfg.get("loss_function") != "weighted_bce", (
            "Shipped config combines sampling_strategy=oversample with "
            "loss_function=weighted_bce — this double-corrects for imbalance"
        )
