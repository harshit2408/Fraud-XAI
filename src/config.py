"""
src/config.py

Validated pydantic Settings model (Phase D1, docs/IMPLEMENTATION_PLAN.md).

MEDIUM finding: an identical three-line `yaml.safe_load` `load_config()` was
copy-pasted across six modules (preprocess.py, train_xgb.py, train_lgbm.py,
train_tft.py, tune_xgb.py, tune_tft.py) with zero schema validation. A missing
or mistyped config key surfaced as a `KeyError` deep inside a long-running
training job instead of failing at startup.

This module is the single source of truth for config/config.yaml's schema.
`load_settings()` parses and validates the file once (fail-fast, with a clear
pydantic `ValidationError` naming the offending field) and is cached so
repeated calls with the same path don't re-parse/re-validate. Call sites that
need the legacy dict shape (`config["data"]["processed_dir"]`, etc.) get it
via `Settings.model_dump()`, which reproduces the same nested-dict structure
`yaml.safe_load()` always produced — so downstream code did not need to be
rewritten field-by-field to adopt this.
"""

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from src.device import DeviceLiteral


class _StrictModel(BaseModel):
    """Base for every config section: unknown keys (typos) fail validation
    instead of being silently ignored, the way plain dict access would.

    `protected_namespaces=()` disables pydantic's default "model_*" field
    name warning — config.yaml's `serving.model_path` is a legitimate field
    name here, not an accidental collision with pydantic internals.

    `frozen=True` makes the cached `Settings` singleton returned by
    `load_settings()` (below) immutable: a stray `settings.project.random_seed
    = 1` raises instead of silently corrupting the shared cache for every
    other caller in the process.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=(), frozen=True)


class ProjectConfig(_StrictModel):
    name: str
    version: str
    random_seed: int


class DataConfig(_StrictModel):
    raw_dir: str
    processed_dir: str
    train_file: str
    identity_file: str
    target_col: str
    temporal_col: str
    train_split_ratio: float
    val_split_ratio: float
    sequence_length: int


class FeaturesConfig(_StrictModel):
    drop_cols: List[str]
    v_features_pca_components: int
    categorical_cols: List[str]
    numerical_cols: List[str] = []


class XGBoostConfig(_StrictModel):
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    reg_alpha: float
    reg_lambda: float
    gamma: float
    device: DeviceLiteral
    tree_method: str
    scale_pos_weight: float
    eval_metric: str
    early_stopping_rounds: int


class LightGBMConfig(_StrictModel):
    n_estimators: int
    max_depth: int
    num_leaves: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_data_in_leaf: int
    reg_alpha: float
    reg_lambda: float
    scale_pos_weight: float
    eval_metric: str
    early_stopping_rounds: int


class TFTConfig(_StrictModel):
    max_encoder_length: int
    max_prediction_length: int
    hidden_size: int
    attention_head_size: int
    num_lstm_layers: int
    dropout: float
    hidden_continuous_size: int
    learning_rate: float
    max_epochs: int
    batch_size: int
    patience: int
    gradient_clip_val: float
    device: DeviceLiteral
    use_amp: bool


class ModelConfig(_StrictModel):
    xgboost: XGBoostConfig
    lightgbm: LightGBMConfig
    tft: TFTConfig


class ImbalanceConfig(_StrictModel):
    # Resampling and loss weighting are independent knobs (Phase B1) — see
    # train_tft.py:resolve_imbalance_config for why these are two keys, not one.
    sampling_strategy: Literal["none", "oversample", "smote"]
    loss_function: Literal["focal_loss", "weighted_bce", "bce"]
    smote_k_neighbors: int
    focal_loss_gamma: float
    focal_loss_alpha: float


class ThresholdsConfig(_StrictModel):
    default: float
    cost_fn: float
    cost_fp: float
    revenue_tp: float


class MLflowConfig(_StrictModel):
    tracking_uri: str
    experiment_name: str


class ServingConfig(_StrictModel):
    host: str
    port: int
    model_path: str
    tft_model_path: str
    log_file: str


class KafkaConfig(_StrictModel):
    bootstrap_servers: str
    input_topic: str
    output_topic: str
    consumer_group: str
    producer_rate_per_second: int


class MonitoringConfig(_StrictModel):
    reference_data_path: str
    evidently_report_dir: str
    drift_check_interval_hours: int
    prometheus_port: int


class Settings(_StrictModel):
    """Root schema for config/config.yaml."""

    project: ProjectConfig
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    imbalance: ImbalanceConfig
    thresholds: ThresholdsConfig
    mlflow: MLflowConfig
    serving: ServingConfig
    kafka: KafkaConfig
    monitoring: MonitoringConfig


@cache
def load_settings(config_path: str = "config/config.yaml") -> Settings:
    """
    Load, parse, and validate a config YAML file into a `Settings` model.

    Cached per `config_path` so repeated calls (every one of the six former
    `load_config()` call sites, plus each Optuna trial in tune_tft.py) don't
    re-read and re-validate the file from disk. The returned `Settings` is
    frozen (see `_StrictModel`), so it's safe to share as a singleton across
    every caller — callers that need to mutate a config value per-trial
    (tune_tft.py) mutate `.model_dump()`'s output, a fresh plain dict on
    every call, instead.

    Raises:
        FileNotFoundError: `config_path` does not exist.
        pydantic.ValidationError: the file is missing a required key, has a
            key of the wrong type, contains an unrecognized (mistyped) key,
            or fails a field constraint (e.g. an invalid `device` value).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Settings.model_validate(raw)


def config_hash(settings: Settings) -> str:
    """
    Deterministic sha256 hex digest of a `Settings` instance's validated content.

    Phase D4 (docs/IMPLEMENTATION_PLAN.md): the model manifest written next to
    every trained artifact needs a `config_hash` field so an artifact can be
    traced back to the exact configuration it was trained with. Hashing is
    done over the *parsed and validated* model (`model_dump(mode="json")`,
    serialized with sorted keys) rather than the raw YAML file bytes, so
    cosmetic changes to config.yaml — comments, key ordering, whitespace —
    don't spuriously change the hash, while any change that actually affects
    a validated field always does.

    Args:
        settings: A validated `Settings` instance (e.g. from `load_settings()`).

    Returns:
        A 64-character lowercase hex sha256 digest.
    """
    canonical = json.dumps(settings.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
