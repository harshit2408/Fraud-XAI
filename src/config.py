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
import logging
import os
from functools import cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.device import DeviceLiteral

logger = logging.getLogger(__name__)

# Environment overrides applied to the parsed YAML before validation.
# Keys are `(section, field)`; the value is the env var that wins when set and
# non-empty. Kept to deployment-topology values that legitimately differ
# between `docker compose` and a local/CI run and must NOT be committed to
# config.yaml — the Kafka broker address is the whole set today. `config_hash`
# then reflects the value the process actually used, which is the point.
_ENV_OVERRIDES: Dict[tuple, str] = {
    ("kafka", "bootstrap_servers"): "KAFKA_BOOTSTRAP_SERVERS",
}


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
    # Finding F6 (2026-08-19 metrics audit): expanding target encoding
    # (create_target_encoding) previously assumed same-window label history
    # was instantly available, which a live model never has. 0.0 preserves
    # that original (leakage-optimistic) behavior for any config that
    # doesn't set this explicitly; config/config.yaml sets a documented
    # positive value. See FeatureEngineer.create_target_encoding's
    # docstring for the full rationale.
    target_encoding_label_lag_days: float = 0.0


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
    # PRD Phase 9 P9-6. Optional with the previously hardcoded default, so
    # configs written before this key remain valid.
    weight_decay: float = 1e-5


class GNNConfig(_StrictModel):
    """PRD Phase 12 — GNN-GraphSAGE. Required keys mirror TFTConfig's style;
    the rest are Optional-with-default so a config file written before Phase 12
    still validates (ModelConfig.gnn is itself Optional for the same reason)."""

    hidden_dims: List[int]
    mlp_hidden_dims: List[int]
    dropout: float
    aggr: Literal["mean", "max", "sum", "lstm"]
    num_neighbors: List[int]
    learning_rate: float
    max_epochs: int
    batch_size: int
    patience: int
    gradient_clip_val: float
    device: DeviceLiteral
    weight_decay: float = 1e-5
    # null in YAML -> compute the empirical train neg/pos ratio at train time.
    pos_weight: Optional[float] = None
    card1_max_neighbors: int = 10
    addr_product_max_neighbors: int = 5
    graph_cache_path: str = "data/processed/graph/fraud_graph.pt"
    # ADR-006 §3.1 Arm A — richer edges, frozen architecture. "legacy"
    # reproduces the pre-Arm-A graph exactly; "card_full" / "addr_card" each
    # add one new refinement key alongside the legacy pair (never a
    # replacement — see src.data.graph_builder.arm_a_edge_specs). The two
    # *_max_neighbors knobs default to their corresponding legacy cap
    # (resolved in GraphBuildConfig.from_config) when left null.
    edge_spec_set: Literal["legacy", "card_full", "addr_card"] = "legacy"
    card_full_max_neighbors: Optional[int] = None
    addr_card_max_neighbors: Optional[int] = None
    # ADR-006 §3.2 Arm B — deeper/regularized architecture knobs. All default
    # to their pre-Arm-B no-op values so a config predating this change still
    # reproduces identical behavior (see src.models.gnn_model.GraphSAGEModel).
    input_dropout: float = 0.0
    l2_normalize: bool = False
    residual: bool = False
    # Eval-time (per-epoch val PR-AUC + final scoring) NeighborLoader knobs.
    # A full-graph forward over 590k nodes / ~16M edges thrashes an 8 GB card;
    # scoring is batched with a wide deterministic fan-out instead. Defaulted
    # (train_gnn.py derives eval_num_neighbors from num_neighbors when unset).
    eval_num_neighbors: Optional[List[int]] = None
    eval_batch_size: int = 4096
    use_graph_cache: bool = True


class ModelConfig(_StrictModel):
    xgboost: XGBoostConfig
    lightgbm: LightGBMConfig
    tft: TFTConfig
    # Defaulted so a config predating Phase 12 still validates. From Phase 12
    # on, config/config.yaml sets model.gnn explicitly.
    gnn: Optional[GNNConfig] = None


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
    lgbm_model_path: str
    log_file: str
    # PRD Phase 12. Defaulted so a config predating Phase 12 still validates;
    # only read once a GNN artifact exists and is wired into the ensemble.
    gnn_model_path: str = "models/gnn_model.pt"
    # Phase E (ADR-001 §3.5/§4.3). Defaulted so a config written before Phase E
    # still validates; config/config.yaml sets both explicitly.
    transformer_path: str = "models/transformers"
    ensemble_spec_path: str = "models/ensemble.json"


class EnsembleConfig(_StrictModel):
    """2026-08-20 (mle-reviewer review, docs/IMPLEMENTATION_PLAN.md Phase
    F-Audit): parameters for scripts/run_ensemble_eval.py's simplex weight
    search, added when LightGBM was proposed as a 3rd ensemble input.
    """

    # Grid step for the primary (validation) weight search — see
    # src/models/ensemble.py:grid_search_simplex_weights.
    grid_step: float
    refine_step: float
    # LightGBM is only kept in the production blend if the 3-way ensemble
    # beats the 2-way (XGBoost+TFT) baseline's TEST PR-AUC by at least this
    # much, measured once, not tuned against (mle-reviewer: XGBoost and
    # LightGBM share the same features/imbalance strategy, so their
    # diversity — and thus any lift — is not guaranteed the way TFT's was).
    min_lightgbm_lift: float
    bootstrap_resamples: int
    bootstrap_grid_step: float


class ExplainabilityConfig(_StrictModel):
    """PRD Phase 4 (P4-3): TreeSHAP explanations for the served XGBoost
    component. Defaulted end-to-end so a config written before Phase 4 still
    validates; config/config.yaml sets these explicitly.

    `enabled=false` is a real switch — the registry then builds no explainer,
    skips the `shap` import, and every prediction returns an empty
    `explanation`. `enabled=true` with no loadable `xgb` model fails startup
    rather than serving predictions that can never carry an explanation
    (ADR-001 §3.4).
    """

    enabled: bool = True
    # How many risk / mitigating factors to surface per prediction.
    top_k: int = Field(default=5, ge=1)
    # Per-request wall-clock budget for the SHAP call on the scoring path.
    # TreeSHAP on the 171-feature booster is ~140 ms p50 (ecc:mle-reviewer,
    # P4-8), which would otherwise dominate `/predict` latency. On timeout the
    # explanation is dropped exactly like any other explainer failure — the
    # score is unaffected — and `explanation_failures` increments. 0 disables
    # the budget (explain synchronously with no cap).
    timeout_ms: int = Field(default=50, ge=0)


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
    # Phase E / E6. Defaulted so a config written before E6 still validates;
    # config/config.yaml sets both explicitly.
    drift_share_threshold: float = 0.30
    drift_window_rows: int = 5000
    missing_share_threshold: float = 0.20
    # PRD Phase 7.4. Where drift_scheduler.py writes a JSON alert when a run
    # detects drift. Defaulted so a config written before Phase 7 still
    # validates; config/config.yaml sets it explicitly.
    alerts_dir: str = "monitoring/alerts"


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
    ensemble: EnsembleConfig
    kafka: KafkaConfig
    monitoring: MonitoringConfig
    # Defaulted: a config written before PRD Phase 4 still validates.
    explainability: ExplainabilityConfig = ExplainabilityConfig()


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

    raw = _apply_env_overrides(raw)
    return Settings.model_validate(raw)


def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay `_ENV_OVERRIDES` onto the parsed YAML.

    `docker-compose.yml` sets `KAFKA_BOOTSTRAP_SERVERS=kafka:9092`, but nothing
    read it — the in-process consumer used `config.yaml`'s `localhost:9092` and
    could not reach the broker under compose. This is the one code path
    `load_settings` has, so the override applies to every caller (API, producer,
    consumer) identically and flows through `config_hash`.

    Only a set, non-empty env var wins; an unset or blank one leaves the YAML
    value untouched, so local and CI runs are unaffected. Mutates a copy — the
    caller's `raw` dict is left alone.
    """
    if not any(os.environ.get(var, "").strip() for var in _ENV_OVERRIDES.values()):
        return raw

    updated = {**raw}
    for (section, field), env_var in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var, "").strip()
        if not value:
            continue
        current = updated.get(section) or {}
        if current.get(field) == value:
            continue
        updated[section] = {**current, field: value}
        logger.info(
            "config: %s.%s overridden from %s (%r -> %r)",
            section,
            field,
            env_var,
            current.get(field),
            value,
        )
    return updated


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
