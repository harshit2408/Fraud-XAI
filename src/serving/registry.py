"""
src/serving/registry.py

Load-time ownership of every serving artifact (ADR-001 §4.2).

The registry does its work once, at process startup, and then hands out an
immutable snapshot. Its most important job is not loading — it is REFUSING to
load: a container that will not start is strictly better than one quietly
serving an ensemble whose weights came from a different training run than its
models. Every check below fails startup rather than degrading.

Model rollout is therefore a process restart, not an in-place swap. At this
project's scale that is the right trade (ADR-001 §5).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.feature_engineering import FeatureEngineer
from src.serving.ensemble_spec import EnsembleSpec, load_ensemble_spec
from src.serving.metrics import ServingMetrics

logger = logging.getLogger(__name__)

# Model artifacts the blend can name, mapped to their trainer classes and the
# config key holding their path. Imported lazily inside `_load_models` so that
# importing this module does not drag in torch.
_TRAINER_IMPORTS = {
    "xgb": ("src.training.train_xgb", "XGBTrainer", "model_path"),
    "lgbm": ("src.training.train_lgbm", "LGBMTrainer", "lgbm_model_path"),
    "tft": ("src.training.train_tft", "TFTTrainer", "tft_model_path"),
}


@dataclass(frozen=True)
class LoadedModels:
    """Immutable snapshot handed to `InferenceService` (ADR-001 §4.2 step 5)."""

    trainers: Dict[str, Any]
    feature_engineer: FeatureEngineer
    feature_names: List[str]
    ensemble: EnsembleSpec
    model_version: str
    dataset_hash: str
    config_hash: str
    git_sha: Optional[str]
    # PRD Phase 4 (P4-3). TreeSHAP over the XGBoost component only, built once
    # from the loaded booster at startup so it holds no per-request state
    # (safe for the FastAPI sync threadpool). `None` when
    # `explainability.enabled` is false — the serving path then returns
    # predictions with an empty `explanation`.
    explainer: Optional[Any] = None


def build_model_version(
    dataset_hash: str, config_hash: str, git_sha: Optional[str]
) -> str:
    """Compose the serving version string (ADR-001 §4.4).

    Format: ``{dataset_hash[:12]}-{config_hash[:12]}-{git_sha[:7]}``.

    One string resolves the whole serving stack, because startup already
    validated that every artifact shares these hashes. It is returned in every
    prediction response and log line so a decision can be traced back to the
    exact code, config and data that produced it.
    """
    return f"{dataset_hash[:12]}-{config_hash[:12]}-{(git_sha or 'nogit')[:7]}"


class ModelRegistry:
    """Loads, validates, and owns the serving artifacts."""

    def __init__(
        self,
        config: Dict[str, Any],
        transformer_dir: Optional[str] = None,
        ensemble_spec_path: Optional[str] = None,
        metrics: Optional[ServingMetrics] = None,
    ) -> None:
        self.config = config
        # PRD Phase 4 (P4-3). Defaulted so a config written before Phase 4
        # still loads; config/config.yaml sets `explainability` explicitly.
        explain_cfg = config.get("explainability", {})
        self.explainability_enabled: bool = bool(explain_cfg.get("enabled", True))
        self.explainability_top_k: int = int(explain_cfg.get("top_k", 5))
        self.explanation_timeout_ms: int = int(explain_cfg.get("timeout_ms", 50))
        # A config_hash mismatch is a WARNING, not a startup failure (see
        # _validate_consistency). That decision left a genuinely bad deploy
        # discoverable only by grepping logs, so it is also recorded here.
        self.metrics = metrics
        serving_cfg = config.get("serving", {})
        # ADR-001 §3.5: serving reads only from models/, which docker-compose
        # already mounts. data/processed/ is a TRAINING output and is not
        # present in the fraud-api container at all.
        self.transformer_dir = Path(
            transformer_dir
            or serving_cfg.get("transformer_path", "models/transformers")
        )
        self.ensemble_spec_path = Path(
            ensemble_spec_path
            or serving_cfg.get("ensemble_spec_path", "models/ensemble.json")
        )

    def load(self) -> LoadedModels:
        """Load everything, validate cross-artifact consistency, snapshot.

        Raises:
            FileNotFoundError: an artifact or its checksum manifest is missing.
            ValueError: a checksum mismatch, a missing calibrator, or artifacts
                that do not all originate from the same training run.
        """
        spec = load_ensemble_spec(self.ensemble_spec_path)
        required = spec.models_required()
        trainers = self._load_models(required)
        manifests = self._load_manifests(required)
        self._validate_consistency(manifests, spec, trainers)

        feature_engineer = FeatureEngineer()
        feature_engineer.load_transformers(str(self.transformer_dir))

        reference = manifests[required[0]]
        feature_names = self._resolve_feature_names(trainers)
        explainer = self._build_explainer(
            trainers,
            feature_names=feature_names,
            xgb_weight=spec.mode().weights.get("xgb"),
        )
        snapshot = LoadedModels(
            trainers=trainers,
            feature_engineer=feature_engineer,
            feature_names=feature_names,
            ensemble=spec,
            model_version=build_model_version(
                reference["dataset_hash"],
                reference["config_hash"],
                reference.get("git_sha"),
            ),
            dataset_hash=reference["dataset_hash"],
            config_hash=reference["config_hash"],
            git_sha=reference.get("git_sha"),
            explainer=explainer,
        )
        logger.info(
            "ModelRegistry loaded: models=%s, features=%d, model_version=%s, "
            "explainer=%s",
            sorted(trainers),
            len(snapshot.feature_names),
            snapshot.model_version,
            "on" if explainer is not None else "off",
        )
        return snapshot

    def _build_explainer(
        self,
        trainers: Dict[str, Any],
        feature_names: List[str],
        xgb_weight: Optional[float],
    ) -> Optional[Any]:
        """Construct the TreeSHAP explainer over the XGBoost booster, once.

        Returns None when explainability is disabled in config. Raises when it
        is enabled but `xgb` was not loaded — an enabled-but-unexplained
        deployment is a silent gap, so startup fails rather than serving
        predictions with a permanently empty `explanation` (ADR-001 §3.4).
        """
        if not self.explainability_enabled:
            return None
        xgb_trainer = trainers.get("xgb")
        if xgb_trainer is None or getattr(xgb_trainer, "model", None) is None:
            raise ValueError(
                "explainability.enabled is true but no 'xgb' model with a "
                "fitted booster was loaded. TreeSHAP explanations are defined "
                "over the XGBoost component (ADR-001 §3.4); refusing to serve "
                "predictions that can never carry an explanation. Set "
                "explainability.enabled=false to serve without them."
            )
        from src.explainability.shap_explainer import FraudExplainer

        return FraudExplainer(
            xgb_trainer.model,
            feature_names=feature_names,
            top_k=self.explainability_top_k,
            explained_weight=xgb_weight,
        )

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load_models(self, names: List[str]) -> Dict[str, Any]:
        import importlib

        trainers: Dict[str, Any] = {}
        for name in names:
            if name not in _TRAINER_IMPORTS:
                raise ValueError(
                    f"Ensemble spec names unknown model '{name}' "
                    f"(known: {sorted(_TRAINER_IMPORTS)})."
                )
            module_name, class_name, config_key = _TRAINER_IMPORTS[name]
            trainer_cls = getattr(importlib.import_module(module_name), class_name)
            path = self.config.get("serving", {}).get(config_key)
            if not path:
                raise ValueError(
                    f"config serving.{config_key} is not set for model '{name}'."
                )
            trainers[name] = trainer_cls.load(path)
            logger.info("Loaded %s from %s", name, path)
        return trainers

    def _load_manifests(self, names: List[str]) -> Dict[str, Dict[str, Any]]:
        manifests: Dict[str, Dict[str, Any]] = {}
        for name in names:
            _, _, config_key = _TRAINER_IMPORTS[name]
            model_path = Path(self.config["serving"][config_key])
            manifest_path = model_path.parent / f"{model_path.stem}.manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"No provenance manifest for model '{name}' at "
                    f"{manifest_path}. Serving cannot verify that this artifact "
                    "belongs with the others."
                )
            manifests[name] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifests

    # ── Validation (ADR-001 §4.2 step 3) ─────────────────────────────────────

    def _validate_consistency(
        self,
        manifests: Dict[str, Dict[str, Any]],
        spec: EnsembleSpec,
        trainers: Dict[str, Any],
    ) -> None:
        dataset_hashes = {name: m["dataset_hash"] for name, m in manifests.items()}
        if len(set(dataset_hashes.values())) > 1:
            raise ValueError(
                "Model artifacts were trained on different datasets: "
                f"{dataset_hashes}. Refusing to serve a mixed-vintage ensemble."
            )

        # config_hash is deliberately a WARNING, not a hard failure, while
        # dataset_hash above is fatal. `config_hash` covers the whole of
        # config.yaml, so editing one model's hyperparameters changes it for
        # every artifact trained afterwards even though the earlier models are
        # untouched and still valid. That is exactly what the shipped
        # artifacts show: XGBoost and TFT were trained at one config, then
        # `model.lightgbm.early_stopping_rounds` was changed (see the comment
        # in config/config.yaml) and LightGBM and the ensemble were trained at
        # the next. Refusing to serve that would be a false positive.
        #
        # What actually must hold is that every model saw the same DATA and
        # that the blend was fitted against these very models — both checked
        # here as hard failures. A per-model config hash would let this be
        # exact; until then the mismatch is surfaced, not swallowed.
        config_hashes = {name: m["config_hash"] for name, m in manifests.items()}
        if len(set(config_hashes.values())) > 1:
            if self.metrics is not None:
                self.metrics.mark_config_hash_mismatch(config_hashes)
            logger.warning(
                "Model artifacts were trained under different config hashes: %s. "
                "This is expected when config.yaml was edited between training "
                "runs (config_hash covers the whole file, not just the model's "
                "own section). Verify the difference is confined to sections "
                "that do not affect the earlier models.",
                config_hashes,
            )

        model_dataset_hash = next(iter(dataset_hashes.values()))
        if spec.dataset_hash != model_dataset_hash:
            raise ValueError(
                f"Ensemble spec was fitted on dataset {spec.dataset_hash} but "
                f"the models were trained on {model_dataset_hash}. The blend "
                "weights and threshold do not belong to these models."
            )

        # The blend is defined over calibrated probabilities. The training
        # script tolerates a raw fallback for pre-Phase-C4 artifacts; serving
        # must not (ADR-001 §4.2).
        uncalibrated = [
            n for n, t in trainers.items() if getattr(t, "calibrator", None) is None
        ]
        if uncalibrated:
            raise ValueError(
                f"Models {sorted(uncalibrated)} carry no frozen calibrator, but "
                "the ensemble threshold was selected over calibrated "
                "probabilities. Re-train them before serving."
            )

        no_threshold = [
            n for n, t in trainers.items() if getattr(t, "threshold", None) is None
        ]
        if no_threshold:
            logger.warning(
                "Models %s carry no frozen per-model threshold. The ensemble "
                "uses its own threshold, so this does not block serving.",
                sorted(no_threshold),
            )

    @staticmethod
    def _resolve_feature_names(trainers: Dict[str, Any]) -> List[str]:
        """The trained column list, taken from a tree model's persisted names."""
        for name in ("xgb", "lgbm"):
            trainer = trainers.get(name)
            names = getattr(trainer, "feature_names", None) if trainer else None
            if names:
                return list(names)
        raise ValueError(
            "No loaded model carries a persisted feature_names list — the "
            "serving transform has no column contract to align against."
        )
