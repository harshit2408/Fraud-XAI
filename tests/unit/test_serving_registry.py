"""
tests/unit/test_serving_registry.py

Phase E task E3, and the fail-closed guarantees ADR-001 §4.2 depends on.

The registry's job is mostly REFUSAL, so that is what these tests exercise.
Each rejected case below is a deployment that would otherwise come up healthy
and serve confident, wrong scores:

  - models from different training runs blended with one set of weights,
  - an ensemble spec fitted against a different dataset than the models,
  - a blend applied in a probability space its threshold was not fitted in,
  - a degradation mode invented at request time with no frozen threshold.

They use stub trainers rather than the real 14 MB artifacts: the behaviour
under test is the validation logic, and binding it to real files would make it
a slow integration test that could not exercise the failure cases at all.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.ensemble_spec import (
    CALIBRATED_SPACE,
    EnsembleMode,
    load_ensemble_spec,
    parse_ensemble_spec,
)
from src.serving.registry import ModelRegistry, build_model_version
from src.utils.checksums import write_checksums

DATASET_HASH = "4c1059dcd5d48c8bb45300e2b5da8c54"
CONFIG_HASH = "24e774db15acc563a48443268b9f31ef"
GIT_SHA = "f73b8af1cb0d1a77a7f9228b91c51bd26055b92d"


def _spec_payload(**overrides):
    payload = {
        "schema_version": "1.0",
        "modes": {
            "full": {
                "models": ["xgb", "tft", "lgbm"],
                "weights": {"xgb": 0.692, "tft": 0.134, "lgbm": 0.174},
                "threshold": 0.006123399927495639,
            },
            "no_tft": {
                "models": ["xgb", "lgbm"],
                "weights": {"xgb": 0.8, "lgbm": 0.2},
                "threshold": 0.0071,
            },
        },
        "default_mode": "full",
        "probability_space": CALIBRATED_SPACE,
        "dataset_hash": DATASET_HASH,
        "mlflow_run_id": "0fb89a9beb3f4dbd8883dce82819979f",
    }
    payload.update(overrides)
    return payload


class TestModelVersion:
    def test_composes_the_documented_format(self):
        """ADR-001 §4.4: dataset[:12]-config[:12]-git[:7]."""
        assert build_model_version(DATASET_HASH, CONFIG_HASH, GIT_SHA) == (
            "4c1059dcd5d4-24e774db15ac-f73b8af"
        )

    def test_missing_git_sha_is_marked_not_omitted(self):
        """`resolve_git_sha` may legitimately return None; the version must
        still be a complete, unambiguous string."""
        version = build_model_version(DATASET_HASH, CONFIG_HASH, None)
        assert version.endswith("-nogit")


class TestEnsembleSpecInvariants:
    def test_valid_spec_round_trips(self):
        spec = parse_ensemble_spec(_spec_payload())
        assert spec.mode().name == "full"
        assert spec.mode("no_tft").threshold == pytest.approx(0.0071)
        assert set(spec.models_required()) == {"xgb", "tft", "lgbm"}

    def test_unregistered_mode_is_refused(self):
        """ADR-001 §3.3: serving never improvises an operating point."""
        spec = parse_ensemble_spec(_spec_payload())
        with pytest.raises(KeyError, match="not pre-registered"):
            spec.mode("xgb_only")

    def test_raw_probability_space_is_refused(self):
        """The weights and threshold were fitted over calibrated probabilities;
        applying them to raw output silently scores a different quantity."""
        with pytest.raises(ValueError, match="probability_space"):
            parse_ensemble_spec(_spec_payload(probability_space="per_model_raw"))

    def test_weights_that_do_not_sum_to_one_are_refused(self):
        """`blend()` does not renormalize, so a non-convex combination would
        silently shift every score."""
        payload = _spec_payload()
        payload["modes"]["full"]["weights"] = {"xgb": 0.5, "tft": 0.1, "lgbm": 0.1}
        with pytest.raises(ValueError, match="sum to"):
            parse_ensemble_spec(payload)

    def test_weight_keys_must_match_the_named_models(self):
        payload = _spec_payload()
        payload["modes"]["full"]["models"] = ["xgb", "tft"]
        with pytest.raises(ValueError, match="do not match"):
            parse_ensemble_spec(payload)

    def test_default_mode_must_exist(self):
        with pytest.raises(ValueError, match="default_mode"):
            parse_ensemble_spec(_spec_payload(default_mode="ghost"))

    def test_threshold_outside_zero_one_is_refused(self):
        payload = _spec_payload()
        payload["modes"]["full"]["threshold"] = 1.5
        with pytest.raises(ValueError, match="outside"):
            parse_ensemble_spec(payload)

    def test_mode_is_frozen(self):
        mode = EnsembleMode("full", ["xgb"], {"xgb": 1.0}, 0.5)
        with pytest.raises((AttributeError, TypeError)):
            mode.threshold = 0.9  # type: ignore[misc]


class TestEnsembleSpecLoading:
    def test_checksum_is_verified_before_parsing(self, tmp_path):
        spec_path = tmp_path / "ensemble.json"
        spec_path.write_text(json.dumps(_spec_payload()), encoding="utf-8")
        write_checksums(tmp_path / "ensemble.checksums.json", {"ensemble": spec_path})

        assert load_ensemble_spec(spec_path).default_mode == "full"

        spec_path.write_text(
            json.dumps(_spec_payload(default_mode="no_tft")), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Checksum mismatch"):
            load_ensemble_spec(spec_path)

    def test_missing_spec_names_the_producing_script(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run_ensemble_eval"):
            load_ensemble_spec(tmp_path / "absent.json")


class _StubTrainer:
    def __init__(self, calibrator="fitted", threshold=0.5, feature_names=None):
        self.calibrator = calibrator
        self.threshold = threshold
        self.feature_names = feature_names if feature_names is not None else ["f1", "f2"]


class TestCrossArtifactValidation:
    """ADR-001 §4.2 step 3 — every one of these must fail startup."""

    @pytest.fixture()
    def registry(self):
        return ModelRegistry(config={"serving": {}})

    def _manifests(self, **overrides):
        base = {
            name: {
                "dataset_hash": DATASET_HASH,
                "config_hash": CONFIG_HASH,
                "git_sha": GIT_SHA,
            }
            for name in ("xgb", "tft", "lgbm")
        }
        for name, patch in overrides.items():
            base[name] = {**base[name], **patch}
        return base

    def test_consistent_artifacts_pass(self, registry):
        registry._validate_consistency(
            self._manifests(),
            parse_ensemble_spec(_spec_payload()),
            {n: _StubTrainer() for n in ("xgb", "tft", "lgbm")},
        )

    def test_mixed_dataset_hashes_are_refused(self, registry):
        with pytest.raises(ValueError, match="different datasets"):
            registry._validate_consistency(
                self._manifests(tft={"dataset_hash": "deadbeef"}),
                parse_ensemble_spec(_spec_payload()),
                {n: _StubTrainer() for n in ("xgb", "tft", "lgbm")},
            )

    def test_mixed_config_hashes_warn_but_do_not_block(self, registry, caplog):
        """config_hash covers the WHOLE config file, so editing one model's
        hyperparameters changes it for every artifact trained afterwards even
        though the earlier models are untouched. The shipped artifacts show
        exactly this: XGBoost/TFT at ad07f4af8389, then LightGBM's
        early_stopping_rounds was changed, then LightGBM/ensemble at
        24e774db15ac. Failing closed on that would be a false positive; the
        dataset_hash check above is the one that must be fatal."""
        with caplog.at_level("WARNING"):
            registry._validate_consistency(
                self._manifests(lgbm={"config_hash": "deadbeef"}),
                parse_ensemble_spec(_spec_payload()),
                {n: _StubTrainer() for n in ("xgb", "tft", "lgbm")},
            )
        assert "different config hashes" in caplog.text

    def test_spec_fitted_on_another_dataset_is_refused(self, registry):
        with pytest.raises(ValueError, match="do not belong to these models"):
            registry._validate_consistency(
                self._manifests(),
                parse_ensemble_spec(_spec_payload(dataset_hash="deadbeef")),
                {n: _StubTrainer() for n in ("xgb", "tft", "lgbm")},
            )

    def test_uncalibrated_model_is_refused(self, registry):
        """run_ensemble_eval tolerates a raw fallback for pre-Phase-C4
        artifacts; serving must not (ADR-001 §4.2)."""
        trainers = {n: _StubTrainer() for n in ("xgb", "tft", "lgbm")}
        trainers["tft"] = _StubTrainer(calibrator=None)
        with pytest.raises(ValueError, match="no frozen calibrator"):
            registry._validate_consistency(
                self._manifests(), parse_ensemble_spec(_spec_payload()), trainers
            )

    def test_missing_per_model_threshold_only_warns(self, registry, caplog):
        """The ensemble uses its own threshold, so this is not fatal — but it
        must be visible."""
        trainers = {n: _StubTrainer() for n in ("xgb", "tft", "lgbm")}
        trainers["lgbm"] = _StubTrainer(threshold=None)
        with caplog.at_level("WARNING"):
            registry._validate_consistency(
                self._manifests(), parse_ensemble_spec(_spec_payload()), trainers
            )
        assert "no frozen per-model threshold" in caplog.text


class TestFeatureNameResolution:
    def test_prefers_a_tree_model_feature_list(self):
        names = ModelRegistry._resolve_feature_names(
            {"xgb": _StubTrainer(feature_names=["a", "b", "c"]), "tft": _StubTrainer()}
        )
        assert names == ["a", "b", "c"]

    def test_raises_when_no_model_carries_a_column_contract(self):
        with pytest.raises(ValueError, match="feature_names"):
            ModelRegistry._resolve_feature_names(
                {"tft": _StubTrainer(feature_names=[])}
            )
