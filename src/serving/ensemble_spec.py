"""
src/serving/ensemble_spec.py

The ensemble's serving contract (ADR-001 §4.3).

`reports/ensemble_results.json` is an OUTPUT of evaluation. Serving must not
read from `reports/` — it reads a checksummed artifact under `models/` that
carries only what a scoring path needs. This module defines that artifact's
schema and the invariants `ModelRegistry` enforces before any traffic is
served.

Two invariants are worth stating up front, because both are easy to violate by
accident and neither fails visibly at runtime:

  - `probability_space` names WHICH per-model method the weights and threshold
    were fitted against. The deployed blend is fitted over per-model
    *calibrated* probabilities (`scripts/run_ensemble_eval.py`'s `_calibrated`
    helper), so applying its threshold to raw probabilities silently
    thresholds a different quantity. A spec whose `probability_space` this
    code does not implement is rejected rather than assumed.
  - Every degradation mode carries its OWN threshold. Renormalizing weights
    over a smaller model set changes the score distribution, which invalidates
    a threshold selected for the full blend — so a mode that was not
    pre-registered with its own threshold is not servable (ADR-001 §3.3).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from src.utils.checksums import verify_checksums

logger = logging.getLogger(__name__)

SPEC_SCHEMA_VERSION = "1.0"

# The only probability space this serving code implements. Matches
# run_ensemble_eval.py's `_calibrated` helper (and export_test_probabilities).
CALIBRATED_SPACE = "per_model_calibrated"

WEIGHT_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class EnsembleMode:
    """One pre-registered blend: which models, at what weights, at what
    operating point. Frozen — a mode is a promotion decision, not runtime state.
    """

    name: str
    models: List[str]
    weights: Dict[str, float]
    threshold: float

    def validate(self) -> None:
        if not self.models:
            raise ValueError(f"Ensemble mode '{self.name}' names no models.")
        if set(self.models) != set(self.weights):
            raise ValueError(
                f"Ensemble mode '{self.name}': models {sorted(self.models)} do "
                f"not match weight keys {sorted(self.weights)}."
            )
        total = sum(self.weights.values())
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"Ensemble mode '{self.name}': weights sum to {total!r}, not 1.0. "
                "The blend must be a convex combination — `blend()` does not "
                "renormalize."
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"Ensemble mode '{self.name}': threshold {self.threshold!r} is "
                "outside [0, 1]."
            )


@dataclass(frozen=True)
class EnsembleSpec:
    """The full serving contract loaded from `models/ensemble.json`."""

    modes: Dict[str, EnsembleMode]
    default_mode: str
    probability_space: str
    dataset_hash: str
    mlflow_run_id: Optional[str] = None
    schema_version: str = SPEC_SCHEMA_VERSION

    def mode(self, name: Optional[str] = None) -> EnsembleMode:
        """Look up a pre-registered mode.

        Raises:
            KeyError: the mode was never registered with its own threshold —
                serving refuses to improvise an operating point (ADR-001 §3.3).
        """
        key = name or self.default_mode
        if key not in self.modes:
            raise KeyError(
                f"Ensemble mode '{key}' is not pre-registered (available: "
                f"{sorted(self.modes)}). A mode without its own frozen "
                "threshold is not servable."
            )
        return self.modes[key]

    def models_required(self) -> List[str]:
        """Every model any registered mode can call for — the set startup must
        load. A fallback mode is unservable if its models were never loaded, so
        the union across modes (not just the default mode) is what the registry
        needs.
        """
        ordered: List[str] = []
        for mode in self.modes.values():
            for name in mode.models:
                if name not in ordered:
                    ordered.append(name)
        return ordered

    def validate(self) -> None:
        if self.probability_space != CALIBRATED_SPACE:
            raise ValueError(
                f"Ensemble spec declares probability_space="
                f"{self.probability_space!r}, but this serving code only "
                f"implements {CALIBRATED_SPACE!r}. The blend weights and "
                "threshold were fitted in a specific probability space; "
                "applying them in another silently scores the wrong quantity."
            )
        if not self.modes:
            raise ValueError("Ensemble spec registers no modes.")
        if self.default_mode not in self.modes:
            raise ValueError(
                f"default_mode '{self.default_mode}' is not among the "
                f"registered modes {sorted(self.modes)}."
            )
        for mode in self.modes.values():
            mode.validate()


def parse_ensemble_spec(payload: Mapping[str, Any]) -> EnsembleSpec:
    """Build and validate an `EnsembleSpec` from a parsed JSON mapping."""
    try:
        raw_modes = payload["modes"]
        spec = EnsembleSpec(
            modes={
                name: EnsembleMode(
                    name=name,
                    models=list(body["models"]),
                    weights={k: float(v) for k, v in body["weights"].items()},
                    threshold=float(body["threshold"]),
                )
                for name, body in raw_modes.items()
            },
            default_mode=payload["default_mode"],
            probability_space=payload["probability_space"],
            dataset_hash=payload["dataset_hash"],
            mlflow_run_id=payload.get("mlflow_run_id"),
            schema_version=payload.get("schema_version", SPEC_SCHEMA_VERSION),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed ensemble spec: {exc}") from exc

    # A spec written by a newer, incompatible schema must be REJECTED, not
    # silently downgraded. Without this, a future 2.0 spec (e.g. one carrying
    # cascade fields) parses as 1.0 with its extra fields dropped, and serving
    # would score a single stage at what was meant to be a first-stage gate
    # threshold — flagging a third of all traffic while looking healthy. Every
    # other integrity check in this module fails loud (checksum manifest,
    # dataset vintage); this one read the field and ignored it.
    spec_major = str(spec.schema_version).split(".")[0]
    supported_major = SPEC_SCHEMA_VERSION.split(".")[0]
    if spec_major != supported_major:
        raise ValueError(
            f"Unsupported ensemble spec schema_version "
            f"{spec.schema_version!r}: this build supports "
            f"{SPEC_SCHEMA_VERSION!r} (major version {supported_major}). "
            "Refusing to load a spec whose fields this code may not understand."
        )

    spec.validate()
    return spec


def load_ensemble_spec(path: Union[str, Path]) -> EnsembleSpec:
    """Load `models/ensemble.json`, verifying its checksum manifest first.

    Consistent with every other artifact loader added in Phase D6: integrity is
    checked before the bytes are parsed, and a failed check raises rather than
    falling back.

    Raises:
        FileNotFoundError: the spec or its checksum manifest is absent.
        ValueError: checksum mismatch, or the spec violates an invariant.
    """
    spec_path = Path(path)
    checksum_path = spec_path.with_suffix(".checksums.json")

    if not spec_path.exists():
        raise FileNotFoundError(
            f"Ensemble spec not found: {spec_path}. Run "
            "scripts/run_ensemble_eval.py to produce it — serving must not "
            "read blend weights from reports/ (ADR-001 §4.3)."
        )

    verify_checksums(checksum_path, {"ensemble": spec_path})
    return parse_ensemble_spec(json.loads(spec_path.read_text(encoding="utf-8")))
