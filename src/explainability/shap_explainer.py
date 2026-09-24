"""
src/explainability/shap_explainer.py

TreeSHAP explanations for the XGBoost component of the fraud ensemble
(PRD Phase 4, ADR-001 §3.4).

Scope, stated once so it is not lost downstream:

  - This explains **the XGBoost model only**, which carries 0.692 of the blend
    (`models/ensemble.json`). It is *not* an explanation of the ensemble
    decision. ADR-001 §3.4 rejected KernelSHAP-over-the-blend on latency and
    chose this; every `ExplanationResult` carries `explained_model="xgb"` so
    the API response can say so honestly.
  - TreeSHAP is **exact** for a tree model — no background dataset, no
    sampling. `base_value + Σ contributions` reconstructs the model's raw
    margin (log-odds) for the row. The contributions are therefore in
    log-odds units, not probability units; the API surfaces them as signed
    `FeatureContribution` values, largest-magnitude first.

The explainer is built **once**, from the already-loaded booster, at
`ModelRegistry` startup (ADR-001 §4.2 step 4) — not per request. It holds no
mutable per-request state, so it is safe to call from FastAPI's sync
threadpool without a lock (the thread-safety lesson from the Phase E review
rounds).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The component this explainer speaks for. Kept as a constant so the API
# response and the dashboard cannot disagree about which model was explained.
EXPLAINED_MODEL = "xgb"
DEFAULT_TOP_K = 5

# `base_value + Σcontributions` must reconstruct the booster's raw margin to
# within this (log-odds units). TreeSHAP is exact, so the residual is pure
# float error — observed max ~3e-5 over 5,000 real rows (ecc:mle-reviewer,
# P4-8). A residual above this means the constant-offset assumption in
# `_calibrate_base_value` no longer holds (a shap/xgboost change, a categorical
# feature entering the set) and the explanation would be quietly wrong.
ADDITIVITY_TOLERANCE = 1e-2


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's signed push on the XGBoost log-odds for a single row.

    Mirrors `src.api.schemas.prediction.FeatureContribution` field-for-field so
    `ExplanationResult.as_api_contributions()` validates against it directly.
    """

    feature: str
    contribution: float


@dataclass(frozen=True)
class ExplanationResult:
    """The explanation for one scored transaction.

    `base_value` and `contributions` are in **raw margin (log-odds) space**:
    ``base_value + sum(c.contribution) == model.predict(row, output_margin=True)``.
    """

    base_value: float
    contributions: List[FeatureContribution]
    top_risk_factors: List[FeatureContribution]
    top_mitigating_factors: List[FeatureContribution]
    explained_model: str = EXPLAINED_MODEL
    explained_weight: float | None = None

    def as_api_contributions(self) -> List[Dict[str, Any]]:
        """The payload for `PredictionResponse.explanation`: the union of the
        top risk and mitigating factors, largest magnitude first, as plain
        dicts ready for `FeatureContribution(**item)`."""
        merged = sorted(
            [*self.top_risk_factors, *self.top_mitigating_factors],
            key=lambda c: abs(c.contribution),
            reverse=True,
        )
        return [
            {"feature": c.feature, "contribution": float(c.contribution)}
            for c in merged
        ]


class FraudExplainer:
    """TreeSHAP over the XGBoost booster.

    Args:
        model: a fitted `xgboost.XGBClassifier` (the `XGBTrainer.model` object).
        feature_names: the trained column order. An incoming frame is aligned
            to this; a missing column raises rather than being reindexed to NaN.
        top_k: how many risk / mitigating factors to surface per explanation.
        explained_weight: the model's blend weight, carried into the result for
            the API to report ("this explains 69.2% of the decision").
    """

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        top_k: int = DEFAULT_TOP_K,
        explained_weight: float | None = None,
    ) -> None:
        # Local import: keep module import cheap for callers that never explain.
        import shap

        if not feature_names:
            raise ValueError("feature_names must be a non-empty sequence.")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        self.feature_names: List[str] = list(feature_names)
        self.top_k = int(top_k)
        self.explained_weight = explained_weight
        self._model = model
        self._explainer = shap.TreeExplainer(model)
        self.base_value: float = self._calibrate_base_value(model)
        logger.info(
            "FraudExplainer ready: %d features, base_value=%.6f (raw margin), "
            "explains model=%s weight=%s",
            len(self.feature_names),
            self.base_value,
            EXPLAINED_MODEL,
            explained_weight,
        )

    def _calibrate_base_value(self, model: Any) -> float:
        """The additive intercept, derived so ``base_value + Σ contributions``
        equals the model's raw margin *exactly*.

        `shap 0.44`'s `TreeExplainer.expected_value` for an `XGBClassifier` is
        unreliable — on the shipped 171-feature booster (shap 0.44.0 / xgboost
        2.0.3) it returns ``[0.]``, not the booster's ~0.86 log-odds intercept
        (ecc:mle-reviewer, P4-8). SHAP still returns correct *contributions*
        (path-dependent Shapley values are internally consistent), so
        ``margin_row - Σshap_row`` is a constant across rows — measure it on a
        few reference rows and use that. Any row works; a spread of rows
        (not just zeros) is used so a future shap version that breaks the
        constant-offset property is caught here rather than in production.
        """
        reference = pd.DataFrame(
            [
                [0.0] * len(self.feature_names),
                [1.0] * len(self.feature_names),
                [-1.0] * len(self.feature_names),
            ],
            columns=self.feature_names,
        )
        shap_ref = self._shap_values(reference)
        margin_ref = np.asarray(
            model.predict(reference, output_margin=True), dtype=float
        )
        offsets = margin_ref - shap_ref.sum(axis=1)
        spread = float(np.ptp(offsets))
        if spread > ADDITIVITY_TOLERANCE:
            raise ValueError(
                "TreeSHAP additivity offset is not constant across reference "
                f"rows (spread {spread:.3g} > {ADDITIVITY_TOLERANCE:g} log-odds). "
                "`_calibrate_base_value` assumes a single row-invariant "
                "intercept; this shap/xgboost combination has broken that "
                "assumption and explanations would not reconstruct the model "
                "decision."
            )
        return float(offsets.mean())

    def _assert_additive(self, aligned: pd.DataFrame, shap_row: np.ndarray) -> None:
        """Verify ``base_value + Σshap`` reconstructs the booster margin for the
        *served* row, not just the calibration reference.

        `_shap_values` passes ``check_additivity=False`` for speed, so without
        this a shap/xgboost change that broke the constant-offset assumption
        would let the API emit contributions that do not sum to the model's
        decision — a silent wrong explanation (ecc:mle-reviewer, P4-8). Raising
        here is caught by `InferenceService._explain`, which drops the
        explanation and increments `explanation_failures`; the score is
        unaffected.
        """
        recon = self.base_value + float(np.sum(shap_row))
        margin = float(self._model.predict(aligned, output_margin=True)[0])
        residual = abs(recon - margin)
        if residual > ADDITIVITY_TOLERANCE:
            raise ValueError(
                f"SHAP additivity check failed: base_value + Σcontributions "
                f"({recon:.6f}) does not reconstruct the XGBoost margin "
                f"({margin:.6f}); residual {residual:.3g} > "
                f"{ADDITIVITY_TOLERANCE:g} log-odds. The explanation would not "
                "match the model decision — refusing to return it."
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def explain_single(self, X: pd.DataFrame) -> ExplanationResult:
        """Explain exactly one transaction.

        Raises:
            ValueError: `X` does not have exactly one row, or a trained
                feature column is missing.
        """
        if len(X) != 1:
            raise ValueError(
                f"explain_single expects exactly one row, got {len(X)}. "
                "Use explain_batch for multiple."
            )
        aligned = self._align(X)
        shap_row = self._shap_values(aligned)[0]
        self._assert_additive(aligned, shap_row)

        contributions = [
            FeatureContribution(feature=name, contribution=float(value))
            for name, value in zip(self.feature_names, shap_row)
        ]
        positives = sorted(
            (c for c in contributions if c.contribution > 0),
            key=lambda c: c.contribution,
            reverse=True,
        )
        negatives = sorted(
            (c for c in contributions if c.contribution < 0),
            key=lambda c: c.contribution,  # most negative first
        )
        return ExplanationResult(
            base_value=self.base_value,
            contributions=contributions,
            top_risk_factors=positives[: self.top_k],
            top_mitigating_factors=negatives[: self.top_k],
            explained_model=EXPLAINED_MODEL,
            explained_weight=self.explained_weight,
        )

    def explain_batch(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values for every row: an ``(n_rows, n_features)`` array in the
        trained column order. Used for summary / beeswarm / dependence plots."""
        aligned = self.align(X)
        return self._shap_values(aligned)

    def align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Public: project a frame to the trained feature order (see `_align`).

        Exposed so a caller that needs both the SHAP matrix *and* the aligned
        feature frame (the dashboard's beeswarm) derives them from one
        alignment rather than re-implementing it and risking divergence.
        """
        return self._align(X)

    # ── Internals ────────────────────────────────────────────────────────────

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project to the trained feature order; raise on a missing column.

        Extra columns are dropped (the served frame may legitimately carry
        more than the model uses); a *missing* trained column is the loud
        failure the rest of the serving path also takes (ADR-001 §4.5 step 3).
        This is a column selection, never a pandas reindex — an absent trained
        column is rejected, not filled with NaN.
        """
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(
                f"Cannot explain: input is missing trained feature column(s) "
                f"{missing}."
            )
        return X.loc[:, self.feature_names]

    def _shap_values(self, aligned: pd.DataFrame) -> np.ndarray:
        """`(n_rows, n_features)` TreeSHAP array.

        For a binary `XGBClassifier`, shap 0.44's `TreeExplainer.shap_values`
        returns a single 2-D array (the positive class). Older/other shap
        builds return a length-2 list; take element 1 defensively.
        """
        values = self._explainer.shap_values(aligned, check_additivity=False)
        if isinstance(values, list):
            values = values[1] if len(values) == 2 else values[0]
        return np.asarray(values, dtype=float)
