"""
tests/unit/test_shap_explainer.py

TDD spec for `src.explainability.shap_explainer.FraudExplainer` (PRD Phase 4).

Written before the implementation. Every test states the property it pins:

  - **Shape / contract.** `explain_single` returns exactly the structure the
    API response schema (`src/api/schemas/prediction.py`) and ADR-001 §3.4
    require: a base value, a full signed contribution vector, and a top-K
    split into risk-increasing and risk-decreasing factors.
  - **Additivity.** TreeSHAP is *exact* for a tree model: base_value plus the
    sum of contributions must reconstruct the model's raw margin for that row.
    A ranking without additivity is not an explanation.
  - **Direction / order.** The top risk factors are the largest positive
    contributions in descending magnitude; the top mitigating factors the
    largest negative ones. Signs are not swapped.
  - **Column alignment.** The explainer aligns an incoming frame to the
    trained feature order and raises on a missing column — the same loud
    failure the rest of the serving path takes, never a silent reindex-to-NaN.
  - **Determinism.** Same row in, identical contributions out.
  - **Single-row / degenerate input.** The explain step must survive a
    one-row frame whose columns arrive in the wrong order or with an all-NaN
    column (the train/serve-skew class of bug Phase E hit repeatedly).

The XGBoost model here is tiny and trained in-test, so the suite stays a unit
test with no dependency on the 14MB shipped artifact.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

xgb = pytest.importorskip("xgboost")
pytest.importorskip("shap")

from src.explainability.shap_explainer import FraudExplainer  # noqa: E402

FEATURES = ["amount_log", "amount_zscore_per_card", "pca_v_1", "hour_sin", "card1_freq"]


@pytest.fixture(scope="module")
def trained_model() -> "xgb.XGBClassifier":
    """A small, deterministic binary classifier over `FEATURES`.

    `amount_log` and `amount_zscore_per_card` carry the signal so the tests
    can assert *which* features dominate the explanation.
    """
    rng = np.random.default_rng(42)
    n = 600
    X = pd.DataFrame(
        {
            "amount_log": rng.normal(0, 1, n),
            "amount_zscore_per_card": rng.normal(0, 1, n),
            "pca_v_1": rng.normal(0, 1, n),
            "hour_sin": rng.uniform(-1, 1, n),
            "card1_freq": rng.uniform(0, 1, n),
        }
    )
    logit = 2.5 * X["amount_log"] + 1.8 * X["amount_zscore_per_card"] - 0.5
    y = (1 / (1 + np.exp(-logit)) > rng.uniform(0, 1, n)).astype(int)
    model = xgb.XGBClassifier(
        n_estimators=40, max_depth=3, learning_rate=0.2, random_state=0
    )
    model.fit(X[FEATURES], y)
    return model


@pytest.fixture(scope="module")
def explainer(trained_model) -> FraudExplainer:
    return FraudExplainer(trained_model, feature_names=FEATURES, top_k=3)


def _row(**overrides) -> pd.DataFrame:
    base = {
        "amount_log": 2.0,
        "amount_zscore_per_card": 1.5,
        "pca_v_1": 0.1,
        "hour_sin": -0.3,
        "card1_freq": 0.4,
    }
    base.update(overrides)
    return pd.DataFrame([base])[FEATURES]


# ── Contract / shape ────────────────────────────────────────────────────────


class TestExplainSingleContract:
    def test_returns_base_value_and_full_contribution_vector(self, explainer):
        result = explainer.explain_single(_row())

        assert hasattr(result, "base_value")
        assert isinstance(result.base_value, float)
        # One contribution per trained feature, keyed by feature name.
        assert {c.feature for c in result.contributions} == set(FEATURES)
        assert len(result.contributions) == len(FEATURES)

    def test_top_k_split_is_signed_and_bounded(self, explainer):
        result = explainer.explain_single(_row())

        assert len(result.top_risk_factors) <= 3
        assert len(result.top_mitigating_factors) <= 3
        # Risk factors push toward fraud (positive), mitigating away (negative).
        assert all(c.contribution > 0 for c in result.top_risk_factors)
        assert all(c.contribution < 0 for c in result.top_mitigating_factors)

    def test_top_risk_factors_ordered_by_descending_magnitude(self, explainer):
        result = explainer.explain_single(_row())
        magnitudes = [c.contribution for c in result.top_risk_factors]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_top_mitigating_factors_ordered_by_descending_magnitude(self, explainer):
        result = explainer.explain_single(_row())
        magnitudes = [abs(c.contribution) for c in result.top_mitigating_factors]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_explained_model_labelled_xgb_only(self, explainer):
        """ADR-001 §3.4: the response must not imply it explains the blend."""
        result = explainer.explain_single(_row())
        assert result.explained_model == "xgb"

    def test_as_api_contributions_matches_response_schema(self, explainer):
        """The list handed to `PredictionResponse.explanation` must validate
        against `FeatureContribution` (feature: str, contribution: float)."""
        from src.api.schemas.prediction import FeatureContribution

        result = explainer.explain_single(_row())
        payload = result.as_api_contributions()

        assert isinstance(payload, list)
        assert len(payload) > 0
        for item in payload:
            fc = FeatureContribution(**item)
            assert isinstance(fc.feature, str)
            assert isinstance(fc.contribution, float)


# ── Additivity (TreeSHAP is exact for trees) ───────────────────────────────


class TestAdditivity:
    def test_base_plus_contributions_reconstructs_raw_margin(
        self, explainer, trained_model
    ):
        row = _row()
        result = explainer.explain_single(row)

        expected_margin = float(trained_model.predict(row, output_margin=True)[0])
        recon = result.base_value + sum(c.contribution for c in result.contributions)
        assert recon == pytest.approx(expected_margin, abs=1e-4)

    def test_batch_additivity_holds_row_by_row(self, explainer, trained_model):
        rng = np.random.default_rng(7)
        X = pd.DataFrame(rng.normal(0, 1, (25, len(FEATURES))), columns=FEATURES)

        sv = explainer.explain_batch(X)
        assert sv.shape == (25, len(FEATURES))

        margins = trained_model.predict(X, output_margin=True)
        recon = sv.sum(axis=1) + explainer.base_value
        np.testing.assert_allclose(recon, margins, atol=1e-4)

    def test_additivity_violation_is_not_silently_returned(
        self, explainer, trained_model, monkeypatch
    ):
        """`check_additivity=False` is passed to `shap` for speed, so a future
        shap/xgboost change (or a categorical feature entering the set) that
        broke the constant-offset assumption would let the API emit
        contributions that do not reconstruct the model's decision — a silent
        wrong explanation. `explain_single` must verify additivity on the
        served row and raise instead (ecc:mle-reviewer P4-8 MEDIUM). The raise
        is caught upstream by `InferenceService._explain`, which drops the
        explanation and counts the failure — the score is unaffected."""
        real = explainer._shap_values

        def _corrupt(aligned):
            return real(aligned) * 1.5  # break the sum, keep the shape

        monkeypatch.setattr(explainer, "_shap_values", _corrupt)

        with pytest.raises(ValueError, match="additiv"):
            explainer.explain_single(_row())


# ── Direction ──────────────────────────────────────────────────────────────


class TestDirection:
    def test_high_signal_features_dominate_a_clear_fraud_row(self, explainer):
        """A row with large positive `amount_log` / `amount_zscore_per_card`
        should have those two as its dominant risk factors."""
        result = explainer.explain_single(
            _row(amount_log=4.0, amount_zscore_per_card=3.5)
        )
        top_two = {c.feature for c in result.top_risk_factors[:2]}
        assert "amount_log" in top_two
        assert "amount_zscore_per_card" in top_two

    def test_flipping_the_signal_flips_the_contribution_sign(self, explainer):
        hi = explainer.explain_single(_row(amount_log=4.0))
        lo = explainer.explain_single(_row(amount_log=-4.0))

        def contrib(res, name):
            return next(c.contribution for c in res.contributions if c.feature == name)

        assert contrib(hi, "amount_log") > 0
        assert contrib(lo, "amount_log") < 0


# ── Column alignment ───────────────────────────────────────────────────────


class TestColumnAlignment:
    def test_reorders_columns_to_the_trained_order(self, explainer, trained_model):
        row = _row()
        shuffled = row[list(reversed(FEATURES))]

        r_ordered = explainer.explain_single(row)
        r_shuffled = explainer.explain_single(shuffled)

        for name in FEATURES:
            a = next(
                c.contribution for c in r_ordered.contributions if c.feature == name
            )
            b = next(
                c.contribution for c in r_shuffled.contributions if c.feature == name
            )
            assert a == pytest.approx(b, abs=1e-9)

    def test_missing_trained_column_raises(self, explainer):
        incomplete = _row().drop(columns=["pca_v_1"])
        with pytest.raises((ValueError, KeyError)):
            explainer.explain_single(incomplete)

    def test_extra_columns_are_ignored_not_fed_to_the_model(self, explainer):
        row = _row()
        row["some_unrelated_column"] = 999.0
        result = explainer.explain_single(row)
        assert {c.feature for c in result.contributions} == set(FEATURES)


# ── Determinism ────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_row_same_contributions(self, explainer):
        row = _row()
        a = explainer.explain_single(row)
        b = explainer.explain_single(row.copy())
        for ca, cb in zip(a.contributions, b.contributions):
            assert ca.feature == cb.feature
            assert ca.contribution == pytest.approx(cb.contribution, abs=1e-12)


# ── Degenerate single-row input ────────────────────────────────────────────


class TestDegenerateInput:
    def test_all_nan_column_does_not_crash_the_explainer(self, explainer):
        """XGBoost treats NaN as a routed missing value; the explainer must
        not raise on it (batch never sees an all-NaN column, a one-row
        request can)."""
        row = _row()
        row["pca_v_1"] = np.nan
        result = explainer.explain_single(row)
        assert len(result.contributions) == len(FEATURES)

    def test_rejects_a_multi_row_frame_for_explain_single(self, explainer):
        two = pd.concat([_row(), _row()], ignore_index=True)
        with pytest.raises(ValueError):
            explainer.explain_single(two)
