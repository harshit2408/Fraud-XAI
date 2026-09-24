"""
tests/unit/test_serving_explainability.py

TDD spec for wiring `FraudExplainer` into the serving path (PRD Phase 4,
tasks P4-3 / P4-4).

Written before the implementation. The `FraudExplainer` unit itself is
already covered by `tests/unit/test_shap_explainer.py`; this file pins the
*integration* contract:

  - **Registry builds it once.** `ModelRegistry.load()` constructs the
    explainer from the already-loaded XGBoost booster at startup — not per
    request — and carries the XGBoost blend weight into it so the response
    can say "this explains 69.2% of the decision" honestly.
  - **Fail loud, never silent.** If explainability is enabled in config but
    the `xgb` model was not loaded, startup raises. It does not quietly
    serve un-explained predictions.
  - **Disabled is a real switch.** `explainability.enabled: false` yields a
    registry with `explainer is None` and predictions with an empty
    `explanation` — no SHAP import cost, no per-request work.
  - **Predict populates the response.** With an explainer present,
    `InferenceService.predict()` fills `explanation`, `explained_model`
    (`"xgb"`), and `explained_weight` on the result, and `as_dict()`
    surfaces all three for `PredictionResponse(**result.as_dict())`.
  - **The explanation is of the served vector.** It explains the exact
    engineered feature frame the model scored, not the raw request — same
    contract as the prediction log (E6).
  - **Explainer failure degrades the explanation, not the score.** A raising
    explainer must not take down a scoring request: the decision is still
    returned, with an empty explanation and a logged warning. (A wrong score
    is far worse than a missing explanation.)

Stub trainers are used, as in `test_serving_registry.py` — the behaviour
under test is the wiring, not the 14 MB shipped artifacts.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

xgb = pytest.importorskip("xgboost")
pytest.importorskip("shap")

from src.explainability.shap_explainer import FraudExplainer  # noqa: E402
from src.serving.inference import InferenceService  # noqa: E402
from src.serving.registry import LoadedModels  # noqa: E402

FEATURES = ["amount_log", "amount_zscore_per_card", "pca_v_1", "hour_sin", "card1_freq"]


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def xgb_model() -> "xgb.XGBClassifier":
    rng = np.random.default_rng(42)
    n = 400
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
        n_estimators=30, max_depth=3, learning_rate=0.2, random_state=0
    )
    model.fit(X[FEATURES], y)
    return model


class _StubTrainer:
    """Minimal stand-in for XGBTrainer/LGBMTrainer: a calibrated probability
    and (for xgb) a `.model` the explainer can wrap."""

    def __init__(self, p: float, model: Any = None) -> None:
        self._p = p
        self.model = model
        self.calibrator = object()  # non-None: passes the serving invariant
        self.threshold = 0.5
        self.feature_names = list(FEATURES)

    def predict_proba_calibrated(
        self, X: pd.DataFrame, history_X: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        return np.full(len(X), self._p, dtype=float)


class _StubEnsembleMode:
    def __init__(self) -> None:
        self.name = "full"
        self.models = ["xgb", "lgbm"]
        self.weights = {"xgb": 0.7, "lgbm": 0.3}
        self.threshold = 0.5


class _StubEnsembleSpec:
    def __init__(self) -> None:
        self._mode = _StubEnsembleMode()
        self.default_mode = "full"

    def mode(self, name: Optional[str] = None) -> _StubEnsembleMode:
        return self._mode


class _StubFeatureEngineer:
    card_agg_state: Dict[str, Any] = {}


class _StubStateStore:
    """No-history store: predict() runs end-to-end without a real feature
    state backend."""

    def bind_metrics(self, _metrics: Any) -> None:
        pass

    def snapshot(self, _card_id: Any) -> Any:
        return type("S", (), {"n": 0, "last_dt": None})()

    def sequence(self, _card_id: Any) -> List[Any]:
        return []

    def observe(self, **_kwargs: Any) -> bool:
        return True

    def append_sequence(self, _card_id: Any, _vec: Any) -> None:
        pass


@pytest.fixture
def transaction() -> Dict[str, Any]:
    return {
        "TransactionID": "t-1",
        "TransactionDT": 100_000.0,
        "TransactionAmt": 250.0,
        "card1": 1234,
    }


def _loaded_models(
    xgb_model: "xgb.XGBClassifier",
    *,
    explainer: Optional[FraudExplainer],
) -> LoadedModels:
    return LoadedModels(
        trainers={
            "xgb": _StubTrainer(0.8, model=xgb_model),
            "lgbm": _StubTrainer(0.4),
        },
        feature_engineer=_StubFeatureEngineer(),
        feature_names=list(FEATURES),
        ensemble=_StubEnsembleSpec(),
        model_version="dh-ch-sha",
        dataset_hash="dh",
        config_hash="ch",
        git_sha="sha",
        explainer=explainer,
    )


def _service(models: LoadedModels, **kwargs: Any) -> InferenceService:
    svc = InferenceService(models=models, state_store=_StubStateStore(), **kwargs)
    # Bypass the real ServingFeatureTransformer, which needs fitted
    # transformers on disk. predict() only needs a frame with the trained
    # columns.
    svc.transformer = type(
        "T",
        (),
        {
            "transform": staticmethod(
                lambda raw: pd.DataFrame([[2.0, 1.5, 0.1, -0.3, 0.4]], columns=FEATURES)
            )
        },
    )()
    return svc


# ── LoadedModels carries the explainer ─────────────────────────────────────


class TestLoadedModelsContract:
    def test_loaded_models_has_explainer_field(self, xgb_model):
        models = _loaded_models(
            xgb_model,
            explainer=FraudExplainer(xgb_model, FEATURES, top_k=5),
        )
        assert models.explainer is not None

    def test_explainer_may_be_none_when_disabled(self, xgb_model):
        models = _loaded_models(xgb_model, explainer=None)
        assert models.explainer is None


# ── predict() populates the response ──────────────────────────────────────


class TestPredictPopulatesExplanation:
    def test_explanation_present_when_explainer_loaded(self, xgb_model, transaction):
        explainer = FraudExplainer(xgb_model, FEATURES, top_k=5, explained_weight=0.7)
        svc = _service(_loaded_models(xgb_model, explainer=explainer))

        result = svc.predict(transaction)

        assert result.explanation, "explanation should be a non-empty list"
        assert result.explained_model == "xgb"
        assert result.explained_weight == pytest.approx(0.7)

    def test_explanation_items_have_feature_and_contribution(
        self, xgb_model, transaction
    ):
        explainer = FraudExplainer(xgb_model, FEATURES, top_k=5)
        svc = _service(_loaded_models(xgb_model, explainer=explainer))

        result = svc.predict(transaction)

        for item in result.explanation:
            assert set(item) == {"feature", "contribution"}
            assert isinstance(item["feature"], str)
            assert isinstance(item["contribution"], float)

    def test_as_dict_surfaces_the_new_fields(self, xgb_model, transaction):
        explainer = FraudExplainer(xgb_model, FEATURES, top_k=5, explained_weight=0.7)
        svc = _service(_loaded_models(xgb_model, explainer=explainer))

        d = svc.predict(transaction).as_dict()

        assert "explanation" in d
        assert d["explained_model"] == "xgb"
        assert d["explained_weight"] == pytest.approx(0.7)

    def test_response_schema_accepts_the_result(self, xgb_model, transaction):
        from src.api.schemas.prediction import PredictionResponse

        explainer = FraudExplainer(xgb_model, FEATURES, top_k=5, explained_weight=0.7)
        svc = _service(_loaded_models(xgb_model, explainer=explainer))

        resp = PredictionResponse(**svc.predict(transaction).as_dict())

        assert resp.explained_model == "xgb"
        assert len(resp.explanation) >= 1

    def test_empty_explanation_when_no_explainer(self, xgb_model, transaction):
        svc = _service(_loaded_models(xgb_model, explainer=None))

        result = svc.predict(transaction)

        assert result.explanation == []
        assert result.explained_model is None
        assert result.explained_weight is None

    def test_explains_the_served_vector_not_the_raw_request(
        self, xgb_model, transaction
    ):
        """The contribution set must be over engineered feature names, never
        the raw IEEE-CIS payload keys."""
        explainer = FraudExplainer(xgb_model, FEATURES, top_k=5)
        svc = _service(_loaded_models(xgb_model, explainer=explainer))

        result = svc.predict(transaction)

        explained = {item["feature"] for item in result.explanation}
        assert explained.issubset(set(FEATURES))
        assert "TransactionAmt" not in explained


# ── Failure isolation ────────────────────────────────────────────────────


class TestExplainerFailureIsolation:
    def test_raising_explainer_does_not_break_scoring(
        self, xgb_model, transaction, caplog
    ):
        class _Boom:
            explained_model = "xgb"
            explained_weight = 0.7

            def explain_single(self, _X: pd.DataFrame) -> Any:
                raise RuntimeError("shap blew up")

        svc = _service(_loaded_models(xgb_model, explainer=_Boom()))

        result = svc.predict(transaction)

        # Score still returned; explanation just absent.
        assert result.fraud_probability == pytest.approx(0.8 * 0.7 + 0.4 * 0.3)
        assert result.explanation == []
        assert any(
            "explan" in r.message.lower() for r in caplog.records
        ), "a warning about the failed explanation should be logged"

    def test_raising_explainer_increments_the_failure_counter(
        self, xgb_model, transaction
    ):
        """The 'counts' half of the best-effort contract — every other
        silent-degradation counter in the serving path is pinned this way
        (`test_serving_hardening.py`); the SHAP one must be too, or a refactor
        that drops the increment leaves `/health` reading a permanent zero for
        a globally broken explainer."""

        class _Boom:
            explained_model = "xgb"
            explained_weight = 0.7

            def explain_single(self, _X: pd.DataFrame) -> Any:
                raise RuntimeError("shap blew up")

        svc = _service(_loaded_models(xgb_model, explainer=_Boom()))
        assert svc.metrics.snapshot()["explanation_failures"] == 0

        svc.predict(transaction)

        assert svc.metrics.snapshot()["explanation_failures"] == 1

    def test_successful_explanation_does_not_increment_the_failure_counter(
        self, xgb_model, transaction
    ):
        explainer = FraudExplainer(xgb_model, FEATURES, top_k=5)
        svc = _service(_loaded_models(xgb_model, explainer=explainer))

        svc.predict(transaction)

        assert svc.metrics.snapshot()["explanation_failures"] == 0

    def test_slow_explainer_is_dropped_at_the_time_budget(
        self, xgb_model, transaction, caplog
    ):
        """A SHAP call slower than `explanation_timeout_ms` must not extend the
        scoring latency: the score is returned within budget, the explanation
        is dropped, and the failure is counted (ecc:mle-reviewer P4-8 HIGH —
        unbounded synchronous ~140 ms SHAP on every /predict)."""

        class _Slow:
            explained_model = "xgb"
            explained_weight = 0.7

            def explain_single(self, _X: pd.DataFrame) -> Any:
                time.sleep(1.0)
                raise AssertionError("should have been abandoned at the budget")

        svc = _service(
            _loaded_models(xgb_model, explainer=_Slow()),
            explanation_timeout_ms=50,
        )

        started = time.perf_counter()
        result = svc.predict(transaction)
        elapsed_ms = (time.perf_counter() - started) * 1000

        assert result.fraud_probability == pytest.approx(0.8 * 0.7 + 0.4 * 0.3)
        assert result.explanation == []
        assert result.explained_model is None
        assert (
            elapsed_ms < 500
        ), f"predict() waited on the slow explainer ({elapsed_ms:.0f} ms)"
        assert svc.metrics.snapshot()["explanation_failures"] == 1

    def test_zero_timeout_means_no_budget(self, xgb_model, transaction):
        """`explanation_timeout_ms=0` disables the budget — a real explainer
        still runs to completion."""
        explainer = FraudExplainer(xgb_model, FEATURES, top_k=5)
        svc = _service(
            _loaded_models(xgb_model, explainer=explainer),
            explanation_timeout_ms=0,
        )

        result = svc.predict(transaction)

        assert result.explanation
        assert result.explained_model == "xgb"


# ── Registry construction ────────────────────────────────────────────────


class TestRegistryBuildsExplainer:
    """`ModelRegistry` builds the explainer at load() time. These exercise the
    private builder so the heavy load() path (real artifacts) is not needed.
    """

    def test_builder_returns_none_when_disabled(self, xgb_model):
        from src.serving.registry import ModelRegistry

        reg = ModelRegistry({"serving": {}})
        reg.explainability_enabled = False
        out = reg._build_explainer(
            {"xgb": _StubTrainer(0.8, model=xgb_model)},
            feature_names=list(FEATURES),
            xgb_weight=0.7,
        )
        assert out is None

    def test_builder_builds_from_xgb_when_enabled(self, xgb_model):
        from src.serving.registry import ModelRegistry

        reg = ModelRegistry({"serving": {}})
        reg.explainability_enabled = True
        reg.explainability_top_k = 5
        out = reg._build_explainer(
            {"xgb": _StubTrainer(0.8, model=xgb_model)},
            feature_names=list(FEATURES),
            xgb_weight=0.7,
        )
        assert isinstance(out, FraudExplainer)
        assert out.explained_weight == pytest.approx(0.7)

    def test_builder_raises_when_enabled_but_xgb_absent(self, xgb_model):
        from src.serving.registry import ModelRegistry

        reg = ModelRegistry({"serving": {}})
        reg.explainability_enabled = True
        reg.explainability_top_k = 5
        with pytest.raises(ValueError, match="xgb"):
            reg._build_explainer(
                {"lgbm": _StubTrainer(0.4)},
                feature_names=list(FEATURES),
                xgb_weight=None,
            )
