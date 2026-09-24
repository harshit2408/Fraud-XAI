"""
tests/unit/test_shap_dashboard_generator.py

TDD spec for the standalone SHAP dashboard (PRD Phase 4, task P4-5).

Written before the implementation. `FraudExplainer` itself is covered by
`tests/unit/test_shap_explainer.py`; this file pins the *dashboard* contract:

  - **One self-contained artifact.** `generate_dashboard()` writes a single
    HTML file with every figure inlined as a base64 data URI — no sidecar
    PNGs, no CDN, no network. It opens from a file:// path with nothing else
    on disk. (Same "one file, no dependencies" rule the repo already applies
    to the drift report and the future Kafka dashboards.)
  - **Global importance is mean(|SHAP|) over the sample**, in descending
    order, and the table in the HTML lists exactly the top `max_features` of
    them.
  - **It says what it explains.** The XGBoost-only scope (ADR-001 §3.4) and
    the blend weight are printed in the page, so a reader can never mistake it
    for an explanation of the ensemble decision.
  - **The row count it explained is reported**, so a dashboard built from a
    truncated sample is self-evidently that.
  - **Empty input fails loudly** rather than emitting a blank page.

The generator is handed an already-constructed `FraudExplainer` and an
engineered feature frame; it does no model loading and no feature
engineering of its own (that is the CLI script's job, P4-5 part 2).
"""

import base64
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

xgb = pytest.importorskip("xgboost")
pytest.importorskip("shap")
pytest.importorskip("matplotlib")

from src.explainability.shap_explainer import FraudExplainer  # noqa: E402
from src.explainability.dashboard_generator import (  # noqa: E402
    DashboardData,
    build_dashboard_data,
    generate_dashboard,
)

FEATURES = ["amount_log", "amount_zscore_per_card", "pca_v_1", "hour_sin", "card1_freq"]


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "amount_log": rng.normal(0, 1, n),
            "amount_zscore_per_card": rng.normal(0, 1, n),
            "pca_v_1": rng.normal(0, 1, n),
            "hour_sin": rng.uniform(-1, 1, n),
            "card1_freq": rng.uniform(0, 1, n),
        }
    )[FEATURES]


@pytest.fixture(scope="module")
def explainer(sample: pd.DataFrame) -> FraudExplainer:
    rng = np.random.default_rng(0)
    logit = 2.5 * sample["amount_log"] + 1.8 * sample["amount_zscore_per_card"] - 0.5
    y = (1 / (1 + np.exp(-logit)) > rng.uniform(0, 1, len(sample))).astype(int)
    model = xgb.XGBClassifier(
        n_estimators=40, max_depth=3, learning_rate=0.2, random_state=0
    )
    model.fit(sample, y)
    return FraudExplainer(model, FEATURES, top_k=5, explained_weight=0.692)


# ── build_dashboard_data ───────────────────────────────────────────────────


class TestBuildDashboardData:
    def test_returns_dashboard_data(self, explainer, sample):
        data = build_dashboard_data(explainer, sample)
        assert isinstance(data, DashboardData)

    def test_global_importance_is_mean_abs_shap_descending(self, explainer, sample):
        data = build_dashboard_data(explainer, sample)

        shap_values = explainer.explain_batch(sample)
        expected = pd.Series(
            np.abs(shap_values).mean(axis=0), index=FEATURES
        ).sort_values(ascending=False)

        got = data.global_importance
        assert list(got.index) == list(expected.index)
        np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-6)
        # The signal features the fixture actually drives on should top it.
        assert set(got.index[:2]) == {"amount_log", "amount_zscore_per_card"}

    def test_reports_row_count_and_scope(self, explainer, sample):
        data = build_dashboard_data(explainer, sample)
        assert data.n_rows == len(sample)
        assert data.explained_model == "xgb"
        assert data.explained_weight == pytest.approx(0.692)

    def test_empty_sample_raises(self, explainer):
        with pytest.raises(ValueError, match="empty|no rows"):
            build_dashboard_data(explainer, pd.DataFrame(columns=FEATURES))


# ── generate_dashboard ─────────────────────────────────────────────────────


class TestGenerateDashboard:
    def test_writes_single_self_contained_html(self, explainer, sample, tmp_path):
        out = tmp_path / "shap_dashboard.html"
        generate_dashboard(explainer, sample, out)

        assert out.exists()
        # Nothing else written beside it.
        assert [p.name for p in tmp_path.iterdir()] == ["shap_dashboard.html"]

        html = out.read_text(encoding="utf-8")
        assert html.lstrip().lower().startswith("<!doctype html>")
        # Every image is an inline data URI — no external src.
        srcs = re.findall(r"<img[^>]+src=\"([^\"]+)\"", html)
        assert srcs, "dashboard should embed at least one figure"
        assert all(s.startswith("data:image/") for s in srcs)
        # And each one decodes as real PNG bytes.
        for s in srcs:
            payload = s.split(",", 1)[1]
            assert base64.b64decode(payload)[:8] == b"\x89PNG\r\n\x1a\n"

    def test_html_states_scope_and_weight_and_rowcount(
        self, explainer, sample, tmp_path
    ):
        out = tmp_path / "d.html"
        generate_dashboard(explainer, sample, out)
        html = out.read_text(encoding="utf-8")

        assert "XGBoost" in html
        assert "69.2%" in html or "0.692" in html
        assert str(len(sample)) in html
        # ADR reference so the scope caveat is traceable.
        assert "ADR-001" in html

    def test_top_features_table_lists_max_features_rows(
        self, explainer, sample, tmp_path
    ):
        out = tmp_path / "d.html"
        generate_dashboard(explainer, sample, out, max_features=3)
        html = out.read_text(encoding="utf-8")

        rows = re.findall(r"<tr[^>]*>.*?</tr>", html, flags=re.DOTALL)
        body = [r for r in rows if "<td" in r]
        assert len(body) == 3
        assert "amount_log" in html

    def test_returns_the_output_path(self, explainer, sample, tmp_path):
        out = tmp_path / "d.html"
        returned = generate_dashboard(explainer, sample, out)
        assert Path(returned) == out

    def test_creates_parent_directory(self, explainer, sample, tmp_path):
        out = tmp_path / "nested" / "deeper" / "d.html"
        generate_dashboard(explainer, sample, out)
        assert out.exists()
