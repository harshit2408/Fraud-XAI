"""
src/explainability/dashboard_generator.py

The standalone SHAP dashboard (PRD Phase 4, task P4-5).

Renders one self-contained HTML file — global feature importance, a SHAP
beeswarm, and a top-factors table — from a `FraudExplainer` and a sample of
engineered feature rows. Every figure is inlined as a base64 PNG: the file
opens from `file://` with nothing else on disk, no CDN, no network. This is
the same "one artifact, no dependencies" rule the drift report already
follows, and it means the dashboard can be attached to a review or a
regulator packet as a single download.

Scope, restated so the artifact cannot be misread (ADR-001 §3.4): this
explains the **XGBoost component only** (0.692 of the blend). It is not an
explanation of the ensemble decision. The scope line and the weight are
printed on the page.

The generator does no model loading and no feature engineering — it is handed
a ready `FraudExplainer` and an already-transformed frame. Wiring it to the
real artifacts is `scripts/generate_shap_dashboard.py`.
"""

from __future__ import annotations

import base64
import html
import io
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: figures render straight to PNG, no display

if TYPE_CHECKING:  # avoid importing shap at module import time
    from src.explainability.shap_explainer import FraudExplainer

logger = logging.getLogger(__name__)

# Global-importance bars and the beeswarm both get truncated to this many
# features by default — a 171-feature axis is unreadable and the tail is
# noise. Overridable per call.
DEFAULT_MAX_FEATURES = 15

# Beeswarm point budget. TreeSHAP is exact and cheap, but a scatter of
# 100k points renders to a smear and bloats the base64 payload; sample down.
BEESWARM_MAX_POINTS = 2_000


@dataclass(frozen=True)
class DashboardData:
    """Everything the HTML template needs, computed once from the sample.

    `global_importance` is mean(|SHAP|) per feature over the sample, in
    descending order (a `pd.Series` indexed by feature name). `shap_values`
    and `feature_sample` are the full aligned arrays kept for the beeswarm.
    """

    global_importance: pd.Series
    shap_values: np.ndarray
    feature_sample: pd.DataFrame
    n_rows: int
    explained_model: str
    explained_weight: float | None
    base_value: float


def build_dashboard_data(
    explainer: "FraudExplainer", sample: pd.DataFrame
) -> DashboardData:
    """Run TreeSHAP over `sample` and reduce it to the dashboard's inputs.

    Raises:
        ValueError: `sample` has no rows. A dashboard from an empty sample
            would be a blank page asserting nothing — fail instead.
    """
    if sample is None or len(sample) == 0:
        raise ValueError(
            "Cannot build a SHAP dashboard from an empty sample (no rows). "
            "Pass a non-empty frame of engineered feature rows."
        )

    from src.explainability.shap_explainer import EXPLAINED_MODEL

    # One alignment for both the SHAP matrix and the feature frame the beeswarm
    # colours by — `explain_batch` aligns internally the same way, so deriving
    # `aligned` from the same helper keeps the two from drifting apart.
    aligned = explainer.align(sample)
    shap_values = explainer.explain_batch(aligned)

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=explainer.feature_names
    ).sort_values(ascending=False)

    logger.info(
        "SHAP dashboard data: %d rows, %d features, top factor=%s (mean|SHAP|=%.4f)",
        len(sample),
        len(explainer.feature_names),
        importance.index[0],
        importance.iloc[0],
    )
    return DashboardData(
        global_importance=importance,
        shap_values=np.asarray(shap_values, dtype=float),
        feature_sample=aligned,
        n_rows=len(sample),
        explained_model=EXPLAINED_MODEL,
        explained_weight=explainer.explained_weight,
        base_value=float(explainer.base_value),
    )


def generate_dashboard(
    explainer: "FraudExplainer",
    sample: pd.DataFrame,
    out_path: str | Path,
    *,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> Path:
    """Write the self-contained HTML dashboard and return its path.

    Args:
        explainer: a constructed `FraudExplainer` (XGBoost component).
        sample: engineered feature rows, trained column order (extra columns
            are tolerated and dropped by the explainer's own alignment).
        out_path: destination `.html`. Parent directories are created.
        max_features: rows in the importance table / bars, and features on
            the beeswarm axis.

    Raises:
        ValueError: `sample` is empty, or `max_features < 1`.
    """
    if max_features < 1:
        raise ValueError(f"max_features must be >= 1, got {max_features}.")

    data = build_dashboard_data(explainer, sample)
    top = data.global_importance.head(max_features)

    bar_png = _bar_figure(top)
    beeswarm_png = _beeswarm_figure(data, list(top.index))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _render_html(data, top, bar_png, beeswarm_png, max_features),
        encoding="utf-8",
    )
    logger.info("SHAP dashboard written to %s", out)
    return out


# ── Figures ────────────────────────────────────────────────────────────────


def _figure_to_data_uri(fig) -> str:
    """Serialise a Matplotlib figure to a base64 PNG data URI, always closing
    it — a `savefig` error must not leak the figure into pyplot's global
    registry (a batch caller would then accumulate open figures until it
    warns / leaks)."""
    import matplotlib.pyplot as plt

    try:
        with io.BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    finally:
        plt.close(fig)


def _bar_figure(top: pd.Series) -> str:
    """Horizontal bar chart of mean(|SHAP|), most important at the top."""
    import matplotlib.pyplot as plt

    ordered = top.iloc[::-1]  # barh puts the first row at the bottom
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(ordered))))
    ax.barh(ordered.index.astype(str), ordered.to_numpy(), color="#c0392b")
    ax.set_xlabel("mean(|SHAP value|)  —  average impact on XGBoost log-odds")
    ax.set_title("Global feature importance (XGBoost component)")
    ax.grid(axis="x", alpha=0.3)
    return _figure_to_data_uri(fig)


def _beeswarm_figure(data: DashboardData, feature_order: List[str]) -> str:
    """A compact SHAP beeswarm: per-feature strip of the sample's SHAP values,
    coloured by the (min-max normalised) feature value. Hand-rolled rather than
    `shap.summary_plot` so the figure size, colours, and headless backend are
    under this module's control and match the bar chart."""
    import matplotlib.pyplot as plt

    idx = [data.feature_sample.columns.get_loc(f) for f in feature_order]
    shap_subset = data.shap_values[:, idx]
    feat_subset = data.feature_sample.iloc[:, idx].to_numpy()

    if len(shap_subset) > BEESWARM_MAX_POINTS:
        rng = np.random.default_rng(42)
        pick = rng.choice(len(shap_subset), BEESWARM_MAX_POINTS, replace=False)
        shap_subset = shap_subset[pick]
        feat_subset = feat_subset[pick]

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.42 * len(feature_order))))
    jitter_rng = np.random.default_rng(0)
    for row, feature in enumerate(feature_order[::-1]):
        col = len(feature_order) - 1 - row
        values = shap_subset[:, col]
        colour_raw = feat_subset[:, col].astype(float)
        finite = colour_raw[np.isfinite(colour_raw)]
        spread = float(np.ptp(finite)) if finite.size else 0.0
        if spread > 0:
            colour = np.clip(
                (np.nan_to_num(colour_raw, nan=finite.min()) - finite.min()) / spread,
                0.0,
                1.0,
            )
        else:
            colour = np.zeros_like(colour_raw)
        jitter = jitter_rng.uniform(-0.18, 0.18, size=len(values))
        ax.scatter(
            values,
            np.full(len(values), row) + jitter,
            c=colour,
            cmap="coolwarm",
            s=8,
            alpha=0.6,
            linewidths=0,
        )
    ax.set_yticks(range(len(feature_order)))
    ax.set_yticklabels(feature_order[::-1])
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on XGBoost log-odds)")
    ax.set_title(
        "Per-transaction SHAP distribution — colour = feature value (low→high)"
    )
    ax.grid(axis="x", alpha=0.3)
    return _figure_to_data_uri(fig)


# ── HTML ───────────────────────────────────────────────────────────────────


def _render_html(
    data: DashboardData,
    top: pd.Series,
    bar_png: str,
    beeswarm_png: str,
    max_features: int,
) -> str:
    """Assemble the single-file HTML document. No external CSS or JS."""
    weight = data.explained_weight
    if weight is not None:
        scope_weight = (
            f"which carries <strong>{weight * 100:.1f}%</strong> "
            f"(<code>{weight:.3f}</code>) of the calibrated blend"
        )
    else:
        scope_weight = (
            "whose blend weight could not be read from "
            "<code>models/ensemble.json</code>"
        )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    shown = min(max_features, len(top))

    rows = "\n".join(
        f"      <tr><td>{i + 1}</td><td>{html.escape(str(feat))}</td>"
        f"<td>{val:.5f}</td></tr>"
        for i, (feat, val) in enumerate(top.items())
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraud model — SHAP explainability dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         background: #f7f7f8; color: #1a1a1a; }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 32px 24px 64px; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 4px; }}
  .sub {{ color: #666; font-size: 0.9rem; margin-bottom: 24px; }}
  .scope {{ background: #fff4e5; border: 1px solid #ffd9a8; border-radius: 8px;
           padding: 14px 16px; font-size: 0.9rem; margin-bottom: 28px; }}
  .card {{ background: #fff; border: 1px solid #e3e3e6; border-radius: 10px;
          padding: 20px; margin-bottom: 24px; }}
  .card h2 {{ font-size: 1.1rem; margin: 0 0 12px; }}
  img {{ max-width: 100%; height: auto; display: block; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid #ececef; }}
  th {{ background: #fafafa; }}
  td:first-child {{ color: #999; width: 3rem; }}
  code {{ background: #f0f0f2; padding: 1px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Fraud model — SHAP explainability dashboard</h1>
  <div class="sub">Generated {generated} &nbsp;·&nbsp; {data.n_rows} transactions explained
      &nbsp;·&nbsp; base value (log-odds) {data.base_value:.4f}</div>

  <div class="scope">
    <strong>What this explains.</strong> TreeSHAP over the <strong>XGBoost</strong>
    component only, {scope_weight}. Per <strong>ADR-001 &sect;3.4</strong> this is an
    exact explanation of the XGBoost sub-model, <em>not</em> of the ensemble decision
    (LightGBM and the TFT are not represented here). SHAP values are in raw-margin
    (log-odds) units:
    <code>base_value + &Sigma;contributions = XGBoost margin</code>.
  </div>

  <div class="card">
    <h2>Global feature importance</h2>
    <p class="sub">Mean absolute SHAP value across the {data.n_rows}-transaction sample.
       Top {shown} of {len(data.global_importance)} features.</p>
    <img src="{bar_png}" alt="Global feature importance bar chart">
  </div>

  <div class="card">
    <h2>Per-transaction SHAP distribution</h2>
    <p class="sub">Each point is one transaction's SHAP value for that feature; colour is
       the feature's value (blue = low, red = high). Spread away from zero means the
       feature moved that transaction's score.</p>
    <img src="{beeswarm_png}" alt="SHAP beeswarm distribution">
  </div>

  <div class="card">
    <h2>Top {shown} features by mean |SHAP|</h2>
    <table>
      <tr><th>#</th><th>Feature</th><th>mean(|SHAP|)</th></tr>
{rows}
    </table>
  </div>
</div>
</body>
</html>
"""
