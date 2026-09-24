"""
tests/performance/test_latency.py

PRD Phase 10 done-when: "100 predictions, assert P95 < 100ms" — the PRD's own
§6.1 NFR ("Inference latency: < 100ms P95 per single transaction prediction
(XGBoost)").

Measures `InferenceService.predict()` directly (the real scoring path:
feature transform + model forward passes), not the HTTP round trip — the
100ms NFR is about inference cost, and adding TestClient/ASGI overhead on top
would measure this test's own transport rather than the number the PRD names.
`tests/integration/test_api.py` already covers the HTTP/schema layer with a
fake service; this file is the complementary real-artifact, real-latency
measurement.

Explainability is measured separately from the base P95, not folded into it:
`InferenceService.__init__`'s own docstring records TreeSHAP on this model as
~140ms p50 on its own — well past the 100ms budget by design (it runs off
the request thread with its own timeout, per PRD Phase 4's P4-8 finding) — so
asserting the SHAP-inclusive path against the base-scoring NFR would be
asserting a number the architecture was deliberately built not to guarantee.

Skipped when `models/` artifacts or `data/raw/*.csv` are absent (a fresh
checkout, or CI without a local training run / Kaggle download), mirroring
`tests/integration/test_real_ensemble_artifact_loads.py`'s pattern — this test
never blocks a machine that has not trained or downloaded data.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.performance

_MODELS = PROJECT_ROOT / "models"
_RAW_TX = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
_REQUIRED_MODEL_ARTIFACTS = [
    _MODELS / "ensemble.json",
    _MODELS / "ensemble.checksums.json",
    _MODELS / "xgb_model.ubj",
]
_ARTIFACTS_PRESENT = all(p.exists() for p in _REQUIRED_MODEL_ARTIFACTS) and _RAW_TX.exists()

N_REQUESTS = 100
P95_BUDGET_MS = 100.0


def _tail_rows(n: int) -> pd.DataFrame:
    """The last `n` rows of the ~590k-row raw file, without loading the whole
    file into memory first. `pd.read_csv(...).tail(n)` reads and materialises
    every column (V1-V339 as float64) for all 590,540 rows before discarding
    all but `n` of them — ~1.6 GiB for the V-columns alone, which OOM'd this
    exact machine when the model artifacts (XGBoost + TFT + LightGBM, already
    loaded by the `real_service` fixture) were resident at the same time.
    Counting rows first and skipping straight to the tail keeps peak memory to
    just the `n` rows this test actually needs."""
    with open(_RAW_TX, "r", encoding="utf-8") as f:
        total_rows = sum(1 for _ in f) - 1  # minus the header
    skip = max(1, total_rows - n + 1)  # +1: keep the header at row 0
    return pd.read_csv(_RAW_TX, skiprows=range(1, skip))


def _sample_transactions(n: int) -> List[Dict[str, Any]]:
    """`n` real raw rows from the tail of the training file — the most
    recent transactions, i.e. the closest analogue to what a live serving
    instance actually receives. NaN is replaced with None so the dicts are
    JSON-shaped the way a real request payload would be; the transform layer
    already handles missing raw fields (that is what the ~75%-unmatched
    identity join means in production)."""
    tail = _tail_rows(n).drop(columns=["isFraud"])
    records = tail.where(pd.notnull(tail), None).to_dict(orient="records")
    for i, record in enumerate(records):
        # Distinct, monotonically increasing per-card timestamps so the
        # staleness check (src.serving.staleness) never rejects a request
        # inside this loop — each row already has a real, sorted DT from the
        # source file, so this is a no-op for well-ordered tails and only
        # matters if two sampled rows share a DT.
        record["TransactionDT"] = float(record["TransactionDT"]) + i
    return records


@pytest.fixture(scope="module")
def real_service():
    if not _ARTIFACTS_PRESENT:
        pytest.skip("real models/ artifacts or data/raw/train_transaction.csv not present")

    from src.api.main import build_inference_service
    from src.config import load_settings

    config = load_settings("config/config.yaml").model_dump()
    # Explainability off for the base latency measurement — see module
    # docstring. Disabling it here (rather than trusting a timeout to bound
    # it) keeps this test's result attributable to scoring alone.
    config.setdefault("explainability", {})["enabled"] = False
    return build_inference_service(config)


@pytest.fixture(scope="module")
def sampled_transactions() -> List[Dict[str, Any]]:
    """Read the raw CSV's tail exactly once for the whole module — both
    tests need a sample from the same population, and re-reading (even with
    `_tail_rows`' cheaper skip-to-tail approach) is pure waste alongside the
    already-resident model artifacts."""
    return _sample_transactions(N_REQUESTS)


@pytest.mark.skipif(not _ARTIFACTS_PRESENT, reason="real artifacts not present")
class TestPredictionLatency:
    def test_p95_latency_under_100ms_for_100_predictions(
        self, real_service, sampled_transactions
    ):
        transactions = sampled_transactions

        latencies_ms: List[float] = []
        for tx in transactions:
            started = time.perf_counter()
            real_service.predict(tx)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)

        p95 = float(np.percentile(latencies_ms, 95))
        p50 = float(np.percentile(latencies_ms, 50))

        assert p95 < P95_BUDGET_MS, (
            f"P95 latency {p95:.1f}ms exceeds the {P95_BUDGET_MS}ms NFR "
            f"(PRD §6.1) over {len(latencies_ms)} predictions "
            f"(p50={p50:.1f}ms, max={max(latencies_ms):.1f}ms)"
        )

    def test_all_sampled_predictions_return_a_valid_probability(
        self, real_service, sampled_transactions
    ):
        """A latency measurement over a service that is silently failing (and
        therefore fast for the wrong reason) is not a real latency
        measurement — pin correctness alongside speed."""
        for tx in sampled_transactions[:10]:
            result = real_service.predict(tx)
            assert 0.0 <= result.fraud_probability <= 1.0
            assert result.decision in {"FRAUD", "LEGITIMATE"}
