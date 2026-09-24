"""
scripts/run_ensemble_eval.py

Fit and freeze the deployed 3-way ensemble (XGBoost + TFT + LightGBM).

Rebuilt for PRD Phase 9 (was the stale 2-way XGB+TFT version that wrote no
`models/ensemble.json` — the 3-way code that produced the committed artifact
had never been in the tree; see `docs/IMPLEMENTATION_PLAN.md` Phase 8 note).

Pipeline:
  1. Load the three trained artifacts and score val + test with each model's
     **calibrated** probabilities (the space `models/ensemble.json` declares,
     ADR-001 §4.3).
  2. Diversity check: pairwise probability correlation on val
     (`src.models.ensemble.pairwise_diagnostics`).
  3. Weight search on VALIDATION PR-AUC over the probability simplex
     (`grid_search_simplex_weights`), for both the 2-way (XGB+TFT) baseline
     and the 3-way challenger.
  4. Min-lift gate: keep LightGBM only if the 3-way beats the 2-way
     VALIDATION PR-AUC by >= `config.ensemble.min_lightgbm_lift` (fixed
     2026-09-09 — previously gated on test PR-AUC, a model-selection decision
     that must not read the held-out set; test lift is now a diagnostic only).
  5. Bootstrap weight-stability CIs on the kept blend.
  6. Operating threshold on the VALIDATION blend via
     `ModelEvaluator.find_optimal_threshold` under the **net_of_principal**
     cost convention (ADR-001 §7, PRD Phase 9 step 9.1) — FN term zeroed.
  7. Write `models/ensemble.json` (+ checksum manifest) and
     `reports/ensemble_results.json` (+ provenance manifest).

Test metrics are reported at the frozen validation-selected threshold, never
re-derived from test (Phase C1).

Usage:
    python scripts/run_ensemble_eval.py
    python scripts/run_ensemble_eval.py --no-write   # evaluate, don't touch models/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config_hash, load_settings  # noqa: E402
from src.evaluation.business_impact import (  # noqa: E402
    CONVENTION_NET,
    CostModel,
)
from src.evaluation.evaluator import ModelEvaluator  # noqa: E402
from src.models.ensemble import (  # noqa: E402
    blend,
    bootstrap_weight_ci,
    grid_search_simplex_weights,
    pairwise_diagnostics,
)
from src.training.manifest import compute_dataset_hash, resolve_git_sha  # noqa: E402
from src.training.train_lgbm import LGBMTrainer  # noqa: E402
from src.training.train_tft import TFTTrainer  # noqa: E402
from src.training.train_xgb import XGBTrainer  # noqa: E402
from src.utils.checksums import write_checksums  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

_DATASET_FILES = [
    "train_features.parquet",
    "train_labels.parquet",
    "val_features.parquet",
    "val_labels.parquet",
    "test_features.parquet",
    "test_labels.parquet",
]
_SEQUENCE_GROUP_COL = "card1"


def apply_min_lift_gate(
    pr_auc_2way: float, pr_auc_3way: float, min_lift: float
) -> tuple[bool, float]:
    """Keep LightGBM only if the 3-way beats the 2-way PR-AUC by at least
    `min_lift` (config `ensemble.min_lightgbm_lift`). Returns (keep, lift).

    Boundary is inclusive: `lift == min_lift` keeps LightGBM.

    2026-09-09 (ecc:mle-reviewer, 4-agent ML review): the gate previously took
    the *test*-split PR-AUC for both arguments, which makes a model-selection
    decision (LightGBM's ensemble membership) on the held-out set. That
    decision had flipped across at least four evaluation runs, so test was
    effectively re-consulted for the same decision each time, biasing the
    reported test PR-AUC upward whenever the gate happened to pass. The call
    site now passes VALIDATION PR-AUC; test values are logged only as a
    post-hoc diagnostic and never influence `keep`.
    """
    lift = pr_auc_3way - pr_auc_2way
    return lift >= min_lift, lift


def build_spec(
    final_w: Dict[str, float],
    threshold: float,
    no_tft_threshold: float,
    dataset_hash: str,
    mlflow_run_id: Optional[str],
) -> Dict:
    """The `models/ensemble.json` payload. Always carries a `no_tft` fallback
    mode (XGB-only, weight 1.0, its own threshold) alongside `full`, so
    `InferenceService._fallback_mode` has a pre-registered mode to retry with
    when TFT scoring fails (ADR-001 §3.3, mle-reviewer P9-2 H2)."""
    return {
        "schema_version": "1.0",
        "modes": {
            "full": {
                "models": list(final_w.keys()),
                "weights": {k: float(v) for k, v in final_w.items()},
                "threshold": float(threshold),
            },
            "no_tft": {
                "models": ["xgb"],
                "weights": {"xgb": 1.0},
                "threshold": float(no_tft_threshold),
            },
        },
        "default_mode": "full",
        "probability_space": "per_model_calibrated",
        "dataset_hash": dataset_hash,
        "mlflow_run_id": mlflow_run_id,
    }


def _calibrated(trainer, name: str, *args, **kwargs) -> np.ndarray:
    if trainer.calibrator is None:
        raise RuntimeError(
            f"{name} artifact has no frozen calibrator — re-train it (Phase C4) "
            f"before ensembling; models/ensemble.json declares "
            f"probability_space=per_model_calibrated."
        )
    return np.asarray(trainer.predict_proba_calibrated(*args, **kwargs), dtype=float)


def _sequence_history(
    X_train: pd.DataFrame, X_prev: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Tail of the prior splits the TFT needs to fill the first per-card
    windows of the split being scored (mirrors export_test_probabilities.py)."""
    seq_len = int(config.get("data", {}).get("sequence_length", 10))
    combined = pd.concat([X_train, X_prev], axis=0, ignore_index=True)
    if _SEQUENCE_GROUP_COL not in combined.columns or seq_len <= 1:
        return combined
    return (
        combined.groupby(_SEQUENCE_GROUP_COL, sort=False)
        .tail(seq_len - 1)
        .reset_index(drop=True)
    )


def evaluate_ensemble(write: bool = True) -> Dict:
    config = load_settings("config/config.yaml").model_dump()
    settings = load_settings("config/config.yaml")
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    ev = ModelEvaluator()

    logger.info("Loading splits...")
    X_train = pd.read_parquet(processed_dir / "train_features.parquet")
    X_val = pd.read_parquet(processed_dir / "val_features.parquet")
    y_val = pd.read_parquet(processed_dir / "val_labels.parquet").squeeze().to_numpy()
    X_test = pd.read_parquet(processed_dir / "test_features.parquet")
    y_test = pd.read_parquet(processed_dir / "test_labels.parquet").squeeze().to_numpy()

    serving = config.get("serving", {})
    logger.info("Loading models...")
    xgb = XGBTrainer.load(serving.get("model_path", "models/xgb_model.pkl"))
    tft = TFTTrainer.load(
        serving.get("tft_model_path", "models/tft_model.pt"), config=config
    )
    lgbm = LGBMTrainer.load(serving.get("lgbm_model_path", "models/lgbm_model.pkl"))

    logger.info("Scoring validation split...")
    val_hist = _sequence_history(X_train, X_train.iloc[:0], config)
    val_prob = {
        "xgb": _calibrated(xgb, "XGBoost", X_val),
        "tft": _calibrated(tft, "TFT", X_val, history_X=val_hist),
        "lgbm": _calibrated(lgbm, "LightGBM", X_val),
    }

    logger.info("Scoring test split...")
    test_hist = _sequence_history(X_train, X_val, config)
    test_prob = {
        "xgb": _calibrated(xgb, "XGBoost", X_test),
        "tft": _calibrated(tft, "TFT", X_test, history_X=test_hist),
        "lgbm": _calibrated(lgbm, "LightGBM", X_test),
    }

    # ── Diversity + weight search ──────────────────────────────────────────
    corr = pairwise_diagnostics(val_prob)
    logger.info("Pairwise val-probability correlation: %s", corr)

    ens_cfg = config.get("ensemble", {})
    step = float(ens_cfg.get("grid_step", 0.02))
    refine = float(ens_cfg.get("refine_step", 0.002))
    min_lift = float(ens_cfg.get("min_lightgbm_lift", 0.005))

    two_prob_val = {k: val_prob[k] for k in ("xgb", "tft")}
    two_prob_test = {k: test_prob[k] for k in ("xgb", "tft")}
    w2, val2 = grid_search_simplex_weights(y_val, two_prob_val, step, refine)
    test2 = ev.compute_pr_auc(y_test, blend(two_prob_test, w2))

    w3, val3 = grid_search_simplex_weights(y_val, val_prob, step, refine)
    test3 = ev.compute_pr_auc(y_test, blend(test_prob, w3))

    # Gate on VALIDATION PR-AUC (2026-09-09 fix, see apply_min_lift_gate
    # docstring) — model-selection decisions must not read the held-out set.
    keep_lgbm, lift = apply_min_lift_gate(val2, val3, min_lift)
    test_lift = test3 - test2
    logger.info(
        "2-way val PR-AUC %.6f | 3-way val PR-AUC %.6f | val lift %.6f (gate %.4f) "
        "-> %s | test lift %.6f (diagnostic only, not gating)",
        val2,
        val3,
        lift,
        min_lift,
        "KEEP LightGBM" if keep_lgbm else "DROP LightGBM",
        test_lift,
    )

    final_prob_val = val_prob if keep_lgbm else two_prob_val
    final_prob_test = test_prob if keep_lgbm else two_prob_test
    final_w = w3 if keep_lgbm else w2

    boot = bootstrap_weight_ci(
        y_val,
        final_prob_val,
        step=float(ens_cfg.get("bootstrap_grid_step", 0.05)),
        n_boot=int(ens_cfg.get("bootstrap_resamples", 50)),
    )
    boot_ci = {
        name: {
            "mean": float(np.mean(s)),
            "std": float(np.std(s)),
            "p05": float(np.percentile(s, 5)),
            "p95": float(np.percentile(s, 95)),
        }
        for name, s in boot.items()
    }

    # ── Operating threshold on the VALIDATION blend, net_of_principal ──────
    cm = CostModel.from_config(config)

    def _net_threshold(y, p) -> float:
        # net_of_principal (ADR-001 §7): a missed fraud recovers nothing, so
        # the FN cost is 0. find_optimal_threshold(cost_fn=0, ...) is that
        # objective, selected on the VALIDATION blend and frozen (Phase C1).
        return ev.find_optimal_threshold(
            y, p, cost_fn=0.0, cost_fp=cm.cost_fp, revenue_tp=cm.revenue_tp
        )

    ens_val = blend(final_prob_val, final_w)
    ens_test = blend(final_prob_test, final_w)
    threshold = _net_threshold(y_val, ens_val)

    # ADR-001 §3.3 / §4.3: every degradation mode carries its OWN threshold.
    # `no_tft` (built in `build_spec`) is what `InferenceService._fallback_mode`
    # retries with when TFT scoring fails — XGB-only, its own val-selected
    # threshold.
    no_tft_thr = _net_threshold(y_val, val_prob["xgb"])

    val_pr_auc = ev.compute_pr_auc(y_val, ens_val)
    test_pr_auc = ev.compute_pr_auc(y_test, ens_test)
    test_roc_auc = ev.compute_roc_auc(y_test, ens_test)
    m = ev.compute_metrics_at_threshold(y_test, ens_test, threshold)

    logger.info(
        "FINAL %s blend %s: val PR-AUC %.6f  test PR-AUC %.6f  ROC-AUC %.6f  "
        "threshold %.6f  test P=%.4f R=%.4f  (no_tft fallback threshold %.6f)",
        f"{len(final_w)}-model",
        {k: round(v, 3) for k, v in final_w.items()},
        val_pr_auc,
        test_pr_auc,
        test_roc_auc,
        threshold,
        m["precision"],
        m["recall"],
        no_tft_thr,
    )

    dataset_hash = compute_dataset_hash(processed_dir, _DATASET_FILES)
    results = {
        "final_weights": final_w,
        "keep_lgbm": bool(keep_lgbm),
        "lgbm_lift_val_pr_auc": float(lift),
        "lgbm_lift_test_pr_auc_diagnostic": float(test_lift),
        "min_lightgbm_lift_threshold": min_lift,
        "threshold": float(threshold),
        "no_tft_threshold": float(no_tft_thr),
        "threshold_convention": CONVENTION_NET,
        "diagnostics": {
            "pairwise_correlation": corr,
            "bootstrap_weight_ci": boot_ci,
        },
        "standalone": {
            name: {
                "val_pr_auc": ev.compute_pr_auc(y_val, val_prob[name]),
                "test_pr_auc": ev.compute_pr_auc(y_test, test_prob[name]),
            }
            for name in val_prob
        },
        "two_way_baseline": {
            "weights": w2,
            "val_pr_auc": float(val2),
            "test_pr_auc": float(test2),
        },
        "three_way_challenger": {
            "weights": w3,
            "val_pr_auc": float(val3),
            "test_pr_auc": float(test3),
        },
        "final": {
            "val_pr_auc": float(val_pr_auc),
            "test_pr_auc": float(test_pr_auc),
            "test_roc_auc": float(test_roc_auc),
            "test_TP": int(m["TP"]),
            "test_TN": int(m["TN"]),
            "test_FP": int(m["FP"]),
            "test_FN": int(m["FN"]),
            "test_precision": float(m["precision"]),
            "test_recall": float(m["recall"]),
            "test_f1": float(m["f1"]),
            "test_accuracy": float(m["accuracy"]),
        },
        "dataset_hash": dataset_hash,
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "ensemble_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (reports_dir / "ensemble_results.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "model_type": "ensemble",
                "config_hash": config_hash(settings),
                "git_sha": resolve_git_sha(cwd=PROJECT_ROOT),
                "dataset_hash": dataset_hash,
                "dataset_files": sorted(_DATASET_FILES),
                "random_seed": int(config.get("project", {}).get("random_seed", 42)),
                "keep_lgbm": bool(keep_lgbm),
                "metrics": {
                    "val_pr_auc_final": float(val_pr_auc),
                    "test_pr_auc_final": float(test_pr_auc),
                    "test_roc_auc_final": float(test_roc_auc),
                    "lgbm_lift_val_pr_auc": float(lift),
                    "lgbm_lift_test_pr_auc_diagnostic": float(test_lift),
                    "optimal_threshold": float(threshold),
                    "no_tft_threshold": float(no_tft_thr),
                    **{f"final_weight_{k}": float(v) for k, v in final_w.items()},
                },
                "model_run_ids": _model_run_ids(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote reports/ensemble_results.json (+ manifest)")

    # ── Log the ensemble decision as its own MLflow run ───────────────────
    ens_run_id = _log_mlflow_run(
        config, final_w, threshold, no_tft_thr, keep_lgbm, lift, test_lift,
        val_pr_auc, test_pr_auc, test_roc_auc,
    )

    spec = build_spec(final_w, threshold, no_tft_thr, dataset_hash, ens_run_id)

    if write:
        spec_path = PROJECT_ROOT / "models" / "ensemble.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        write_checksums(
            spec_path.with_suffix(".checksums.json"), {"ensemble": spec_path}
        )
        logger.info("Wrote %s (+ checksum manifest)", spec_path)
    else:
        # Still write the candidate spec somewhere inspectable, just not to
        # the deployed path — a gated step that fails must not touch models/.
        cand = reports_dir / "ensemble_spec_candidate.json"
        cand.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        logger.info(
            "--no-write / gate not passed: models/ensemble.json untouched; "
            "candidate spec written to %s", cand,
        )

    return results


def _model_run_ids() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name, fname in (
        ("xgb", "xgb_model.manifest.json"),
        ("tft", "tft_model.manifest.json"),
        ("lgbm", "lgbm_model.manifest.json"),
    ):
        p = PROJECT_ROOT / "models" / fname
        if p.exists():
            try:
                out[name] = json.loads(p.read_text(encoding="utf-8")).get(
                    "mlflow_run_id"
                )
            except (OSError, json.JSONDecodeError):
                out[name] = None
    return out


def _log_mlflow_run(
    config, final_w, threshold, no_tft_thr, keep_lgbm, lift, test_lift,
    val_pr_auc, test_pr_auc, test_roc_auc,
) -> Optional[str]:
    """Record the ensemble decision (weights, thresholds, gate outcome) as an
    MLflow run so the deployed spec points at a tracking record (mle-reviewer
    H3). A tracking failure must not stop the eval — returns None then."""
    try:
        import mlflow

        mlflow_cfg = config.get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "fraud_detection"))
        with mlflow.start_run(run_name="ensemble_blend") as run:
            mlflow.log_params(
                {
                    "convention": CONVENTION_NET,
                    "keep_lgbm": bool(keep_lgbm),
                    **{f"weight_{k}": v for k, v in final_w.items()},
                    "threshold_full": threshold,
                    "threshold_no_tft": no_tft_thr,
                }
            )
            mlflow.log_metrics(
                {
                    "val_pr_auc": val_pr_auc,
                    "test_pr_auc": test_pr_auc,
                    "test_roc_auc": test_roc_auc,
                    "lgbm_lift_val": lift,
                    "lgbm_lift_test_diagnostic": test_lift,
                }
            )
            for name, rid in _model_run_ids().items():
                if rid:
                    mlflow.set_tag(f"model_run_{name}", rid)
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001 — tracking is best-effort here
        logger.warning("MLflow run for the ensemble not logged: %s", exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the ensemble; --promote to freeze it into models/."
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Write models/ensemble.json (+ checksums). Without this the run "
        "only writes reports/ and a candidate spec — a gated Phase 9 step that "
        "does not meet its band must not touch the deployed artifact "
        "(mle-reviewer H1).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Deprecated alias for the default (no promote). Kept so old "
        "invocations do not error.",
    )
    args = parser.parse_args()
    evaluate_ensemble(write=args.promote and not args.no_write)


if __name__ == "__main__":
    main()
