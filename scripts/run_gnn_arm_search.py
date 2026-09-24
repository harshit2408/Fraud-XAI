"""
scripts/run_gnn_arm_search.py

ADR-006 §3 execution — Arm A (richer edges) and Arm B (deeper/regularized
architecture), evaluated VALIDATION-ONLY per the mle-reviewer's CRITICAL
fix for the "slow-motion G1" failure mode (§3.3): this script never reads
``test_labels.parquet`` and never calls a code path that could. Each arm
variant is trained via ``GNNTrainer.train()`` (which itself never touches
test labels, R7) and scored with ``predict_proba(X_val)`` only.

Gate 0 (§4): Arm C (HPO) only runs if Arm A and/or Arm B beats the
incumbent's validation PR-AUC of 0.5260. This script computes and prints
that verdict; it does NOT invoke Arm C itself (a separate, explicit step
per the ADR's "not automatic" instruction).

Gate 1 (§4): if the single best validation PR-AUC across all arms does not
reach 0.5502 (the baseline's own test level), the revisit stops — no
confirmatory test run. This script reports the Gate 1 verdict too, but
does NOT run a confirmatory test evaluation itself; that is exactly one
run, in train_gnn.py's main(), gated on this script's output and on
explicit human sign-off (§3.4's "no re-running after seeing the test
number" discipline applies at that step, not this one).

Usage:
    conda run -n fraudx python scripts/run_gnn_arm_search.py
    conda run -n fraudx python scripts/run_gnn_arm_search.py --arms A
    conda run -n fraudx python scripts/run_gnn_arm_search.py --arms A,B --epochs 30
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings  # noqa: E402
from src.evaluation.evaluator import ModelEvaluator  # noqa: E402
from src.training.train_gnn import GNNTrainer  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

LOG_DIR = Path("reports/logs")


def _setup_logging(log_path: Path) -> None:
    """File + console logging so a background run can be tailed live
    (``tail -f <log_path>``) from any terminal."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


logger = logging.getLogger(__name__)

INCUMBENT_VAL_PR_AUC: float = 0.5260  # ADR-005 run 6615a0fc..., reports/gnn_results.json
BASELINE_TEST_PR_AUC: float = 0.5502  # frozen 3-way ensemble
GATE1_THRESHOLD: float = BASELINE_TEST_PR_AUC  # §4 Gate 1: match baseline on VAL first

REPORTS_PATH = Path("reports/gnn_arm_search_results.json")
# Per-variant crash-resume checkpoints (see GNNTrainer.train's
# checkpoint_path). Deleted by train() on clean completion; a surviving file
# means that variant did not finish and will resume from it on rerun.
CHECKPOINT_DIR = Path("reports/gnn_arm_search_checkpoints")


def _run_variant(
    name: str,
    base_config: Dict[str, Any],
    gnn_overrides: Dict[str, Any],
    data: Dict[str, Any],
    *,
    seed: int,
) -> Dict[str, Any]:
    """Train one variant to completion and score it on VALIDATION ONLY.
    Returns a result dict; never reads or references y_test."""
    config = copy.deepcopy(base_config)
    config["model"]["gnn"].update(gnn_overrides)

    set_seed(seed)
    trainer = GNNTrainer(config)

    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params(
            {
                "variant": name,
                **{f"override_{k}": str(v) for k, v in gnn_overrides.items()},
                "seed": seed,
                "max_epochs": config["model"]["gnn"].get("max_epochs"),
                "patience": config["model"]["gnn"].get("patience"),
            }
        )

        checkpoint_path = str(CHECKPOINT_DIR / f"{name}.ckpt.pt")
        t0 = time.perf_counter()
        history = trainer.train(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            X_test=data["X_test"], y_test=data["y_test"],
            checkpoint_path=checkpoint_path,
        )
        elapsed = time.perf_counter() - t0

        for epoch_idx, (loss, val_pr_auc) in enumerate(
            zip(history.get("train_loss", []), history.get("val_pr_auc", [])), start=1
        ):
            mlflow.log_metrics(
                {"train_loss": loss, "val_pr_auc": val_pr_auc}, step=epoch_idx
            )

        evaluator = ModelEvaluator()
        y_prob_val = trainer.predict_proba(data["X_val"])
        val_pr_auc = evaluator.compute_pr_auc(data["y_val"].to_numpy(), y_prob_val)
        best_val_pr_auc = max(history.get("val_pr_auc", [val_pr_auc]))

        n_epochs = len(history.get("train_loss", []))
        mean_epoch_seconds = (
            sum(history["epoch_seconds"]) / len(history["epoch_seconds"])
            if history.get("epoch_seconds")
            else None
        )

        result = {
            "variant": name,
            "gnn_overrides": gnn_overrides,
            "val_pr_auc_final_epoch": float(val_pr_auc),
            "val_pr_auc_best_epoch": float(best_val_pr_auc),
            "epochs_run": n_epochs,
            "mean_epoch_seconds": mean_epoch_seconds,
            "wall_clock_seconds": elapsed,
            "beats_incumbent_val": bool(best_val_pr_auc > INCUMBENT_VAL_PR_AUC),
            "clears_gate1": bool(best_val_pr_auc >= GATE1_THRESHOLD),
        }
        mlflow.log_metric("best_val_pr_auc", best_val_pr_auc)
        mlflow.log_metric("epochs_run", n_epochs)
        mlflow.log_metric("wall_clock_seconds", elapsed)
        mlflow.set_tag("beats_incumbent_val", result["beats_incumbent_val"])
        mlflow.set_tag("clears_gate1", result["clears_gate1"])

    logger.info(
        "[%s] best val PR-AUC=%.4f (final=%.4f) over %d epochs, %.1fs/epoch, "
        "beats_incumbent=%s, clears_gate1=%s",
        name, best_val_pr_auc, val_pr_auc, n_epochs,
        mean_epoch_seconds or float("nan"),
        result["beats_incumbent_val"], result["clears_gate1"],
    )
    return result


def arm_a_variants() -> List[Dict[str, Any]]:
    """§3.1 — richer edges, frozen (paper) architecture. Only the graph
    changes; hidden_dims/pos_weight/fan-out stay at config.yaml's frozen
    values."""
    return [
        {"name": "arm_a_card_full", "gnn_overrides": {"edge_spec_set": "card_full"}},
        {"name": "arm_a_addr_card", "gnn_overrides": {"edge_spec_set": "addr_card"}},
    ]


def arm_b_variants() -> List[Dict[str, Any]]:
    """§3.2 — deeper/regularized architecture, frozen (legacy) graph.
    Ordered by expected value per the mle-reviewer's compute-budget finding:
    weight_decay first (the one lever both reviews rate as having a real,
    non-zero mechanism), then dropout.

    NOTE: input_dropout, l2_normalize, residual and num_layers=3 require
    GraphSAGEModel support beyond the current 2-layer, no-normalization
    architecture (src/models/gnn_model.py). This function returns ONLY the
    variants that are runnable against today's model — weight_decay and
    dropout sweeps — and logs a warning for the rest so Arm B's screen is
    not silently incomplete. Extending GraphSAGEModel for the remaining
    variants is model-architecture work, tracked separately from this
    experiment-runner script.
    """
    variants = [
        {"name": "arm_b_weight_decay_1e-4", "gnn_overrides": {"weight_decay": 1e-4}},
        {"name": "arm_b_weight_decay_1e-3", "gnn_overrides": {"weight_decay": 1e-3}},
        {"name": "arm_b_dropout_0.4", "gnn_overrides": {"dropout": 0.4}},
        {"name": "arm_b_dropout_0.6", "gnn_overrides": {"dropout": 0.6}},
        {
            "name": "arm_b_weight_decay_1e-3_dropout_0.5",
            "gnn_overrides": {"weight_decay": 1e-3, "dropout": 0.5},
        },
    ]
    logger.warning(
        "Arm B: input_dropout / l2_normalize / residual / num_layers=3 are NOT "
        "run by this script — they require GraphSAGEModel architecture changes "
        "(src/models/gnn_model.py) not yet implemented. Only weight_decay/"
        "dropout sweeps run here."
    )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADR-006 Arm A / Arm B validation-only screen."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--arms", default="A,B", help="Comma-separated subset of {A,B} to run."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument("--patience", type=int, default=None, help="Override patience")
    parser.add_argument("--results-json", default=str(REPORTS_PATH))
    parser.add_argument(
        "--log-file", default=None,
        help="Path to a log file (default: reports/logs/gnn_arm_search_<timestamp>.log) "
        "— lets a background run be tailed live via `tail -f <path>`.",
    )
    args = parser.parse_args()

    log_path = Path(
        args.log_file
        or LOG_DIR / f"gnn_arm_search_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.log"
    )
    _setup_logging(log_path)
    logger.info("Logging to %s (tail -f this path to watch live)", log_path)

    arms = {a.strip().upper() for a in args.arms.split(",") if a.strip()}
    if not arms <= {"A", "B"}:
        raise ValueError(f"--arms must be a subset of {{A,B}}, got {args.arms!r}")

    settings = load_settings(args.config)
    config = settings.model_dump()
    seed = int(config.get("project", {}).get("random_seed", 42))
    if args.epochs is not None:
        config["model"]["gnn"]["max_epochs"] = args.epochs
    if args.patience is not None:
        config["model"]["gnn"]["patience"] = args.patience

    processed_dir = Path(config["data"]["processed_dir"])
    logger.info("Loading parquet data (val screen — test loaded for graph topology only)...")
    data = {
        "X_train": pd.read_parquet(processed_dir / "train_features.parquet"),
        "y_train": pd.read_parquet(processed_dir / "train_labels.parquet").squeeze(),
        "X_val": pd.read_parquet(processed_dir / "val_features.parquet"),
        "y_val": pd.read_parquet(processed_dir / "val_labels.parquet").squeeze(),
        "X_test": pd.read_parquet(processed_dir / "test_features.parquet"),
        "y_test": pd.read_parquet(processed_dir / "test_labels.parquet").squeeze(),
    }
    logger.info(
        "Data loaded: train=%d, val=%d, test=%d (test labels loaded for R7-"
        "compliant topology only — never scored by this script)",
        len(data["X_train"]), len(data["X_val"]), len(data["X_test"]),
    )

    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "fraud_detection"))

    results: List[Dict[str, Any]] = []
    with mlflow.start_run(run_name=f"gnn_arm_search_{'_'.join(sorted(arms))}") as parent_run:
        logger.info("MLflow parent run ID: %s (each variant is a nested child run)",
                     parent_run.info.run_id)
        mlflow.log_params(
            {
                "adr": "ADR-006",
                "arms_run": ",".join(sorted(arms)),
                "incumbent_val_pr_auc": INCUMBENT_VAL_PR_AUC,
                "gate1_threshold": GATE1_THRESHOLD,
                "seed": seed,
            }
        )

        if "A" in arms:
            logger.info("=" * 60)
            logger.info("ARM A — richer edges, frozen architecture (ADR-006 §3.1)")
            logger.info("=" * 60)
            for v in arm_a_variants():
                results.append(
                    _run_variant(v["name"], config, v["gnn_overrides"], data, seed=seed)
                )

        if "B" in arms:
            logger.info("=" * 60)
            logger.info("ARM B — deeper/regularized architecture (ADR-006 §3.2)")
            logger.info("=" * 60)
            for v in arm_b_variants():
                results.append(
                    _run_variant(v["name"], config, v["gnn_overrides"], data, seed=seed)
                )

        best = max(results, key=lambda r: r["val_pr_auc_best_epoch"]) if results else None
        gate0_pass = any(r["beats_incumbent_val"] for r in results)
        gate1_pass = bool(best and best["clears_gate1"])
        mlflow.log_metric("best_val_pr_auc", best["val_pr_auc_best_epoch"] if best else 0.0)
        mlflow.set_tag("gate0_pass", gate0_pass)
        mlflow.set_tag("gate1_pass", gate1_pass)

    summary = {
        "incumbent_val_pr_auc": INCUMBENT_VAL_PR_AUC,
        "gate1_threshold": GATE1_THRESHOLD,
        "results": results,
        "best_variant": best["variant"] if best else None,
        "best_val_pr_auc": best["val_pr_auc_best_epoch"] if best else None,
        "gate0_pass": gate0_pass,
        "gate1_pass": gate1_pass,
    }

    logger.info("=" * 60)
    logger.info(
        "GATE 0 (per-arm validation screen, vs incumbent %.4f): %s",
        INCUMBENT_VAL_PR_AUC,
        "PASS — Arm C may run" if gate0_pass else "FAIL — Arm C skipped",
    )
    if best:
        logger.info(
            "Best variant: %s (val PR-AUC %.4f)", best["variant"], best["val_pr_auc_best_epoch"]
        )
    logger.info(
        "GATE 1 (stop-before-test screen, >= %.4f): %s",
        GATE1_THRESHOLD,
        "PASS — confirmatory test run may proceed" if gate1_pass else
        "FAIL — revisit stops here, no test run",
    )
    logger.info("=" * 60)

    out_path = Path(args.results_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), "utf-8")
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
