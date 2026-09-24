"""
scripts/run_gnn_hpo.py

ADR-006 §3.3 — Arm C: hyperparameter search on top of the winning axis/axes
from Arms A/B. Runs ONLY after Gate 0 passes (checked here at startup, not
just documented) — Arm A/B's results are read from
``reports/gnn_arm_search_results_arm{A,B}.json`` and this script refuses to
run if neither shows ``beats_incumbent_val: true``.

STRUCTURAL test-blocking (mle-reviewer's CRITICAL fix for the "slow-motion
G1" failure mode): this script runs in ``--search-mode``, which never reads
``test_labels.parquet``. Test FEATURES may still load (needed for graph
topology only, R7-sanctioned — labels never enter the loss), but the label
file path is never opened by this module. The Optuna objective function
returns validation PR-AUC only. This is enforced structurally, not by
discipline: ``_load_data()`` has no code path that reads
``test_labels.parquet``, and this file contains no other reference to that
filename.

Search space (ADR-006 §3.3 table):
    learning_rate       log-uniform [3e-4, 1e-2]
    weight_decay        log-uniform [1e-6, 1e-2]
    dropout             uniform [0.2, 0.7]
    input_dropout       uniform [0.0, 0.2]
    num_layers          categorical {2, 3}
    hidden_dims preset  categorical, length-matched to num_layers
    residual            categorical {True, False}, conditional on num_layers==3
    l2_normalize        categorical {True, False}
    aggr                categorical {mean, max}
    num_neighbors preset  categorical, length-matched to num_layers
    pos_weight_scale    uniform [0.5, 1.5] x empirical train neg/pos
    edge_spec_set       categorical over pre-built, cached graphs from Arm A
                        (never rebuilt inside a trial)

Trial budget: 30 trials, fixed seed 42, max_epochs=50, patience=8 (§3.3 —
reduced from the confirmatory run's 100/10; the incumbent's val curve was
flat from epoch ~25-45, so full-budget trials would waste compute
re-establishing what Arms A/B's runs already showed).

Usage:
    conda run -n fraudx python scripts/run_gnn_hpo.py
    conda run -n fraudx python scripts/run_gnn_hpo.py --n-trials 10 --epochs 20
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import mlflow
import optuna
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings  # noqa: E402
from src.training.train_gnn import GNNTrainer  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

LOG_DIR = Path("reports/logs")


def _setup_logging(log_path: Path) -> None:
    """File + console logging, so a run started in the background can be
    tailed live (``tail -f <log_path>``) from any terminal, not just through
    this session's task-output stream."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,  # override any prior basicConfig from an imported module
    )


logger = logging.getLogger(__name__)

INCUMBENT_VAL_PR_AUC: float = 0.5260
GATE1_THRESHOLD: float = 0.5502
SEED: int = 42
TRIAL_BUDGET: int = 30

ARM_A_RESULTS = Path("reports/gnn_arm_search_results_armA.json")
ARM_B_RESULTS = Path("reports/gnn_arm_search_results_armB.json")
# Per-trial crash-resume checkpoints (see GNNTrainer.train's checkpoint_path).
# Not a permanent artifact — a trial's checkpoint is deleted by train() on
# clean completion; a file surviving here means that trial number did not
# finish and will resume from it on the next run.
CHECKPOINT_DIR = Path("reports/gnn_hpo_checkpoints")

_HIDDEN_DIMS_PRESETS_2LAYER = [[128, 64], [256, 128]]
_HIDDEN_DIMS_PRESETS_3LAYER = [[256, 128, 64], [128, 64, 32]]
_NUM_NEIGHBORS_PRESETS_2LAYER = [[10, 5], [15, 10]]
_NUM_NEIGHBORS_PRESETS_3LAYER = [[15, 10, 5], [10, 8, 5]]
# Only graphs already built and cached by Arm A — never rebuilt inside a
# trial (§3.3's explicit constraint).
_EDGE_SPEC_SET_CHOICES = ["legacy", "card_full", "addr_card"]
_PRESET_LOOKUP = {
    str(p): p
    for p in (
        _HIDDEN_DIMS_PRESETS_2LAYER
        + _HIDDEN_DIMS_PRESETS_3LAYER
        + _NUM_NEIGHBORS_PRESETS_2LAYER
        + _NUM_NEIGHBORS_PRESETS_3LAYER
    )
}


def _check_gate0() -> Dict[str, Any]:
    """Gate 0 (§4): refuse to run unless an Arm A or Arm B result file shows
    at least one variant beating the incumbent's validation PR-AUC. Returns
    a small summary dict for logging; raises if the gate has not been
    cleared."""
    found_pass = False
    summaries = []
    for path in (ARM_A_RESULTS, ARM_B_RESULTS):
        if not path.exists():
            continue
        data = json.loads(path.read_text("utf-8"))
        summaries.append({"path": str(path), "gate0_pass": data.get("gate0_pass")})
        if data.get("gate0_pass"):
            found_pass = True
    if not found_pass:
        raise RuntimeError(
            "Gate 0 (ADR-006 §4) has not passed: neither "
            f"{ARM_A_RESULTS} nor {ARM_B_RESULTS} shows gate0_pass=true. "
            "Arm C must not run — HPO on a direction with no signal is "
            "exactly the wasted-compute failure mode the mle-reviewer's "
            "compute-budget finding warns against. Run scripts/"
            "run_gnn_arm_search.py first."
        )
    logger.info("Gate 0 check: PASS (%s)", summaries)
    return {"summaries": summaries}


def _load_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Loads train/val features+labels and TEST FEATURES ONLY (never
    test_labels.parquet — structural R7/search-mode guarantee, see module
    docstring)."""
    processed_dir = Path(config["data"]["processed_dir"])
    X_test = pd.read_parquet(processed_dir / "test_features.parquet")
    # A zero vector stands in for test labels: GraphBuilder needs a y_test
    # of the right length to build node features/topology (R7-sanctioned),
    # but this value is NEVER read back by anything in this script — no
    # metric, gate, or log line in run_gnn_hpo.py references it.
    y_test_placeholder = pd.Series([0] * len(X_test))
    return {
        "X_train": pd.read_parquet(processed_dir / "train_features.parquet"),
        "y_train": pd.read_parquet(processed_dir / "train_labels.parquet").squeeze(),
        "X_val": pd.read_parquet(processed_dir / "val_features.parquet"),
        "y_val": pd.read_parquet(processed_dir / "val_labels.parquet").squeeze(),
        "X_test": X_test,
        "y_test": y_test_placeholder,
    }


def _sample_params(trial: "optuna.Trial") -> Dict[str, Any]:
    num_layers = trial.suggest_categorical("num_layers", [2, 3])
    if num_layers == 2:
        hidden_key = trial.suggest_categorical(
            "hidden_dims_2layer", [str(p) for p in _HIDDEN_DIMS_PRESETS_2LAYER]
        )
        neighbors_key = trial.suggest_categorical(
            "num_neighbors_2layer", [str(p) for p in _NUM_NEIGHBORS_PRESETS_2LAYER]
        )
        residual = False  # conditional on num_layers==3 per §3.3 table
    else:
        hidden_key = trial.suggest_categorical(
            "hidden_dims_3layer", [str(p) for p in _HIDDEN_DIMS_PRESETS_3LAYER]
        )
        neighbors_key = trial.suggest_categorical(
            "num_neighbors_3layer", [str(p) for p in _NUM_NEIGHBORS_PRESETS_3LAYER]
        )
        residual = trial.suggest_categorical("residual", [True, False])

    return {
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.2, 0.7),
        "input_dropout": trial.suggest_float("input_dropout", 0.0, 0.2),
        "num_layers": num_layers,
        "hidden_dims": _PRESET_LOOKUP[hidden_key],
        "residual": residual,
        "l2_normalize": trial.suggest_categorical("l2_normalize", [True, False]),
        "aggr": trial.suggest_categorical("aggr", ["mean", "max"]),
        "num_neighbors": _PRESET_LOOKUP[neighbors_key],
        "pos_weight_scale": trial.suggest_float("pos_weight_scale", 0.5, 1.5),
        "edge_spec_set": trial.suggest_categorical("edge_spec_set", _EDGE_SPEC_SET_CHOICES),
    }


def _objective(
    trial: "optuna.Trial",
    base_config: Dict[str, Any],
    data: Dict[str, Any],
    *,
    max_epochs: int,
    patience: int,
) -> float:
    params = _sample_params(trial)
    config = copy.deepcopy(base_config)
    g = config["model"]["gnn"]
    g["learning_rate"] = params["learning_rate"]
    g["weight_decay"] = params["weight_decay"]
    g["dropout"] = params["dropout"]
    g["input_dropout"] = params["input_dropout"]
    g["hidden_dims"] = params["hidden_dims"]
    # mlp_hidden_dims is independent of hidden_dims (it's the post-conv
    # head's own hidden sizes) — the config's existing value stays as-is.
    g["residual"] = params["residual"]
    g["l2_normalize"] = params["l2_normalize"]
    g["aggr"] = params["aggr"]
    g["num_neighbors"] = params["num_neighbors"]
    g["edge_spec_set"] = params["edge_spec_set"]
    g["max_epochs"] = max_epochs
    g["patience"] = patience

    set_seed(SEED)
    trainer = GNNTrainer(config)

    # pos_weight_scale is applied on top of the empirical train neg/pos
    # ratio (what GNNTrainer.train() computes internally when pos_weight is
    # None) — recompute the same ratio here and scale it, rather than
    # duplicating GNNTrainer's internal training-loop logic via a hook.
    train_y = pd.read_parquet(
        Path(config["data"]["processed_dir"]) / "train_labels.parquet"
    ).squeeze()
    pos = float((train_y == 1).sum())
    neg = float((train_y == 0).sum())
    empirical_ratio = (neg / pos) if pos > 0 else 1.0
    g["pos_weight"] = empirical_ratio * params["pos_weight_scale"]

    # One nested MLflow run per trial — child of the parent HPO run started
    # in main(). Lets Arm C be watched live in the MLflow UI (per-epoch val
    # PR-AUC as a metric series) exactly like a normal training run, not
    # just via this script's own stdout/log file.
    with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
        mlflow.log_params(
            {
                "trial_number": trial.number,
                **{k: str(v) for k, v in params.items()},
                "max_epochs": max_epochs,
                "patience": patience,
                "empirical_pos_weight": empirical_ratio,
                "scaled_pos_weight": g["pos_weight"],
            }
        )

        # Per-trial resume checkpoint: if this process (or the whole run_gnn_
        # hpo.py process) dies mid-trial, a rerun with the same Optuna study
        # (TPESampler + fixed seed reproduces the same trial number's sampled
        # params) resumes THIS trial from its last completed epoch instead of
        # losing it entirely — the failure mode observed in practice (a
        # 5-epoch trial killed mid-epoch-6, all progress lost).
        checkpoint_path = str(CHECKPOINT_DIR / f"trial_{trial.number:03d}.ckpt.pt")
        try:
            history = trainer.train(
                data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                X_test=data["X_test"], y_test=data["y_test"],
                checkpoint_path=checkpoint_path,
            )
        except Exception as exc:  # noqa: BLE001 — a bad sampled config should prune, not crash the study
            logger.warning("Trial %d failed: %s", trial.number, exc)
            mlflow.set_tag("trial_status", "failed")
            mlflow.log_param("failure_reason", str(exc)[:250])
            raise optuna.TrialPruned() from exc

        val_curve = history.get("val_pr_auc", [])
        train_loss_curve = history.get("train_loss", [])
        for epoch_idx, (loss, val_pr_auc) in enumerate(
            zip(train_loss_curve, val_curve), start=1
        ):
            mlflow.log_metrics(
                {"train_loss": loss, "val_pr_auc": val_pr_auc}, step=epoch_idx
            )

        best_val_pr_auc = max(val_curve) if val_curve else 0.0
        mlflow.log_metric("best_val_pr_auc", best_val_pr_auc)
        mlflow.log_metric("epochs_run", len(train_loss_curve))
        mlflow.set_tag("trial_status", "completed")

        # Report intermediate values for MedianPruner (per-epoch val PR-AUC).
        for epoch_idx, val_pr_auc in enumerate(val_curve):
            trial.report(val_pr_auc, step=epoch_idx)
            if trial.should_prune():
                mlflow.set_tag("trial_status", "pruned")
                raise optuna.TrialPruned()

    trial.set_user_attr("params", params)
    trial.set_user_attr("epochs_run", len(history.get("train_loss", [])))
    return best_val_pr_auc


def main() -> None:
    parser = argparse.ArgumentParser(description="ADR-006 Arm C — Optuna HPO.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--n-trials", type=int, default=TRIAL_BUDGET)
    parser.add_argument("--epochs", type=int, default=50, help="max_epochs per trial")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--study-storage", default="sqlite:///reports/gnn_hpo_study.db",
        help="Optuna storage URL — sqlite makes the study resumable/inspectable.",
    )
    parser.add_argument("--results-json", default="reports/gnn_hpo_results.json")
    parser.add_argument(
        "--log-file", default=None,
        help="Path to a log file to tee stdout into (default: reports/logs/"
        "gnn_hpo_<timestamp>.log) — lets a background run be tailed live "
        "from any terminal via `tail -f <path>`.",
    )
    parser.add_argument(
        "--skip-gate0-check", action="store_true",
        help="DANGEROUS: bypass the Gate 0 pre-check. Only for testing this script.",
    )
    args = parser.parse_args()

    log_path = Path(
        args.log_file
        or LOG_DIR / f"gnn_hpo_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.log"
    )
    _setup_logging(log_path)
    logger.info("Logging to %s (tail -f this path to watch live)", log_path)

    if not args.skip_gate0_check:
        _check_gate0()
    else:
        logger.warning("Gate 0 check SKIPPED (--skip-gate0-check) — testing only.")

    settings = load_settings(args.config)
    config = settings.model_dump()

    logger.info("Loading parquet data (search-mode — test LABELS never read)...")
    data = _load_data(config)
    logger.info(
        "Data loaded: train=%d, val=%d, test_features_only=%d",
        len(data["X_train"]), len(data["X_val"]), len(data["X_test"]),
    )

    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "fraud_detection"))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name="gnn_arm_c_hpo",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.study_storage,
        load_if_exists=True,
    )

    with mlflow.start_run(run_name="gnn_arm_c_hpo") as parent_run:
        logger.info("MLflow parent run ID: %s (each trial is a nested child run)",
                     parent_run.info.run_id)
        mlflow.log_params(
            {
                "adr": "ADR-006",
                "arm": "C",
                "n_trials_requested": args.n_trials,
                "max_epochs_per_trial": args.epochs,
                "patience_per_trial": args.patience,
                "incumbent_val_pr_auc": INCUMBENT_VAL_PR_AUC,
                "gate1_threshold": GATE1_THRESHOLD,
                "sampler": "TPESampler",
                "pruner": "MedianPruner",
                "seed": SEED,
            }
        )

        study.optimize(
            lambda trial: _objective(
                trial, config, data, max_epochs=args.epochs, patience=args.patience
            ),
            n_trials=args.n_trials,
        )

        best = study.best_trial
        result = {
            "n_trials_requested": args.n_trials,
            "n_trials_completed": len(study.trials),
            "incumbent_val_pr_auc": INCUMBENT_VAL_PR_AUC,
            "gate1_threshold": GATE1_THRESHOLD,
            "best_val_pr_auc": best.value,
            "best_params": best.user_attrs.get("params"),
            "beats_incumbent": bool(best.value > INCUMBENT_VAL_PR_AUC),
            "clears_gate1": bool(best.value >= GATE1_THRESHOLD),
        }
        mlflow.log_metric("best_val_pr_auc", best.value)
        mlflow.log_metric("n_trials_completed", len(study.trials))
        mlflow.set_tag("gate1_pass", result["clears_gate1"])

    logger.info("=" * 60)
    logger.info("ARM C — HPO complete: %d trials", len(study.trials))
    logger.info("Best val PR-AUC: %.4f", best.value)
    logger.info("Best params: %s", best.user_attrs.get("params"))
    logger.info(
        "GATE 1 (stop-before-test, >= %.4f): %s",
        GATE1_THRESHOLD,
        "PASS — confirmatory test run may proceed" if result["clears_gate1"] else
        "FAIL — revisit stops here, no test run",
    )
    logger.info("=" * 60)

    out_path = Path(args.results_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), "utf-8")
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
