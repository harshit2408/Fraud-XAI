"""
src/training/train_gnn.py

Training script + ``GNNTrainer`` for the GraphSAGE fraud detector (PRD Phase 12).

Mirrors ``src/training/train_tft.py``:
  * loads the processed parquet frames,
  * builds (or loads a cached) transaction graph via ``src/data/graph_builder.py``,
  * trains GraphSAGE with ``NeighborLoader`` neighbourhood sampling on the
    PHASE-SCOPED, SYMMETRIC train subgraph, class-weighted BCE, early stopping
    on validation PR-AUC,
  * fits an isotonic calibrator on val + freezes a val-selected threshold,
  * writes a 3-file artifact (state_dict only, no pickle of the model),
    checksum-verified, plus a ``build_manifest`` sidecar.

Leakage contract: see ``src/data/graph_builder.py`` and ``docs/adr/ADR-005``.
The mle-reviewer findings this module implements:
  * R2 — SYMMETRIC message passing everywhere (train loader, val loader,
    every eval forward). ``_edge_index_for(phase)`` is the single source of
    truth for which edges a phase may use.
  * R3 — the model has no normalization layers (enforced in ``GraphSAGEModel``).
  * R6 — ``predict_proba(X)`` resolves the split by exact row count and
    raises on zero / ambiguous matches. Whole-split scoring only; Phase 12
    is offline evaluation, not a serving path.
  * R7 — test labels never enter the loss: ``train()`` reads X_test/y_test
    from disk purely for node features + topology; no train/val ``NeighborLoader``
    batch can reach a test node (phase-scoped edge mask).

Usage:
    python src/training/train_gnn.py
    python src/training/train_gnn.py --config config/config.yaml --epochs 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings, load_settings  # noqa: E402
from src.data.graph_builder import GraphBuilder, GraphBuildConfig  # noqa: E402
from src.device import resolve_device  # noqa: E402
from src.evaluation.evaluator import ModelEvaluator  # noqa: E402
from src.models.gnn_model import GraphSAGEModel  # noqa: E402
from src.training.losses import WeightedBCELoss  # noqa: E402
from src.training.manifest import (  # noqa: E402
    build_manifest,
    compute_dataset_hash,
    write_manifest,
)
from src.training.run_logging import RunLogger  # noqa: E402
from src.utils.checksums import verify_checksums, write_checksums  # noqa: E402
from src.utils.seed import seed_worker, set_seed  # noqa: E402

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
_PHASES = ("train", "val", "test")


def _artifact_paths(base_path: Path) -> Dict[str, Path]:
    """The three files a GNNTrainer artifact is split across — parity with
    ``TFTTrainer._artifact_paths`` (Phase D6): the model ``state_dict`` alone
    (loadable with ``weights_only=True``), a joblib metadata sidecar, and a
    sha256 checksum manifest verified before either is deserialized."""
    return {
        "weights": base_path.with_name(f"{base_path.stem}.weights.pt"),
        "metadata": base_path.with_name(f"{base_path.stem}.meta.joblib"),
        "checksums": base_path.with_name(f"{base_path.stem}.checksums.json"),
    }


class GNNTrainer:
    """End-to-end trainer for the GraphSAGE fraud detector.

    Interface parity with ``XGBTrainer`` / ``LGBMTrainer`` / ``TFTTrainer``:
    ``build_model`` / ``train`` / ``predict_proba`` / ``predict_proba_calibrated``
    / ``predict`` / ``set_threshold`` / ``set_calibrator`` / ``save`` / ``load``.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model: Optional[GraphSAGEModel] = None
        self.device = self._get_device()
        self.graph_builder: Optional[GraphBuilder] = None
        self.data = None  # torch_geometric.data.Data, held in memory (D3)
        self.split_sizes: Dict[str, int] = {}
        self._training_history: List[dict] = []
        self.threshold: Optional[float] = None
        self.calibrator: Any = None
        self._graph_cache_path: Optional[str] = None
        self._graph_dataset_hash: Optional[str] = None
        self._phase_data_cache: Dict[str, Any] = {}
        self._eval_loader_cache: Dict[str, Any] = {}

    # ── config / setup ─────────────────────────────────────────────────────
    def _gnn_cfg(self) -> Dict[str, Any]:
        return (self.config.get("model", {}) or {}).get("gnn", {}) or {}

    def _get_device(self, *, force_cpu: bool = False) -> torch.device:
        device_str = resolve_device(
            self._gnn_cfg().get("device", "auto"), force_cpu=force_cpu
        )
        if device_str == "cuda":
            logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        else:
            logger.info("Using device: %s", device_str)
        return torch.device(device_str)

    def set_threshold(self, threshold: float) -> None:
        """Freeze the validation-selected decision threshold (Phase C1/C4)."""
        self.threshold = threshold

    def set_calibrator(self, calibrator: Any) -> None:
        """Attach the validation-fitted probability calibrator (Phase C2/C4)."""
        self.calibrator = calibrator

    def build_model(self, num_node_features: int) -> GraphSAGEModel:
        g = self._gnn_cfg()
        hidden_dims = list(g.get("hidden_dims", [128, 64]))
        num_neighbors = list(g.get("num_neighbors", [10, 5]))
        # ADR-006 §3.2: the fan-out list length must equal len(hidden_dims) —
        # a mismatch silently truncates NeighborLoader sampling at the
        # shorter length instead of raising, which would understate how many
        # hops a deeper (num_layers=3) model actually samples. Fail fast.
        if len(num_neighbors) != len(hidden_dims):
            raise ValueError(
                f"model.gnn.num_neighbors (len={len(num_neighbors)}, "
                f"{num_neighbors}) must have the same length as "
                f"model.gnn.hidden_dims (len={len(hidden_dims)}, {hidden_dims}) "
                "— one fan-out entry per SAGEConv layer (ADR-006 §3.2)."
            )
        self.model = GraphSAGEModel(
            num_node_features=num_node_features,
            hidden_dims=hidden_dims,
            mlp_hidden_dims=list(g.get("mlp_hidden_dims", [64, 32])),
            dropout=float(g.get("dropout", 0.3)),
            aggr=str(g.get("aggr", "mean")),
            input_dropout=float(g.get("input_dropout", 0.0)),
            l2_normalize=bool(g.get("l2_normalize", False)),
            residual=bool(g.get("residual", False)),
        ).to(self.device)
        logger.info(
            "GraphSAGEModel built: %d params, hidden=%s, mlp=%s, aggr=%s, "
            "input_dropout=%.2f, l2_normalize=%s, residual=%s",
            self.model.count_parameters(),
            self.model.hidden_dims,
            self.model.mlp_hidden_dims,
            self.model.aggr,
            self.model.input_dropout,
            self.model.l2_normalize,
            self.model.residual,
        )
        return self.model

    # ── graph prep ─────────────────────────────────────────────────────────
    def _variant_graph_cache_path(self, base_cache_path: str) -> str:
        """ADR-006 §3.1: the legacy / card_full / addr_card graph variants
        are cached separately so they never collide. ``edge_spec_set ==
        "legacy"`` keeps the pre-Arm-A path unchanged (no behavior change for
        existing runs); any other value inserts a ``.<edge_spec_set>`` suffix
        before the extension."""
        edge_spec_set = str(self._gnn_cfg().get("edge_spec_set", "legacy"))
        if edge_spec_set == "legacy":
            return base_cache_path
        p = Path(base_cache_path)
        return str(p.with_name(f"{p.stem}.{edge_spec_set}{p.suffix}"))

    def _prepare_graph(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        *,
        cache_path: Optional[str] = None,
        dataset_hash: Optional[str] = None,
        use_cache: bool = True,
    ):
        """Build the graph once and hold it on the instance. ``X_test`` /
        ``y_test`` are needed here for node features + topology so val/test
        nodes have realistic connectivity (§12.1 caveat 2) — their LABELS
        never enter the loss (R7)."""
        if cache_path is None:
            base_cache_path = self._gnn_cfg().get(
                "graph_cache_path", "data/processed/graph/fraud_graph.pt"
            )
            cache_path = self._variant_graph_cache_path(base_cache_path)
        gb_cfg = GraphBuildConfig.from_config(self.config)

        if (
            use_cache
            and dataset_hash is not None
            and GraphBuilder.is_cache_valid(cache_path, dataset_hash)
        ):
            logger.info("Loading cached graph from %s", cache_path)
            self.data = GraphBuilder.load(cache_path)
            self.graph_builder = GraphBuilder(gb_cfg)
            self.graph_builder.feature_names = list(X_train.columns)
            self.graph_builder.split_sizes = dict(self.data.split_sizes)
        else:
            self.graph_builder = GraphBuilder(gb_cfg)
            self.data = self.graph_builder.build(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            if use_cache:
                self.graph_builder.save(
                    self.data, cache_path, dataset_hash=dataset_hash
                )

        self._graph_cache_path = cache_path
        self._graph_dataset_hash = dataset_hash
        self.split_sizes = dict(self.data.split_sizes)
        self.data = self.data.to(self.device)
        # Both caches are invalidated by a fresh graph — a new self.data means
        # a new edge_mask_* / edge_index, so any previously sliced Data views
        # or NeighborLoaders built against the old tensors would be stale.
        self._phase_data_cache: Dict[str, Any] = {}
        self._eval_loader_cache: Dict[str, Any] = {}
        return self.data

    def _edge_index_for(self, phase: str) -> torch.Tensor:
        """The SYMMETRIC edge_index slice a phase may use (R2). Single source
        of truth — train loader, val loader, and every eval forward call this
        with the same semantics."""
        mask = {
            "train": self.data.edge_mask_train,
            "val": self.data.edge_mask_val,
            "test": self.data.edge_mask_test,
        }[phase]
        return self.data.edge_index[:, mask]

    def _phase_data(self, phase: str):
        """A shallow ``Data`` view carrying the full node set but only the
        phase-eligible edges — what ``NeighborLoader`` samples on.

        Cached per phase for the lifetime of ``self.data``: ``edge_index[:,
        mask]`` is a boolean-mask fancy-index that allocates a fresh
        multi-million-element tensor copy every time it runs, and
        ``_score_nodes`` used to call this (via a fresh ``NeighborLoader``)
        ONCE PER EPOCH for validation scoring — on a long run that is one
        redundant allocation of the same unchanging tensor per epoch. Left
        unbounded, the resulting allocator churn manifested as steadily
        climbing per-epoch time within a single long HPO trial and an
        eventual ``bad_alloc`` (observed empirically: 458s/epoch at epoch 1
        climbing to 1768s/epoch by epoch 29, then a crash, followed by a run
        of allocation failures in the next few trials before the process's
        memory arena recovered). This is a real leak, not a hardware
        ceiling — the fix is to slice once and reuse, since
        ``edge_mask_<phase>`` never changes for the lifetime of one built
        graph.
        """
        if phase in self._phase_data_cache:
            return self._phase_data_cache[phase]
        from torch_geometric.data import Data

        phase_data = Data(
            x=self.data.x,
            y=self.data.y,
            edge_index=self._edge_index_for(phase),
        )
        self._phase_data_cache[phase] = phase_data
        return phase_data

    # ── training ──────────────────────────────────────────────────────────
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        run_logger: Optional[RunLogger] = None,
        *,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, list]:
        """Full training loop, structured like ``TFTTrainer.train``.

        ``X_test`` / ``y_test`` may be passed directly (tests do this); when
        omitted they are read from ``config.data.processed_dir`` — needed for
        graph topology only, never for the loss (R7).

        ``checkpoint_path``, when given, makes this call resumable across a
        process crash: after EVERY epoch, model/optimizer/scheduler state,
        training history, and early-stopping counters are written there
        (``torch.save``, atomically via a temp-file + rename so a crash
        mid-write cannot corrupt the checkpoint the next resume would read).
        If the path already exists when ``train()`` starts, the run resumes
        from the saved epoch instead of epoch 1 — same optimizer momentum,
        same early-stopping state, so a resumed run continues as if it had
        never stopped. On a clean finish the checkpoint file is deleted (its
        job is crash recovery, not a permanent artifact — the real artifact
        is written by ``save()``). Motivated by an observed failure: an
        Optuna HPO trial killed mid-epoch-6 lost all 5 already-trained
        epochs because nothing survived the crash boundary.
        """
        from torch_geometric.loader import NeighborLoader

        processed_dir = Path(self.config["data"]["processed_dir"])
        if X_test is None or y_test is None:
            X_test = pd.read_parquet(processed_dir / "test_features.parquet")
            y_test = pd.read_parquet(processed_dir / "test_labels.parquet").squeeze()

        try:
            dataset_hash = compute_dataset_hash(processed_dir, _DATASET_FILES)
        except Exception:  # noqa: BLE001 — hashing is best-effort for tests
            dataset_hash = None

        self._prepare_graph(
            X_train, y_train, X_val, y_val, X_test, y_test,
            dataset_hash=dataset_hash,
            use_cache=bool(self._gnn_cfg().get("use_graph_cache", True)),
        )

        if self.model is None:
            self.build_model(num_node_features=self.data.x.shape[1])

        g = self._gnn_cfg()
        train_y = self.data.y[self.data.train_mask]
        pos = float((train_y == 1).sum())
        neg = float((train_y == 0).sum())
        pos_weight = g.get("pos_weight")
        if pos_weight is None:
            pos_weight = (neg / pos) if pos > 0 else 1.0
        logger.info(
            "Class imbalance (train nodes): %d neg / %d pos -> pos_weight = %.4f",
            int(neg), int(pos), pos_weight,
        )
        criterion = WeightedBCELoss(pos_weight=float(pos_weight)).to(self.device)

        num_neighbors = list(g.get("num_neighbors", [10, 5]))
        batch_size = int(g.get("batch_size", 1024))
        max_epochs = int(g.get("max_epochs", 100))
        patience = int(g.get("patience", 10))
        lr = float(g.get("learning_rate", 0.005))
        weight_decay = float(g.get("weight_decay", 1e-5))
        grad_clip = float(g.get("gradient_clip_val", 1.0))

        gen = torch.Generator()
        gen.manual_seed(int(self.config.get("project", {}).get("random_seed", 42)))
        train_loader = NeighborLoader(
            self._phase_data("train"),
            num_neighbors=num_neighbors,
            input_nodes=self.data.train_mask.cpu(),
            batch_size=batch_size,
            shuffle=True,
            generator=gen,
            worker_init_fn=seed_worker,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=max(2, patience // 3)
        )
        evaluator = ModelEvaluator()

        best_val_pr_auc = -1.0
        best_state: Optional[Dict] = None
        epochs_no_improve = 0
        history: Dict[str, list] = {"train_loss": [], "val_pr_auc": []}
        start_epoch = 1

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_val_pr_auc = ckpt["best_val_pr_auc"]
            best_state = ckpt["best_state"]
            epochs_no_improve = ckpt["epochs_no_improve"]
            history = ckpt["history"]
            self._training_history = ckpt["training_history"]
            start_epoch = ckpt["epoch"] + 1
            logger.info(
                "Resuming training from checkpoint %s: epoch %d, best val "
                "PR-AUC so far %.4f",
                checkpoint_path, ckpt["epoch"], best_val_pr_auc,
            )

        for epoch in range(start_epoch, max_epochs + 1):
            epoch_start = time.perf_counter()
            self.model.train()
            epoch_loss, n_batches = 0.0, 0
            for batch in train_loader:
                batch = batch.to(self.device)
                # R7 defence-in-depth: seed nodes must be train nodes.
                seed_ids = batch.n_id[: batch.batch_size]
                if int(seed_ids.max()) >= self.split_sizes["train"]:
                    raise RuntimeError(
                        "train NeighborLoader yielded a non-train seed node "
                        "(edge-mask construction bug)"
                    )
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index)["logits"].view(-1)
                seed_logits = out[: batch.batch_size]
                seed_targets = batch.y[: batch.batch_size].to(self.device)
                loss = criterion(seed_logits, seed_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            mean_loss = epoch_loss / max(1, n_batches)
            val_prob = self._score_nodes("val")
            val_pr_auc = evaluator.compute_pr_auc(
                self.data.y[self.data.val_mask].cpu().numpy(), val_prob
            )
            scheduler.step(val_pr_auc)
            epoch_seconds = time.perf_counter() - epoch_start
            history["train_loss"].append(mean_loss)
            history["val_pr_auc"].append(val_pr_auc)
            history.setdefault("epoch_seconds", []).append(epoch_seconds)
            self._training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "val_pr_auc": val_pr_auc,
                    "epoch_seconds": epoch_seconds,
                }
            )
            logger.info(
                "Epoch %3d | train_loss %.4f | val PR-AUC %.4f | %.1fs",
                epoch, mean_loss, val_pr_auc, epoch_seconds,
            )
            if run_logger is not None:
                run_logger.log_scalars(
                    {"train/loss": mean_loss, "val/pr_auc": val_pr_auc}, step=epoch
                )

            if val_pr_auc > best_val_pr_auc + 1e-5:
                best_val_pr_auc = val_pr_auc
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                epochs_no_improve = 0
                if run_logger is not None:
                    run_logger_ckpt = run_logger.checkpoint_path(epoch, suffix="pt")
                    torch.save(best_state, run_logger_ckpt)
                    run_logger.prune_checkpoints(keep_last=3)
            else:
                epochs_no_improve += 1

            if checkpoint_path is not None:
                self._save_resume_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val_pr_auc=best_val_pr_auc,
                    best_state=best_state,
                    epochs_no_improve=epochs_no_improve,
                    history=history,
                )

            if epochs_no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d (no val PR-AUC improvement "
                    "for %d epochs).",
                    epoch, patience,
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        logger.info("Best validation PR-AUC: %.4f", best_val_pr_auc)

        if checkpoint_path is not None:
            # Training finished cleanly (converged or early-stopped, not
            # crashed) — the resume checkpoint has done its job. Remove it
            # so a later, unrelated train() call for the same path doesn't
            # accidentally resume from a finished run's state.
            Path(checkpoint_path).unlink(missing_ok=True)

        return history

    def _save_resume_checkpoint(
        self,
        checkpoint_path: str,
        *,
        epoch: int,
        optimizer: "torch.optim.Optimizer",
        scheduler: "torch.optim.lr_scheduler.ReduceLROnPlateau",
        best_val_pr_auc: float,
        best_state: Optional[Dict],
        epochs_no_improve: int,
        history: Dict[str, list],
    ) -> None:
        """Write ``train()``'s crash-resume checkpoint atomically: save to a
        temp file in the same directory, then ``os.replace`` it over the
        real path. ``os.replace`` is atomic on both POSIX and Windows, so a
        crash mid-write leaves the PREVIOUS (still-valid) checkpoint in
        place rather than a half-written, unreadable one."""
        import os

        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_pr_auc": best_val_pr_auc,
            "best_state": best_state,
            "epochs_no_improve": epochs_no_improve,
            "history": history,
            "training_history": self._training_history,
        }
        p = Path(checkpoint_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.with_suffix(p.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, p)

    # ── scoring ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def _score_nodes(self, phase: str) -> np.ndarray:
        """Eval-mode scoring for one phase's nodes, on that phase's
        SYMMETRIC edge slice, returned in split-row order.

        Batched via ``NeighborLoader`` (no shuffle) rather than one full-graph
        forward: on an 8 GB card the 590k-node / 15.8M-edge full-graph pass
        through 2 SAGEConv layers thrashes GPU memory (the mle-reviewer's
        flagged fallback). A wide, deterministic fan-out reproduces near-
        full-neighbourhood aggregation while keeping the working set bounded.
        Every sampled edge is still phase-eligible, so R2/R7 hold.

        The ``NeighborLoader`` is cached per phase (``_eval_loader_cache``):
        this is called once per epoch during training (val PR-AUC for early
        stopping) and its underlying graph/mask never change within one
        ``train()`` call, so rebuilding it every epoch was pure waste — see
        ``_phase_data``'s docstring for the OOM this caused in practice.
        ``shuffle=False`` makes the loader stateless/re-iterable, so reusing
        one instance across epochs is safe.
        """
        self.model.eval()

        mask = {
            "train": self.data.train_mask,
            "val": self.data.val_mask,
            "test": self.data.test_mask,
        }[phase]
        loader = self._eval_loader_cache.get(phase)
        if loader is None:
            from torch_geometric.loader import NeighborLoader

            g = self._gnn_cfg()
            eval_fanout = list(
                g.get(
                    "eval_num_neighbors",
                    [max(nn * 4, 40) for nn in g.get("num_neighbors", [10, 5])],
                )
            )
            eval_bs = int(g.get("eval_batch_size", 4096))
            loader = NeighborLoader(
                self._phase_data(phase),
                num_neighbors=eval_fanout,
                input_nodes=mask.detach().cpu(),
                batch_size=eval_bs,
                shuffle=False,
            )
            self._eval_loader_cache[phase] = loader
        out = np.empty(int(mask.sum().item()), dtype=np.float64)
        cursor = 0
        for batch in loader:
            batch = batch.to(self.device)
            p = (
                self.model(batch.x, batch.edge_index)["probabilities"]
                .view(-1)[: batch.batch_size]
                .detach()
                .cpu()
                .numpy()
            )
            out[cursor : cursor + len(p)] = p
            cursor += len(p)
        if cursor != len(out):
            raise RuntimeError(
                f"_score_nodes({phase}): scored {cursor} of {len(out)} nodes"
            )
        return out

    def _phase_for_length(self, n: int) -> str:
        """Resolve a whole-split frame to its phase by exact row count (R6).
        Raises on zero or ambiguous matches — no silent default branch."""
        matches = [p for p in _PHASES if self.split_sizes.get(p) == n]
        if len(matches) == 1:
            return matches[0]
        expected = {p: self.split_sizes.get(p) for p in _PHASES}
        raise ValueError(
            f"predict_proba: input has {n} rows, which matches "
            f"{'no' if not matches else 'multiple'} registered split span(s) "
            f"{expected}. Phase 12 GNN scoring is whole-split only — pass a "
            f"full train/val/test feature frame."
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Raw fraud probability for every row of a WHOLE split frame.

        The split is resolved by exact row count against the graph's
        registered span sizes (R6); a non-matching or ambiguous length
        raises. Row-level or arbitrary-subset scoring, and scoring a
        transaction not present in the cached graph, are out of scope for
        Phase 12 (offline evaluation only — no serving path).
        """
        if self.data is None:
            self.ensure_graph_loaded()
        phase = self._phase_for_length(len(X))
        return self._score_nodes(phase)

    def predict_proba_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        """Raw ``predict_proba`` through the frozen calibrator (Phase C2/C4).
        Raises if no calibrator was attached — never silently returns
        uncalibrated output. This is the method ``run_ensemble_eval._calibrated``
        calls."""
        if self.calibrator is None:
            raise ValueError(
                "No calibrator attached. Call set_calibrator() or load a "
                "calibrated artifact."
            )
        return self.calibrator.predict(self.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binarise RAW ``predict_proba`` at the frozen threshold (Phase C1/C4).
        Raises if no threshold was frozen."""
        if self.threshold is None:
            raise ValueError(
                "No threshold frozen. Call set_threshold() or load a "
                "thresholded artifact."
            )
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ── persistence ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """3-file artifact, parity with ``TFTTrainer.save`` (Phase D6 — no
        ``weights_only=False`` on the load path). The ``Data`` object is NOT
        in the artifact; it is regenerated from the graph cache (keyed by
        ``graph_dataset_hash``) on load."""
        if self.model is None:
            raise ValueError("Model not trained. Cannot save.")
        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        paths = _artifact_paths(base_path)

        torch.save(self.model.state_dict(), paths["weights"])

        processed_dir = Path(self.config["data"]["processed_dir"])
        try:
            graph_dataset_hash = self._graph_dataset_hash or compute_dataset_hash(
                processed_dir, _DATASET_FILES
            )
        except Exception:  # noqa: BLE001
            graph_dataset_hash = None

        # The Data object is not in the artifact — load() reconstructs it from
        # the graph cache. Guarantee that cache exists so a loaded artifact is
        # always usable, even when training ran with use_graph_cache=False.
        cache_path = self._graph_cache_path or self._gnn_cfg().get(
            "graph_cache_path", "data/processed/graph/fraud_graph.pt"
        )
        if (
            self.data is not None
            and self.graph_builder is not None
            and not GraphBuilder.is_cache_valid(cache_path, graph_dataset_hash or "")
        ):
            self.graph_builder.save(
                self.data.cpu() if hasattr(self.data, "cpu") else self.data,
                cache_path,
                dataset_hash=graph_dataset_hash,
            )
        self._graph_cache_path = cache_path

        metadata = {
            "config": self.config,
            "model_config": {
                "num_node_features": self.model.num_node_features,
                "hidden_dims": self.model.hidden_dims,
                "mlp_hidden_dims": self.model.mlp_hidden_dims,
                "dropout": self.model.dropout,
                "aggr": self.model.aggr,
            },
            "graph_builder": (
                self.graph_builder.state_dict() if self.graph_builder else {}
            ),
            "graph_cache_path": self._graph_cache_path
            or self._gnn_cfg().get(
                "graph_cache_path", "data/processed/graph/fraud_graph.pt"
            ),
            "graph_dataset_hash": graph_dataset_hash,
            "split_sizes": dict(self.split_sizes),
            "training_history": self._training_history,
            "threshold": self.threshold,
            "calibrator": self.calibrator,
        }
        joblib.dump(metadata, paths["metadata"])
        write_checksums(
            paths["checksums"],
            {"weights": paths["weights"], "metadata": paths["metadata"]},
        )
        logger.info("GNN model saved to %s (+ metadata, checksums)", paths["weights"])

    @classmethod
    def load(cls, path: str, config: Optional[Dict] = None) -> "GNNTrainer":
        """Load an artifact saved by ``save()``. Verifies the sha256 manifest
        before deserializing anything (fail-closed). The graph is loaded
        LAZILY on the first ``predict_proba`` call — reading parquet from
        ``config.data.processed_dir`` and validating against the stored
        ``graph_dataset_hash``."""
        base_path = Path(path)
        paths = _artifact_paths(base_path)
        verify_checksums(
            paths["checksums"],
            {"weights": paths["weights"], "metadata": paths["metadata"]},
        )
        metadata = joblib.load(paths["metadata"])
        cfg = config or metadata["config"]

        trainer = cls(cfg)
        mc = metadata["model_config"]
        trainer.build_model(num_node_features=mc["num_node_features"])
        state_dict = torch.load(
            paths["weights"], map_location="cpu", weights_only=True
        )
        trainer.model.load_state_dict(state_dict)
        trainer.model.to(trainer.device)

        trainer.graph_builder = GraphBuilder.from_state_dict(
            metadata.get("graph_builder", {})
        )
        trainer.split_sizes = dict(metadata.get("split_sizes", {}))
        trainer._training_history = metadata.get("training_history", [])
        trainer.threshold = metadata.get("threshold")
        trainer.calibrator = metadata.get("calibrator")
        trainer._graph_cache_path = metadata.get("graph_cache_path")
        trainer._graph_dataset_hash = metadata.get("graph_dataset_hash")
        if trainer.threshold is None or trainer.calibrator is None:
            logger.warning(
                "Loaded GNN artifact has no frozen threshold/calibrator — "
                "predict()/predict_proba_calibrated() will raise."
            )
        logger.info("GNN model loaded from %s (graph loads lazily)", paths["weights"])
        return trainer

    def ensure_graph_loaded(self) -> None:
        """Populate ``self.data`` from the graph cache if not already in
        memory — used by ``predict_proba`` after ``load()``."""
        if self.data is not None:
            return
        cache_path = self._graph_cache_path or self._gnn_cfg().get(
            "graph_cache_path", "data/processed/graph/fraud_graph.pt"
        )
        expected_hash = self._graph_dataset_hash
        processed_dir = Path(self.config["data"]["processed_dir"])
        if expected_hash is not None:
            actual = compute_dataset_hash(processed_dir, _DATASET_FILES)
            if actual != expected_hash:
                raise RuntimeError(
                    "Processed dataset hash changed since the GNN was trained "
                    f"({expected_hash[:12]} -> {actual[:12]}) — the graph "
                    "topology is stale; retrain the GNN."
                )
        self.data = GraphBuilder.load(cache_path).to(self.device)
        if not self.split_sizes:
            self.split_sizes = dict(self.data.split_sizes)


# ── training entry point ──────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GNN-GraphSAGE model.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument(
        "--results-json",
        default="reports/gnn_results.json",
        help="Where to write the standalone-eval result summary (PRD 12.2.3).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Resume-on-crash checkpoint path (see GNNTrainer.train). If the "
        "path exists, training resumes from its saved epoch instead of "
        "epoch 1; the file is deleted on a clean finish.",
    )
    args = parser.parse_args()

    import json

    import mlflow

    settings = load_settings(args.config)
    config = settings.model_dump()
    seed = set_seed(config.get("project", {}).get("random_seed", 42))
    if args.epochs is not None:
        config["model"]["gnn"]["max_epochs"] = args.epochs
        settings = Settings.model_validate(config)

    processed_dir = Path(config["data"]["processed_dir"])
    logger.info("Loading parquet data...")
    X_train = pd.read_parquet(processed_dir / "train_features.parquet")
    y_train = pd.read_parquet(processed_dir / "train_labels.parquet").squeeze()
    X_val = pd.read_parquet(processed_dir / "val_features.parquet")
    y_val = pd.read_parquet(processed_dir / "val_labels.parquet").squeeze()
    X_test = pd.read_parquet(processed_dir / "test_features.parquet")
    y_test = pd.read_parquet(processed_dir / "test_labels.parquet").squeeze()
    logger.info(
        "Data loaded: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test)
    )

    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "fraud_detection"))

    with mlflow.start_run(run_name="gnn_graphsage") as run, RunLogger(
        run_type="gnn", run_name=f"gnn_{run.info.run_id[:8]}"
    ) as run_logger:
        run_id = run.info.run_id
        logger.info("MLflow run ID: %s", run_id)

        trainer = GNNTrainer(config)
        g = config["model"]["gnn"]
        mlflow.log_params(
            {
                "model_type": "GNN-GraphSAGE",
                "hidden_dims": str(g["hidden_dims"]),
                "mlp_hidden_dims": str(g["mlp_hidden_dims"]),
                "aggr": g["aggr"],
                "dropout": g["dropout"],
                "num_neighbors": str(g["num_neighbors"]),
                "learning_rate": g["learning_rate"],
                "weight_decay": g["weight_decay"],
                "max_epochs": g["max_epochs"],
                "batch_size": g["batch_size"],
                "patience": g["patience"],
                "card1_max_neighbors": g["card1_max_neighbors"],
                "addr_product_max_neighbors": g["addr_product_max_neighbors"],
                "random_seed": seed,
                "device": str(trainer.device),
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
                "tensorboard_log_dir": str(run_logger.log_dir),
            }
        )

        history = trainer.train(
            X_train, y_train, X_val, y_val,
            run_logger=run_logger, X_test=X_test, y_test=y_test,
            checkpoint_path=args.checkpoint_path,
        )
        mlflow.log_param("total_parameters", trainer.model.count_parameters())

        evaluator = ModelEvaluator()
        y_prob_train = trainer.predict_proba(X_train)
        y_prob_val = trainer.predict_proba(X_val)
        y_prob_test = trainer.predict_proba(X_test)

        pr_auc_train = evaluator.compute_pr_auc(y_train.to_numpy(), y_prob_train)
        pr_auc_val = evaluator.compute_pr_auc(y_val.to_numpy(), y_prob_val)
        pr_auc_test = evaluator.compute_pr_auc(y_test.to_numpy(), y_prob_test)
        roc_auc_test = evaluator.compute_roc_auc(y_test.to_numpy(), y_prob_test)
        overfit_gap = pr_auc_train - pr_auc_test

        calibrator = evaluator.fit_calibrator(
            y_val.to_numpy(), y_prob_val, method="isotonic"
        )
        y_prob_val_cal = evaluator.apply_calibration(calibrator, y_prob_val)
        y_prob_test_cal = evaluator.apply_calibration(calibrator, y_prob_test)
        brier_test_before = evaluator.compute_brier_score(y_test.to_numpy(), y_prob_test)
        brier_test_after = evaluator.compute_brier_score(
            y_test.to_numpy(), y_prob_test_cal
        )

        thresh_cfg = config.get("thresholds", {})
        cost_fn = thresh_cfg.get("cost_fn", 500)
        cost_fp = thresh_cfg.get("cost_fp", 5)
        revenue_tp = thresh_cfg.get("revenue_tp", 480)
        optimal_t = evaluator.find_optimal_threshold(
            y_val.to_numpy(), y_prob_val, cost_fn, cost_fp, revenue_tp
        )
        metrics = evaluator.compute_metrics_at_threshold(
            y_test.to_numpy(), y_prob_test, optimal_t
        )

        logger.info("=" * 60)
        logger.info("GNN-GraphSAGE — standalone evaluation (PRD 12.2.3)")
        logger.info("  Train PR-AUC:  %.4f", pr_auc_train)
        logger.info("  Val PR-AUC:    %.4f", pr_auc_val)
        logger.info("  Test PR-AUC:   %.4f   (baseline ensemble: 0.5502)", pr_auc_test)
        logger.info("  Test ROC-AUC:  %.4f", roc_auc_test)
        logger.info("  Overfit gap (train-test): %.4f", overfit_gap)
        logger.info(
            "  Test P=%.4f R=%.4f F1=%.4f at threshold %.4f",
            metrics["precision"], metrics["recall"], metrics["f1"], optimal_t,
        )
        logger.info("=" * 60)

        for k, v in {
            "gnn_pr_auc_train": pr_auc_train,
            "gnn_pr_auc_val": pr_auc_val,
            "gnn_pr_auc_test": pr_auc_test,
            "gnn_roc_auc_test": roc_auc_test,
            "gnn_overfit_gap": overfit_gap,
            "gnn_optimal_threshold": optimal_t,
            "gnn_calibration_brier_test_before": brier_test_before,
            "gnn_calibration_brier_test_after": brier_test_after,
        }.items():
            mlflow.log_metric(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(f"gnn_test_{k}", v)

        reports_dir = Path("reports/figures")
        reports_dir.mkdir(parents=True, exist_ok=True)
        y_pred = (y_prob_test >= optimal_t).astype(int)
        evaluator.plot_pr_curve(
            y_test.to_numpy(), y_prob_test, "GNN-GraphSAGE",
            str(reports_dir / "gnn_pr_curve.png"),
        )
        evaluator.plot_roc_curve(
            y_test.to_numpy(), y_prob_test, "GNN-GraphSAGE",
            str(reports_dir / "gnn_roc_curve.png"),
        )
        evaluator.plot_confusion_matrix(
            y_test.to_numpy(), y_pred, str(reports_dir / "gnn_confusion_matrix.png")
        )
        evaluator.plot_reliability_curve(
            y_test.to_numpy(), y_prob_test,
            str(reports_dir / "gnn_reliability_curve.png"),
            y_prob_calibrated=y_prob_test_cal,
        )
        for fig in [
            "gnn_pr_curve.png", "gnn_roc_curve.png",
            "gnn_confusion_matrix.png", "gnn_reliability_curve.png",
        ]:
            mlflow.log_artifact(str(reports_dir / fig))

        trainer.set_threshold(optimal_t)
        trainer.set_calibrator(calibrator)

        model_path = config.get("serving", {}).get(
            "gnn_model_path", "models/gnn_model.pt"
        )
        trainer.save(model_path)
        for artifact_path in _artifact_paths(Path(model_path)).values():
            mlflow.log_artifact(str(artifact_path))

        manifest = build_manifest(
            model_type="gnn",
            mlflow_run_id=run_id,
            settings=settings,
            dataset_dir=processed_dir,
            dataset_files=_DATASET_FILES,
            random_seed=seed,
            metrics={
                "pr_auc_train": pr_auc_train,
                "pr_auc_val": pr_auc_val,
                "pr_auc_test": pr_auc_test,
                "roc_auc_test": roc_auc_test,
                "overfit_gap": overfit_gap,
                "optimal_threshold": optimal_t,
            },
        )
        manifest_path = write_manifest(model_path, manifest)
        mlflow.log_artifact(str(manifest_path))

        # PRD 12.2.3: record the standalone result regardless of outcome.
        baseline = 0.5502
        margin = float(config.get("ensemble", {}).get("min_lightgbm_lift", 0.005))
        result = {
            "phase": "12.2.3",
            "model": "GNN-GraphSAGE",
            "mlflow_run_id": run_id,
            "random_seed": seed,
            "pr_auc_train": pr_auc_train,
            "pr_auc_val": pr_auc_val,
            "pr_auc_test": pr_auc_test,
            "roc_auc_test": roc_auc_test,
            "overfit_gap": overfit_gap,
            "optimal_threshold": optimal_t,
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "brier_test_before": brier_test_before,
            "brier_test_after": brier_test_after,
            "ensemble_baseline_test_pr_auc": baseline,
            "preregistered_margin": margin,
            "beats_baseline_by_margin": bool(pr_auc_test > baseline + margin),
            "history": history,
        }
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_json).write_text(json.dumps(result, indent=2), "utf-8")
        logger.info("Wrote %s", args.results_json)
        logger.info(
            "12.2.4 ensemble integration trigger: test PR-AUC %.4f vs "
            "baseline+margin %.4f -> %s",
            pr_auc_test, baseline + margin,
            "PROCEED" if result["beats_baseline_by_margin"] else "DO NOT PROCEED",
        )


if __name__ == "__main__":
    main()
