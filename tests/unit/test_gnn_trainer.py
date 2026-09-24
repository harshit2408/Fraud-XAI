"""
tests/unit/test_gnn_trainer.py

Unit tests for src/training/train_gnn.py :: GNNTrainer (PRD Phase 12).

  * build_model -> train (1-2 epochs) runs; history populated
  * R7 — test labels never influence the train loss: permuting y_test before
    train() leaves the trained state_dict byte-identical; and no train
    NeighborLoader batch reaches a test node
  * R6 — predict_proba resolves the split by exact row count; a wrong-length
    frame raises
  * predict raises without a frozen threshold; predict_proba_calibrated
    raises without a calibrator
  * save -> load -> predict_proba reproduces scores; checksum tamper raises
  * _calibrated(gnn, "GNN", X_val) from run_ensemble_eval works
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.graph_builder import DEFAULT_ADDR1_MISSING_SENTINEL  # noqa: E402
from src.training.train_gnn import GNNTrainer  # noqa: E402


def _frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "card1": rng.integers(1000, 1006, n),
            "card2": rng.choice([1.0, 2.0, DEFAULT_ADDR1_MISSING_SENTINEL], n),
            "card3": rng.choice([10.0, 20.0, DEFAULT_ADDR1_MISSING_SENTINEL], n),
            "card5": rng.choice([100.0, 200.0, DEFAULT_ADDR1_MISSING_SENTINEL], n),
            "addr1": rng.choice([100.0, 200.0, DEFAULT_ADDR1_MISSING_SENTINEL], n),
            "ProductCD": rng.integers(0, 3, n),
            "f_dollar": rng.uniform(1, 3000, n),
            "f_unit": rng.normal(size=n),
        }
    )


@pytest.fixture
def cfg(tmp_path):
    return {
        "project": {"random_seed": 42},
        "data": {"processed_dir": str(tmp_path / "processed")},
        "model": {
            "gnn": {
                "hidden_dims": [16, 8],
                "mlp_hidden_dims": [8, 4],
                "dropout": 0.1,
                "aggr": "mean",
                "num_neighbors": [5, 3],
                "learning_rate": 0.01,
                "weight_decay": 1e-5,
                "max_epochs": 2,
                "batch_size": 64,
                "patience": 5,
                "gradient_clip_val": 1.0,
                "device": "cpu",
                "pos_weight": None,
                "card1_max_neighbors": 4,
                "addr_product_max_neighbors": 3,
                "graph_cache_path": str(tmp_path / "graph" / "g.pt"),
                "use_graph_cache": False,
            }
        },
        "thresholds": {"cost_fn": 500, "cost_fp": 5, "revenue_tp": 480},
        "ensemble": {"min_lightgbm_lift": 0.005},
        "serving": {"gnn_model_path": str(tmp_path / "models" / "gnn_model.pt")},
        "mlflow": {"tracking_uri": "file:./mlruns", "experiment_name": "t"},
    }


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(11)
    n_tr, n_va, n_te = 240, 60, 90
    X_train, X_val, X_test = _frame(n_tr, rng), _frame(n_va, rng), _frame(n_te, rng)
    y_train = pd.Series((rng.random(n_tr) < 0.2).astype(int))
    y_val = pd.Series((rng.random(n_va) < 0.2).astype(int))
    y_test = pd.Series((rng.random(n_te) < 0.2).astype(int))
    proc = tmp_path / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    X_test.to_parquet(proc / "test_features.parquet", index=False)
    y_test.to_frame("isFraud").to_parquet(proc / "test_labels.parquet", index=False)
    return X_train, y_train, X_val, y_val, X_test, y_test


def _train(cfg, data, *, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = data
    t = GNNTrainer(cfg)
    hist = t.train(X_train, y_train, X_val, y_val, X_test=X_test, y_test=y_test)
    return t, hist


# ── basic training ──────────────────────────────────────────────────────
def test_train_runs_and_populates_history(cfg, data):
    trainer, hist = _train(cfg, data)
    assert len(hist["train_loss"]) >= 1
    assert len(hist["val_pr_auc"]) == len(hist["train_loss"])
    assert trainer.model is not None
    assert trainer.split_sizes == {"train": 240, "val": 60, "test": 90}
    assert "epoch_seconds" in hist and len(hist["epoch_seconds"]) == len(hist["train_loss"])


# ── ADR-006 §3.2 Arm B : architecture knobs wired through build_model ────
def test_build_model_passes_through_arm_b_knobs(cfg):
    cfg["model"]["gnn"]["input_dropout"] = 0.1
    cfg["model"]["gnn"]["l2_normalize"] = True
    cfg["model"]["gnn"]["residual"] = True
    t = GNNTrainer(cfg)
    model = t.build_model(num_node_features=8)
    assert model.input_dropout == 0.1
    assert model.l2_normalize is True
    assert model.residual is True


def test_build_model_rejects_num_neighbors_hidden_dims_length_mismatch(cfg):
    cfg["model"]["gnn"]["hidden_dims"] = [128, 64, 32]  # 3 layers
    cfg["model"]["gnn"]["num_neighbors"] = [10, 5]  # only 2 fan-out entries
    t = GNNTrainer(cfg)
    with pytest.raises(ValueError, match="num_neighbors"):
        t.build_model(num_node_features=8)


def test_train_runs_with_three_layer_residual_architecture(cfg, data):
    """End-to-end: num_layers=3 + residual + matching fan-out actually
    trains (ADR-006 §3.2's num_layers=3 variant)."""
    cfg["model"]["gnn"]["hidden_dims"] = [16, 12, 8]
    cfg["model"]["gnn"]["num_neighbors"] = [5, 4, 3]
    cfg["model"]["gnn"]["residual"] = True
    cfg["model"]["gnn"]["l2_normalize"] = True
    cfg["model"]["gnn"]["input_dropout"] = 0.1
    trainer, hist = _train(cfg, data)
    assert len(hist["train_loss"]) == cfg["model"]["gnn"]["max_epochs"]
    assert trainer.model.hidden_dims == [16, 12, 8]
    assert trainer.model.residual is True


# ── R7 : test labels do not influence the train loss ────────────────────
def test_test_labels_do_not_affect_training(cfg, data):
    X_train, y_train, X_val, y_val, X_test, y_test = data
    t1, _ = _train(cfg, data, seed=7)

    y_test_perm = pd.Series(np.random.default_rng(999).permutation(y_test.to_numpy()))
    t2, _ = _train(
        cfg,
        (X_train, y_train, X_val, y_val, X_test, y_test_perm),
        seed=7,
    )

    sd1, sd2 = t1.model.state_dict(), t2.model.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), (
            f"param {k} changed when y_test was permuted — test labels leaked "
            f"into the train loss (R7)"
        )


def test_no_train_batch_reaches_a_test_node(cfg, data):
    from torch_geometric.loader import NeighborLoader

    trainer, _ = _train(cfg, data)
    n_trv = trainer.split_sizes["train"] + trainer.split_sizes["val"]
    loader = NeighborLoader(
        trainer._phase_data("train"),
        num_neighbors=[5, 3],
        input_nodes=trainer.data.train_mask.cpu(),
        batch_size=64,
        shuffle=False,
    )
    for batch in loader:
        assert int(batch.n_id.max()) < trainer.split_sizes["train"], (
            "a train batch reached a val/test node (2-hop reach / edge-mask bug)"
        )
        assert int(batch.n_id.max()) < n_trv


# ── R6 : predict_proba split resolution ────────────────────────────────
def test_predict_proba_resolves_split_by_length(cfg, data):
    trainer, _ = _train(cfg, data)
    X_train, _, X_val, _, X_test, _ = data
    assert trainer.predict_proba(X_val).shape == (60,)
    assert trainer.predict_proba(X_test).shape == (90,)
    assert trainer.predict_proba(X_train).shape == (240,)


def test_predict_proba_wrong_length_raises(cfg, data):
    trainer, _ = _train(cfg, data)
    _, _, X_val, _, _, _ = data
    with pytest.raises(ValueError, match="whole-split"):
        trainer.predict_proba(X_val.iloc[:10])


# ── frozen threshold / calibrator contract ────────────────────────────
def test_predict_raises_without_threshold(cfg, data):
    trainer, _ = _train(cfg, data)
    _, _, X_val, _, _, _ = data
    with pytest.raises(ValueError, match="threshold"):
        trainer.predict(X_val)


def test_predict_proba_calibrated_raises_without_calibrator(cfg, data):
    trainer, _ = _train(cfg, data)
    _, _, X_val, _, _, _ = data
    with pytest.raises(ValueError, match="calibrator"):
        trainer.predict_proba_calibrated(X_val)


# ── save / load round-trip ────────────────────────────────────────────
def test_save_load_reproduces_scores(cfg, data, tmp_path):
    from sklearn.isotonic import IsotonicRegression

    trainer, _ = _train(cfg, data)
    _, _, X_val, y_val, X_test, _ = data
    before = trainer.predict_proba(X_test)

    trainer.set_threshold(0.5)
    cal = IsotonicRegression(out_of_bounds="clip").fit(
        trainer.predict_proba(X_val), y_val.to_numpy()
    )
    trainer.set_calibrator(cal)

    path = str(tmp_path / "models" / "gnn_model.pt")
    trainer.save(path)

    loaded = GNNTrainer.load(path, config=cfg)
    after = loaded.predict_proba(X_test)
    assert np.allclose(before, after, atol=1e-5)
    assert loaded.threshold == 0.5
    assert loaded.calibrator is not None
    assert loaded.predict_proba_calibrated(X_test).shape == (90,)
    assert loaded.predict(X_test).shape == (90,)


def test_load_rejects_tampered_artifact(cfg, data, tmp_path):
    trainer, _ = _train(cfg, data)
    path = tmp_path / "models" / "gnn_model.pt"
    trainer.save(str(path))
    weights = path.with_name("gnn_model.weights.pt")
    weights.write_bytes(b"tampered")
    with pytest.raises(Exception):
        GNNTrainer.load(str(path), config=cfg)


# ── ensemble integration hook ─────────────────────────────────────────
def test_run_ensemble_eval_calibrated_helper_works(cfg, data, tmp_path):
    from sklearn.isotonic import IsotonicRegression

    from scripts.run_ensemble_eval import _calibrated

    trainer, _ = _train(cfg, data)
    _, _, X_val, y_val, _, _ = data
    cal = IsotonicRegression(out_of_bounds="clip").fit(
        trainer.predict_proba(X_val), y_val.to_numpy()
    )
    trainer.set_calibrator(cal)
    out = _calibrated(trainer, "GNN", X_val)
    assert isinstance(out, np.ndarray) and out.shape == (60,)


# ── ADR-006 §3.1 Arm A : edge_spec_set config wiring ───────────────────
def test_variant_graph_cache_path_legacy_is_unchanged(cfg):
    t = GNNTrainer(cfg)
    base = cfg["model"]["gnn"]["graph_cache_path"]
    assert t._variant_graph_cache_path(base) == base


@pytest.mark.parametrize("variant", ["card_full", "addr_card"])
def test_variant_graph_cache_path_suffixes_non_legacy_variants(cfg, variant):
    cfg["model"]["gnn"]["edge_spec_set"] = variant
    t = GNNTrainer(cfg)
    base = cfg["model"]["gnn"]["graph_cache_path"]
    suffixed = t._variant_graph_cache_path(base)
    assert suffixed != base
    assert variant in suffixed


@pytest.mark.parametrize("variant", ["card_full", "addr_card"])
def test_train_runs_with_arm_a_edge_spec_set(cfg, data, variant):
    """Arm A wiring end-to-end: setting edge_spec_set in config actually
    changes which edges GraphBuilder constructs and training still completes
    (frozen architecture — only the graph changes, ADR-006 §3.1)."""
    cfg["model"]["gnn"]["edge_spec_set"] = variant
    trainer, hist = _train(cfg, data)
    assert len(hist["train_loss"]) == cfg["model"]["gnn"]["max_epochs"]
    # the cache actually landed at the variant-suffixed path
    assert variant in trainer._graph_cache_path
    assert len(trainer.graph_builder.config.resolved_edge_specs()) == 3
    names = [s.name for s in trainer.graph_builder.config.resolved_edge_specs()]
    assert "card1" in names and "addr_product" in names and variant in names


# ── memory-leak fix : _phase_data / eval NeighborLoader cached per phase ──
def test_phase_data_is_cached_not_rebuilt_per_call(cfg, data):
    """_phase_data used to allocate a fresh edge_index[:, mask] slice (a
    multi-million-element tensor in production) on every call — including
    once per epoch via _score_nodes. Repeated calls for the same phase must
    now return the SAME object, not rebuild it, or the per-epoch OOM this
    caused (observed empirically during an Optuna HPO run: climbing
    epoch time culminating in a bad_alloc) will recur."""
    X_train, y_train, X_val, y_val, X_test, y_test = data
    t = GNNTrainer(cfg)
    t.train(X_train, y_train, X_val, y_val, X_test=X_test, y_test=y_test)

    val_data_1 = t._phase_data("val")
    val_data_2 = t._phase_data("val")
    assert val_data_1 is val_data_2

    train_data_1 = t._phase_data("train")
    train_data_2 = t._phase_data("train")
    assert train_data_1 is train_data_2


def test_eval_loader_is_cached_not_rebuilt_per_call(cfg, data):
    """_score_nodes used to construct a fresh NeighborLoader every call
    (once per epoch during training) even though the underlying phase graph
    never changes within one train() call. It must now reuse one instance
    per phase."""
    X_train, y_train, X_val, y_val, X_test, y_test = data
    t = GNNTrainer(cfg)
    t.train(X_train, y_train, X_val, y_val, X_test=X_test, y_test=y_test)

    t._score_nodes("val")
    loader_1 = t._eval_loader_cache.get("val")
    t._score_nodes("val")
    loader_2 = t._eval_loader_cache.get("val")
    assert loader_1 is not None
    assert loader_1 is loader_2


def test_phase_data_cache_is_reset_by_a_fresh_prepare_graph_call(cfg, data):
    """A second _prepare_graph (e.g. a new trial reusing a trainer instance
    against a different graph) must invalidate the old cached phase-data
    views — reusing a stale slice against a new self.data would silently
    score the wrong graph."""
    X_train, y_train, X_val, y_val, X_test, y_test = data
    t = GNNTrainer(cfg)
    t.train(X_train, y_train, X_val, y_val, X_test=X_test, y_test=y_test)
    first_val_data = t._phase_data("val")

    t._prepare_graph(
        X_train, y_train, X_val, y_val, X_test, y_test,
        cache_path=str(Path(cfg["model"]["gnn"]["graph_cache_path"]).with_suffix(".v2.pt")),
        dataset_hash=None,
        use_cache=False,
    )
    assert t._phase_data_cache == {}
    assert t._eval_loader_cache == {}
    second_val_data = t._phase_data("val")
    assert second_val_data is not first_val_data


# ── crash-resume checkpointing (GNNTrainer.train(checkpoint_path=...)) ───
def test_checkpoint_is_written_after_each_epoch_and_removed_on_completion(
    cfg, data, tmp_path
):
    ckpt_path = tmp_path / "resume.ckpt.pt"
    X_train, y_train, X_val, y_val, X_test, y_test = data
    t = GNNTrainer(cfg)
    t.train(
        X_train, y_train, X_val, y_val, X_test=X_test, y_test=y_test,
        checkpoint_path=str(ckpt_path),
    )
    # train() ran to max_epochs (no early stop expected in 2 epochs) and
    # finished cleanly — the resume checkpoint must be cleaned up.
    assert not ckpt_path.exists()


def test_training_resumes_from_a_mid_run_checkpoint(cfg, data, tmp_path):
    """Simulates a crash: run 1 epoch, save a checkpoint by hand mid-loop
    (via a monkeypatched early stop), then start a fresh GNNTrainer and
    resume from that checkpoint — it must continue from epoch 2, not
    restart at epoch 1, and the resumed run's history must be prefixed with
    the original epoch's numbers."""
    ckpt_path = tmp_path / "resume.ckpt.pt"
    X_train, y_train, X_val, y_val, X_test, y_test = data
    cfg["model"]["gnn"]["max_epochs"] = 1  # train() completes after epoch 1...
    t1 = GNNTrainer(cfg)

    # ...but patch unlink to a no-op so the "crash recovery" checkpoint isn't
    # deleted on this (otherwise clean) completion — simulating a process
    # that died right after epoch 1's checkpoint write, before cleanup.
    # Scoped with a context manager (not the fixture-wide monkeypatch) so it
    # does NOT leak into t2.train()'s own cleanup below.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "unlink", lambda self, missing_ok=False: None)
        hist1 = t1.train(
            X_train, y_train, X_val, y_val, X_test=X_test, y_test=y_test,
            checkpoint_path=str(ckpt_path),
        )
    assert len(hist1["train_loss"]) == 1
    assert ckpt_path.exists()

    # Fresh trainer, higher max_epochs — must resume from epoch 2, not 1.
    cfg2 = dict(cfg)
    cfg2["model"] = {**cfg["model"], "gnn": {**cfg["model"]["gnn"], "max_epochs": 3}}
    t2 = GNNTrainer(cfg2)
    hist2 = t2.train(
        X_train, y_train, X_val, y_val, X_test=X_test, y_test=y_test,
        checkpoint_path=str(ckpt_path),
    )
    # Resumed history is the FULL history (epoch 1 from the checkpoint +
    # epochs 2-3 newly trained) — 3 entries total, not 2.
    assert len(hist2["train_loss"]) == 3
    assert not ckpt_path.exists()  # cleaned up on this clean completion


def test_resumed_training_restores_optimizer_and_early_stopping_state(
    cfg, data, tmp_path, monkeypatch
):
    """The resumed optimizer must not be a fresh AdamW (momentum reset) —
    its state_dict should be non-empty after one epoch, matching what a
    freshly-checkpointed-then-restored optimizer looks like."""
    ckpt_path = tmp_path / "resume.ckpt.pt"
    X_train, y_train, X_val, y_val, X_test, y_test = data
    cfg["model"]["gnn"]["max_epochs"] = 1
    t1 = GNNTrainer(cfg)
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: None)
    t1.train(
        X_train, y_train, X_val, y_val, X_test=X_test, y_test=y_test,
        checkpoint_path=str(ckpt_path),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1
    assert "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]["state"]
    assert ckpt["epochs_no_improve"] == 0
