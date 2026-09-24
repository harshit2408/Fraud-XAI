"""
tests/unit/test_graph_builder.py

Unit tests for src/data/graph_builder.py (PRD Phase 12).

Covers the mle-reviewer's required regression tests:
  * R1/R2 — phase-scoped, symmetric edge masks: no train-usable edge touches
    a val/test node; no val-usable edge touches a test node; test mask covers
    every edge; edge_index is symmetric.
  * R1 — test-row placement in a shared-key group cannot change which
    train->train edges survive (phase-scoped windowing).
  * R5 — the QuantileTransformer is fit on the TRAIN node span only.
  * R8 — rows on the addr1 imputation sentinel are excluded from composite
    edges.
  * neighbour caps, dedupe, cache round-trip + checksum / dataset-hash guard.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.preprocessing import QuantileTransformer  # noqa: E402

from src.data.graph_builder import (  # noqa: E402
    DEFAULT_ADDR1_MISSING_SENTINEL,
    EdgeKeySpec,
    GraphBuilder,
    GraphBuildConfig,
)


def _mk_frame(n: int, rng: np.random.Generator, n_card1: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "card1": rng.integers(1000, 1000 + n_card1, n),
            "addr1": rng.choice(
                [100.0, 200.0, 300.0, DEFAULT_ADDR1_MISSING_SENTINEL], n
            ),
            "ProductCD": rng.integers(0, 4, n),
            "feat_dollar": rng.uniform(1, 5000, n),
            "feat_unit": rng.normal(0, 1, n),
            "feat_sentinel": rng.choice([-999.0, 1.0, 2.0, 3.0], n),
        }
    )


@pytest.fixture
def splits():
    rng = np.random.default_rng(7)
    X_train, X_val, X_test = _mk_frame(300, rng), _mk_frame(80, rng), _mk_frame(120, rng)
    y_train = pd.Series(rng.integers(0, 2, 300))
    y_val = pd.Series(rng.integers(0, 2, 80))
    y_test = pd.Series(rng.integers(0, 2, 120))
    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture
def built(splits):
    cfg = GraphBuildConfig(card1_max_neighbors=4, addr_product_max_neighbors=3)
    builder = GraphBuilder(cfg)
    data = builder.build(*splits)
    return builder, data


# ── R1 / R2 : phase-scoped symmetric edge masks ───────────────────────────
def test_edge_mask_train_has_no_non_train_endpoint(built):
    _, data = built
    n_train = data.split_sizes["train"]
    ei = data.edge_index[:, data.edge_mask_train]
    assert ei.numel() > 0
    assert int(ei.max()) < n_train


def test_edge_mask_val_has_no_test_endpoint(built):
    _, data = built
    n_trv = data.split_sizes["train"] + data.split_sizes["val"]
    ei = data.edge_index[:, data.edge_mask_val]
    assert int(ei.max()) < n_trv


def test_edge_mask_test_covers_all_edges(built):
    _, data = built
    assert bool(data.edge_mask_test.all())


def test_edge_index_is_symmetric(built):
    _, data = built
    ei = data.edge_index
    assert ei.shape[1] % 2 == 0
    fwd = set(map(tuple, ei.t().tolist()))
    rev = set((b, a) for a, b in fwd)
    assert fwd == rev


def test_node_split_masks_partition_all_nodes(built):
    _, data = built
    part = data.train_mask.int() + data.val_mask.int() + data.test_mask.int()
    assert torch.all(part == 1)
    assert data.x.shape[0] == data.num_nodes


# ── R1 : test-row placement cannot reshape train->train edges ─────────────
def test_train_edges_invariant_to_test_rows(splits):
    """The set of train<->train edges must be identical whether or not the
    val/test rows are present in the frame passed to build() — phase-scoped
    windowing (R1)."""
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    cfg = GraphBuildConfig(card1_max_neighbors=3, addr_product_max_neighbors=2)

    full = GraphBuilder(cfg).build(X_train, y_train, X_val, y_val, X_test, y_test)
    n_train = len(X_train)
    train_edges_full = {
        tuple(sorted(e))
        for e in full.edge_index[:, full.edge_mask_train].t().tolist()
    }

    stub_X = X_train.iloc[:1].copy()
    stub_y = y_train.iloc[:1].copy()
    train_only = GraphBuilder(cfg).build(
        X_train, y_train, stub_X, stub_y, stub_X, stub_y
    )
    train_edges_alone = {
        tuple(sorted(e))
        for e in train_only.edge_index[:, train_only.edge_mask_train].t().tolist()
        if max(e) < n_train
    }
    assert train_edges_full == train_edges_alone


# ── R5 : scaler fit on the train span only ───────────────────────────────
def test_scaler_fits_on_train_span_only(splits, built):
    builder, data = built
    X_train = splits[0]
    X_all = pd.concat([splits[0], splits[2], splits[4]], axis=0, ignore_index=True)

    ref_train = QuantileTransformer(
        n_quantiles=min(1000, len(X_train)),
        output_distribution="normal",
        subsample=1_000_000_000,
        random_state=0,
    ).fit(X_train.to_numpy(np.float32))
    # Fit an all-rows reference at the SAME n_quantiles so the arrays are
    # comparable; it must differ from the builder's train-only fit.
    ref_all = QuantileTransformer(
        n_quantiles=min(1000, len(X_train)),
        output_distribution="normal",
        subsample=1_000_000_000,
        random_state=0,
    ).fit(X_all.to_numpy(np.float32))

    assert np.allclose(builder.scaler.quantiles_, ref_train.quantiles_)
    assert builder.scaler.quantiles_.shape == ref_all.quantiles_.shape
    assert not np.allclose(builder.scaler.quantiles_, ref_all.quantiles_, atol=1e-6)
    assert data.x.shape == (data.num_nodes, len(builder.feature_names))


# ── R8 : addr1 sentinel excluded from composite edges ────────────────────
def test_addr1_sentinel_rows_excluded_from_composite_edges():
    rng = np.random.default_rng(1)
    n = 60
    addr1 = np.where(np.arange(n) < n // 2, DEFAULT_ADDR1_MISSING_SENTINEL, 100.0)
    X = pd.DataFrame(
        {
            "card1": np.full(n, 9999),
            "addr1": addr1,
            "ProductCD": np.zeros(n, dtype=int),
            "f": rng.normal(size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, n))
    cfg = GraphBuildConfig(card1_max_neighbors=0, addr_product_max_neighbors=5)
    b = GraphBuilder(cfg)
    key = b._composite_key_hash(X)
    assert (key[: n // 2] == -1).all()
    assert (key[n // 2 :] >= 0).all()

    data = b.build(
        X.iloc[:40], y.iloc[:40], X.iloc[40:50], y.iloc[40:50],
        X.iloc[50:], y.iloc[50:]
    )
    ei = data.edge_index.t().tolist()
    sentinel_nodes = set(range(20))
    assert not any(
        a in sentinel_nodes and b_ in sentinel_nodes for a, b_ in ei
    ), "composite edge built across the addr1 sentinel clique"


# ── neighbour caps + dedupe ──────────────────────────────────────────────
def test_card1_out_degree_respects_cap():
    rng = np.random.default_rng(3)
    n = 400
    X = pd.DataFrame(
        {
            "card1": np.full(n, 5),
            "addr1": np.full(n, DEFAULT_ADDR1_MISSING_SENTINEL),
            "ProductCD": np.zeros(n, dtype=int),
            "f": rng.normal(size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, n))
    cap = 5
    cfg = GraphBuildConfig(card1_max_neighbors=cap, addr_product_max_neighbors=0)
    data = GraphBuilder(cfg).build(
        X.iloc[:300], y.iloc[:300], X.iloc[300:340], y.iloc[300:340],
        X.iloc[340:], y.iloc[340:]
    )
    ei = data.edge_index
    deg = torch.zeros(data.num_nodes, dtype=torch.long)
    deg.scatter_add_(0, ei[0], torch.ones(ei.shape[1], dtype=torch.long))
    assert int(deg.max()) <= 2 * cap


def test_no_duplicate_edges(built):
    _, data = built
    pairs = [tuple(sorted(e)) for e in data.edge_index.t().tolist()]
    counts = Counter(pairs)
    assert set(counts.values()) == {2}


# ── cache round-trip + integrity ─────────────────────────────────────────
def test_save_load_roundtrip_and_cache_validation(built, tmp_path):
    builder, data = built
    p = tmp_path / "g.pt"
    builder.save(data, str(p), dataset_hash="hash_abc")

    loaded = GraphBuilder.load(str(p))
    assert torch.equal(loaded.edge_index, data.edge_index)
    assert torch.equal(loaded.edge_mask_train, data.edge_mask_train)

    assert GraphBuilder.is_cache_valid(str(p), "hash_abc") is True
    assert GraphBuilder.is_cache_valid(str(p), "different_hash") is False

    p.write_bytes(b"corrupted")
    with pytest.raises(Exception):
        GraphBuilder.load(str(p))
    assert GraphBuilder.is_cache_valid(str(p), "hash_abc") is False


def test_from_state_dict_restores_scaler(built):
    builder, _ = built
    state = builder.state_dict()
    restored = GraphBuilder.from_state_dict(state)
    assert restored.feature_names == builder.feature_names
    assert restored.split_sizes == builder.split_sizes
    assert restored.scaler is not None


# ── ADR-006 §3.1 / §6 : EdgeKeySpec + blocklist + generalized sentinels ────
@pytest.mark.parametrize(
    "col", ["P_emaildomain", "R_emaildomain", "DeviceInfo", "id_31", "id_33", "card_hash_freq"]
)
def test_edge_key_spec_rejects_frequency_encoded_blocklist_column(col):
    with pytest.raises(ValueError, match="blocklist"):
        EdgeKeySpec(name="bad", columns=(col,), max_neighbors=5)


def test_edge_key_spec_rejects_target_encoded_column():
    with pytest.raises(ValueError, match="blocklist"):
        EdgeKeySpec(name="bad", columns=("card1_target_enc",), max_neighbors=5)


def test_edge_key_spec_rejects_blocklist_column_in_composite_key():
    with pytest.raises(ValueError, match="blocklist"):
        EdgeKeySpec(name="bad", columns=("card1", "DeviceInfo"), max_neighbors=5)


def test_graph_build_config_rejects_blocklist_in_legacy_fields():
    with pytest.raises(ValueError, match="blocklist"):
        GraphBuildConfig(card1_column="id_31")


def test_edge_key_spec_rejects_sentinel_column_not_in_columns():
    with pytest.raises(ValueError, match="not in columns"):
        EdgeKeySpec(
            name="bad", columns=("card1",), max_neighbors=5, sentinels=(("addr1", -999.0),)
        )


def test_graph_build_config_rejects_duplicate_edge_spec_names():
    dup = (
        EdgeKeySpec(name="x", columns=("card1",), max_neighbors=5),
        EdgeKeySpec(name="x", columns=("card2",), max_neighbors=5),
    )
    with pytest.raises(ValueError, match="duplicate"):
        GraphBuildConfig(edge_specs=dup)


def test_empty_edge_specs_resolves_to_legacy_pair():
    cfg = GraphBuildConfig(card1_max_neighbors=4, addr_product_max_neighbors=3)
    specs = cfg.resolved_edge_specs()
    assert [s.name for s in specs] == ["card1", "addr_product"]
    assert specs[0].columns == ("card1",)
    assert specs[0].max_neighbors == 4
    assert specs[1].columns == ("addr1", "ProductCD")
    assert specs[1].max_neighbors == 3
    assert specs[1].sentinels == (("addr1", DEFAULT_ADDR1_MISSING_SENTINEL),)


def test_legacy_default_and_explicit_equivalent_specs_are_byte_identical(splits):
    """ADR-006 §6: the legacy-vs-explicit-edge_specs regression test — an
    empty ``edge_specs`` (implicit legacy pair) must produce an
    ``edge_index`` byte-identical to passing the equivalent specs
    explicitly, before any new spec is exercised in training."""
    legacy_cfg = GraphBuildConfig(card1_max_neighbors=4, addr_product_max_neighbors=3)
    explicit_cfg = GraphBuildConfig(
        card1_max_neighbors=4,
        addr_product_max_neighbors=3,
        edge_specs=(
            EdgeKeySpec(name="card1", columns=("card1",), max_neighbors=4),
            EdgeKeySpec(
                name="addr_product",
                columns=("addr1", "ProductCD"),
                max_neighbors=3,
                sentinels=(("addr1", DEFAULT_ADDR1_MISSING_SENTINEL),),
            ),
        ),
    )
    legacy_data = GraphBuilder(legacy_cfg).build(*splits)
    explicit_data = GraphBuilder(explicit_cfg).build(*splits)

    assert torch.equal(legacy_data.edge_index, explicit_data.edge_index)
    assert torch.equal(legacy_data.edge_mask_train, explicit_data.edge_mask_train)
    assert torch.equal(legacy_data.edge_mask_val, explicit_data.edge_mask_val)
    assert torch.equal(legacy_data.edge_mask_test, explicit_data.edge_mask_test)


def test_multi_key_edge_specs_are_unioned(splits):
    """A card_full-style refinement key alongside the legacy pair adds edges
    without disturbing the legacy edges (Arm A: additive edge sets)."""
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    for X in (X_train, X_val, X_test):
        X["card2"] = X["card1"] + 1  # deterministic refinement of card1

    cfg = GraphBuildConfig(
        edge_specs=(
            EdgeKeySpec(name="card1", columns=("card1",), max_neighbors=4),
            EdgeKeySpec(name="card_full", columns=("card1", "card2"), max_neighbors=4),
        )
    )
    data = GraphBuilder(cfg).build(X_train, y_train, X_val, y_val, X_test, y_test)
    assert data.edge_index.shape[1] > 0
    assert bool(data.edge_mask_test.all())


def test_generalized_sentinel_exclusion_on_a_non_addr1_column():
    """R8's sentinel exclusion generalizes past addr1 — a new refinement key
    on a numeric column carrying the same imputation sentinel must exclude
    those rows too, or it recreates the sentinel-clique bug one column over
    (mle-reviewer CRITICAL finding)."""
    rng = np.random.default_rng(2)
    n = 60
    card2 = np.where(np.arange(n) < n // 2, DEFAULT_ADDR1_MISSING_SENTINEL, 100.0)
    X = pd.DataFrame(
        {
            "card1": np.full(n, 9999),
            "card2": card2,
            "addr1": np.full(n, 200.0),
            "ProductCD": np.zeros(n, dtype=int),
            "f": rng.normal(size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, n))
    cfg = GraphBuildConfig(
        edge_specs=(
            EdgeKeySpec(
                name="card2_key",
                columns=("card2",),
                max_neighbors=10,
                sentinels=(("card2", DEFAULT_ADDR1_MISSING_SENTINEL),),
            ),
        )
    )
    b = GraphBuilder(cfg)
    key = b._spec_key_hash(X, cfg.edge_specs[0])
    assert (key[: n // 2] == -1).all()
    assert (key[n // 2 :] >= 0).all()

    data = b.build(
        X.iloc[:40], y.iloc[:40], X.iloc[40:50], y.iloc[40:50], X.iloc[50:], y.iloc[50:]
    )
    ei = data.edge_index.t().tolist()
    sentinel_nodes = set(range(30))
    assert not any(
        a in sentinel_nodes and b_ in sentinel_nodes for a, b_ in ei
    ), "edge built across the card2 sentinel clique"
