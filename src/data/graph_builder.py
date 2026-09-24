"""
src/data/graph_builder.py

Converts the flat, train-only-fitted transaction feature frame into a
transaction-level graph for the GNN-GraphSAGE model (PRD Phase 12).

Nodes are transactions. Edges connect transactions that share an identity
signal: the same ``card1`` (up to ``card1_max_neighbors`` per node) or the
same composite ``(addr1, ProductCD)`` key (up to ``addr_product_max_neighbors``
per node). This is the graph-engineering analogue of ``sequence_builder.py``:
it turns the same feature frame into a model-specific structure and adds
nothing that is not already derivable from a single transaction's own past.

LEAKAGE CONTRACT (mirrors src/data/preprocess.py; reviewed by ecc:mle-reviewer
2026-09-09, findings R1/R2/R3/R5/R8):

  * The processed parquet frames are ALREADY train-only fitted (PCA, target
    encoding, imputation, categorical encoding). GraphBuilder REFITS NOTHING
    of that kind.

  * The one stateful object it fits is a ``QuantileTransformer`` over TRAIN
    NODE ROWS ONLY (identical scope discipline to
    ``SequenceBuilder.fit_scaler_on`` — the 2026-09-09 HIGH scaling-leak fix,
    finding G2). Val/test node features are transformed, never used to fit.
    (R5)

  * Edges are built over ``concat(train, val, test)`` so val/test nodes have
    realistic connectivity, BUT with PHASE-SCOPED NEIGHBOUR WINDOWING (R1):
    the per-node neighbour cap for the *train* edge set is computed over
    train-span nodes only, for the *val* edge set over the train+val span,
    for the *test* edge set over everything. Each ``edge_mask_<phase>`` graph
    is therefore a pure function of that phase's rows plus all earlier rows —
    a test row's position can never reshape which train->train edges survive.

  * Message passing is SYMMETRIC and phase-scoped (R2): ``NeighborLoader``
    for the train phase samples on ``edge_index[:, edge_mask_train]``, whose
    edges by construction have both endpoints in the train span, so a train
    seed node can only ever reach other train nodes at any hop. There is no
    ``directed_past_to_any`` refinement — ``card1`` / ``(addr1, ProductCD)``
    edges are IDENTITY links, not temporal-causal ones, and cross-split
    leakage is fully handled by the phase scoping above. This matches
    ``preprocess.py``'s "full-frame causal features, split-scoped fit"
    pattern.

``card1`` is the RAW integer card code (never re-encoded, never NaN in
IEEE-CIS). ``ProductCD`` in the parquet is a label-encoded int (equality is
still meaningful — same raw value -> same code, fit on train). ``addr1`` in
the parquet is post-imputation numeric: missing values were replaced with the
imputation sentinel (``-999.0`` in the shipped pipeline). Rows on that
sentinel are EXCLUDED from composite ``(addr1, ProductCD)`` edges (R8) — an
"unknown address" is not a shared address, and ~11.5% of rows carry it, which
would otherwise form one dominating clique.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch_geometric.data import Data

from src.utils.checksums import verify_checksums, write_checksums

logger = logging.getLogger(__name__)

# Paper (Uddin & Aziz, arXiv:2604.14231) per-node neighbour caps.
CARD1_MAX_NEIGHBORS: int = 10
ADDR_PRODUCT_MAX_NEIGHBORS: int = 5

# The numeric value src/data/feature_engineering.handle_missing_values writes
# for a missing numeric column (``self._num_fill_values.get(col, -999.0)``).
# Composite (addr1, ProductCD) edges are NOT built across rows carrying this
# in ``addr1``. Passed into ``GraphBuildConfig`` so a future change to the
# imputation fill cannot silently re-enable the sentinel clique (R8).
DEFAULT_ADDR1_MISSING_SENTINEL: float = -999.0

DEFAULT_GRAPH_CACHE: str = "data/processed/graph/fraud_graph.pt"

# Split names, in temporal order. Node ids are assigned in this order, so a
# node id IS a position on the global timeline.
_PHASES: Tuple[str, ...] = ("train", "val", "test")

# ADR-006 §3.1 hard-fail blocklist. These columns are already frequency-
# encoded (or otherwise non-injective / label-correlated) upstream in
# feature_engineering.py's encode_categoricals / create_target_encoding /
# create_card_hash_freq — every train-unseen value collapses to the same
# float, so an equality-based edge key on them silently forms one dominating,
# label-correlated clique (a larger, worse version of the R8 addr1-sentinel
# bug). This is a hard-fail in GraphBuildConfig.__post_init__, not a
# documentation convention, per the mle-reviewer's CRITICAL finding.
FREQUENCY_ENCODED_BLOCKLIST: Tuple[str, ...] = (
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_31",
    "id_33",
    "card_hash_freq",
)
# Any column ending in this suffix is a per-entity target-encoded prior
# (create_target_encoding) — using it as an edge key would link transactions
# by their own leaked label signal.
BLOCKED_COLUMN_SUFFIX: str = "_target_enc"


def _is_blocked_column(col: str) -> bool:
    return col in FREQUENCY_ENCODED_BLOCKLIST or col.endswith(BLOCKED_COLUMN_SUFFIX)


@dataclass(frozen=True)
class EdgeKeySpec:
    """One edge-construction key: nodes sharing the same value(s) across
    ``columns`` are linked, capped at ``max_neighbors`` per node (identical
    windowing discipline to the legacy ``card1`` / ``(addr1, ProductCD)``
    keys — see ``GraphBuilder._edges_from_key``).

    ``sentinels`` maps a column name in ``columns`` to the numeric value that
    marks "missing" for that column (e.g. the ``-999.0`` imputation fill).
    Rows carrying a sentinel in ANY of their key's sentinel-bearing columns
    are excluded from that key's edges (R8, generalized past ``addr1``) — an
    unknown value is not a shared value, and skipping this recreates the
    sentinel-clique bug one column over (mle-reviewer CRITICAL finding).

    ``name`` is a short identifier used only in logging/manifests.
    """

    name: str
    columns: Tuple[str, ...]
    max_neighbors: int
    sentinels: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError(f"EdgeKeySpec {self.name!r}: columns must be non-empty")
        for col in self.columns:
            if _is_blocked_column(col):
                raise ValueError(
                    f"EdgeKeySpec {self.name!r}: column {col!r} is on the "
                    "frequency-encoded/target-encoded blocklist (ADR-006 §3.1) "
                    "— it is non-injective or label-correlated upstream and "
                    "cannot be used as an edge key."
                )
        sentinel_cols = {c for c, _ in self.sentinels}
        unknown = sentinel_cols - set(self.columns)
        if unknown:
            raise ValueError(
                f"EdgeKeySpec {self.name!r}: sentinel column(s) {sorted(unknown)} "
                "not in columns"
            )


# ADR-006 §3.1 Arm A — named edge-spec sets. Each is a refinement of the
# existing card1 key (never a new linking modality — the architect's review
# found email/device columns already destroyed as equality keys upstream).
# "legacy" reproduces the pre-Arm-A graph exactly (empty edge_specs ->
# GraphBuildConfig.resolved_edge_specs() default). "card_full" and
# "addr_card" each ADD one new key alongside the two legacy keys so the
# comparison is additive, not a replacement — the legacy edges stay
# available and the new key's isolated effect is visible against the same
# incumbent graph.
ARM_A_EDGE_SPEC_SETS: Tuple[str, ...] = ("legacy", "card_full", "addr_card")

# Columns new to the card_full key. Each gets its own sentinel exclusion at
# the same imputation fill value addr1 uses (mle-reviewer CRITICAL finding —
# omitting this recreates R8 one column over).
CARD_FULL_COLUMNS: Tuple[str, ...] = ("card1", "card2", "card3", "card5")


def arm_a_edge_specs(
    edge_spec_set: str,
    *,
    card1_max_neighbors: int,
    addr_product_max_neighbors: int,
    addr1_missing_sentinel: float,
    card_full_max_neighbors: int,
    addr_card_max_neighbors: int,
) -> Tuple[EdgeKeySpec, ...]:
    """Build the named Arm A edge-spec set (ADR-006 §3.1).

    * ``"legacy"`` -> ``()`` (empty), so ``GraphBuildConfig.resolved_edge_specs``
      falls back to reconstructing the two legacy keys itself — this keeps
      the legacy path's byte-identical guarantee anchored to one function,
      not duplicated here.
    * ``"card_full"`` -> the two legacy keys PLUS
      ``(card1, card2, card3, card5)``, each new numeric key column
      sentinel-excluded at ``addr1_missing_sentinel`` (the shared imputation
      fill value).
    * ``"addr_card"`` -> the two legacy keys PLUS ``(addr1, card1)``,
      sentinel-excluded on ``addr1`` (reusing the same exclusion the legacy
      composite key already applies).
    """
    if edge_spec_set not in ARM_A_EDGE_SPEC_SETS:
        raise ValueError(
            f"edge_spec_set={edge_spec_set!r} not one of {ARM_A_EDGE_SPEC_SETS}"
        )
    if edge_spec_set == "legacy":
        return ()

    legacy_pair = (
        EdgeKeySpec(name="card1", columns=("card1",), max_neighbors=card1_max_neighbors),
        EdgeKeySpec(
            name="addr_product",
            columns=("addr1", "ProductCD"),
            max_neighbors=addr_product_max_neighbors,
            sentinels=(("addr1", addr1_missing_sentinel),),
        ),
    )
    if edge_spec_set == "card_full":
        card_full = EdgeKeySpec(
            name="card_full",
            columns=CARD_FULL_COLUMNS,
            max_neighbors=card_full_max_neighbors,
            sentinels=tuple((col, addr1_missing_sentinel) for col in CARD_FULL_COLUMNS[1:]),
        )
        return legacy_pair + (card_full,)

    # edge_spec_set == "addr_card"
    addr_card = EdgeKeySpec(
        name="addr_card",
        columns=("addr1", "card1"),
        max_neighbors=addr_card_max_neighbors,
        sentinels=(("addr1", addr1_missing_sentinel),),
    )
    return legacy_pair + (addr_card,)


@dataclass(frozen=True)
class GraphBuildConfig:
    """Immutable knobs for a graph build. Sourced from ``config.model.gnn`` so
    a rebuild is reproducible and the values land in the model manifest.

    ``edge_specs`` (ADR-006 §3.1, Arm A) is the general edge-key list. Leave
    it empty (the default) to get EXACTLY the legacy behaviour — one
    ``card1`` key and one ``(addr1, ProductCD)`` composite key, built from
    ``card1_column`` / ``composite_columns`` / the two ``*_max_neighbors``
    fields / ``addr1_missing_sentinel`` below — byte-identical ``edge_index``
    to every build before this field existed (see
    ``tests/unit/test_graph_builder.py``'s legacy-vs-explicit-specs
    regression test, ADR-006 §6). Pass explicit ``EdgeKeySpec`` entries to add
    Arm A's new refinement keys (``card_full``, ``addr_card``) alongside or
    instead of the legacy pair.
    """

    card1_max_neighbors: int = CARD1_MAX_NEIGHBORS
    addr_product_max_neighbors: int = ADDR_PRODUCT_MAX_NEIGHBORS
    card1_column: str = "card1"
    composite_columns: Tuple[str, str] = ("addr1", "ProductCD")
    addr1_missing_sentinel: float = DEFAULT_ADDR1_MISSING_SENTINEL
    scale_features: bool = True
    edge_specs: Tuple[EdgeKeySpec, ...] = ()
    # ADR-006 §6: a degenerate edge key (e.g. one group spanning most of a
    # phase's eligible prefix) must fail fast and clearly instead of building
    # a multi-hour-OOM edge tensor. None of the legacy/Arm-A specs are
    # expected to come close to this; it exists purely as a guard rail for a
    # future spec authored with a bad max_neighbors or a low-cardinality key.
    max_undirected_edges_per_phase: int = 50_000_000

    def __post_init__(self) -> None:
        for col in (self.card1_column, *self.composite_columns):
            if _is_blocked_column(col):
                raise ValueError(
                    f"GraphBuildConfig: column {col!r} is on the frequency-"
                    "encoded/target-encoded blocklist (ADR-006 §3.1) and "
                    "cannot be used as an edge key."
                )
        names = [s.name for s in self.edge_specs]
        if len(names) != len(set(names)):
            raise ValueError(f"GraphBuildConfig: duplicate edge_specs names: {names}")

    def resolved_edge_specs(self) -> Tuple[EdgeKeySpec, ...]:
        """``edge_specs`` if explicitly set, else the two legacy keys
        rebuilt from the legacy fields — so an empty ``edge_specs`` always
        reproduces pre-Arm-A behaviour exactly."""
        if self.edge_specs:
            return self.edge_specs
        addr_col, prod_col = self.composite_columns
        return (
            EdgeKeySpec(
                name="card1",
                columns=(self.card1_column,),
                max_neighbors=self.card1_max_neighbors,
            ),
            EdgeKeySpec(
                name="addr_product",
                columns=(addr_col, prod_col),
                max_neighbors=self.addr_product_max_neighbors,
                sentinels=((addr_col, self.addr1_missing_sentinel),),
            ),
        )

    @classmethod
    def from_config(cls, config: dict) -> "GraphBuildConfig":
        gnn = (config.get("model", {}) or {}).get("gnn", {}) or {}
        card1_max_neighbors = int(gnn.get("card1_max_neighbors", CARD1_MAX_NEIGHBORS))
        addr_product_max_neighbors = int(
            gnn.get("addr_product_max_neighbors", ADDR_PRODUCT_MAX_NEIGHBORS)
        )
        addr1_missing_sentinel = float(
            gnn.get("addr1_missing_sentinel", DEFAULT_ADDR1_MISSING_SENTINEL)
        )
        edge_spec_set = str(gnn.get("edge_spec_set", "legacy"))
        # `.get(key, default)` only falls back when the key is ABSENT — a
        # config with the key present but explicitly null (YAML `null` /
        # Pydantic Optional[int] = None) still needs the legacy-cap default,
        # hence the extra `or`.
        card_full_max_neighbors = int(
            gnn.get("card_full_max_neighbors") or card1_max_neighbors
        )
        addr_card_max_neighbors = int(
            gnn.get("addr_card_max_neighbors") or addr_product_max_neighbors
        )
        edge_specs = arm_a_edge_specs(
            edge_spec_set,
            card1_max_neighbors=card1_max_neighbors,
            addr_product_max_neighbors=addr_product_max_neighbors,
            addr1_missing_sentinel=addr1_missing_sentinel,
            card_full_max_neighbors=card_full_max_neighbors,
            addr_card_max_neighbors=addr_card_max_neighbors,
        )
        return cls(
            card1_max_neighbors=card1_max_neighbors,
            addr_product_max_neighbors=addr_product_max_neighbors,
            addr1_missing_sentinel=addr1_missing_sentinel,
            edge_specs=edge_specs,
        )


class GraphBuilder:
    """Builds a :class:`torch_geometric.data.Data` transaction graph from the
    flat feature frames, with a train/val/test node mask and phase-scoped
    edge masks.

    Usage::

        builder = GraphBuilder(GraphBuildConfig())
        data = builder.build(X_train, y_train, X_val, y_val, X_test, y_test)
        builder.save(data, "data/processed/graph/fraud_graph.pt")
        # later, in GNNTrainer.load():
        data = GraphBuilder.load("data/processed/graph/fraud_graph.pt")
    """

    def __init__(self, config: Optional[GraphBuildConfig] = None) -> None:
        self.config = config or GraphBuildConfig()
        self.scaler: Optional[QuantileTransformer] = None
        self.feature_names: List[str] = []
        self.split_sizes: Dict[str, int] = {}

    # ── top-level ────────────────────────────────────────────────────────────
    def build(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Data:
        """Concatenate the three splits IN TEMPORAL ORDER (train, val, test),
        build node features ``x``, labels ``y``, ``edge_index``, split masks
        and phase-scoped edge masks, and return one :class:`Data`.

        Node id == global row position in ``concat(train, val, test)``. Each
        split is a contiguous temporal block and rows are already in
        ``TransactionDT`` order within a split, so node-id order IS global
        temporal order.
        """
        for name, (X, y) in {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }.items():
            if len(X) != len(y):
                raise ValueError(f"{name}: X has {len(X)} rows but y has {len(y)}")

        if not (list(X_train.columns) == list(X_val.columns) == list(X_test.columns)):
            raise ValueError("train/val/test feature columns differ")

        self.feature_names = list(X_train.columns)
        n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
        self.split_sizes = {"train": n_train, "val": n_val, "test": n_test}
        n_total = n_train + n_val + n_test

        X_all = pd.concat([X_train, X_val, X_test], axis=0, ignore_index=True)
        y_all = pd.concat(
            [pd.Series(y_train), pd.Series(y_val), pd.Series(y_test)],
            axis=0,
            ignore_index=True,
        )

        logger.info(
            "Building transaction graph: %d nodes (train=%d, val=%d, test=%d), "
            "%d features",
            n_total,
            n_train,
            n_val,
            n_test,
            len(self.feature_names),
        )

        # ── node features (scaler fit on the train span only, R5) ───────────
        x = self._build_node_features(X_all, train_len=n_train)
        y = torch.as_tensor(y_all.to_numpy(dtype=np.float32), dtype=torch.float32)

        # ── phase-scoped edges (R1) ────────────────────────────────────────
        ceilings = {
            "train": n_train,
            "val": n_train + n_val,
            "test": n_total,
        }
        edge_specs = self.config.resolved_edge_specs()
        spec_keys: Dict[str, np.ndarray] = {
            spec.name: self._spec_key_hash(X_all, spec) for spec in edge_specs
        }

        # Build one edge set per phase over that phase's eligible prefix, then
        # union. edge_mask_<phase> marks the edges that phase is allowed to use.
        phase_edges: Dict[str, np.ndarray] = {}
        for phase, ceil in ceilings.items():
            per_spec_edges = [
                self._edges_from_key(
                    spec_keys[spec.name], spec.max_neighbors, eligible_prefix=ceil
                )
                for spec in edge_specs
            ]
            merged = _dedupe_undirected_pairs(
                np.concatenate(per_spec_edges, axis=1)
                if per_spec_edges
                else np.empty((2, 0), dtype=np.int64)
            )
            if merged.shape[1] > self.config.max_undirected_edges_per_phase:
                raise ValueError(
                    f"phase={phase!r}: {merged.shape[1]} undirected edges exceeds "
                    f"max_undirected_edges_per_phase="
                    f"{self.config.max_undirected_edges_per_phase} — a degenerate "
                    "edge key (near-global group, or max_neighbors set too high "
                    "for a low-cardinality column) would otherwise build an "
                    "oversized edge tensor silently (ADR-006 §6)."
                )
            phase_edges[phase] = merged
            logger.info(
                "  phase=%s ceiling=%d -> %d undirected edges (%s, deduped)",
                phase,
                ceil,
                merged.shape[1],
                ", ".join(
                    f"{spec.name}={e.shape[1]}" for spec, e in zip(edge_specs, per_spec_edges)
                ),
            )

        # Union of all phases' pairs -> the master edge list. Since the train
        # prefix is a subset of the val prefix is a subset of test, the union
        # equals phase_edges["test"], but computing it explicitly keeps the
        # invariant obvious and survives a future non-nested change.
        all_pairs = _dedupe_undirected_pairs(
            np.concatenate([phase_edges[p] for p in _PHASES], axis=1)
        )
        edge_index_np = _symmetrize(all_pairs)  # [2, 2 * E_undirected]
        edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)

        # Per-phase boolean edge masks over the symmetric edge_index. An edge
        # is usable by phase P iff its undirected pair is in phase_edges[P].
        pair_to_phase = {phase: _pairs_as_set(phase_edges[phase]) for phase in _PHASES}
        edge_masks = self._build_edge_masks(edge_index_np, pair_to_phase)

        # ── node split masks ───────────────────────────────────────────────
        node_id = np.arange(n_total)
        train_mask = torch.as_tensor(node_id < n_train)
        val_mask = torch.as_tensor((node_id >= n_train) & (node_id < n_train + n_val))
        test_mask = torch.as_tensor(node_id >= n_train + n_val)

        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.edge_mask_train = torch.as_tensor(edge_masks["train"])
        data.edge_mask_val = torch.as_tensor(edge_masks["val"])
        data.edge_mask_test = torch.as_tensor(edge_masks["test"])
        data.node_row_pos = torch.as_tensor(node_id, dtype=torch.long)
        data.split_sizes = dict(self.split_sizes)

        self._assert_invariants(data)
        logger.info(
            "Graph built: %d nodes, %d directed edges "
            "(train-usable=%d, val-usable=%d, test-usable=%d)",
            data.num_nodes,
            data.edge_index.shape[1],
            int(data.edge_mask_train.sum()),
            int(data.edge_mask_val.sum()),
            int(data.edge_mask_test.sum()),
        )
        return data

    # ── node features ───────────────────────────────────────────────────────
    def _build_node_features(self, X_all: pd.DataFrame, train_len: int) -> torch.Tensor:
        """All feature columns -> ``float32`` ``[N, F]``.

        The frame mixes raw-scale columns (dollar ``TransactionAmt``,
        cumulative per-card sums, ``-999`` imputation sentinels) with
        unit-scale PCA components, so a ``QuantileTransformer`` is fit on the
        TRAIN NODE ROWS ONLY (``X_all.iloc[:train_len]``) and then applied to
        the whole frame — the same choice ``sequence_builder.py`` makes for
        the identical raw-scale-plus-sentinels problem. Never fit on val/test
        rows (mle-reviewer R5, finding G2).
        """
        values = X_all.to_numpy(dtype=np.float32)
        if not self.config.scale_features:
            return torch.as_tensor(values, dtype=torch.float32)

        self.scaler = QuantileTransformer(
            n_quantiles=min(1000, train_len),
            output_distribution="normal",
            subsample=1_000_000_000,
            random_state=0,
        )
        self.scaler.fit(values[:train_len])
        scaled = self.scaler.transform(values).astype(np.float32)
        return torch.as_tensor(scaled, dtype=torch.float32)

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted scaler to an arbitrary frame (inference helper).
        Raises if no scaler was fit / restored."""
        if self.config.scale_features and self.scaler is None:
            raise RuntimeError("Scaler not fitted/restored — call build() or load state.")
        values = X.to_numpy(dtype=np.float32)
        if not self.config.scale_features:
            return values
        return self.scaler.transform(values).astype(np.float32)

    # ── edges ───────────────────────────────────────────────────────────────
    def _spec_key_hash(self, X_all: pd.DataFrame, spec: EdgeKeySpec) -> np.ndarray:
        """Map each row's value(s) across ``spec.columns`` to one int64 key.

        For a single-column spec the key is just that column's integer value.
        For a multi-column (composite) spec, ``key = col0_int * K0 + col1_int
        * K1 + ...`` with each ``Ki`` larger than any code seen in that
        column, using the same base-``max+1`` construction the legacy
        ``(addr1, ProductCD)`` composite key used.

        A row is EXCLUDED (key ``-1``) if ANY of its key's sentinel-bearing
        columns (``spec.sentinels``) carries that column's sentinel value —
        the generalized R8 exclusion (mle-reviewer CRITICAL finding: skipping
        this per-column recreates the addr1-sentinel-clique bug one column
        over). The count is logged.
        """
        missing_cols = [c for c in spec.columns if c not in X_all.columns]
        if missing_cols:
            logger.warning(
                "EdgeKeySpec %s: column(s) %s not present — no edges for this spec.",
                spec.name,
                missing_cols,
            )
            return np.full(len(X_all), -1, dtype=np.int64)

        sentinel_map = dict(spec.sentinels)
        missing = np.zeros(len(X_all), dtype=bool)
        col_ints: List[np.ndarray] = []
        for col in spec.columns:
            vals = X_all[col].to_numpy()
            if col in sentinel_map:
                col_missing = np.isclose(vals.astype(np.float64), sentinel_map[col])
                missing = missing | col_missing
                col_ints.append(np.where(col_missing, 0, vals).astype(np.int64))
            else:
                col_ints.append(vals.astype(np.int64))

        n_missing = int(missing.sum())
        if n_missing:
            logger.info(
                "EdgeKeySpec %s: excluding %d rows (%.1f%%) on a sentinel value "
                "(imputation fill).",
                spec.name,
                n_missing,
                100.0 * n_missing / len(X_all),
            )

        key = col_ints[0]
        for col_int in col_ints[1:]:
            base = int(col_int.max()) + 1 if len(col_int) else 1
            key = key * base + col_int
        key = key.copy()
        key[missing] = -1
        return key

    def _composite_key_hash(self, X_all: pd.DataFrame) -> np.ndarray:
        """Backward-compat shim over ``_spec_key_hash`` for the legacy
        ``(addr1, ProductCD)`` composite key — kept because existing tests
        call this method by name."""
        addr_col, prod_col = self.config.composite_columns
        spec = EdgeKeySpec(
            name="addr_product",
            columns=(addr_col, prod_col),
            max_neighbors=self.config.addr_product_max_neighbors,
            sentinels=((addr_col, self.config.addr1_missing_sentinel),),
        )
        return self._spec_key_hash(X_all, spec)

    def _edges_from_key(
        self, key_values: np.ndarray, max_neighbors: int, eligible_prefix: int
    ) -> np.ndarray:
        """Capped undirected edges among node ids that share a key value,
        computed WITHIN the first ``eligible_prefix`` node ids only (R1).

        Per key group (node ids are ascending == temporal order):

          * key ``< 0`` -> skipped (missing/sentinel).
          * group of size ``g <= max_neighbors + 1`` -> complete subgraph.
          * larger group -> sliding window: connect ``id[i]`` to
            ``id[i+1 .. i+max_neighbors]``. Because ids are time-ordered the
            retained neighbours are the temporally nearest ones, which is
            what the paper's "up to N neighbours" implies and which keeps the
            train edge set stable regardless of where later-phase rows fall.

        Returns an ``[2, E']`` int64 array of ``(src, dst)`` pairs with
        ``src < dst`` (pre-symmetrisation).
        """
        if eligible_prefix <= 1 or max_neighbors < 1:
            return np.empty((2, 0), dtype=np.int64)

        keys = key_values[:eligible_prefix]
        order = np.argsort(keys, kind="stable")
        keys_sorted = keys[order]

        # group boundaries in the sorted view
        boundaries = np.flatnonzero(np.diff(keys_sorted)) + 1
        groups = np.split(order, boundaries)

        src_list: List[np.ndarray] = []
        dst_list: List[np.ndarray] = []
        for g_idx in groups:
            if len(g_idx) < 2:
                continue
            if keys[g_idx[0]] < 0:
                continue
            ids = np.sort(g_idx)  # ascending node id == temporal order
            g = len(ids)
            if g <= max_neighbors + 1:
                ii, jj = np.triu_indices(g, k=1)
                src_list.append(ids[ii])
                dst_list.append(ids[jj])
            else:
                for offset in range(1, max_neighbors + 1):
                    src_list.append(ids[:-offset])
                    dst_list.append(ids[offset:])

        if not src_list:
            return np.empty((2, 0), dtype=np.int64)
        src = np.concatenate(src_list)
        dst = np.concatenate(dst_list)
        lo = np.minimum(src, dst)
        hi = np.maximum(src, dst)
        return np.stack([lo, hi], axis=0).astype(np.int64)

    @staticmethod
    def _build_edge_masks(
        edge_index_np: np.ndarray, pair_to_phase: Dict[str, set]
    ) -> Dict[str, np.ndarray]:
        """For each directed edge in ``edge_index_np`` ([2, 2E]), mark whether
        its undirected pair is usable by each phase."""
        if edge_index_np.shape[1] == 0:
            return {p: np.zeros(0, dtype=bool) for p in pair_to_phase}
        lo = np.minimum(edge_index_np[0], edge_index_np[1]).astype(np.int64)
        hi = np.maximum(edge_index_np[0], edge_index_np[1]).astype(np.int64)
        packed = (lo << np.int64(32)) | hi
        masks: Dict[str, np.ndarray] = {}
        for phase, pairset in pair_to_phase.items():
            masks[phase] = np.fromiter(
                (int(p) in pairset for p in packed), dtype=bool, count=len(packed)
            )
        return masks

    # ── invariants ──────────────────────────────────────────────────────────
    def _assert_invariants(self, data: Data) -> None:
        n = data.num_nodes
        assert data.x.shape == (n, len(self.feature_names)), data.x.shape
        assert data.y.shape == (n,), data.y.shape
        # masks partition the nodes exactly
        part = data.train_mask.int() + data.val_mask.int() + data.test_mask.int()
        assert torch.all(part == 1), "node split masks do not partition"
        # phase-scoped edge masks: NO train-usable edge touches a val/test node
        n_train = self.split_sizes["train"]
        n_trv = n_train + self.split_sizes["val"]
        ei = data.edge_index
        tr = data.edge_mask_train
        if tr.any():
            assert int(ei[:, tr].max().item()) < n_train, (
                "edge_mask_train includes an edge with a non-train endpoint "
                "(R1/R2 violation)"
            )
        va = data.edge_mask_val
        if va.any():
            assert int(ei[:, va].max().item()) < n_trv, (
                "edge_mask_val includes an edge with a test endpoint (R1/R2 violation)"
            )
        assert bool(data.edge_mask_test.all()), "edge_mask_test must cover all edges"
        # symmetry
        assert ei.shape[1] % 2 == 0, "edge_index is not symmetric (odd column count)"

    # ── caching / integrity ─────────────────────────────────────────────────
    def state_dict(self) -> dict:
        """Everything needed to reconstruct feature transforms at inference
        time — travels in the GNNTrainer artifact's ``.meta.joblib``, exactly
        like the TFT scaler."""
        return {
            "config": asdict(self.config),
            "feature_names": list(self.feature_names),
            "split_sizes": dict(self.split_sizes),
            "scaler": self.scaler,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "GraphBuilder":
        cfg = dict(state.get("config", {}))
        # A Tuple-typed dataclass field survives asdict() as an actual tuple
        # when the state is kept in-memory (e.g. the GNNTrainer artifact's
        # .meta.joblib, written by GraphBuilder.state_dict()'s own asdict()
        # call) — nested EdgeKeySpec dataclasses become plain dicts either
        # way. Accept both list and tuple containers here.
        if isinstance(cfg.get("composite_columns"), (list, tuple)):
            cfg["composite_columns"] = tuple(cfg["composite_columns"])
        edge_specs = cfg.get("edge_specs")
        if isinstance(edge_specs, (list, tuple)) and edge_specs and isinstance(
            edge_specs[0], dict
        ):
            cfg["edge_specs"] = tuple(
                EdgeKeySpec(
                    name=spec["name"],
                    columns=tuple(spec["columns"]),
                    max_neighbors=spec["max_neighbors"],
                    sentinels=tuple(tuple(s) for s in spec.get("sentinels", ())),
                )
                for spec in edge_specs
            )
        elif isinstance(edge_specs, list):
            cfg["edge_specs"] = tuple(edge_specs)
        builder = cls(GraphBuildConfig(**cfg))
        builder.feature_names = list(state.get("feature_names", []))
        builder.split_sizes = dict(state.get("split_sizes", {}))
        builder.scaler = state.get("scaler")
        return builder

    def save(
        self,
        data: Data,
        path: str = DEFAULT_GRAPH_CACHE,
        *,
        dataset_hash: Optional[str] = None,
    ) -> None:
        """``torch.save`` the ``Data`` blob + a sha256 checksum sidecar + a
        ``.meta.json`` build manifest (config, dataset_hash, node/edge counts).
        The fitted scaler is NOT written here — it travels in the trainer
        artifact, same as the TFT scaler."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, p)
        write_checksums(p.with_suffix(".checksums.json"), {"graph": p})
        meta = {
            "schema_version": "1.0",
            "config": asdict(self.config),
            "dataset_hash": dataset_hash,
            "num_nodes": int(data.num_nodes),
            "num_directed_edges": int(data.edge_index.shape[1]),
            "split_sizes": dict(self.split_sizes),
            "edges_usable": {
                "train": int(data.edge_mask_train.sum()),
                "val": int(data.edge_mask_val.sum()),
                "test": int(data.edge_mask_test.sum()),
            },
            "feature_count": len(self.feature_names),
        }
        p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
        logger.info("Graph cached to %s (+ checksums, meta)", p)

    @classmethod
    def load(cls, path: str = DEFAULT_GRAPH_CACHE) -> Data:
        """Verify the checksum sidecar BEFORE ``torch.load`` (fail-closed —
        never fall back). ``weights_only=False`` is required because a
        :class:`Data` object is not a bare ``state_dict``; this is acceptable
        ONLY because the checksum gate runs first and the file is produced by
        this class's own ``save()``."""
        p = Path(path)
        verify_checksums(p.with_suffix(".checksums.json"), {"graph": p})
        data = torch.load(p, map_location="cpu", weights_only=False)
        logger.info("Graph loaded from %s", p)
        return data

    @staticmethod
    def is_cache_valid(path: str, dataset_hash: str) -> bool:
        """True iff the cache exists, its checksum verifies, and its
        ``meta.json`` ``dataset_hash`` matches ``dataset_hash``."""
        p = Path(path)
        meta_p = p.with_suffix(".meta.json")
        chk_p = p.with_suffix(".checksums.json")
        if not (p.exists() and meta_p.exists() and chk_p.exists()):
            return False
        try:
            verify_checksums(chk_p, {"graph": p})
        except Exception as exc:  # noqa: BLE001 — any failure invalidates the cache
            logger.warning("Graph cache checksum check failed: %s", exc)
            return False
        try:
            meta = json.loads(meta_p.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return meta.get("dataset_hash") == dataset_hash


# ── module-level helpers ────────────────────────────────────────────────────
def _dedupe_undirected_pairs(pairs: np.ndarray) -> np.ndarray:
    """``pairs`` is ``[2, E]`` with each column ``(lo, hi)``, ``lo < hi``.
    Returns the unique columns."""
    if pairs.shape[1] == 0:
        return pairs.astype(np.int64)
    lo = np.minimum(pairs[0], pairs[1]).astype(np.int64)
    hi = np.maximum(pairs[0], pairs[1]).astype(np.int64)
    packed = (lo << np.int64(32)) | hi
    uniq = np.unique(packed)
    out_lo = (uniq >> np.int64(32)).astype(np.int64)
    out_hi = (uniq & np.int64(0xFFFFFFFF)).astype(np.int64)
    return np.stack([out_lo, out_hi], axis=0)


def _symmetrize(pairs: np.ndarray) -> np.ndarray:
    """``[2, E]`` undirected pairs -> ``[2, 2E]`` directed edge_index."""
    if pairs.shape[1] == 0:
        return pairs.astype(np.int64)
    src = np.concatenate([pairs[0], pairs[1]])
    dst = np.concatenate([pairs[1], pairs[0]])
    return np.stack([src, dst], axis=0).astype(np.int64)


def _pairs_as_set(pairs: np.ndarray) -> set:
    """Pack ``[2, E]`` (lo, hi) pairs into a set of int64 for O(1) membership."""
    if pairs.shape[1] == 0:
        return set()
    lo = np.minimum(pairs[0], pairs[1]).astype(np.int64)
    hi = np.maximum(pairs[0], pairs[1]).astype(np.int64)
    return set(((lo << np.int64(32)) | hi).tolist())
