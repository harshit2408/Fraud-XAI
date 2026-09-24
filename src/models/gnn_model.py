"""
src/models/gnn_model.py

GraphSAGE fraud classifier (PRD Phase 12, arXiv:2604.14231 Table II config).

Two ``SAGEConv`` layers (128 -> 64 hidden by default) over the transaction
graph from ``src/data/graph_builder.py``, followed by a 3-layer MLP head ->
1 logit. Structured to match ``src/models/tft_model.py``: a plain
``nn.Module`` whose ``__init__`` takes explicit dims, a ``forward`` returning
``{'logits', 'probabilities'}``, and a ``count_parameters`` helper.

HARD ARCHITECTURAL CONSTRAINT (mle-reviewer R3, 2026-09-09) — NO NORMALIZATION
LAYERS. This model must not contain ``BatchNorm``/``LayerNorm``/``GraphNorm``,
nor pass ``norm=`` to ``SAGEConv``. Rationale: ``GNNTrainer`` runs a full-graph
eval-mode forward over ``edge_mask_val`` / ``edge_mask_test`` slices that span
val/test nodes; a normalization layer with running statistics would update
those stats from val/test activations, and the buffers are shared parameters
reloaded on the next ``.train()`` call — a silent val/test -> train leak that
the temporal edge masking cannot catch. ``SAGEConv -> ReLU -> Dropout`` carry
no cross-sample state. Any future addition of a norm layer REQUIRES re-review
of the Phase 12 leakage argument (see ``docs/adr/ADR-005``).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)

# Enforced by GraphSAGEModel.__init__ (R3). Kept as a module constant so a
# test can import and assert against it.
FORBIDDEN_MODULE_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
)


class GraphSAGEModel(nn.Module):
    """GraphSAGE node classifier: ``len(hidden_dims)`` × ``SAGEConv`` + a
    3-layer MLP head producing one fraud logit per node.

    Args:
        num_node_features: input feature dim per node (184 for this dataset).
        hidden_dims: ``SAGEConv`` output sizes, in order. Paper: ``[128, 64]``.
        mlp_hidden_dims: hidden sizes of the post-conv MLP head. Paper implies
            a 3-layer head, e.g. ``[64, 32]`` then a final ``Linear(32, 1)``.
        dropout: applied after each ``SAGEConv`` activation and between MLP
            layers.
        aggr: ``SAGEConv`` aggregation (``"mean"`` == GraphSAGE-mean / paper).
        input_dropout: feature-level dropout applied to ``x`` BEFORE the
            first ``SAGEConv`` (ADR-006 §3.2 Arm B). ``0.0`` (default)
            reproduces pre-Arm-B behaviour exactly — dropout at rate 0 is a
            no-op.
        l2_normalize: if ``True``, apply ``F.normalize(h, p=2, dim=-1)`` after
            each conv's activation+dropout (ADR-006 §3.2 / D5). This is a
            pure per-node, buffer-free function — it holds no running
            statistics and therefore passes ``_assert_no_norm_layers`` (R3):
            it is scale control, not a normalization LAYER in the R3 sense
            (BatchNorm/LayerNorm/GraphNorm, which accumulate cross-sample
            state). Never replace this with an ``nn.Module``-based norm.
        residual: if ``True``, add an additive skip connection around each
            conv layer when the input/output dims match, else project the
            input through an ``nn.Linear`` shortcut first (ADR-006 §3.2,
            targets oversmoothing at ``num_layers=3``). Stateless, R3-compliant
            (plain ``nn.Linear`` carries no cross-sample statistics).
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dims: Optional[List[int]] = None,
        mlp_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        aggr: str = "mean",
        input_dropout: float = 0.0,
        l2_normalize: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims) if hidden_dims else [128, 64]
        mlp_hidden_dims = list(mlp_hidden_dims) if mlp_hidden_dims else [64, 32]

        self.num_node_features = int(num_node_features)
        self.hidden_dims = hidden_dims
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = float(dropout)
        self.aggr = str(aggr)
        self.input_dropout = float(input_dropout)
        self.l2_normalize = bool(l2_normalize)
        self.residual = bool(residual)

        # ── SAGEConv stack (+ optional residual shortcuts) ─────────────────
        self.convs = nn.ModuleList()
        self.shortcuts = nn.ModuleList()  # Identity or Linear per conv layer
        in_dim = self.num_node_features
        for h in hidden_dims:  # [128, 64]
            self.convs.append(SAGEConv(in_dim, h, aggr=aggr))
            if self.residual:
                self.shortcuts.append(
                    nn.Identity() if in_dim == h else nn.Linear(in_dim, h)
                )
            in_dim = h

        # ── 3-layer MLP head -> 1 logit ──────────────────────────────────
        mlp_layers: List[nn.Module] = []
        prev = hidden_dims[-1]  # 64
        for h in mlp_hidden_dims:  # [64, 32]
            mlp_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        mlp_layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*mlp_layers)

        self._assert_no_norm_layers()  # R3
        self._init_weights()

    # ── R3 enforcement ──────────────────────────────────────────────────────
    def _assert_no_norm_layers(self) -> None:
        offenders = [
            type(m).__name__
            for m in self.modules()
            if isinstance(m, FORBIDDEN_MODULE_TYPES)
        ]
        if offenders:
            raise AssertionError(
                "GraphSAGEModel contains normalization layers "
                f"{offenders} — forbidden by Phase 12 leakage contract (R3). "
                "See src/models/gnn_model.py module docstring."
            )

    def _init_weights(self) -> None:
        """Xavier on 2-D weights, zeros on biases — same convention as
        ``tft_model.py:_init_weights``."""
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:          ``[n_nodes, num_node_features]``
            edge_index: ``[2, n_edges]`` — a sampled subgraph from
                ``NeighborLoader`` (train) or a full masked graph (eval).

        Returns:
            ``{'logits': [n_nodes, 1], 'probabilities': [n_nodes, 1]}``.

        The module returns outputs for every node it was given; the trainer
        slices ``out[:batch_size]`` for the seed nodes (``NeighborLoader``
        places seed nodes first) — mirroring how ``tft_model.forward`` returns
        all timesteps and the trainer picks the last.
        """
        h = F.dropout(x, p=self.input_dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual:
                h = h + self.shortcuts[i](h_in)
            if self.l2_normalize:
                h = F.normalize(h, p=2, dim=-1)
        logits = self.head(h)
        return {"logits": logits, "probabilities": torch.sigmoid(logits)}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
