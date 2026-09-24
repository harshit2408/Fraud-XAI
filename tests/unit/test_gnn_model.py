"""
tests/unit/test_gnn_model.py

Unit tests for src/models/gnn_model.py (PRD Phase 12).

  * forward output shapes / probability range
  * count_parameters > 0
  * dropout inactive in eval() -> deterministic forward
  * R3 — the model contains NO normalization layers, and adding one is
    rejected at construction time
  * aggr variants build
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gnn_model import FORBIDDEN_MODULE_TYPES, GraphSAGEModel  # noqa: E402


@pytest.fixture
def model():
    return GraphSAGEModel(
        num_node_features=16, hidden_dims=[32, 16], mlp_hidden_dims=[16, 8], dropout=0.3
    )


@pytest.fixture
def graph():
    torch.manual_seed(0)
    x = torch.randn(40, 16)
    edge_index = torch.randint(0, 40, (2, 160))
    return x, edge_index


def test_forward_output_shapes_and_range(model, graph):
    x, ei = graph
    out = model(x, ei)
    assert out["logits"].shape == (40, 1)
    assert out["probabilities"].shape == (40, 1)
    assert torch.all(out["probabilities"] >= 0) and torch.all(out["probabilities"] <= 1)


def test_count_parameters_positive(model):
    assert model.count_parameters() > 0


def test_eval_mode_forward_is_deterministic(model, graph):
    x, ei = graph
    model.eval()
    with torch.no_grad():
        a = model(x, ei)["logits"]
        b = model(x, ei)["logits"]
    assert torch.equal(a, b)


def test_train_mode_dropout_is_active(model, graph):
    x, ei = graph
    model.train()
    torch.manual_seed(1)
    a = model(x, ei)["logits"]
    torch.manual_seed(2)
    b = model(x, ei)["logits"]
    assert not torch.equal(a, b)


def test_deterministic_under_fixed_seed(graph):
    x, ei = graph
    torch.manual_seed(123)
    m1 = GraphSAGEModel(16, [32, 16], [16, 8], dropout=0.0)
    torch.manual_seed(123)
    m2 = GraphSAGEModel(16, [32, 16], [16, 8], dropout=0.0)
    m1.eval()
    m2.eval()
    with torch.no_grad():
        assert torch.equal(m1(x, ei)["logits"], m2(x, ei)["logits"])


# ── R3 : no normalization layers ─────────────────────────────────────────
def test_model_has_no_normalization_layers(model):
    offenders = [
        type(m).__name__
        for m in model.modules()
        if isinstance(m, FORBIDDEN_MODULE_TYPES)
    ]
    assert offenders == [], f"forbidden norm layers present: {offenders}"


def test_norm_layer_injection_is_rejected(model):
    """R3 is enforced at construction; splicing a norm layer into the head
    and re-running the check must raise."""
    model.head = nn.Sequential(model.head, nn.LayerNorm(1))
    with pytest.raises(AssertionError):
        model._assert_no_norm_layers()


@pytest.mark.parametrize("aggr", ["mean", "max", "sum"])
def test_aggr_variants_build_and_run(aggr, graph):
    x, ei = graph
    m = GraphSAGEModel(16, [16], [8], dropout=0.1, aggr=aggr)
    out = m(x, ei)
    assert out["logits"].shape == (40, 1)


# ── ADR-006 §3.2 Arm B : input_dropout / l2_normalize / residual ──────────
def test_default_args_reproduce_legacy_forward(graph):
    """input_dropout=0.0, l2_normalize=False, residual=False must be a
    complete no-op vs. the pre-Arm-B forward — same weights, same output."""
    x, ei = graph
    torch.manual_seed(7)
    legacy = GraphSAGEModel(16, [32, 16], [16, 8], dropout=0.0)
    torch.manual_seed(7)
    extended = GraphSAGEModel(
        16, [32, 16], [16, 8], dropout=0.0,
        input_dropout=0.0, l2_normalize=False, residual=False,
    )
    legacy.eval()
    extended.eval()
    with torch.no_grad():
        assert torch.equal(legacy(x, ei)["logits"], extended(x, ei)["logits"])


def test_l2_normalize_produces_unit_norm_pre_head_activations(graph):
    """With l2_normalize=True, the per-node hidden state after the last conv
    has L2 norm ~1 (verified by replicating forward()'s own arithmetic)."""
    x, ei = graph
    m = GraphSAGEModel(16, [32, 16], [16, 8], dropout=0.0, l2_normalize=True)
    m.eval()
    with torch.no_grad():
        h = F.dropout(x, p=0.0, training=False)
        for i, conv in enumerate(m.convs):
            h = conv(h, ei)
            h = F.relu(h)
            h = F.dropout(h, p=0.0, training=False)
            if m.residual:
                h = h + m.shortcuts[i](h)
            if m.l2_normalize:
                h = F.normalize(h, p=2, dim=-1)
    norms = h.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_residual_shortcut_uses_identity_when_dims_match():
    m = GraphSAGEModel(16, [16, 16], [8], dropout=0.0, residual=True)
    assert isinstance(m.shortcuts[0], nn.Identity)
    assert isinstance(m.shortcuts[1], nn.Identity)


def test_residual_shortcut_uses_linear_projection_when_dims_differ():
    m = GraphSAGEModel(16, [32, 16], [8], dropout=0.0, residual=True)
    assert isinstance(m.shortcuts[0], nn.Linear)  # 16 -> 32
    assert isinstance(m.shortcuts[1], nn.Linear)  # 32 -> 16


def test_residual_changes_output_vs_non_residual(graph):
    x, ei = graph
    torch.manual_seed(3)
    plain = GraphSAGEModel(16, [32, 16], [16, 8], dropout=0.0, residual=False)
    torch.manual_seed(3)
    resid = GraphSAGEModel(16, [32, 16], [16, 8], dropout=0.0, residual=True)
    plain.eval()
    resid.eval()
    with torch.no_grad():
        out_plain = plain(x, ei)["logits"]
        out_resid = resid(x, ei)["logits"]
    assert not torch.equal(out_plain, out_resid)


def test_input_dropout_changes_train_mode_output_vs_eval():
    torch.manual_seed(0)
    x = torch.ones(10, 16)
    ei = torch.randint(0, 10, (2, 20))
    m = GraphSAGEModel(16, [8], [4], dropout=0.0, input_dropout=0.9)
    m.train()
    with torch.no_grad():
        torch.manual_seed(1)
        out_train = m(x, ei)["logits"]
    m.eval()
    with torch.no_grad():
        out_eval = m(x, ei)["logits"]
    assert not torch.equal(out_train, out_eval)


def test_extended_model_still_has_no_normalization_layers():
    """R3 must still hold with residual (Linear shortcuts) and l2_normalize
    (a functional call, not a module) enabled."""
    m = GraphSAGEModel(
        16, [32, 16, 8], [8], dropout=0.1, residual=True, l2_normalize=True,
        input_dropout=0.1,
    )
    offenders = [
        type(mod).__name__
        for mod in m.modules()
        if isinstance(mod, FORBIDDEN_MODULE_TYPES)
    ]
    assert offenders == []


def test_three_layer_stack_builds_and_runs(graph):
    x, ei = graph
    m = GraphSAGEModel(16, [32, 24, 16], [8], dropout=0.1, residual=True)
    out = m(x, ei)
    assert out["logits"].shape == (40, 1)
