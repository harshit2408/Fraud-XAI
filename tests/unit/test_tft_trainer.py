"""
tests/unit/test_tft_trainer.py

Unit tests for the TFT model and trainer.
Tests model instantiation, forward pass shapes, save/load, and prediction output.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.tft_model import (
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    TemporalFusionTransformer,
    VariableSelectionNetwork,
)


# ── Component Tests ──────────────────────────────────────────────────────────


class TestGatedLinearUnit:
    def test_output_shape(self):
        glu = GatedLinearUnit(input_dim=32, output_dim=16)
        x = torch.randn(4, 32)
        out = glu(x)
        assert out.shape == (4, 16)

    def test_output_range(self):
        """GLU output should be bounded (sigmoid * linear)."""
        glu = GatedLinearUnit(input_dim=16, output_dim=8)
        x = torch.randn(10, 16)
        out = glu(x)
        # Output should be finite
        assert torch.isfinite(out).all()


class TestGatedResidualNetwork:
    def test_output_shape_matching_dims(self):
        grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=32)
        x = torch.randn(4, 32)
        out = grn(x)
        assert out.shape == (4, 32)

    def test_output_shape_different_dims(self):
        grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=16)
        x = torch.randn(4, 32)
        out = grn(x)
        assert out.shape == (4, 16)

    def test_with_context(self):
        grn = GatedResidualNetwork(
            input_dim=32, hidden_dim=64, output_dim=32, context_dim=16
        )
        x = torch.randn(4, 32)
        ctx = torch.randn(4, 16)
        out = grn(x, context=ctx)
        assert out.shape == (4, 32)

    def test_3d_input(self):
        grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=32)
        x = torch.randn(4, 10, 32)  # (batch, seq, features)
        out = grn(x)
        assert out.shape == (4, 10, 32)


class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = InterpretableMultiHeadAttention(hidden_size=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out, weights = attn(x, x, x)
        assert out.shape == (2, 10, 64)
        assert weights.shape == (2, 4, 10, 10)

    def test_with_mask(self):
        attn = InterpretableMultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 8, 32)
        mask = torch.ones(2, 8)
        mask[0, :3] = 0  # Pad first 3 positions for batch 0

        out, weights = attn(x, x, x, mask=mask)
        assert out.shape == (2, 8, 32)
        # Attention to padded positions should be ~0
        assert weights[0, :, :, :3].max() < 0.01

    def test_attention_sums_to_one(self):
        attn = InterpretableMultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 8, 32)
        _, weights = attn(x, x, x)
        # Each row of attention should sum to ~1
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# ── Full TFT Model Tests ─────────────────────────────────────────────────────


class TestTemporalFusionTransformer:
    @pytest.fixture
    def model_config(self):
        return {
            "num_numeric_features": 20,
            "num_static_features": 3,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "dropout": 0.1,
            "sequence_length": 5,
        }

    @pytest.fixture
    def model(self, model_config):
        return TemporalFusionTransformer(**model_config)

    def test_forward_pass_shape(self, model, model_config):
        """Test that forward pass produces correct output shapes."""
        batch_size = 4
        seq_len = model_config["sequence_length"]

        sequences = torch.randn(batch_size, seq_len, model_config["num_numeric_features"])
        static = torch.randn(batch_size, model_config["num_static_features"])
        mask = torch.ones(batch_size, seq_len)

        output = model(sequences, static, mask)

        assert "logits" in output
        assert "probabilities" in output
        assert "attention_weights" in output

        assert output["logits"].shape == (batch_size, 1)
        assert output["probabilities"].shape == (batch_size, 1)

    def test_probabilities_in_range(self, model, model_config):
        """Test that probabilities are in [0, 1]."""
        batch_size = 8
        seq_len = model_config["sequence_length"]

        sequences = torch.randn(batch_size, seq_len, model_config["num_numeric_features"])
        static = torch.randn(batch_size, model_config["num_static_features"])
        mask = torch.ones(batch_size, seq_len)

        output = model(sequences, static, mask)
        probs = output["probabilities"]

        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_with_padding(self, model, model_config):
        """Test that model handles padded sequences correctly."""
        batch_size = 4
        seq_len = model_config["sequence_length"]

        sequences = torch.randn(batch_size, seq_len, model_config["num_numeric_features"])
        static = torch.randn(batch_size, model_config["num_static_features"])
        mask = torch.ones(batch_size, seq_len)
        # Pad first 3 positions for batch 0
        mask[0, :3] = 0

        output = model(sequences, static, mask)
        assert output["logits"].shape == (batch_size, 1)
        assert torch.isfinite(output["logits"]).all()

    def test_no_static_features(self):
        """Test model with zero static features."""
        model = TemporalFusionTransformer(
            num_numeric_features=15,
            num_static_features=0,
            hidden_size=32,
            num_attention_heads=4,
        )
        sequences = torch.randn(4, 5, 15)
        static = torch.zeros(4, 0)
        mask = torch.ones(4, 5)

        output = model(sequences, static, mask)
        assert output["logits"].shape == (4, 1)

    def test_gradient_flow(self, model, model_config):
        """Test that gradients flow through the model."""
        sequences = torch.randn(4, 5, model_config["num_numeric_features"], requires_grad=True)
        static = torch.randn(4, model_config["num_static_features"])
        mask = torch.ones(4, 5)

        output = model(sequences, static, mask)
        loss = output["logits"].sum()
        loss.backward()

        assert sequences.grad is not None
        assert sequences.grad.shape == sequences.shape

    def test_parameter_count(self, model):
        """Test that model has a reasonable number of parameters."""
        n_params = model.count_parameters()
        assert n_params > 0
        # With hidden_size=32, should be in the thousands range
        assert n_params < 10_000_000  # Not too large

    def test_save_and_load(self, model, model_config):
        """Test model save/load roundtrip."""
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            path = f.name

        # Save
        torch.save(model.state_dict(), path)

        # Load
        model2 = TemporalFusionTransformer(**model_config)
        model2.load_state_dict(torch.load(path, weights_only=True))

        # Compare outputs
        sequences = torch.randn(2, 5, model_config["num_numeric_features"])
        static = torch.randn(2, model_config["num_static_features"])
        mask = torch.ones(2, 5)

        model.eval()
        model2.eval()

        with torch.no_grad():
            out1 = model(sequences, static, mask)
            out2 = model2(sequences, static, mask)

        assert torch.allclose(out1["logits"], out2["logits"], atol=1e-6)

        # Cleanup
        Path(path).unlink(missing_ok=True)

    def test_attention_weights_extraction(self, model, model_config):
        """Test attention weight extraction for interpretability."""
        sequences = torch.randn(4, 5, model_config["num_numeric_features"])
        static = torch.randn(4, model_config["num_static_features"])
        mask = torch.ones(4, 5)

        attn_weights = model.get_attention_weights(sequences, static, mask)
        assert attn_weights.shape == (4, 5)  # (batch, seq_len)

    def test_batch_size_1(self, model, model_config):
        """Test with batch size of 1."""
        sequences = torch.randn(1, 5, model_config["num_numeric_features"])
        static = torch.randn(1, model_config["num_static_features"])
        mask = torch.ones(1, 5)

        output = model(sequences, static, mask)
        assert output["logits"].shape == (1, 1)


# ── Phase B5 TDD — categorical embeddings for static inputs ────────────────────
#
# HIGH finding (adjacent to no-scaling): static categorical features
# (ProductCD, card4, card6, DeviceType) are label-encoded integer codes.
# Feeding them into a plain nn.Linear as if continuous treats "category 7"
# as numerically greater than "category 2", which is meaningless for
# unordered categories. They must go through an embedding table instead.


class TestCategoricalEmbeddings:
    def test_model_with_categorical_static_embeddings_forward_shape(self):
        model = TemporalFusionTransformer(
            num_numeric_features=10,
            num_static_features=3,  # 1 continuous + 2 categorical slots
            hidden_size=16,
            num_attention_heads=2,
            static_categorical_indices=[1, 2],
            static_cardinalities=[5, 4],
        )
        sequences = torch.randn(4, 5, 10)
        static = torch.zeros(4, 3)
        static[:, 0] = torch.randn(4)
        static[:, 1] = torch.randint(0, 5, (4,)).float()
        static[:, 2] = torch.randint(0, 4, (4,)).float()
        mask = torch.ones(4, 5)

        output = model(sequences, static, mask)
        assert output["logits"].shape == (4, 1)
        assert torch.isfinite(output["logits"]).all()

    def test_embedding_tables_sized_by_cardinality(self):
        model = TemporalFusionTransformer(
            num_numeric_features=5,
            num_static_features=2,
            hidden_size=16,
            num_attention_heads=2,
            static_categorical_indices=[0, 1],
            static_cardinalities=[10, 6],
        )
        assert len(model.categorical_embeddings) == 2
        assert model.categorical_embeddings[0].num_embeddings == 10
        assert model.categorical_embeddings[1].num_embeddings == 6

    def test_gradient_flows_through_embeddings(self):
        model = TemporalFusionTransformer(
            num_numeric_features=5,
            num_static_features=2,
            hidden_size=16,
            num_attention_heads=2,
            static_categorical_indices=[0, 1],
            static_cardinalities=[10, 6],
        )
        sequences = torch.randn(3, 4, 5)
        static = torch.stack([
            torch.randint(0, 10, (3,)).float(),
            torch.randint(0, 6, (3,)).float(),
        ], dim=1)
        mask = torch.ones(3, 4)

        output = model(sequences, static, mask)
        output["logits"].sum().backward()

        for emb in model.categorical_embeddings:
            assert emb.weight.grad is not None
            assert torch.isfinite(emb.weight.grad).all()

    def test_backward_compatible_without_categorical_config(self):
        """Default construction (no categorical indices) is unaffected — all
        pre-existing callers/tests keep working exactly as before."""
        model = TemporalFusionTransformer(
            num_numeric_features=20,
            num_static_features=3,
            hidden_size=32,
            num_attention_heads=4,
            dropout=0.1,
            sequence_length=5,
        )
        assert len(model.categorical_embeddings) == 0


class TestFraudSequenceDataset:
    def test_dataset_creation(self):
        from src.training.train_tft import FraudSequenceDataset

        n = 20
        seq_len = 5
        feat_dim = 10
        static_dim = 3

        dataset = FraudSequenceDataset(
            sequences=np.random.randn(n, seq_len, feat_dim).astype(np.float32),
            static_features=np.random.randn(n, static_dim).astype(np.float32),
            targets=np.random.choice([0, 1], n).astype(np.float32),
            mask=np.ones((n, seq_len), dtype=np.float32),
        )

        assert len(dataset) == n
        seqs, static, mask, target = dataset[0]
        assert seqs.shape == (seq_len, feat_dim)
        assert static.shape == (static_dim,)
        assert mask.shape == (seq_len,)
        assert target.shape == ()
