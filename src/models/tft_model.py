"""
src/models/tft_model.py

Custom Temporal Fusion Transformer (TFT) for binary fraud classification.

Implements the key components from the Google Research TFT paper
(Lim et al., 2021), adapted for binary classification on transaction
sequences rather than time-series forecasting.

Architecture:
    1. Input Embedding: Projects numeric features to hidden_size
    2. Static Covariate Encoder: Processes card-level static features
    3. Variable Selection Networks (VSN): Learn which features matter per timestep
    4. LSTM Encoder: Captures sequential dependencies
    5. Multi-Head Attention: Self-attention over temporal context
    6. Gated Residual Networks (GRN): Non-linear processing with skip connections
    7. Output Layer: Sigmoid for binary fraud probability

Reference:
    Lim et al. "Temporal Fusion Transformers for Interpretable Multi-horizon
    Time Series Forecasting" (2021)
    https://arxiv.org/abs/1912.09363
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) — controls information flow via a learned gate.

    GLU(x) = sigmoid(W_gate * x + b_gate) ⊙ (W_main * x + b_main)

    Provides a learnable skip mechanism where the gate can shut off
    contributions from less useful inputs.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc_main = nn.Linear(input_dim, output_dim)
        self.fc_gate = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc_gate(x)) * self.fc_main(x)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) — core building block of TFT.

    Applies non-linear processing with a gating mechanism and residual connection:
        GRN(a, c) = LayerNorm(a + GLU(η₁))
        η₁ = W₁ * η₂ + b₁
        η₂ = ELU(W₂ * a + W₃ * c + b₂)

    where 'a' is the primary input and 'c' is an optional context vector.

    Args:
        input_dim: Size of the primary input.
        hidden_dim: Size of the intermediate layer.
        output_dim: Size of the output (must match input_dim for residual).
        context_dim: Size of the optional static context vector.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GatedLinearUnit(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Optional context projection
        self.context_fc = None
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)

        # Residual projection if input != output size
        self.residual_fc = None
        if input_dim != output_dim:
            self.residual_fc = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Primary path
        eta2 = self.fc1(x)

        # Add static context if available
        if self.context_fc is not None and context is not None:
            eta2 = eta2 + self.context_fc(context)

        eta2 = F.elu(eta2)
        eta1 = self.dropout(self.fc2(eta2))

        # Gate
        gated = self.glu(eta1)

        # Residual
        residual = x if self.residual_fc is None else self.residual_fc(x)

        return self.layer_norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) — learns which input features are important.

    Produces variable-wise weights (softmax over features) that scale each
    feature's GRN output before aggregation. This provides built-in
    feature importance for interpretability.

    Args:
        input_dim: Number of input features.
        hidden_dim: Hidden size for each feature's GRN.
        num_features: Number of distinct features (each gets its own GRN).
        context_dim: Optional static context dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_features: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Flattened GRN for variable selection weights
        self.weight_grn = GatedResidualNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            context_dim=context_dim,
            dropout=dropout,
        )

        # Per-feature transformation GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=1,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_features)
        ])

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features, shape (batch, features) or (batch, seq_len, features).
            context: Optional static context, shape (batch, context_dim).

        Returns:
            Tuple of:
                - Selected output, shape (batch, hidden_dim) or (batch, seq_len, hidden_dim)
                - Variable weights, shape (batch, num_features) or (batch, seq_len, num_features)
        """
        # Compute selection weights
        flat_x = x
        if context is not None and x.dim() == 3:
            # Expand context to match sequence dimension
            ctx = context.unsqueeze(1).expand(-1, x.size(1), -1)
        else:
            ctx = context

        weights = torch.softmax(self.weight_grn(flat_x, ctx), dim=-1)

        # Transform each feature independently
        # x shape: (..., num_features)
        transformed = []
        for i in range(min(self.num_features, x.size(-1))):
            feat = x[..., i:i+1]  # (..., 1)
            transformed.append(self.feature_grns[i](feat))  # (..., hidden_dim)

        # Stack and weight
        transformed = torch.stack(transformed, dim=-2)  # (..., num_features, hidden_dim)
        weights_expanded = weights.unsqueeze(-1)  # (..., num_features, 1)

        selected = (transformed * weights_expanded).sum(dim=-2)  # (..., hidden_dim)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention adapted for interpretability.

    Uses additive attention weights (shared value projections across heads)
    so that attention weights are directly interpretable as temporal importance.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: shape (batch, seq_len, hidden_size)
            mask: shape (batch, seq_len), 1 for real tokens, 0 for padding

        Returns:
            Tuple of:
                - Output: shape (batch, seq_len, hidden_size)
                - Attention weights: shape (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape

        # Project and reshape for multi-head
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Apply padding mask
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            # Use -1e4 instead of -inf to prevent overflow in float16 (AMP)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e4)

        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))

        # Handle NaN from all-padding sequences
        attn_weights = attn_weights.nan_to_num(0.0)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        output = self.out_proj(attn_output)

        return output, attn_weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for binary fraud classification.

    Processes a sequence of N historical transactions per card and produces
    a single fraud probability for the most recent transaction.

    Args:
        num_numeric_features: Number of time-varying numeric features per timestep.
        num_static_features: Number of static (card-level) features.
        hidden_size: Model hidden dimension.
        num_attention_heads: Number of attention heads.
        dropout: Dropout rate throughout the model.
        num_lstm_layers: Number of LSTM encoder layers.
        sequence_length: Expected sequence length.
        static_categorical_indices: Indices into the static feature vector
            that are label-encoded multi-class categoricals (e.g. ProductCD,
            card4, card6). Phase B5: these must go through an embedding
            table, not a Linear layer as if they were continuous — treating
            "category 7" as numerically greater than "category 2" is
            meaningless for unordered categories. Remaining static indices
            are treated as continuous (binary flags included). Defaults to
            None/empty, which reproduces the original all-continuous behavior.
        static_cardinalities: Number of distinct categories for each index in
            `static_categorical_indices`, in the same order. Required
            (same length) when `static_categorical_indices` is non-empty.
        categorical_embedding_dim: Embedding dimension per categorical column.
    """

    def __init__(
        self,
        num_numeric_features: int,
        num_static_features: int,
        hidden_size: int = 64,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        num_lstm_layers: int = 1,
        sequence_length: int = 10,
        static_categorical_indices: Optional[List[int]] = None,
        static_cardinalities: Optional[List[int]] = None,
        categorical_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_numeric_features = num_numeric_features
        self.num_static_features = num_static_features

        static_categorical_indices = static_categorical_indices or []
        static_cardinalities = static_cardinalities or []
        if len(static_categorical_indices) != len(static_cardinalities):
            raise ValueError(
                "static_categorical_indices and static_cardinalities must have "
                f"the same length, got {len(static_categorical_indices)} and "
                f"{len(static_cardinalities)}"
            )
        self.static_categorical_indices = static_categorical_indices
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, categorical_embedding_dim)
            for cardinality in static_cardinalities
        ])

        # ── 1. Input Embedding ──────────────────────────────────────────
        # Project each numeric feature to hidden_size independently
        self.numeric_embedding = nn.Linear(num_numeric_features, hidden_size)

        # Static feature embedding — continuous slots pass through raw,
        # categorical slots are replaced by their embedding vectors first
        # (see forward()), so the Linear layer's input dim must account for
        # the embedding expansion rather than the raw static feature count.
        num_static_continuous = num_static_features - len(static_categorical_indices)
        static_input_dim = (
            num_static_continuous + len(static_cardinalities) * categorical_embedding_dim
            if static_cardinalities
            else max(num_static_features, 1)
        )
        self.static_embedding = nn.Linear(
            static_input_dim, hidden_size
        ) if num_static_features > 0 else None

        # Static context vectors for different parts of the network
        self.static_context_variable_selection = nn.Linear(hidden_size, hidden_size) if num_static_features > 0 else None
        self.static_context_enrichment = nn.Linear(hidden_size, hidden_size) if num_static_features > 0 else None
        self.static_context_state_h = nn.Linear(hidden_size, hidden_size) if num_static_features > 0 else None
        self.static_context_state_c = nn.Linear(hidden_size, hidden_size) if num_static_features > 0 else None

        # ── 2. Variable Selection ───────────────────────────────────────
        self.temporal_vsn = VariableSelectionNetwork(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            num_features=hidden_size,  # After embedding, each "feature" is a hidden dim
            context_dim=hidden_size if num_static_features > 0 else None,
            dropout=dropout,
        )

        # ── 3. LSTM Encoder ─────────────────────────────────────────────
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # ── 4. Post-LSTM Gate + Residual ────────────────────────────────
        self.post_lstm_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.post_lstm_norm = nn.LayerNorm(hidden_size)

        # ── 5. Static Enrichment ────────────────────────────────────────
        self.enrichment_grn = GatedResidualNetwork(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            context_dim=hidden_size if num_static_features > 0 else None,
            dropout=dropout,
        )

        # ── 6. Self-Attention ───────────────────────────────────────────
        self.attention = InterpretableMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.post_attn_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.post_attn_norm = nn.LayerNorm(hidden_size)

        # ── 7. Position-wise Feed-Forward ───────────────────────────────
        self.ff_grn = GatedResidualNetwork(
            input_dim=hidden_size,
            hidden_dim=hidden_size * 4,
            output_dim=hidden_size,
            dropout=dropout,
        )

        # ── 8. Output Layer ─────────────────────────────────────────────
        # Takes the last timestep's representation → fraud probability
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for stable training."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _embed_static_categoricals(self, static_features: torch.Tensor) -> torch.Tensor:
        """
        Split raw static features into continuous vs. categorical slots,
        replacing each categorical slot's integer code with its embedding
        vector before concatenation. No-op (returns input unchanged) when
        the model was built without static_categorical_indices — this keeps
        the original all-continuous behavior for existing callers.
        """
        if not self.categorical_embeddings:
            return static_features

        cat_idx = self.static_categorical_indices
        cont_idx = [i for i in range(static_features.size(-1)) if i not in cat_idx]

        parts = []
        if cont_idx:
            parts.append(static_features[..., cont_idx])
        for emb_layer, idx in zip(self.categorical_embeddings, cat_idx):
            codes = static_features[..., idx].long().clamp(min=0)
            parts.append(emb_layer(codes))

        return torch.cat(parts, dim=-1)

    def forward(
        self,
        sequences: torch.Tensor,
        static_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            sequences: Time-varying features, shape (batch, seq_len, num_features)
            static_features: Static features, shape (batch, num_static)
            mask: Padding mask, shape (batch, seq_len), 1=real, 0=padding

        Returns:
            Dict with:
                'logits': Raw logits, shape (batch, 1)
                'probabilities': Sigmoid probabilities, shape (batch, 1)
                'attention_weights': Temporal attention, shape (batch, heads, seq, seq)
                'variable_weights': Feature importance, shape (batch, seq, features)
        """
        batch_size, seq_len, _ = sequences.shape

        # 1. Input embedding
        embedded = self.numeric_embedding(sequences)  # (B, S, H)

        # 2. Static context
        static_context_vs = None
        static_context_enrich = None
        initial_h = None
        initial_c = None

        if self.static_embedding is not None and static_features.size(-1) > 0:
            static_input = self._embed_static_categoricals(static_features)
            static_embedded = self.static_embedding(static_input)  # (B, H)

            static_context_vs = self.static_context_variable_selection(static_embedded)
            static_context_enrich = self.static_context_enrichment(static_embedded)
            num_layers = self.lstm_encoder.num_layers
            initial_h = self.static_context_state_h(static_embedded).unsqueeze(0).expand(num_layers, -1, -1).contiguous()
            initial_c = self.static_context_state_c(static_embedded).unsqueeze(0).expand(num_layers, -1, -1).contiguous()

        # 3. Variable selection on embedded features
        selected, variable_weights = self.temporal_vsn(embedded, static_context_vs)

        # 4. LSTM encoder
        if initial_h is not None and initial_c is not None:
            lstm_out, _ = self.lstm_encoder(selected, (initial_h, initial_c))
        else:
            lstm_out, _ = self.lstm_encoder(selected)

        # Post-LSTM gate + residual
        gated_lstm = self.post_lstm_gate(lstm_out)
        lstm_enriched = self.post_lstm_norm(selected + gated_lstm)

        # 5. Static enrichment
        if static_context_enrich is not None:
            ctx = static_context_enrich.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            ctx = None
        enriched = self.enrichment_grn(lstm_enriched, ctx)

        # 6. Self-attention
        attn_out, attention_weights = self.attention(
            enriched, enriched, enriched, mask=mask
        )
        gated_attn = self.post_attn_gate(attn_out)
        attn_enriched = self.post_attn_norm(enriched + gated_attn)

        # 7. Position-wise feed-forward
        ff_out = self.ff_grn(attn_enriched)

        # 8. Output: take the last real timestep for each sample
        # Find index of last non-padded timestep
        # mask shape: (B, S) — sum gives length, -1 gives last index
        lengths = mask.sum(dim=1).long().clamp(min=1) - 1  # (B,)
        batch_indices = torch.arange(batch_size, device=ff_out.device)
        last_hidden = ff_out[batch_indices, lengths]  # (B, H)

        logits = self.output_layer(last_hidden)  # (B, 1)
        probabilities = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "attention_weights": attention_weights,
            "variable_weights": variable_weights,
        }

    def get_attention_weights(
        self,
        sequences: torch.Tensor,
        static_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract attention weights for interpretability.

        Returns mean attention weights across heads, shape (batch, seq_len).
        Higher weights indicate which historical transactions influenced the
        fraud decision most.
        """
        with torch.no_grad():
            output = self.forward(sequences, static_features, mask)
            # Average across heads and query positions
            attn = output["attention_weights"]  # (B, heads, S, S)
            # Take attention from the last real position to all others
            batch_size = sequences.size(0)
            lengths = mask.sum(dim=1).long().clamp(min=1) - 1
            batch_idx = torch.arange(batch_size, device=attn.device)

            # Last position's attention to all positions, averaged over heads
            last_attn = attn[batch_idx, :, lengths, :].mean(dim=1)  # (B, S)
            return last_attn

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
