"""
src/training/losses.py

Custom loss functions for neural network training on imbalanced datasets.

Focal Loss reduces the relative loss for well-classified examples (p_t > 0.5),
focusing training on hard, misclassified examples — critical for fraud detection
where the vast majority of examples are easy negatives.

Reference:
    Lin et al. "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance in binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default 2.0). Higher values increase focus
               on hard examples. gamma=0 recovers standard BCE.
        alpha: Class balance weight for the positive (fraud) class (default 0.25).
               The negative class gets weight (1 - alpha).
        reduction: Reduction mode ('mean', 'sum', or 'none').
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Raw logits (before sigmoid), shape (N,) or (N, 1).
            targets: Binary targets {0, 1}, same shape as inputs.

        Returns:
            Scalar loss if reduction='mean' or 'sum', otherwise per-sample losses.
        """
        # Flatten to 1D
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Numerically stable sigmoid + BCE
        # Using F.binary_cross_entropy_with_logits avoids log(sigmoid(x)) instability
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # Probability of correct class
        p = torch.sigmoid(inputs)
        p_t = targets * p + (1 - targets) * (1 - p)

        # Alpha weighting: alpha for positive class, (1-alpha) for negative
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal modulation: down-weight easy examples
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy with class-specific weights.

    Simpler alternative to Focal Loss when the focus should be purely
    on re-weighting classes rather than modulating by prediction difficulty.

    Args:
        pos_weight: Weight for positive (fraud) class. Set to
                    neg_count / pos_count for balanced training.
    """

    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            inputs: Raw logits (before sigmoid), shape (N,) or (N, 1).
            targets: Binary targets {0, 1}, same shape as inputs.

        Returns:
            Scalar loss (mean reduction).
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        return F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=torch.tensor(
                self.pos_weight, device=inputs.device, dtype=inputs.dtype
            ),
        )
