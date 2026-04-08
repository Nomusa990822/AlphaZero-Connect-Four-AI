"""
Loss functions for AlphaZero-style training.

Total loss:
    policy loss + value loss + weight decay handled by optimizer
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def policy_loss_fn(policy_logits: torch.Tensor, target_policy: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy style loss for a target distribution.

    Args:
        policy_logits: (B, 7)
        target_policy: (B, 7), probability targets

    Returns:
        scalar loss
    """
    if policy_logits.shape != target_policy.shape:
        raise ValueError(
            f"policy_logits shape {policy_logits.shape} must match target_policy shape {target_policy.shape}."
        )

    log_probs = F.log_softmax(policy_logits, dim=1)
    loss = -(target_policy * log_probs).sum(dim=1).mean()
    return loss


def value_loss_fn(pred_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error loss for scalar value prediction.

    Args:
        pred_value: (B, 1) or (B,)
        target_value: (B, 1) or (B,)
    """
    pred_value = pred_value.view(-1)
    target_value = target_value.view(-1)
    return F.mse_loss(pred_value, target_value)


def alphazero_loss(
    policy_logits: torch.Tensor,
    pred_value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined AlphaZero loss.

    Returns:
        total_loss, policy_loss, value_loss
    """
    p_loss = policy_loss_fn(policy_logits, target_policy)
    v_loss = value_loss_fn(pred_value, target_value)
    total = policy_weight * p_loss + value_weight * v_loss
    return total, p_loss, v_loss
