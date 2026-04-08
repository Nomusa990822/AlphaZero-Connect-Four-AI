"""
AlphaZero-style neural network for Connect Four.

Architecture:
- Input: (B, 3, ROWS, COLS)
- Shared convolutional trunk
- Policy head -> logits over 7 moves
- Value head -> scalar in [-1, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.constants import COLS
from src.neural.policy_head import PolicyHead
from src.neural.value_head import ValueHead


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv -> BN -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AlphaZeroNet(nn.Module):
    """
    Compact AlphaZero-style network for Connect Four.
    """

    def __init__(
        self,
        input_channels: int = 3,
        trunk_channels: int = 64,
        num_blocks: int = 3,
        head_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")

        layers = [ConvBlock(input_channels, trunk_channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBlock(trunk_channels, trunk_channels))

        self.trunk = nn.Sequential(*layers)
        self.policy_head = PolicyHead(in_channels=trunk_channels, hidden_dim=head_hidden_dim)
        self.value_head = ValueHead(in_channels=trunk_channels, hidden_dim=head_hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: tensor of shape (B, 3, 6, 7)

        Returns:
            policy_logits: (B, 7)
            value: (B, 1)
        """
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper.

        Returns:
            policy_probs: (B, 7)
            values: (B, 1)
        """
        self.eval()
        policy_logits, value = self.forward(x)
        policy_probs = torch.softmax(policy_logits, dim=1)
        return policy_probs, value

    @torch.no_grad()
    def predict_single(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Predict for a single input sample.

        Args:
            x: tensor of shape (1, 3, 6, 7)

        Returns:
            policy_probs: tensor of shape (7,)
            value: float
        """
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError("predict_single expects input shape (1, C, H, W).")

        probs, value = self.predict(x)
        return probs[0], float(value[0].item())

    @staticmethod
    def masked_policy(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply a legal-move mask to policy logits and return probabilities.

        Args:
            policy_logits: (B, 7)
            legal_mask: (B, 7), 1 for legal, 0 for illegal

        Returns:
            masked probabilities: (B, 7)
        """
        if policy_logits.shape != legal_mask.shape:
            raise ValueError(
                f"policy_logits shape {policy_logits.shape} must match legal_mask shape {legal_mask.shape}."
            )

        masked_logits = policy_logits.masked_fill(legal_mask == 0, float("-inf"))
        probs = torch.softmax(masked_logits, dim=1)

        # Handle edge case where all moves are masked
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        return probs
