"""
Policy head for AlphaZero-style Connect Four.

Outputs unnormalized logits for each column.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.constants import COLS, ROWS


class PolicyHead(nn.Module):
    """
    Maps shared features to policy logits over the 7 columns.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(2 * ROWS * COLS, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, COLS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shared feature tensor of shape (B, C, ROWS, COLS)

        Returns:
            logits of shape (B, COLS)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        logits = self.fc(x)
        return logits
