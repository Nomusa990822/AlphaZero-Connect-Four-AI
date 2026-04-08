"""
Value head for AlphaZero-style Connect Four.

Outputs a scalar in [-1, 1] estimating the position value.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.constants import ROWS, COLS


class ValueHead(nn.Module):
    """
    Maps shared features to a scalar value prediction.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(ROWS * COLS, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shared feature tensor of shape (B, C, ROWS, COLS)

        Returns:
            values of shape (B, 1)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        value = self.fc(x)
        return value
