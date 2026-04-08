"""
PyTorch dataset for AlphaZero-style training samples.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ConnectFourDataset(Dataset):
    """
    Dataset wrapping (state, policy, value) samples.
    """

    def __init__(self, samples: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
        if not samples:
            raise ValueError("samples cannot be empty.")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        state, policy, value = self.samples[idx]

        state_tensor = torch.tensor(state, dtype=torch.float32)
        policy_tensor = torch.tensor(policy, dtype=torch.float32)
        value_tensor = torch.tensor([value], dtype=torch.float32)

        return state_tensor, policy_tensor, value_tensor
