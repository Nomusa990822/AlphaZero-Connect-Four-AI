"""
Replay buffer for AlphaZero-style Connect Four.

Stores training samples of the form:
    (state, policy, value)

Where:
- state: np.ndarray of shape (3, ROWS, COLS)
- policy: np.ndarray of shape (COLS,)
- value: float in [-1, 1]
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable
import random

import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity replay buffer for self-play samples.
    """

    def __init__(self, capacity: int = 10000, seed: int | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")

        self.capacity = capacity
        self.buffer: Deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float) -> None:
        """
        Add a single training sample.
        """
        state = np.asarray(state, dtype=np.float32)
        policy = np.asarray(policy, dtype=np.float32)
        value = float(value)

        self.buffer.append((state, policy, value))

    def extend(self, samples: Iterable[tuple[np.ndarray, np.ndarray, float]]) -> None:
        """
        Add multiple samples.
        """
        for state, policy, value in samples:
            self.add(state, policy, value)

    def sample(self, batch_size: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Randomly sample a batch without replacement.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if batch_size > len(self.buffer):
            raise ValueError("batch_size cannot exceed current buffer size.")

        return self._rng.sample(list(self.buffer), batch_size)

    def clear(self) -> None:
        """
        Remove all stored samples.
        """
        self.buffer.clear()

    def as_list(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Return all samples as a list.
        """
        return list(self.buffer)
