"""
Move encoding utilities for Connect Four.

Moves are represented by columns:
    0, 1, 2, 3, 4, 5, 6
"""

from __future__ import annotations

import numpy as np
import torch

from src.core.constants import COLS


def encode_move(move: int) -> int:
    """
    Validate and return a move index.
    """
    if not 0 <= move < COLS:
        raise ValueError(f"Move must be between 0 and {COLS - 1}, got {move}.")
    return move


def decode_move(index: int) -> int:
    """
    Decode a policy index back into a move.
    """
    if not 0 <= index < COLS:
        raise ValueError(f"Index must be between 0 and {COLS - 1}, got {index}.")
    return index


def one_hot_move(move: int) -> np.ndarray:
    """
    One-hot encode a move into shape (COLS,).
    """
    move = encode_move(move)
    vec = np.zeros(COLS, dtype=np.float32)
    vec[move] = 1.0
    return vec


def legal_moves_mask(valid_moves: list[int]) -> np.ndarray:
    """
    Create a binary mask over legal moves.

    Args:
        valid_moves: list of playable columns

    Returns:
        np.ndarray of shape (COLS,), dtype float32
    """
    mask = np.zeros(COLS, dtype=np.float32)
    for move in valid_moves:
        mask[encode_move(move)] = 1.0
    return mask


def legal_moves_mask_tensor(valid_moves: list[int]) -> torch.Tensor:
    """
    PyTorch tensor version of legal_moves_mask.
    """
    return torch.from_numpy(legal_moves_mask(valid_moves)).float()


def normalize_policy(policy: np.ndarray, valid_moves: list[int]) -> np.ndarray:
    """
    Normalize a raw policy over valid moves only.

    Args:
        policy: raw probabilities or scores of shape (COLS,)
        valid_moves: legal move indices

    Returns:
        normalized policy of shape (COLS,)
    """
    if policy.shape != (COLS,):
        raise ValueError(f"policy must have shape ({COLS},), got {policy.shape}.")

    mask = legal_moves_mask(valid_moves)
    masked = policy * mask
    total = masked.sum()

    if total <= 0:
        if not valid_moves:
            raise ValueError("Cannot normalize policy with no valid moves.")
        masked = mask / mask.sum()
    else:
        masked = masked / total

    return masked.astype(np.float32)
