"""
State encoding utilities for AlphaZero-style Connect Four.

Encodes a game state into tensor planes suitable for a neural network.

Output shape:
    (3, ROWS, COLS)

Planes:
    0 -> current player's pieces
    1 -> opponent player's pieces
    2 -> current-player indicator plane
         filled with 1.0 if current player is PLAYER_ONE
         filled with 0.0 if current player is PLAYER_TWO
"""

from __future__ import annotations

import numpy as np
import torch

from src.core.constants import COLS, PLAYER_ONE, PLAYER_TWO, ROWS
from src.core.game import ConnectFourGame


def encode_state(game: ConnectFourGame) -> np.ndarray:
    """
    Encode the current game state into a NumPy tensor.

    Args:
        game: Current Connect Four game.

    Returns:
        np.ndarray of shape (3, ROWS, COLS), dtype float32.
    """
    board = game.board.grid
    current_player = game.current_player
    opponent = PLAYER_TWO if current_player == PLAYER_ONE else PLAYER_ONE

    current_plane = (board == current_player).astype(np.float32)
    opponent_plane = (board == opponent).astype(np.float32)

    if current_player == PLAYER_ONE:
        player_plane = np.ones((ROWS, COLS), dtype=np.float32)
    else:
        player_plane = np.zeros((ROWS, COLS), dtype=np.float32)

    encoded = np.stack([current_plane, opponent_plane, player_plane], axis=0)
    return encoded.astype(np.float32)


def encode_state_tensor(game: ConnectFourGame, device: str | torch.device | None = None) -> torch.Tensor:
    """
    Encode state directly as a PyTorch tensor.

    Returns:
        torch.Tensor of shape (1, 3, ROWS, COLS)
    """
    array = encode_state(game)
    tensor = torch.from_numpy(array).unsqueeze(0)  # add batch dimension
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def encode_batch(games: list[ConnectFourGame]) -> torch.Tensor:
    """
    Encode a list of games into a batch tensor.

    Returns:
        torch.Tensor of shape (B, 3, ROWS, COLS)
    """
    if not games:
        raise ValueError("games list cannot be empty.")

    batch = np.stack([encode_state(game) for game in games], axis=0)
    return torch.from_numpy(batch).float()
