"""
Game rule checks for Connect Four:
- win detection
- draw detection
- terminal-state checks
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.core.board import Board
from src.core.constants import COLS, CONNECT_N, DRAW, EMPTY, PLAYER_ONE, PLAYER_TWO, ROWS, VALID_PLAYERS


def check_winner(board: Board, player: int) -> bool:
    """
    Check whether the given player has a connect-four on the board.

    Args:
        board: Board instance.
        player: PLAYER_ONE or PLAYER_TWO.

    Returns:
        True if the player has four connected pieces in any direction.
    """
    if player not in VALID_PLAYERS:
        raise ValueError(f"Invalid player value: {player}.")

    grid = board.grid

    # Horizontal check
    for row in range(ROWS):
        for col in range(COLS - CONNECT_N + 1):
            window = grid[row, col:col + CONNECT_N]
            if np.all(window == player):
                return True

    # Vertical check
    for row in range(ROWS - CONNECT_N + 1):
        for col in range(COLS):
            window = grid[row:row + CONNECT_N, col]
            if np.all(window == player):
                return True

    # Positive diagonal (\) check
    for row in range(ROWS - CONNECT_N + 1):
        for col in range(COLS - CONNECT_N + 1):
            if all(grid[row + i, col + i] == player for i in range(CONNECT_N)):
                return True

    # Negative diagonal (/) check
    for row in range(CONNECT_N - 1, ROWS):
        for col in range(COLS - CONNECT_N + 1):
            if all(grid[row - i, col + i] == player for i in range(CONNECT_N)):
                return True

    return False


def check_draw(board: Board) -> bool:
    """
    Check whether the board is full and no player has won.

    Returns:
        True if the game is a draw.
    """
    if check_winner(board, PLAYER_ONE):
        return False
    if check_winner(board, PLAYER_TWO):
        return False
    return board.is_full()


def is_terminal_state(board: Board) -> bool:
    """
    Return True if the board is in a terminal state:
    - player one win
    - player two win
    - draw
    """
    return (
        check_winner(board, PLAYER_ONE)
        or check_winner(board, PLAYER_TWO)
        or check_draw(board)
    )


def get_winner(board: Board) -> Optional[int]:
    """
    Return the winner if one exists.

    Returns:
        PLAYER_ONE, PLAYER_TWO, or None
    """
    if check_winner(board, PLAYER_ONE):
        return PLAYER_ONE
    if check_winner(board, PLAYER_TWO):
        return PLAYER_TWO
    return None


def get_game_result(board: Board) -> Optional[int]:
    """
    Return the terminal result of the board.

    Returns:
        PLAYER_ONE if player one wins,
        PLAYER_TWO if player two wins,
        DRAW if draw,
        None if game is not terminal.
    """
    winner = get_winner(board)
    if winner is not None:
        return winner
    if check_draw(board):
        return DRAW
    return None


def get_terminal_info(board: Board) -> Tuple[bool, Optional[int]]:
    """
    Convenience helper returning:
        (is_terminal, result)

    result is:
        PLAYER_ONE / PLAYER_TWO / DRAW / None
    """
    result = get_game_result(board)
    return result is not None, result