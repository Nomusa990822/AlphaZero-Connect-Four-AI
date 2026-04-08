import numpy as np
import pytest

from src.core.board import Board
from src.core.constants import DRAW, PLAYER_ONE, PLAYER_TWO
from src.core.rules import (
    check_draw,
    check_winner,
    get_game_result,
    get_terminal_info,
    get_winner,
    is_terminal_state,
)


def test_check_winner_horizontal():
    board = Board()
    board.grid[5, 0:4] = PLAYER_ONE

    assert check_winner(board, PLAYER_ONE) is True
    assert check_winner(board, PLAYER_TWO) is False


def test_check_winner_vertical():
    board = Board()
    board.grid[2:6, 3] = PLAYER_TWO

    assert check_winner(board, PLAYER_TWO) is True
    assert check_winner(board, PLAYER_ONE) is False


def test_check_winner_positive_diagonal():
    board = Board()
    board.grid[0, 0] = PLAYER_ONE
    board.grid[1, 1] = PLAYER_ONE
    board.grid[2, 2] = PLAYER_ONE
    board.grid[3, 3] = PLAYER_ONE

    assert check_winner(board, PLAYER_ONE) is True


def test_check_winner_negative_diagonal():
    board = Board()
    board.grid[3, 0] = PLAYER_TWO
    board.grid[2, 1] = PLAYER_TWO
    board.grid[1, 2] = PLAYER_TWO
    board.grid[0, 3] = PLAYER_TWO

    assert check_winner(board, PLAYER_TWO) is True


def test_get_winner_returns_correct_player():
    board = Board()
    board.grid[5, 0:4] = PLAYER_ONE

    assert get_winner(board) == PLAYER_ONE


def test_get_winner_returns_none_when_no_winner():
    board = Board()
    assert get_winner(board) is None


def test_check_draw_true_for_full_board_without_winner():
    pattern = np.array([
        [1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1],
        [1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1],
        [1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1],
    ], dtype=np.int8)
    board = Board(pattern)

    assert check_draw(board) is True
    assert is_terminal_state(board) is True
    assert get_game_result(board) == DRAW


def test_check_draw_false_when_board_not_full():
    board = Board()
    board.grid[5, 0] = PLAYER_ONE

    assert check_draw(board) is False


def test_check_draw_false_when_there_is_a_winner():
    board = Board()
    board.grid[5, 0:4] = PLAYER_ONE

    assert check_draw(board) is False


def test_is_terminal_state_false_for_non_terminal_board():
    board = Board()
    board.grid[5, 0] = PLAYER_ONE

    assert is_terminal_state(board) is False
    assert get_game_result(board) is None


def test_get_terminal_info_for_winner():
    board = Board()
    board.grid[5, 0:4] = PLAYER_ONE

    is_terminal, result = get_terminal_info(board)

    assert is_terminal is True
    assert result == PLAYER_ONE


def test_get_terminal_info_for_non_terminal_board():
    board = Board()
    is_terminal, result = get_terminal_info(board)

    assert is_terminal is False
    assert result is None


def test_invalid_player_in_check_winner_raises_error():
    board = Board()

    with pytest.raises(ValueError):
        check_winner(board, 99)