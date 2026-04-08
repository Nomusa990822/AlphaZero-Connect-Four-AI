import pytest

from src.core.constants import DRAW, PLAYER_ONE, PLAYER_TWO
from src.core.game import ConnectFourGame


def test_new_game_starts_correctly():
    game = ConnectFourGame()

    assert game.current_player == PLAYER_ONE
    assert game.winner is None
    assert game.done is False
    assert game.move_count == 0
    assert game.last_move is None
    assert game.get_valid_moves() == [0, 1, 2, 3, 4, 5, 6]


def test_apply_move_updates_board_and_switches_player():
    game = ConnectFourGame()

    move_info = game.apply_move(3)

    assert move_info["col"] == 3
    assert move_info["player"] == PLAYER_ONE
    assert move_info["row"] == 5
    assert move_info["done"] is False
    assert game.board[5, 3] == PLAYER_ONE
    assert game.current_player == PLAYER_TWO
    assert game.move_count == 1
    assert game.last_move == {"row": 5, "col": 3, "player": PLAYER_ONE}


def test_invalid_move_out_of_bounds_raises_error():
    game = ConnectFourGame()

    with pytest.raises(ValueError):
        game.apply_move(-1)

    with pytest.raises(ValueError):
        game.apply_move(7)


def test_cannot_play_in_full_column():
    game = ConnectFourGame()

    for _ in range(6):
        game.apply_move(0)

    assert game.board.is_column_full(0) is True

    with pytest.raises(ValueError):
        game.apply_move(0)


def test_horizontal_win_sets_game_done_and_winner():
    game = ConnectFourGame()

    # Sequence creates a horizontal win for PLAYER_ONE on bottom row
    moves = [0, 0, 1, 1, 2, 2, 3]
    for move in moves:
        info = game.apply_move(move)

    assert info["done"] is True
    assert info["winner"] == PLAYER_ONE
    assert info["is_draw"] is False
    assert game.done is True
    assert game.winner == PLAYER_ONE


def test_vertical_win_sets_game_done_and_winner():
    game = ConnectFourGame()

    # PLAYER_ONE wins vertically in column 0
    moves = [0, 1, 0, 1, 0, 1, 0]
    for move in moves:
        info = game.apply_move(move)

    assert info["done"] is True
    assert info["winner"] == PLAYER_ONE
    assert game.done is True
    assert game.winner == PLAYER_ONE


def test_cannot_apply_move_after_game_is_over():
    game = ConnectFourGame()

    moves = [0, 1, 0, 1, 0, 1, 0]
    for move in moves:
        game.apply_move(move)

    assert game.done is True

    with pytest.raises(ValueError):
        game.apply_move(2)


def test_reset_restores_initial_state():
    game = ConnectFourGame()
    game.apply_move(3)
    game.apply_move(4)

    game.reset()

    assert game.current_player == PLAYER_ONE
    assert game.winner is None
    assert game.done is False
    assert game.move_count == 0
    assert game.last_move is None
    assert game.get_valid_moves() == [0, 1, 2, 3, 4, 5, 6]
    assert game.board.grid.sum() == 0


def test_copy_creates_independent_game_instance():
    game = ConnectFourGame()
    game.apply_move(2)

    copied = game.copy()
    copied.apply_move(3)

    assert game.move_count == 1
    assert copied.move_count == 2
    assert game.board[5, 3] == 0
    assert copied.board[5, 3] != 0


def test_get_state_returns_expected_keys():
    game = ConnectFourGame()
    state = game.get_state()

    expected_keys = {
        "board",
        "current_player",
        "winner",
        "done",
        "move_count",
        "last_move",
        "valid_moves",
    }

    assert set(state.keys()) == expected_keys


def test_draw_state_in_game_object():
    game = ConnectFourGame()

    # Manually construct a full board with no winner
    pattern = [
        [1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1],
        [1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1],
        [1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1],
    ]
    game.board.grid[:, :] = pattern
    game._refresh_game_state()

    assert game.done is True
    assert game.winner == DRAW