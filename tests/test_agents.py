import pytest

from src.agents.base_agent import BaseAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.core.constants import PLAYER_ONE, PLAYER_TWO
from src.core.game import ConnectFourGame


class DummyAgent(BaseAgent):
    def select_move(self, game: ConnectFourGame) -> int:
        return game.get_valid_moves()[0]


def test_base_agent_get_valid_moves():
    game = ConnectFourGame()
    agent = DummyAgent(name="Dummy")

    assert agent.get_valid_moves(game) == [0, 1, 2, 3, 4, 5, 6]


def test_random_agent_selects_valid_move():
    game = ConnectFourGame()
    agent = RandomAgent(seed=42)

    move = agent.select_move(game)

    assert move in game.get_valid_moves()


def test_random_agent_raises_if_no_moves():
    game = ConnectFourGame()

    # Fill board manually with a no-win full pattern
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

    agent = RandomAgent(seed=42)

    with pytest.raises(ValueError):
        agent.select_move(game)


def test_heuristic_agent_selects_valid_move():
    game = ConnectFourGame()
    agent = HeuristicAgent()

    move = agent.select_move(game)

    assert move in game.get_valid_moves()


def test_heuristic_agent_takes_immediate_winning_move():
    game = ConnectFourGame()
    agent = HeuristicAgent()

    # Bottom row setup: P1 can win by playing col 3
    game.apply_move(0)  # P1
    game.apply_move(6)  # P2
    game.apply_move(1)  # P1
    game.apply_move(6)  # P2
    game.apply_move(2)  # P1
    game.apply_move(5)  # P2

    assert game.current_player == PLAYER_ONE
    move = agent.select_move(game)

    assert move == 3


def test_heuristic_agent_blocks_opponent_winning_move():
    game = ConnectFourGame()
    agent = HeuristicAgent()

    # Make it PLAYER_ONE's turn while PLAYER_TWO threatens column 3
    game.board.grid[5, 0] = PLAYER_TWO
    game.board.grid[5, 1] = PLAYER_TWO
    game.board.grid[5, 2] = PLAYER_TWO
    game.current_player = PLAYER_ONE

    move = agent.select_move(game)

    assert move == 3


def test_heuristic_agent_prefers_center_on_empty_board():
    game = ConnectFourGame()
    agent = HeuristicAgent()

    move = agent.select_move(game)

    assert move == 3


def test_minimax_agent_selects_valid_move():
    game = ConnectFourGame()
    agent = MinimaxAgent(depth=3, seed=42)

    move = agent.select_move(game)

    assert move in game.get_valid_moves()


def test_minimax_agent_finds_immediate_winning_move():
    game = ConnectFourGame()
    agent = MinimaxAgent(depth=3, seed=42)

    # P1 can win with column 3
    game.apply_move(0)  # P1
    game.apply_move(6)  # P2
    game.apply_move(1)  # P1
    game.apply_move(6)  # P2
    game.apply_move(2)  # P1
    game.apply_move(5)  # P2

    move = agent.select_move(game)

    assert move == 3


def test_minimax_agent_blocks_opponent_immediate_win():
    game = ConnectFourGame()
    agent = MinimaxAgent(depth=4, seed=42)

    # PLAYER_TWO threatens to win at col 3; PLAYER_ONE must block
    game.board.grid[5, 0] = PLAYER_TWO
    game.board.grid[5, 1] = PLAYER_TWO
    game.board.grid[5, 2] = PLAYER_TWO
    game.current_player = PLAYER_ONE

    move = agent.select_move(game)

    assert move == 3


def test_minimax_agent_prefers_center_on_empty_board():
    game = ConnectFourGame()
    agent = MinimaxAgent(depth=2, seed=42)

    move = agent.select_move(game)

    assert move == 3


def test_minimax_invalid_depth_raises_error():
    with pytest.raises(ValueError):
        MinimaxAgent(depth=0)


def test_minimax_agent_raises_if_no_moves():
    game = ConnectFourGame()

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

    agent = MinimaxAgent(depth=2, seed=42)

    with pytest.raises(ValueError):
        agent.select_move(game)
