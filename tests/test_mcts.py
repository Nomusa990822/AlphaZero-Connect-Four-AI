from src.core.game import ConnectFourGame
from src.search.mcts import MCTS


def test_mcts_returns_valid_move():
    game = ConnectFourGame()
    mcts = MCTS(simulations=50)

    move = mcts.search(game)

    assert move in game.get_valid_moves()


def test_mcts_finds_winning_move():
    game = ConnectFourGame()
    mcts = MCTS(simulations=100)

    # Setup: P1 can win at column 3
    game.apply_move(0)
    game.apply_move(6)
    game.apply_move(1)
    game.apply_move(6)
    game.apply_move(2)
    game.apply_move(5)

    move = mcts.search(game)

    assert move == 3


def test_mcts_blocks_opponent_win():
    game = ConnectFourGame()
    mcts = MCTS(simulations=100)

    # Opponent about to win
    game.board.grid[5, 0] = -1
    game.board.grid[5, 1] = -1
    game.board.grid[5, 2] = -1

    move = mcts.search(game)

    assert move == 3
