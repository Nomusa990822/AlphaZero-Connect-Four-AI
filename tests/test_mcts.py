import numpy as np
import torch

from src.core.game import ConnectFourGame
from src.neural.network import AlphaZeroNet
from src.search.mcts import MCTS, MCTSResult


def test_mcts_returns_result_object():
    model = AlphaZeroNet()
    game = ConnectFourGame()
    mcts = MCTS(model=model, simulations=25)

    result = mcts.search(game)

    assert isinstance(result, MCTSResult)
    assert result.selected_move in game.get_valid_moves()


def test_mcts_policy_target_shape_and_sum():
    model = AlphaZeroNet()
    game = ConnectFourGame()
    mcts = MCTS(model=model, simulations=25)

    result = mcts.search(game)

    assert result.policy_target.shape == (7,)
    assert np.isclose(result.policy_target.sum(), 1.0, atol=1e-5)


def test_mcts_visit_counts_cover_only_legal_moves():
    model = AlphaZeroNet()
    game = ConnectFourGame()
    mcts = MCTS(model=model, simulations=25)

    result = mcts.search(game)

    legal_moves = set(game.get_valid_moves())
    visited_moves = set(result.visit_counts.keys())

    assert visited_moves.issubset(legal_moves)


def test_mcts_handles_midgame_position():
    model = AlphaZeroNet()
    game = ConnectFourGame()

    game.apply_move(3)
    game.apply_move(2)
    game.apply_move(3)
    game.apply_move(2)
    game.apply_move(4)

    mcts = MCTS(model=model, simulations=30)
    result = mcts.search(game)

    assert result.selected_move in game.get_valid_moves()
    assert result.policy_target.shape == (7,)
    assert np.isclose(result.policy_target.sum(), 1.0, atol=1e-5)


def test_mcts_root_value_in_range():
    model = AlphaZeroNet()
    game = ConnectFourGame()
    mcts = MCTS(model=model, simulations=20)

    result = mcts.search(game)

    assert -1.0 <= result.root_value <= 1.0


def test_mcts_with_root_noise_still_returns_valid_output():
    model = AlphaZeroNet()
    game = ConnectFourGame()
    mcts = MCTS(
        model=model,
        simulations=20,
        add_root_noise=True,
        seed=42,
    )

    result = mcts.search(game)

    assert result.selected_move in game.get_valid_moves()
    assert np.isclose(result.policy_target.sum(), 1.0, atol=1e-5)
