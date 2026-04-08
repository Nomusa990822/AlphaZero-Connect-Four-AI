import numpy as np
import torch

from src.core.constants import COLS, PLAYER_ONE, PLAYER_TWO, ROWS
from src.core.game import ConnectFourGame
from src.core.move_encoder import (
    decode_move,
    encode_move,
    legal_moves_mask,
    normalize_policy,
    one_hot_move,
)
from src.core.state_encoder import encode_batch, encode_state, encode_state_tensor
from src.neural.losses import alphazero_loss, policy_loss_fn, value_loss_fn
from src.neural.network import AlphaZeroNet


def test_encode_state_shape():
    game = ConnectFourGame()
    encoded = encode_state(game)

    assert encoded.shape == (3, ROWS, COLS)
    assert encoded.dtype == np.float32


def test_encode_state_planes_on_empty_board():
    game = ConnectFourGame()
    encoded = encode_state(game)

    assert np.all(encoded[0] == 0.0)
    assert np.all(encoded[1] == 0.0)
    assert np.all(encoded[2] == 1.0)  # PLAYER_ONE to move


def test_encode_state_after_move_switches_perspective():
    game = ConnectFourGame()
    game.apply_move(3)  # PLAYER_ONE plays, then PLAYER_TWO to move

    encoded = encode_state(game)

    # Current player is PLAYER_TWO, so plane 0 should show PLAYER_TWO pieces
    assert encoded[0].sum() == 0.0
    # Opponent plane should show PLAYER_ONE's single piece
    assert encoded[1].sum() == 1.0
    # PLAYER_TWO plane indicator should be zeros
    assert np.all(encoded[2] == 0.0)


def test_encode_state_tensor_shape():
    game = ConnectFourGame()
    tensor = encode_state_tensor(game)

    assert tensor.shape == (1, 3, ROWS, COLS)
    assert tensor.dtype == torch.float32


def test_encode_batch_shape():
    games = [ConnectFourGame(), ConnectFourGame()]
    batch = encode_batch(games)

    assert batch.shape == (2, 3, ROWS, COLS)
    assert batch.dtype == torch.float32


def test_move_encoder_round_trip():
    for move in range(COLS):
        assert decode_move(encode_move(move)) == move


def test_one_hot_move():
    vec = one_hot_move(3)

    assert vec.shape == (COLS,)
    assert vec.sum() == 1.0
    assert vec[3] == 1.0


def test_legal_moves_mask():
    mask = legal_moves_mask([0, 3, 6])

    expected = np.array([1, 0, 0, 1, 0, 0, 1], dtype=np.float32)
    assert np.array_equal(mask, expected)


def test_normalize_policy_masks_illegal_moves():
    raw = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    normalized = normalize_policy(raw, [2, 3])

    assert np.isclose(normalized.sum(), 1.0)
    assert normalized[2] > 0
    assert normalized[3] > 0
    assert normalized[0] == 0.0
    assert normalized[6] == 0.0


def test_network_forward_shapes():
    net = AlphaZeroNet()
    x = torch.randn(4, 3, ROWS, COLS)

    policy_logits, value = net(x)

    assert policy_logits.shape == (4, COLS)
    assert value.shape == (4, 1)


def test_network_predict_shapes_and_ranges():
    net = AlphaZeroNet()
    x = torch.randn(2, 3, ROWS, COLS)

    probs, value = net.predict(x)

    assert probs.shape == (2, COLS)
    assert value.shape == (2, 1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_predict_single():
    net = AlphaZeroNet()
    x = torch.randn(1, 3, ROWS, COLS)

    probs, value = net.predict_single(x)

    assert probs.shape == (COLS,)
    assert isinstance(value, float)
    assert abs(float(probs.sum().item()) - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0


def test_masked_policy_zeroes_illegal_moves():
    net = AlphaZeroNet()

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 0.0]])
    mask = torch.tensor([[0, 0, 1, 1, 0, 0, 0]], dtype=torch.float32)

    probs = net.masked_policy(logits, mask)

    assert probs.shape == (1, COLS)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert probs[0, 0] == 0.0
    assert probs[0, 1] == 0.0
    assert probs[0, 2] > 0.0
    assert probs[0, 3] > 0.0


def test_policy_loss_fn_returns_scalar():
    logits = torch.randn(3, COLS)
    target_policy = torch.softmax(torch.randn(3, COLS), dim=1)

    loss = policy_loss_fn(logits, target_policy)

    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_value_loss_fn_returns_scalar():
    pred = torch.tensor([[0.2], [-0.5], [0.8]])
    target = torch.tensor([[0.0], [-1.0], [1.0]])

    loss = value_loss_fn(pred, target)

    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_alphazero_loss_returns_three_scalars():
    logits = torch.randn(5, COLS)
    pred_value = torch.randn(5, 1).clamp(-1, 1)
    target_policy = torch.softmax(torch.randn(5, COLS), dim=1)
    target_value = torch.randn(5, 1).clamp(-1, 1)

    total, p_loss, v_loss = alphazero_loss(
        policy_logits=logits,
        pred_value=pred_value,
        target_policy=target_policy,
        target_value=target_value,
    )

    assert total.ndim == 0
    assert p_loss.ndim == 0
    assert v_loss.ndim == 0
    assert torch.isclose(total, p_loss + v_loss, atol=1e-6)


def test_network_can_process_real_encoded_game_state():
    game = ConnectFourGame()
    game.apply_move(3)
    game.apply_move(2)
    game.apply_move(3)

    x = encode_state_tensor(game)
    net = AlphaZeroNet()

    policy_logits, value = net(x)

    assert policy_logits.shape == (1, COLS)
    assert value.shape == (1, 1)
