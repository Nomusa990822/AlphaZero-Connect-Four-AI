import numpy as np
import torch

from src.core.constants import COLS, ROWS
from src.neural.network import AlphaZeroNet
from src.training.dataset import ConnectFourDataset
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import SelfPlay


def test_replay_buffer_add_and_length():
    buffer = ReplayBuffer(capacity=5)

    state = np.zeros((3, ROWS, COLS), dtype=np.float32)
    policy = np.ones(COLS, dtype=np.float32) / COLS
    value = 1.0

    buffer.add(state, policy, value)
    assert len(buffer) == 1


def test_replay_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=2)

    state = np.zeros((3, ROWS, COLS), dtype=np.float32)
    policy = np.ones(COLS, dtype=np.float32) / COLS

    buffer.add(state, policy, 1.0)
    buffer.add(state, policy, 0.0)
    buffer.add(state, policy, -1.0)

    assert len(buffer) == 2


def test_replay_buffer_sample_size():
    buffer = ReplayBuffer(capacity=10)
    state = np.zeros((3, ROWS, COLS), dtype=np.float32)
    policy = np.ones(COLS, dtype=np.float32) / COLS

    for i in range(5):
        buffer.add(state, policy, float(i % 3 - 1))

    sample = buffer.sample(3)
    assert len(sample) == 3


def test_self_play_single_game_generates_samples():
    model = AlphaZeroNet()
    self_play = SelfPlay(model=model, simulations=15, seed=42)

    samples = self_play.play_single_game()

    assert len(samples) > 0

    state, policy, value = samples[0]
    assert state.shape == (3, ROWS, COLS)
    assert policy.shape == (COLS,)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)
    assert value in (-1.0, 0.0, 1.0)


def test_self_play_multiple_games_generates_more_samples():
    model = AlphaZeroNet()
    self_play = SelfPlay(model=model, simulations=10, seed=42)

    samples = self_play.generate_games(2)

    assert len(samples) > 0
    assert all(len(sample) == 3 for sample in samples)


def test_dataset_returns_tensors():
    state = np.zeros((3, ROWS, COLS), dtype=np.float32)
    policy = np.ones(COLS, dtype=np.float32) / COLS
    value = 1.0

    dataset = ConnectFourDataset([(state, policy, value)])
    s, p, v = dataset[0]

    assert isinstance(s, torch.Tensor)
    assert isinstance(p, torch.Tensor)
    assert isinstance(v, torch.Tensor)

    assert s.shape == (3, ROWS, COLS)
    assert p.shape == (COLS,)
    assert v.shape == (1,)
