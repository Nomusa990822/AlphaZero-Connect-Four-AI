import numpy as np

from torch.utils.data import DataLoader

from src.core.constants import COLS, ROWS
from src.neural.network import AlphaZeroNet
from src.training.dataset import ConnectFourDataset
from src.training.loop import TrainingLoop
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import SelfPlay
from src.training.trainer import Trainer


def make_dummy_samples(n: int = 8):
    samples = []
    for _ in range(n):
        state = np.zeros((3, ROWS, COLS), dtype=np.float32)
        policy = np.ones(COLS, dtype=np.float32) / COLS
        value = 0.0
        samples.append((state, policy, value))
    return samples


def test_trainer_train_epoch_returns_metrics():
    model = AlphaZeroNet()
    trainer = Trainer(model=model)

    dataset = ConnectFourDataset(make_dummy_samples(8))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    metrics = trainer.train_epoch(dataloader)

    assert "total_loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics

    assert metrics["total_loss"] >= 0.0
    assert metrics["policy_loss"] >= 0.0
    assert metrics["value_loss"] >= 0.0


def test_trainer_evaluate_epoch_returns_metrics():
    model = AlphaZeroNet()
    trainer = Trainer(model=model)

    dataset = ConnectFourDataset(make_dummy_samples(8))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    metrics = trainer.evaluate_epoch(dataloader)

    assert "total_loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics


def test_training_loop_runs_and_returns_history():
    model = AlphaZeroNet()
    replay_buffer = ReplayBuffer(capacity=500, seed=42)
    self_play = SelfPlay(model=model, seed=42)
    trainer = Trainer(model=model)

    loop = TrainingLoop(
        model=model,
        replay_buffer=replay_buffer,
        self_play=self_play,
        trainer=trainer,
    )

    history = loop.run(
        iterations=2,
        self_play_games_per_iteration=2,
        batch_size=4,
        epochs_per_iteration=1,
    )

    assert isinstance(history, list)
    assert len(replay_buffer) > 0
