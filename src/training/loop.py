"""
High-level AlphaZero-style training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from src.neural.network import AlphaZeroNet
from src.training.dataset import ConnectFourDataset
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import SelfPlay
from src.training.trainer import Trainer


@dataclass
class TrainingLoopConfig:
    iterations: int = 3
    self_play_games_per_iteration: int = 5
    batch_size: int = 32
    epochs_per_iteration: int = 2


class TrainingLoop:
    """
    Coordinates:
    - self-play generation
    - replay buffer updates
    - dataset creation
    - model training
    """

    def __init__(
        self,
        model: AlphaZeroNet,
        replay_buffer: ReplayBuffer,
        self_play: SelfPlay,
        trainer: Trainer,
    ) -> None:
        self.model = model
        self.replay_buffer = replay_buffer
        self.self_play = self_play
        self.trainer = trainer

    def run(
        self,
        iterations: int,
        self_play_games_per_iteration: int,
        batch_size: int,
        epochs_per_iteration: int,
    ) -> list[dict]:
        """
        Run the training loop and return metrics history.
        """
        if iterations < 1:
            raise ValueError("iterations must be at least 1.")
        if self_play_games_per_iteration < 1:
            raise ValueError("self_play_games_per_iteration must be at least 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if epochs_per_iteration < 1:
            raise ValueError("epochs_per_iteration must be at least 1.")

        history: list[dict] = []

        for iteration in range(1, iterations + 1):
            new_samples = self.self_play.generate_games(self_play_games_per_iteration)
            self.replay_buffer.extend(new_samples)

            if len(self.replay_buffer) < batch_size:
                continue

            dataset = ConnectFourDataset(self.replay_buffer.as_list())
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(1, epochs_per_iteration + 1):
                metrics = self.trainer.train_epoch(dataloader)
                history.append({
                    "iteration": iteration,
                    "epoch": epoch,
                    "buffer_size": len(self.replay_buffer),
                    **metrics,
                })

        return history
