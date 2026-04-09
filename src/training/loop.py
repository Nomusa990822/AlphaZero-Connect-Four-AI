"""
High-level AlphaZero-style training loop with live terminal logging.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

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
        total_start_time = time.time()

        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Iterations: {iterations}")
        print(f"Self-play games per iteration: {self_play_games_per_iteration}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs per iteration: {epochs_per_iteration}")
        print("=" * 70)

        for iteration in range(1, iterations + 1):
            iteration_start_time = time.time()

            print("\n" + "=" * 70)
            print(f"ITERATION {iteration}")
            print("=" * 70)

            # -------------------------------------------------
            # Self-play
            # -------------------------------------------------
            print("\nGenerating self-play games...")
            new_samples = []

            for game_idx in range(self_play_games_per_iteration):
                game_start_time = time.time()

                game_samples = self.self_play.play_single_game(game_idx=game_idx)
                new_samples.extend(game_samples)

                game_duration = time.time() - game_start_time
                print(
                    f"Game {game_idx + 1}: "
                    f"moves={len(game_samples)}, "
                    f"samples={len(game_samples)}, "
                    f"time={game_duration:.2f}s"
                )

            self.replay_buffer.extend(new_samples)

            print(f"\nReplay buffer size: {len(self.replay_buffer)}")
            print(f"New samples added this iteration: {len(new_samples)}")

            # -------------------------------------------------
            # Training
            # -------------------------------------------------
            if len(self.replay_buffer) < batch_size:
                iteration_duration = time.time() - iteration_start_time
                print("\nNot enough data to train yet. Skipping training...")
                print(f"Iteration {iteration} time: {iteration_duration:.2f}s")
                continue

            print("\nBuilding dataset...")
            dataset = ConnectFourDataset(self.replay_buffer.as_list())
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            print("Training model...")
            epoch_times: list[float] = []

            for epoch in range(1, epochs_per_iteration + 1):
                epoch_start_time = time.time()

                metrics = self.trainer.train_epoch(dataloader)

                epoch_duration = time.time() - epoch_start_time
                epoch_times.append(epoch_duration)

                print(
                    f"Epoch {epoch}: "
                    f"total={metrics['total_loss']:.4f}, "
                    f"policy={metrics['policy_loss']:.4f}, "
                    f"value={metrics['value_loss']:.4f}, "
                    f"time={epoch_duration:.2f}s"
                )

                history.append({
                    "iteration": iteration,
                    "epoch": epoch,
                    "buffer_size": len(self.replay_buffer),
                    "epoch_time_seconds": epoch_duration,
                    **metrics,
                })

            iteration_duration = time.time() - iteration_start_time

            print("\nIteration summary:")
            print(f"- Samples added: {len(new_samples)}")
            print(f"- Replay buffer size: {len(self.replay_buffer)}")
            print(f"- Epochs run: {epochs_per_iteration}")
            print(f"- Average epoch time: {sum(epoch_times) / len(epoch_times):.2f}s")
            print(f"- Iteration time: {iteration_duration:.2f}s")

        total_duration = time.time() - total_start_time

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total iterations run: {iterations}")
        print(f"Final replay buffer size: {len(self.replay_buffer)}")
        print(f"History entries: {len(history)}")
        print(f"Total training time: {total_duration:.2f}s ({total_duration / 60:.2f} min)")
        print("=" * 70)

        return history
