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
        overall_start_time = time.time()

        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Iterations: {iterations}")
        print(f"Self-play games / iteration: {self_play_games_per_iteration}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs / iteration: {epochs_per_iteration}")
        print("=" * 70)

        for iteration in range(1, iterations + 1):
            iteration_start_time = time.time()

            print("\n" + "=" * 70)
            print(f"ITERATION {iteration}")
            print("=" * 70)

            # =========================
            # SELF-PLAY
            # =========================
            print("\nGenerating self-play games...")
            self_play_start_time = time.time()

            new_samples = []

            for game_idx in range(self_play_games_per_iteration):
                game_start_time = time.time()

                samples = self.self_play.play_single_game(game_idx=game_idx)
                new_samples.extend(samples)

                game_elapsed = time.time() - game_start_time
                print(
                    f"Game {game_idx + 1}/{self_play_games_per_iteration}: "
                    f"moves={len(samples)}, samples={len(samples)}, "
                    f"time={game_elapsed:.2f}s"
                )

            self.replay_buffer.extend(new_samples)
            self_play_elapsed = time.time() - self_play_start_time

            print(f"\nSelf-play complete.")
            print(f"New samples generated: {len(new_samples)}")
            print(f"Replay buffer size: {len(self.replay_buffer)}")
            print(f"Self-play time: {self_play_elapsed:.2f}s")

            # =========================
            # TRAINING
            # =========================
            if len(self.replay_buffer) < batch_size:
                iteration_elapsed = time.time() - iteration_start_time
                print("\nNot enough data to train yet. Skipping training.")
                print(f"Iteration {iteration} time: {iteration_elapsed:.2f}s")
                continue

            print("\nBuilding dataset...")
            dataset = ConnectFourDataset(self.replay_buffer.as_list())
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            print("Training model...")
            training_start_time = time.time()

            for epoch in range(1, epochs_per_iteration + 1):
                epoch_start_time = time.time()

                metrics = self.trainer.train_epoch(dataloader)

                epoch_elapsed = time.time() - epoch_start_time

                print(
                    f"Epoch {epoch}/{epochs_per_iteration}: "
                    f"total={metrics['total_loss']:.4f}, "
                    f"policy={metrics['policy_loss']:.4f}, "
                    f"value={metrics['value_loss']:.4f}, "
                    f"time={epoch_elapsed:.2f}s"
                )

                history.append({
                    "iteration": iteration,
                    "epoch": epoch,
                    "buffer_size": len(self.replay_buffer),
                    "epoch_time_seconds": epoch_elapsed,
                    **metrics,
                })

            training_elapsed = time.time() - training_start_time
            iteration_elapsed = time.time() - iteration_start_time

            print("\nIteration complete.")
            print(f"Training time: {training_elapsed:.2f}s")
            print(f"Iteration {iteration} total time: {iteration_elapsed:.2f}s")

        total_elapsed = time.time() - overall_start_time

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total iterations run: {iterations}")
        print(f"Final replay buffer size: {len(self.replay_buffer)}")
        print(f"History entries: {len(history)}")
        print(f"Total training time: {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)")
        print("=" * 70)

        return history
