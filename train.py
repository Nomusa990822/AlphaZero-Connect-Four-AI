"""
Train an AlphaZero-style Connect Four model.

This script:
1. Creates the model
2. Builds replay buffer, self-play, and trainer objects
3. Runs the training loop
4. Saves a final checkpoint

Example:
    python train.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.neural.network import AlphaZeroNet
from src.training.loop import TrainingLoop
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import SelfPlay
from src.training.trainer import Trainer


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlphaZeroNet()
    replay_buffer = ReplayBuffer(capacity=5000, seed=42)
    self_play = SelfPlay(
        model=model,
        device=device,
        simulations=40,
        c_puct=1.5,
        temperature=1.0,
        add_root_noise=True,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    loop = TrainingLoop(
        model=model,
        replay_buffer=replay_buffer,
        self_play=self_play,
        trainer=trainer,
    )

    history = loop.run(
        iterations=10,
        self_play_games_per_iteration=20,
        batch_size=16,
        epochs_per_iteration=2,
    )

    print("\nTraining complete.")
    print(f"Replay buffer size: {len(replay_buffer)}")
    print(f"History entries: {len(history)}")

    for row in history[-5:]:
        print(row)

    output_dir = Path("models/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "alphazero_connect4_latest.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "buffer_size": len(replay_buffer),
        },
        checkpoint_path,
    )

    print(f"\nSaved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
