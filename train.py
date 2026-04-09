"""
Train an AlphaZero-style Connect Four model.

This version is aligned with the stronger upgrades:
- tactical MCTS
- temperature decay
- symmetry augmentation
- longer self-play / training runs

Example:
    python train.py
"""

from __future__ import annotations

from pathlib import Path
import time

import torch

from src.neural.network import AlphaZeroNet
from src.training.loop import TrainingLoop
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import SelfPlay
from src.training.trainer import Trainer


def main() -> None:
    overall_start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # -------------------------
    # Model
    # -------------------------
    model = AlphaZeroNet()

    # -------------------------
    # Core training components
    # -------------------------
    replay_buffer = ReplayBuffer(capacity=20000, seed=42)

    self_play = SelfPlay(
        model=model,
        device=device,
        simulations=60,
        c_puct=1.5,
        add_root_noise=True,
        seed=42,
        early_temperature=1.0,
        late_temperature=0.1,
        temperature_drop_move=10,
        augment_symmetry=True,
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

    # -------------------------
    # Run training
    # -------------------------
    history = loop.run(
        iterations=15,
        self_play_games_per_iteration=30,
        batch_size=32,
        epochs_per_iteration=5,
    )

    total_elapsed = time.time() - overall_start_time

    # -------------------------
    # Final summary
    # -------------------------
    print("\n" + "=" * 70)
    print("FINAL TRAINING SUMMARY")
    print("=" * 70)
    print(f"Replay buffer size: {len(replay_buffer)}")
    print(f"History entries: {len(history)}")
    print(f"Total runtime: {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)")

    if history:
        print("\nLast 5 history entries:")
        for row in history[-5:]:
            print(row)

        best_total = min(item["total_loss"] for item in history)
        best_policy = min(item["policy_loss"] for item in history)
        best_value = min(item["value_loss"] for item in history)

        print("\nBest observed losses:")
        print(f"Best total loss:  {best_total:.4f}")
        print(f"Best policy loss: {best_policy:.4f}")
        print(f"Best value loss:  {best_value:.4f}")

    # -------------------------
    # Save checkpoint
    # -------------------------
    output_dir = Path("models/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "alphazero_connect4_latest.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "buffer_size": len(replay_buffer),
            "device": str(device),
            "total_runtime_seconds": total_elapsed,
            "training_config": {
                "replay_buffer_capacity": 20000,
                "iterations": 15,
                "self_play_games_per_iteration": 30,
                "batch_size": 32,
                "epochs_per_iteration": 5,
                "simulations": 60,
                "c_puct": 1.5,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "early_temperature": 1.0,
                "late_temperature": 0.1,
                "temperature_drop_move": 10,
                "augment_symmetry": True,
                "add_root_noise": True,
                "seed": 42,
            },
        },
        checkpoint_path,
    )

    print(f"\nSaved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
