"""
Evaluate an AlphaZero-style Connect Four model against baseline agents.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.agents.alphazero_agent import AlphaZeroAgent
from src.evaluation.baseline_matches import (
    evaluate_against_heuristic,
    evaluate_against_minimax,
    evaluate_against_random,
)
from src.neural.network import AlphaZeroNet


def load_model_if_available(model: AlphaZeroNet, checkpoint_path: Path, device: torch.device) -> bool:
    """
    Load model weights if checkpoint exists.

    Returns:
        True if loaded, else False
    """
    if not checkpoint_path.exists():
        return False

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return True


def print_summary(title: str, summary: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Games: {summary['games']}")
    print(f"Wins: {summary['wins']}")
    print(f"Losses: {summary['losses']}")
    print(f"Draws: {summary['draws']}")
    print(f"Win rate: {summary['win_rate']:.2%}")
    print(f"Loss rate: {summary['loss_rate']:.2%}")
    print(f"Draw rate: {summary['draw_rate']:.2%}")
    print(f"Average game length: {summary['average_game_length']:.2f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlphaZeroNet()
    checkpoint_path = Path("models/checkpoints/alphazero_connect4_latest.pt")

    loaded = load_model_if_available(model, checkpoint_path, device)
    if loaded:
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print("No checkpoint found. Evaluating with current untrained model.")

    agent = AlphaZeroAgent(
        model=model,
        simulations=20,
        c_puct=1.5,
        add_root_noise=False,
        device=device,
        seed=42,
        name="AlphaZeroAgent",
    )

    random_summary = evaluate_against_random(agent, num_games=6, seed=42)
    heuristic_summary = evaluate_against_heuristic(agent, num_games=6)
    minimax_summary = evaluate_against_minimax(agent, num_games=6, depth=2, seed=42)

    print_summary("AlphaZeroAgent vs RandomAgent", random_summary)
    print_summary("AlphaZeroAgent vs HeuristicAgent", heuristic_summary)
    print_summary("AlphaZeroAgent vs MinimaxAgent(depth=2)", minimax_summary)


if __name__ == "__main__":
    main()
