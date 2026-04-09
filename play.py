"""
Play Connect Four against the AlphaZero-style AI in the terminal.

Human is PLAYER_ONE (X)
AI is PLAYER_TWO (O)

Example:
    python play.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.agents.alphazero_agent import AlphaZeroAgent
from src.core.constants import PLAYER_ONE, PLAYER_TWO
from src.core.game import ConnectFourGame
from src.neural.network import AlphaZeroNet


def load_model_if_available(model: AlphaZeroNet, checkpoint_path: Path, device: torch.device) -> bool:
    """
    Load model weights if checkpoint exists.

    Returns:
        True if model was loaded, otherwise False.
    """
    if not checkpoint_path.exists():
        return False

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return True


def print_instructions() -> None:
    print("\nConnect Four: Human vs AlphaZero-style AI")
    print("You are X and play first.")
    print("Enter a column number from 0 to 6.")
    print("Columns: 0 1 2 3 4 5 6\n")


def prompt_human_move(game: ConnectFourGame) -> int:
    """
    Prompt the human until a valid move is entered.
    """
    valid_moves = game.get_valid_moves()

    while True:
        raw = input(f"Your move {valid_moves}: ").strip()

        try:
            move = int(raw)
        except ValueError:
            print("Please enter an integer between 0 and 6.")
            continue

        if move not in valid_moves:
            print(f"Invalid move. Legal moves are: {valid_moves}")
            continue

        return move


def announce_result(game: ConnectFourGame) -> None:
    print("\nFinal board:")
    game.render()

    if game.winner == PLAYER_ONE:
        print("\nYou win.")
    elif game.winner == PLAYER_TWO:
        print("\nAI wins.")
    else:
        print("\nDraw.")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlphaZeroNet()
    checkpoint_path = Path("models/checkpoints/alphazero_connect4_latest.pt")
    loaded = load_model_if_available(model, checkpoint_path, device)

    if loaded:
        print(f"Loaded AI checkpoint from: {checkpoint_path}")
    else:
        print("No trained checkpoint found. Using current untrained model.")

    ai_agent = AlphaZeroAgent(
        model=model,
        simulations=25,
        c_puct=1.5,
        add_root_noise=False,
        device=device,
        seed=42,
        name="AlphaZeroAgent",
    )

    game = ConnectFourGame()
    print_instructions()

    while not game.done:
        print("\nCurrent board:")
        game.render()

        if game.current_player == PLAYER_ONE:
            move = prompt_human_move(game)
            game.apply_move(move)
        else:
            ai_move = ai_agent.select_move(game)
            print(f"AI plays column: {ai_move}")
            game.apply_move(ai_move)

    announce_result(game)


if __name__ == "__main__":
    main()
