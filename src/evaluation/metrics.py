"""
Metrics for head-to-head agent evaluation.
"""

from __future__ import annotations

from src.core.constants import DRAW, PLAYER_ONE, PLAYER_TWO


def summarize_results(results: list, agent_a_name: str, agent_b_name: str) -> dict[str, float | int]:
    """
    Summarize a list of MatchResult objects from the perspective of agent_a_name.

    Handles alternating start order by reading agent names from each MatchResult.
    """
    if not results:
        raise ValueError("results cannot be empty.")

    wins = 0
    losses = 0
    draws = 0
    total_moves = 0

    for result in results:
        total_moves += result.num_moves

        if result.winner == DRAW:
            draws += 1
            continue

        if result.winner == PLAYER_ONE:
            winner_name = result.agent_player_one_name
        elif result.winner == PLAYER_TWO:
            winner_name = result.agent_player_two_name
        else:
            raise ValueError(f"Unexpected winner value: {result.winner}")

        if winner_name == agent_a_name:
            wins += 1
        elif winner_name == agent_b_name:
            losses += 1
        else:
            raise ValueError("Winner name does not match either agent name.")

    total_games = len(results)

    return {
        "games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total_games,
        "loss_rate": losses / total_games,
        "draw_rate": draws / total_games,
        "average_game_length": total_moves / total_games,
    }
