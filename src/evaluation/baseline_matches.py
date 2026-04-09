"""
Convenience helpers for evaluating an AlphaZero agent against baseline agents.
"""

from __future__ import annotations

from src.agents.heuristic_agent import HeuristicAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.arena import Arena
from src.evaluation.metrics import summarize_results


def evaluate_against_random(agent, num_games: int = 10, seed: int | None = 42) -> dict:
    arena = Arena()
    opponent = RandomAgent(seed=seed, name="RandomAgent")
    results = arena.play_games(agent, opponent, num_games=num_games, alternate_starts=True)
    return summarize_results(results, agent.name, opponent.name)


def evaluate_against_heuristic(agent, num_games: int = 10) -> dict:
    arena = Arena()
    opponent = HeuristicAgent(name="HeuristicAgent")
    results = arena.play_games(agent, opponent, num_games=num_games, alternate_starts=True)
    return summarize_results(results, agent.name, opponent.name)


def evaluate_against_minimax(agent, num_games: int = 10, depth: int = 3, seed: int | None = 42) -> dict:
    arena = Arena()
    opponent = MinimaxAgent(depth=depth, seed=seed, name=f"MinimaxAgent(depth={depth})")
    results = arena.play_games(agent, opponent, num_games=num_games, alternate_starts=True)
    return summarize_results(results, agent.name, opponent.name)
