"""
Simple round-robin tournament utilities for Connect Four agents.
"""

from __future__ import annotations

from itertools import combinations

from src.evaluation.arena import Arena
from src.evaluation.metrics import summarize_results


class Tournament:
    """
    Runs pairwise evaluations among a set of agents.
    """

    def __init__(self, arena: Arena | None = None) -> None:
        self.arena = arena or Arena()

    def run(self, agents: list, games_per_pairing: int = 4) -> list[dict]:
        """
        Run a round-robin tournament across all unique agent pairings.

        Returns:
            list of records
        """
        if len(agents) < 2:
            raise ValueError("At least two agents are required for a tournament.")
        if games_per_pairing < 1:
            raise ValueError("games_per_pairing must be at least 1.")

        standings: list[dict] = []

        for agent_a, agent_b in combinations(agents, 2):
            results = self.arena.play_games(
                agent_a,
                agent_b,
                num_games=games_per_pairing,
                alternate_starts=True,
            )

            summary_a = summarize_results(results, agent_a.name, agent_b.name)
            summary_b = summarize_results(results, agent_b.name, agent_a.name)

            standings.append({
                "agent": agent_a.name,
                "opponent": agent_b.name,
                **summary_a,
            })
            standings.append({
                "agent": agent_b.name,
                "opponent": agent_a.name,
                **summary_b,
            })

        return standings
