"""
Random agent for Connect Four.
"""

from __future__ import annotations

import random

from src.agents.base_agent import BaseAgent
from src.core.game import ConnectFourGame


class RandomAgent(BaseAgent):
    """
    Agent that selects a move uniformly at random from legal moves.
    """

    def __init__(self, name: str = "RandomAgent", seed: int | None = None) -> None:
        super().__init__(name=name)
        self._rng = random.Random(seed)

    def select_move(self, game: ConnectFourGame) -> int:
        """
        Select a random legal move.
        """
        valid_moves = self.get_valid_moves(game)
        if not valid_moves:
            raise ValueError("No valid moves available for RandomAgent.")
        return self._rng.choice(valid_moves)
