"""
Base agent interface for Connect Four agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.core.game import ConnectFourGame


class BaseAgent(ABC):
    """
    Abstract base class for all game-playing agents.
    """

    def __init__(self, name: str = "BaseAgent") -> None:
        self.name = name

    @abstractmethod
    def select_move(self, game: ConnectFourGame) -> int:
        """
        Choose a legal move for the current game state.

        Args:
            game: Current Connect Four game state.

        Returns:
            A valid column index.
        """
        raise NotImplementedError

    def get_valid_moves(self, game: ConnectFourGame) -> List[int]:
        """
        Return the currently legal moves.
        """
        return game.get_valid_moves()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
