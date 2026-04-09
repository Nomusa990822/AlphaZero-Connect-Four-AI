"""
Arena utilities for evaluating Connect Four agents against each other.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.constants import DRAW, PLAYER_ONE, PLAYER_TWO
from src.core.game import ConnectFourGame


@dataclass
class MatchResult:
    winner: int
    num_moves: int
    starting_player: int
    agent_player_one_name: str
    agent_player_two_name: str


class Arena:
    """
    Runs head-to-head matches between two agents.
    """

    def __init__(self) -> None:
        pass

    def play_game(self, agent_player_one, agent_player_two) -> MatchResult:
        """
        Play one game between two agents.

        Args:
            agent_player_one: agent controlling PLAYER_ONE
            agent_player_two: agent controlling PLAYER_TWO

        Returns:
            MatchResult
        """
        game = ConnectFourGame()

        while not game.done:
            if game.current_player == PLAYER_ONE:
                move = agent_player_one.select_move(game.copy())
            else:
                move = agent_player_two.select_move(game.copy())

            if move not in game.get_valid_moves():
                raise ValueError(
                    f"Agent selected illegal move {move}. "
                    f"Legal moves are {game.get_valid_moves()}."
                )

            game.apply_move(move)

        return MatchResult(
            winner=game.winner,
            num_moves=game.move_count,
            starting_player=PLAYER_ONE,
            agent_player_one_name=agent_player_one.name,
            agent_player_two_name=agent_player_two.name,
        )

    def play_games(self, agent_a, agent_b, num_games: int = 10, alternate_starts: bool = True) -> list[MatchResult]:
        """
        Play multiple games between two agents.

        If alternate_starts=True, agents alternate roles as PLAYER_ONE / PLAYER_TWO.

        Returns:
            list of MatchResult
        """
        if num_games < 1:
            raise ValueError("num_games must be at least 1.")

        results: list[MatchResult] = []

        for game_idx in range(num_games):
            if alternate_starts and game_idx % 2 == 1:
                result = self.play_game(agent_b, agent_a)
            else:
                result = self.play_game(agent_a, agent_b)

            results.append(result)

        return results
