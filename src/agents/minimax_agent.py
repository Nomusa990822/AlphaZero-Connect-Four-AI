"""
Minimax agent with alpha-beta pruning for Connect Four.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from src.agents.base_agent import BaseAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.core.constants import DRAW, PLAYER_ONE, PLAYER_TWO
from src.core.game import ConnectFourGame


class MinimaxAgent(BaseAgent):
    """
    Minimax-based Connect Four agent with alpha-beta pruning.

    Features:
    - configurable search depth
    - alpha-beta pruning
    - tactical win/block handling through search
    - heuristic leaf evaluation
    """

    def __init__(
        self,
        depth: int = 4,
        name: str = "MinimaxAgent",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        self.depth = depth
        self._rng = random.Random(seed)
        self._heuristic = HeuristicAgent()

    def select_move(self, game: ConnectFourGame) -> int:
        """
        Select the best move using minimax search.
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available for MinimaxAgent.")

        maximizing_player = game.current_player
        score, best_move = self._minimax(
            game=game.copy(),
            depth=self.depth,
            alpha=-math.inf,
            beta=math.inf,
            maximizing=True,
            root_player=maximizing_player,
        )

        if best_move is None:
            return self._choose_fallback_move(valid_moves)

        return best_move

    def _minimax(
        self,
        game: ConnectFourGame,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        root_player: int,
    ) -> Tuple[float, int | None]:
        """
        Recursive minimax with alpha-beta pruning.

        Returns:
            (score, best_move)
        """
        valid_moves = game.get_valid_moves()

        # Terminal / leaf handling
        if game.done:
            return self._terminal_score(game, root_player, depth), None

        if depth == 0:
            score = self._heuristic.evaluate_board(game.board, root_player)
            return score, None

        if not valid_moves:
            return 0.0, None

        ordered_moves = self._order_moves(valid_moves)

        if maximizing:
            value = -math.inf
            best_moves: List[int] = []

            for col in ordered_moves:
                child = game.copy()
                child.apply_move(col)

                child_score, _ = self._minimax(
                    game=child,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing=False,
                    root_player=root_player,
                )

                if child_score > value:
                    value = child_score
                    best_moves = [col]
                elif child_score == value:
                    best_moves.append(col)

                alpha = max(alpha, value)
                if alpha >= beta:
                    break

            return value, self._choose_fallback_move(best_moves)

        else:
            value = math.inf
            best_moves: List[int] = []

            for col in ordered_moves:
                child = game.copy()
                child.apply_move(col)

                child_score, _ = self._minimax(
                    game=child,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing=True,
                    root_player=root_player,
                )

                if child_score < value:
                    value = child_score
                    best_moves = [col]
                elif child_score == value:
                    best_moves.append(col)

                beta = min(beta, value)
                if alpha >= beta:
                    break

            return value, self._choose_fallback_move(best_moves)

    def _terminal_score(self, game: ConnectFourGame, root_player: int, depth: int) -> float:
        """
        Score terminal states from the root player's perspective.

        Depth bonus rewards faster wins and delays losses.
        """
        if game.winner == root_player:
            return 1_000_000.0 + depth
        if game.winner == DRAW:
            return 0.0
        if game.winner is None:
            return 0.0
        return -1_000_000.0 - depth

    def _order_moves(self, moves: List[int]) -> List[int]:
        """
        Order moves by center preference to improve pruning.
        """
        center = 3
        return sorted(moves, key=lambda col: abs(col - center))

    def _choose_fallback_move(self, moves: List[int]) -> int:
        """
        Choose among equally good moves.
        Prefers center-biased random tie-breaking for variety.
        """
        if not moves:
            raise ValueError("Cannot choose from an empty move list.")

        best_distance = min(abs(col - 3) for col in moves)
        center_preferred = [col for col in moves if abs(col - 3) == best_distance]
        return self._rng.choice(center_preferred)
