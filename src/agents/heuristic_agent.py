"""
Heuristic agent for Connect Four.

This agent uses simple tactical reasoning:
1. Play a winning move if available.
2. Block the opponent's winning move if necessary.
3. Prefer central columns.
4. Score candidate moves using a board evaluation function.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.agents.base_agent import BaseAgent
from src.core.board import Board
from src.core.constants import COLS, CONNECT_N, EMPTY, PLAYER_ONE, PLAYER_TWO, ROWS
from src.core.game import ConnectFourGame
from src.core.rules import check_winner


class HeuristicAgent(BaseAgent):
    """
    Rule-based heuristic agent for Connect Four.
    """

    def __init__(self, name: str = "HeuristicAgent") -> None:
        super().__init__(name=name)

    def select_move(self, game: ConnectFourGame) -> int:
        """
        Select a move using tactical checks + heuristic evaluation.
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available for HeuristicAgent.")

        player = game.current_player
        opponent = PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE

        # 1. Immediate winning move
        for col in valid_moves:
            if self._is_winning_move(game, col, player):
                return col

        # 2. Immediate block
        for col in valid_moves:
            if self._is_winning_move(game, col, opponent):
                return col

        # 3. Score moves and prefer best
        scored_moves = []
        for col in valid_moves:
            temp_game = game.copy()
            temp_game.apply_move(col)
            score = self.evaluate_board(temp_game.board, player)
            center_bonus = -abs(col - (COLS // 2)) * 0.1
            scored_moves.append((score + center_bonus, col))

        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return scored_moves[0][1]

    def _is_winning_move(self, game: ConnectFourGame, col: int, player: int) -> bool:
        """
        Check whether dropping into col for 'player' creates an immediate win.
        """
        temp_game = game.copy()
        temp_game.current_player = player
        temp_game.apply_move(col)
        return temp_game.winner == player

    def evaluate_board(self, board: Board, player: int) -> float:
        """
        Score a board position from the perspective of 'player'.
        Higher is better for 'player'.
        """
        opponent = PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE
        grid = board.grid
        score = 0.0

        # Center column preference
        center_col = COLS // 2
        center_array = grid[:, center_col]
        center_count = np.count_nonzero(center_array == player)
        score += center_count * 3.0

        # Score all windows of length 4
        # Horizontal
        for row in range(ROWS):
            row_array = grid[row, :]
            for col in range(COLS - CONNECT_N + 1):
                window = list(row_array[col:col + CONNECT_N])
                score += self._score_window(window, player, opponent)

        # Vertical
        for col in range(COLS):
            col_array = grid[:, col]
            for row in range(ROWS - CONNECT_N + 1):
                window = list(col_array[row:row + CONNECT_N])
                score += self._score_window(window, player, opponent)

        # Positive diagonal (\)
        for row in range(ROWS - CONNECT_N + 1):
            for col in range(COLS - CONNECT_N + 1):
                window = [grid[row + i, col + i] for i in range(CONNECT_N)]
                score += self._score_window(window, player, opponent)

        # Negative diagonal (/)
        for row in range(CONNECT_N - 1, ROWS):
            for col in range(COLS - CONNECT_N + 1):
                window = [grid[row - i, col + i] for i in range(CONNECT_N)]
                score += self._score_window(window, player, opponent)

        return score

    @staticmethod
    def _score_window(window: List[int], player: int, opponent: int) -> float:
        """
        Score a 4-cell window.
        """
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(EMPTY)

        # Strong positive patterns
        if player_count == 4:
            return 100000.0
        if player_count == 3 and empty_count == 1:
            return 100.0
        if player_count == 2 and empty_count == 2:
            return 10.0

        # Strong negative patterns (must defend)
        if opponent_count == 4:
            return -100000.0
        if opponent_count == 3 and empty_count == 1:
            return -120.0
        if opponent_count == 2 and empty_count == 2:
            return -12.0

        return 0.0
