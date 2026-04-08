from __future__ import annotations

import random

from src.core.constants import PLAYER_ONE, PLAYER_TWO
from src.core.game import ConnectFourGame
from src.search.node import Node


class MCTS:
    """
    Monte Carlo Tree Search for Connect Four.

    This Stage 3 version includes tactical safeguards:
    1. Play an immediate winning move if available.
    2. Block the opponent's immediate winning move if necessary.
    3. Otherwise run standard MCTS with random rollouts.

    This makes the baseline search much more reliable for tactical positions.
    """

    def __init__(self, simulations: int = 100, c_puct: float = 1.4, seed: int | None = None):
        if simulations < 1:
            raise ValueError("simulations must be at least 1.")
        if c_puct <= 0:
            raise ValueError("c_puct must be positive.")

        self.simulations = simulations
        self.c_puct = c_puct
        self._rng = random.Random(seed)

    def search(self, game: ConnectFourGame) -> int:
        """
        Run MCTS from the current position and return the selected move.
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available for MCTS.")

        # 1. Tactical immediate win
        winning_move = self._find_immediate_winning_move(game, game.current_player)
        if winning_move is not None:
            return winning_move

        # 2. Tactical immediate block
        opponent = PLAYER_TWO if game.current_player == PLAYER_ONE else PLAYER_ONE
        blocking_move = self._find_immediate_winning_move(game, opponent)
        if blocking_move is not None:
            return blocking_move

        # 3. Standard MCTS
        root = Node(game.copy())

        for _ in range(self.simulations):
            node = root

            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.c_puct)

            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation
            value = self.rollout(node.game)

            # Backpropagation
            self.backpropagate(node, value)

        return self.select_best_move(root)

    def rollout(self, game: ConnectFourGame) -> float:
        """
        Play a random simulation until the game ends.

        Returns:
            +1 if the rollout result is good for the player to move at 'game'
            -1 if bad
             0 for draw
        """
        temp_game = game.copy()
        root_player = temp_game.current_player

        while not temp_game.done:
            valid_moves = temp_game.get_valid_moves()

            # Rollout tactical bias:
            # take immediate win if possible
            winning_move = self._find_immediate_winning_move(temp_game, temp_game.current_player)
            if winning_move is not None:
                move = winning_move
            else:
                # otherwise block immediate opponent win if needed
                opponent = PLAYER_TWO if temp_game.current_player == PLAYER_ONE else PLAYER_ONE
                blocking_move = self._find_immediate_winning_move(temp_game, opponent)
                if blocking_move is not None:
                    move = blocking_move
                else:
                    move = self._rng.choice(valid_moves)

            temp_game.apply_move(move)

        if temp_game.winner == 0:
            return 0.0
        if temp_game.winner == root_player:
            return 1.0
        return -1.0

    def backpropagate(self, node: Node, value: float) -> None:
        """
        Backpropagate rollout value up the tree, alternating perspective.
        """
        while node is not None:
            node.update(value)
            value = -value
            node = node.parent

    def select_best_move(self, root: Node) -> int:
        """
        Select the move with the highest visit count.
        Break ties by favoring the more central column.
        """
        if not root.children:
            raise ValueError("Root has no children; cannot select best move.")

        best_visit_count = max(child.visit_count for child in root.children.values())
        best_moves = [
            move for move, child in root.children.items()
            if child.visit_count == best_visit_count
        ]

        return self._choose_center_preferred(best_moves)

    def _find_immediate_winning_move(self, game: ConnectFourGame, player: int) -> int | None:
        """
        Return a move that gives 'player' an immediate win, if one exists.
        """
        for move in self._ordered_moves(game.get_valid_moves()):
            temp_game = game.copy()
            temp_game.current_player = player
            temp_game.apply_move(move)
            if temp_game.winner == player:
                return move
        return None

    def _ordered_moves(self, moves: list[int]) -> list[int]:
        """
        Order moves by center preference to improve search quality.
        """
        return sorted(moves, key=lambda col: abs(col - 3))

    def _choose_center_preferred(self, moves: list[int]) -> int:
        """
        Prefer the most central move among tied candidates.
        """
        ordered = self._ordered_moves(moves)
        return ordered[0]