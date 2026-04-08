from __future__ import annotations

import random

from src.search.node import Node


class MCTS:
    """
    Monte Carlo Tree Search.
    """

    def __init__(self, simulations: int = 100, c_puct: float = 1.4):
        self.simulations = simulations
        self.c_puct = c_puct

    def search(self, game):
        root = Node(game.copy())

        for _ in range(self.simulations):
            node = root

            # 1. Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.c_puct)

            # 2. Expansion
            if not node.is_terminal():
                node = node.expand()

            # 3. Simulation
            value = self.rollout(node.game)

            # 4. Backpropagation
            self.backpropagate(node, value)

        return self.select_best_move(root)

    def rollout(self, game):
        """
        Random simulation until terminal state.
        """
        temp_game = game.copy()

        while not temp_game.done:
            moves = temp_game.get_valid_moves()
            move = random.choice(moves)
            temp_game.apply_move(move)

        winner = temp_game.winner

        if winner == 0:  # draw
            return 0
        elif winner == game.current_player:
            return 1
        else:
            return -1

    def backpropagate(self, node, value):
        while node is not None:
            node.update(value)
            value = -value
            node = node.parent

    def select_best_move(self, root):
        best_move = max(
            root.children.items(),
            key=lambda item: item[1].visit_count,
        )[0]

        return best_move
