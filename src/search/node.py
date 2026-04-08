from __future__ import annotations

from typing import Dict, Optional

from src.core.game import ConnectFourGame


class Node:
    """
    MCTS Node.
    """

    def __init__(
        self,
        game: ConnectFourGame,
        parent: Optional["Node"] = None,
        move: Optional[int] = None,
    ):
        self.game = game
        self.parent = parent
        self.move = move

        self.children: Dict[int, Node] = {}

        self.visit_count = 0
        self.value_sum = 0.0

        self.untried_moves = game.get_valid_moves()

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.game.done

    def expand(self) -> "Node":
        move = self.untried_moves.pop()

        next_game = self.game.copy()
        next_game.apply_move(move)

        child = Node(next_game, parent=self, move=move)
        self.children[move] = child

        return child

    def best_child(self, c_puct: float):
        from src.search.puct import puct_score

        best_score = float("-inf")
        best_node = None

        for child in self.children.values():
            score = puct_score(self, child, c_puct)

            if score > best_score:
                best_score = score
                best_node = child

        return best_node

    def update(self, value: float):
        self.visit_count += 1
        self.value_sum += value

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
