from __future__ import annotations

from typing import Dict, Optional

from src.core.game import ConnectFourGame


class Node:
    """
    AlphaZero-style MCTS node.

    Attributes:
        game: game state at this node
        parent: parent node
        move: move that led from parent -> this node
        prior: prior probability from the policy network
        children: dict[move, child_node]
        visit_count: number of visits
        value_sum: accumulated value from simulations
        is_expanded: whether this node has been expanded with priors
    """

    def __init__(
        self,
        game: ConnectFourGame,
        parent: Optional["Node"] = None,
        move: Optional[int] = None,
        prior: float = 0.0,
    ) -> None:
        self.game = game
        self.parent = parent
        self.move = move
        self.prior = float(prior)

        self.children: Dict[int, Node] = {}

        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def expanded(self) -> bool:
        return self.is_expanded

    def is_terminal(self) -> bool:
        return self.game.done

    def value(self) -> float:
        """
        Mean value from this node's perspective.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, priors: dict[int, float]) -> None:
        """
        Expand this node by creating children for all legal moves.

        Args:
            priors: mapping move -> prior probability
        """
        if self.is_terminal():
            self.is_expanded = True
            return

        valid_moves = self.game.get_valid_moves()

        for move in valid_moves:
            if move not in self.children:
                next_game = self.game.copy()
                next_game.apply_move(move)
                self.children[move] = Node(
                    game=next_game,
                    parent=self,
                    move=move,
                    prior=float(priors.get(move, 0.0)),
                )

        self.is_expanded = True

    def update(self, value: float) -> None:
        """
        Update node statistics with a backed-up value.
        """
        self.visit_count += 1
        self.value_sum += float(value)

    def child_visit_counts(self) -> dict[int, int]:
        return {move: child.visit_count for move, child in self.children.items()}

    def child_priors(self) -> dict[int, float]:
        return {move: child.prior for move, child in self.children.items()}
