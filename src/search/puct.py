import math

from src.search.node import Node


def puct_score(parent: Node, child: Node, c_puct: float) -> float:
    """
    AlphaZero-style PUCT score.

    Q + U
    where:
        Q = child mean value
        U = c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
    """
    q_value = child.value()
    u_value = c_puct * child.prior * math.sqrt(max(parent.visit_count, 1)) / (1 + child.visit_count)
    return q_value + u_value
