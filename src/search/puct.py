import math


def puct_score(parent, child, c_puct: float):
    """
    PUCT formula (simplified for no neural net).
    """

    if child.visit_count == 0:
        q_value = 0
    else:
        q_value = child.value

    u_value = c_puct * math.sqrt(parent.visit_count) / (1 + child.visit_count)

    return q_value + u_value
