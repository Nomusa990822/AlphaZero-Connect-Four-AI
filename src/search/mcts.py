from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.core.constants import COLS, DRAW, PLAYER_ONE, PLAYER_TWO
from src.core.move_encoder import normalize_policy
from src.core.state_encoder import encode_state_tensor
from src.neural.network import AlphaZeroNet
from src.search.dirichlet_noise import add_dirichlet_noise
from src.search.node import Node
from src.search.puct import puct_score


@dataclass
class MCTSResult:
    selected_move: int
    visit_counts: dict[int, int]
    policy_target: np.ndarray
    root_value: float


class MCTS:
    """
    AlphaZero-style MCTS with tactical safeguards.

    Upgrades:
    - immediate winning move detection
    - immediate blocking move detection
    - neural priors + value head
    - visit-count policy targets
    """

    def __init__(
        self,
        model: AlphaZeroNet,
        simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        add_root_noise: bool = False,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be at least 1.")
        if c_puct <= 0:
            raise ValueError("c_puct must be positive.")

        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.add_root_noise = add_root_noise
        self.device = torch.device(device)
        self.seed = seed

    def search(self, game) -> MCTSResult:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available for MCTS.")

        # -------------------------------------------------
        # Tactical overrides: immediate win / immediate block
        # -------------------------------------------------
        winning_move = self._find_immediate_winning_move(game, game.current_player)
        if winning_move is not None:
            return self._build_forced_result(winning_move, valid_moves, root_value=1.0)

        opponent = PLAYER_TWO if game.current_player == PLAYER_ONE else PLAYER_ONE
        blocking_move = self._find_immediate_winning_move(game, opponent)
        if blocking_move is not None:
            return self._build_forced_result(blocking_move, valid_moves, root_value=0.5)

        # -------------------------------------------------
        # Standard neural-guided MCTS
        # -------------------------------------------------
        root = Node(game=game.copy())

        root_priors, root_value = self._evaluate_state(root.game)

        if self.add_root_noise:
            root_priors = add_dirichlet_noise(
                root_priors,
                alpha=self.dirichlet_alpha,
                epsilon=self.dirichlet_epsilon,
                seed=self.seed,
            )

        root.expand(root_priors)
        root.update(root_value)

        for _ in range(self.simulations):
            node = root
            search_path = [node]

            while node.expanded() and not node.is_terminal() and node.children:
                node = self._select_child(node)
                search_path.append(node)

            if node.is_terminal():
                value = self._terminal_value(node)
            else:
                priors, value = self._evaluate_state(node.game)
                node.expand(priors)

            self._backpropagate(search_path, value)

        visit_counts = root.child_visit_counts()
        policy_target = self._visit_count_policy(visit_counts)
        selected_move = int(np.argmax(policy_target))

        return MCTSResult(
            selected_move=selected_move,
            visit_counts=visit_counts,
            policy_target=policy_target,
            root_value=float(root.value()),
        )

    def _build_forced_result(
        self,
        forced_move: int,
        valid_moves: list[int],
        root_value: float,
    ) -> MCTSResult:
        """
        Build a deterministic MCTSResult for forced tactical moves.
        """
        visit_counts = {move: (1 if move == forced_move else 0) for move in valid_moves}
        policy_target = np.zeros(COLS, dtype=np.float32)
        policy_target[forced_move] = 1.0

        return MCTSResult(
            selected_move=forced_move,
            visit_counts=visit_counts,
            policy_target=policy_target,
            root_value=float(root_value),
        )

    def _find_immediate_winning_move(self, game, player: int) -> int | None:
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
        Prefer central columns first.
        """
        center = COLS // 2
        return sorted(moves, key=lambda col: abs(col - center))

    def _select_child(self, node: Node) -> Node:
        best_score = float("-inf")
        best_child: Optional[Node] = None

        for child in node.children.values():
            score = puct_score(node, child, self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise ValueError("No child available during MCTS selection.")

        return best_child

    def _evaluate_state(self, game) -> tuple[dict[int, float], float]:
        state_tensor = encode_state_tensor(game, device=self.device)

        with torch.no_grad():
            policy_probs, value = self.model.predict(state_tensor)

        raw_policy = policy_probs[0].detach().cpu().numpy().astype(np.float32)
        valid_moves = game.get_valid_moves()
        normalized = normalize_policy(raw_policy, valid_moves)

        priors = {move: float(normalized[move]) for move in valid_moves}
        return priors, float(value[0].item())

    def _terminal_value(self, node: Node) -> float:
        winner = node.game.winner
        current_player = node.game.current_player

        if winner == DRAW or winner is None:
            return 0.0
        if winner == current_player:
            return 1.0
        return -1.0

    def _backpropagate(self, search_path: list[Node], value: float) -> None:
        for node in reversed(search_path):
            node.update(value)
            value = -value

    def _visit_count_policy(self, visit_counts: dict[int, int]) -> np.ndarray:
        policy = np.zeros(COLS, dtype=np.float32)

        total_visits = sum(visit_counts.values())
        if total_visits <= 0:
            raise ValueError("Cannot build visit-count policy with zero visits.")

        for move, count in visit_counts.items():
            policy[move] = count / total_visits

        return policy
