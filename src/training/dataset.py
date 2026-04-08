"""
Self-play data generation for AlphaZero-style Connect Four.

This version uses:
- current neural network for policy/value prediction
- legal move masking
- temperature-based move selection
- final game outcome to label training examples

For Stage 5, MCTS is not yet network-guided in a full AlphaZero way.
Instead, we use the network policy directly to generate playable
training data and establish the training loop cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from src.core.game import ConnectFourGame
from src.core.move_encoder import legal_moves_mask, normalize_policy
from src.core.state_encoder import encode_state, encode_state_tensor
from src.neural.network import AlphaZeroNet


@dataclass
class SelfPlayConfig:
    num_games: int = 10
    temperature: float = 1.0
    seed: int | None = None


class SelfPlay:
    """
    Generates self-play games and training samples.
    """

    def __init__(
        self,
        model: AlphaZeroNet,
        device: str | torch.device = "cpu",
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.model = model
        self.device = torch.device(device)
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def generate_games(self, num_games: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate self-play data from multiple games.

        Returns:
            List of (state, policy, value) samples.
        """
        if num_games < 1:
            raise ValueError("num_games must be at least 1.")

        all_samples: list[tuple[np.ndarray, np.ndarray, float]] = []

        for _ in range(num_games):
            game_samples = self.play_single_game()
            all_samples.extend(game_samples)

        return all_samples

    def play_single_game(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Play one self-play game and return labeled training samples.

        During the game, we store:
            (encoded_state, move_policy, player_to_move)

        After the game ends, we convert these into:
            (encoded_state, move_policy, final_value_from_that_player_perspective)
        """
        game = ConnectFourGame()
        trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []

        while not game.done:
            state = encode_state(game)
            policy = self._predict_policy(game)

            move = self._sample_move(policy)
            player = game.current_player

            trajectory.append((state, policy, player))
            game.apply_move(move)

        winner = game.winner
        labeled_samples: list[tuple[np.ndarray, np.ndarray, float]] = []

        for state, policy, player in trajectory:
            value = self._outcome_for_player(winner, player)
            labeled_samples.append((state, policy, value))

        return labeled_samples

    def _predict_policy(self, game: ConnectFourGame) -> np.ndarray:
        """
        Predict a legal move distribution from the network.
        """
        state_tensor = encode_state_tensor(game, device=self.device)

        with torch.no_grad():
            policy_probs, _ = self.model.predict(state_tensor)

        raw_policy = policy_probs[0].detach().cpu().numpy().astype(np.float32)
        valid_moves = game.get_valid_moves()
        policy = normalize_policy(raw_policy, valid_moves)

        if self.temperature != 1.0:
            policy = self._apply_temperature(policy, valid_moves)

        return policy.astype(np.float32)

    def _apply_temperature(self, policy: np.ndarray, valid_moves: list[int]) -> np.ndarray:
        """
        Apply temperature scaling to a policy over valid moves.
        """
        adjusted = np.zeros_like(policy, dtype=np.float32)

        valid_probs = np.array([policy[m] for m in valid_moves], dtype=np.float32)
        valid_probs = np.power(valid_probs, 1.0 / self.temperature)

        total = valid_probs.sum()
        if total <= 0:
            valid_probs = np.ones_like(valid_probs) / len(valid_probs)
        else:
            valid_probs = valid_probs / total

        for move, prob in zip(valid_moves, valid_probs):
            adjusted[move] = prob

        return adjusted

    def _sample_move(self, policy: np.ndarray) -> int:
        """
        Sample a move from the policy distribution.
        """
        moves = np.arange(len(policy))
        return int(self.rng.choice(moves, p=policy))

    @staticmethod
    def _outcome_for_player(winner: int | None, player: int) -> float:
        """
        Convert final winner into value from the given player's perspective.
        """
        if winner == 0:
            return 0.0
        if winner == player:
            return 1.0
        return -1.0
