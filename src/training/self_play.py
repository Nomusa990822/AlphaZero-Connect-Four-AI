"""
AlphaZero-style self-play using neural-guided MCTS.

Each move stores:
- encoded state
- MCTS visit-count policy target
- player-to-move

After game end, each stored position is labeled with final outcome
from that player's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from src.core.game import ConnectFourGame
from src.core.state_encoder import encode_state
from src.neural.network import AlphaZeroNet
from src.search.mcts import MCTS


@dataclass
class SelfPlayConfig:
    num_games: int = 10
    simulations_per_move: int = 100
    c_puct: float = 1.5
    temperature: float = 1.0
    add_root_noise: bool = True
    seed: int | None = None


class SelfPlay:
    """
    Self-play driver using AlphaZero-style MCTS.
    """

    def __init__(
        self,
        model: AlphaZeroNet,
        device: str | torch.device = "cpu",
        simulations: int = 100,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        add_root_noise: bool = True,
        seed: int | None = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if simulations < 1:
            raise ValueError("simulations must be at least 1.")
        if c_puct <= 0:
            raise ValueError("c_puct must be positive.")

        self.model = model
        self.device = torch.device(device)
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.add_root_noise = add_root_noise
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def generate_games(self, num_games: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate self-play samples from multiple games.
        """
        if num_games < 1:
            raise ValueError("num_games must be at least 1.")

        all_samples: list[tuple[np.ndarray, np.ndarray, float]] = []

        for game_idx in range(num_games):
            samples = self.play_single_game(game_idx=game_idx)
            all_samples.extend(samples)

        return all_samples

    def play_single_game(self, game_idx: int = 0) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Play one self-play game and return labeled training samples.

        Each stored position includes:
            - encoded state
            - MCTS visit-count policy target
            - player to move

        After the game ends, each position is labeled with the final outcome
        from that stored player's perspective.
        """
        game = ConnectFourGame()
        trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []

        move_number = 0

        while not game.done:
            state = encode_state(game)

            mcts = MCTS(
                model=self.model,
                simulations=self.simulations,
                c_puct=self.c_puct,
                add_root_noise=self.add_root_noise,
                device=self.device,
                seed=None if self.seed is None else self.seed + game_idx + move_number,
            )

            result = mcts.search(game)

            policy_target = result.policy_target
            move = self._sample_move(policy_target)
            player = game.current_player

            trajectory.append((state, policy_target, player))
            game.apply_move(move)
            move_number += 1

        winner = game.winner
        samples: list[tuple[np.ndarray, np.ndarray, float]] = []

        for state, policy, player in trajectory:
            value = self._outcome_for_player(winner, player)
            samples.append((state, policy, value))

        return samples

    def _sample_move(self, policy: np.ndarray) -> int:
        """
        Sample a move from a temperature-adjusted policy.
        """
        adjusted = self._apply_temperature(policy)
        moves = np.arange(len(adjusted))
        return int(self.rng.choice(moves, p=adjusted))

    def _apply_temperature(self, policy: np.ndarray) -> np.ndarray:
        """
        Apply temperature to a visit-count policy.
        """
        adjusted = np.asarray(policy, dtype=np.float32).copy()

        if adjusted.ndim != 1:
            raise ValueError("policy must be a 1D array.")

        if self.temperature == 1.0:
            total = adjusted.sum()
            if total <= 0:
                raise ValueError("Policy sum must be positive.")
            return adjusted / total

        positive = adjusted > 0
        adjusted[positive] = np.power(adjusted[positive], 1.0 / self.temperature)

        total = adjusted.sum()
        if total <= 0:
            raise ValueError("Temperature-adjusted policy sum must be positive.")

        return adjusted / total

    @staticmethod
    def _outcome_for_player(winner: int | None, player: int) -> float:
        """
        Convert the game winner into a value from one player's perspective.
        """
        if winner == 0:
            return 0.0
        if winner == player:
            return 1.0
        return -1.0
