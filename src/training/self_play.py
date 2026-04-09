"""
AlphaZero-style self-play using neural-guided MCTS.

Upgrades:
- temperature decay
- optional horizontal symmetry augmentation
- deterministic late-game move selection
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
    early_temperature: float = 1.0
    late_temperature: float = 0.1
    temperature_drop_move: int = 6
    add_root_noise: bool = True
    augment_symmetry: bool = True
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
        early_temperature: float = 1.0,
        late_temperature: float = 0.1,
        temperature_drop_move: int = 6,
        augment_symmetry: bool = True,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be at least 1.")
        if c_puct <= 0:
            raise ValueError("c_puct must be positive.")
        if early_temperature <= 0 or late_temperature <= 0:
            raise ValueError("temperatures must be positive.")
        if temperature_drop_move < 0:
            raise ValueError("temperature_drop_move must be non-negative.")

        self.model = model
        self.device = torch.device(device)
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.add_root_noise = add_root_noise
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.early_temperature = early_temperature
        self.late_temperature = late_temperature
        self.temperature_drop_move = temperature_drop_move
        self.augment_symmetry = augment_symmetry

    def generate_games(self, num_games: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
        if num_games < 1:
            raise ValueError("num_games must be at least 1.")

        all_samples: list[tuple[np.ndarray, np.ndarray, float]] = []

        for game_idx in range(num_games):
            samples = self.play_single_game(game_idx=game_idx)
            all_samples.extend(samples)

        return all_samples

    def play_single_game(self, game_idx: int = 0) -> list[tuple[np.ndarray, np.ndarray, float]]:
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

            current_temperature = self._current_temperature(move_number)
            move = self._sample_move(policy_target, current_temperature)
            player = game.current_player

            trajectory.append((state, policy_target, player))
            game.apply_move(move)
            move_number += 1

        winner = game.winner
        samples: list[tuple[np.ndarray, np.ndarray, float]] = []

        for state, policy, player in trajectory:
            value = self._outcome_for_player(winner, player)

            samples.append((state, policy, value))

            if self.augment_symmetry:
                mirrored_state = self._mirror_state(state)
                mirrored_policy = self._mirror_policy(policy)
                samples.append((mirrored_state, mirrored_policy, value))

        return samples

    def _current_temperature(self, move_number: int) -> float:
        if move_number < self.temperature_drop_move:
            return self.early_temperature
        return self.late_temperature

    def _sample_move(self, policy: np.ndarray, temperature: float) -> int:
        """
        Sample move using temperature-adjusted policy.
        Deterministic in late game for stronger play.
        """
        if temperature < 0.2:
            return int(np.argmax(policy))

        adjusted = self._apply_temperature(policy, temperature)
        moves = np.arange(len(adjusted))
        return int(self.rng.choice(moves, p=adjusted))

    def _apply_temperature(self, policy: np.ndarray, temperature: float) -> np.ndarray:
        adjusted = np.asarray(policy, dtype=np.float32).copy()

        if adjusted.ndim != 1:
            raise ValueError("policy must be a 1D array.")

        total = adjusted.sum()
        if total <= 0:
            raise ValueError("Policy sum must be positive.")

        adjusted = adjusted / total

        if temperature == 1.0:
            return adjusted

        positive = adjusted > 0
        adjusted[positive] = np.power(adjusted[positive], 1.0 / temperature)

        total = adjusted.sum()
        if total <= 0:
            raise ValueError("Temperature-adjusted policy sum must be positive.")

        return adjusted / total

    @staticmethod
    def _mirror_state(state: np.ndarray) -> np.ndarray:
        """
        Mirror a state horizontally.
        State shape: (C, H, W)
        """
        return np.flip(state, axis=2).copy().astype(np.float32)

    @staticmethod
    def _mirror_policy(policy: np.ndarray) -> np.ndarray:
        """
        Mirror policy over columns.
        Example: [p0, p1, ..., p6] -> [p6, ..., p1, p0]
        """
        return np.flip(policy).copy().astype(np.float32)

    @staticmethod
    def _outcome_for_player(winner: int | None, player: int) -> float:
        if winner == 0:
            return 0.0
        if winner == player:
            return 1.0
        return -1.0
