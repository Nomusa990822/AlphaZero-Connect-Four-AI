"""
AlphaZero-style agent that selects moves using neural-guided MCTS.
"""

from __future__ import annotations

import torch

from src.agents.base_agent import BaseAgent
from src.core.game import ConnectFourGame
from src.neural.network import AlphaZeroNet
from src.search.mcts import MCTS


class AlphaZeroAgent(BaseAgent):
    """
    Agent wrapper around neural-guided MCTS.
    """

    def __init__(
        self,
        model: AlphaZeroNet,
        simulations: int = 100,
        c_puct: float = 1.5,
        add_root_noise: bool = False,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        name: str = "AlphaZeroAgent",
    ) -> None:
        super().__init__(name=name)
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.add_root_noise = add_root_noise
        self.device = torch.device(device)
        self.seed = seed

    def select_move(self, game: ConnectFourGame) -> int:
        """
        Select a move using neural-guided MCTS.
        """
        mcts = MCTS(
            model=self.model,
            simulations=self.simulations,
            c_puct=self.c_puct,
            add_root_noise=self.add_root_noise,
            device=self.device,
            seed=self.seed,
        )
        result = mcts.search(game)
        return result.selected_move
