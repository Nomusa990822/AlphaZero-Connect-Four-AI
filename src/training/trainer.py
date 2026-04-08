"""
Training utilities for AlphaZero-style Connect Four.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from src.neural.losses import alphazero_loss
from src.neural.network import AlphaZeroNet


@dataclass
class TrainerConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5


class Trainer:
    """
    Handles neural network optimization.
    """

    def __init__(
        self,
        model: AlphaZeroNet,
        device: str | torch.device = "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of average losses.
        """
        self.model.train()

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        for states, target_policies, target_values in dataloader:
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)

            self.optimizer.zero_grad()

            policy_logits, pred_values = self.model(states)
            total_loss, p_loss, v_loss = alphazero_loss(
                policy_logits=policy_logits,
                pred_value=pred_values,
                target_policy=target_policies,
                target_value=target_values,
            )

            total_loss.backward()
            self.optimizer.step()

            total_loss_sum += float(total_loss.item())
            policy_loss_sum += float(p_loss.item())
            value_loss_sum += float(v_loss.item())
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Dataloader produced zero batches.")

        return {
            "total_loss": total_loss_sum / num_batches,
            "policy_loss": policy_loss_sum / num_batches,
            "value_loss": value_loss_sum / num_batches,
        }

    @torch.no_grad()
    def evaluate_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate for one epoch without optimization.
        """
        self.model.eval()

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        for states, target_policies, target_values in dataloader:
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)

            policy_logits, pred_values = self.model(states)
            total_loss, p_loss, v_loss = alphazero_loss(
                policy_logits=policy_logits,
                pred_value=pred_values,
                target_policy=target_policies,
                target_value=target_values,
            )

            total_loss_sum += float(total_loss.item())
            policy_loss_sum += float(p_loss.item())
            value_loss_sum += float(v_loss.item())
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Dataloader produced zero batches.")

        return {
            "total_loss": total_loss_sum / num_batches,
            "policy_loss": policy_loss_sum / num_batches,
            "value_loss": value_loss_sum / num_batches,
        }
