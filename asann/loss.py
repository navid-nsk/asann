import torch
import torch.nn as nn
from typing import Callable, Optional
from .config import ASANNConfig
from .model import ASANNModel


class ASANNLoss:
    """ASANN loss function with real complexity cost and Lagrangian multiplier.

    L_total = L_task + lambda_complexity * C(architecture)

    Where C(architecture) is the REAL computational cost of the current architecture —
    not a differentiable approximation. It reflects the actual tensors in the network.

    lambda_complexity is a Lagrangian multiplier updated via dual ascent toward
    a target budget C_target:

        lambda(t+1) = max(0, lambda(t) + rho * (C_current - C_target))

    Surgery decisions are influenced by complexity: when the network is over-budget,
    add-thresholds increase (harder to add) and remove-thresholds decrease (easier to remove).
    """

    def __init__(
        self,
        task_loss_fn: Callable,
        config: ASANNConfig,
    ):
        self.task_loss_fn = task_loss_fn
        self.config = config

        # Lagrangian multiplier for complexity constraint
        self.lambda_complexity = config.complexity_lambda_init
        self.complexity_target = config.complexity_target
        self.dual_step = config.complexity_dual_step

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model: ASANNModel,
    ) -> tuple:
        """Compute total loss = L_task + lambda * (C_current - C_target).

        Returns (total_loss, task_loss, complexity_cost, lambda_complexity)
        for logging and analysis.
        """
        # Task loss (differentiable - drives weight gradients)
        task_loss = self.task_loss_fn(predictions, targets)

        # Architecture complexity cost (real, not differentiable approximation)
        complexity_cost = model.compute_architecture_cost()

        # Total loss: task loss + Lagrangian penalty
        complexity_penalty = self.lambda_complexity * (complexity_cost - self.complexity_target)
        total_loss = task_loss + complexity_penalty

        return total_loss, task_loss, complexity_cost, self.lambda_complexity

    def update_lambda(self, model: ASANNModel):
        """Dual ascent update for the Lagrangian multiplier.

        lambda(t+1) = max(0, lambda(t) + rho * (C_current - C_target))

        This encourages the architecture to respect the complexity budget.
        When over budget, lambda increases, making complexity more expensive.
        When under budget, lambda decreases, allowing more growth.
        """
        current_cost = model.compute_architecture_cost()
        constraint_violation = current_cost - self.complexity_target

        new_lambda = self.lambda_complexity + self.dual_step * constraint_violation
        self.lambda_complexity = max(0.0, min(new_lambda, self.config.complexity_lambda_max))

    def get_budget_status(self, model: ASANNModel) -> dict:
        """Return the current complexity budget status."""
        current_cost = model.compute_architecture_cost()
        return {
            "current_cost": current_cost,
            "target_cost": self.complexity_target,
            "over_budget": current_cost > self.complexity_target,
            "budget_ratio": current_cost / max(self.complexity_target, 1.0),
            "lambda_complexity": self.lambda_complexity,
        }

    def state_dict(self) -> dict:
        """Save loss state for checkpoint resume."""
        return {
            "lambda_complexity": self.lambda_complexity,
            "complexity_target": self.complexity_target,
            "dual_step": self.dual_step,
        }

    def load_state_dict(self, state: dict):
        """Restore loss state from checkpoint."""
        self.lambda_complexity = state["lambda_complexity"]
        self.complexity_target = state["complexity_target"]
        self.dual_step = state["dual_step"]
