"""Abstract base class for all Physics-Informed Neural Networks."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BasePINN(nn.Module, ABC):
    """
    Abstract base class for PINNs.

    All PINNs map some input (typically time) to a state vector,
    and are trained by minimizing a physics-informed loss that
    penalizes ODE/PDE residuals at collocation points.

    Subclasses must implement:
        - forward(): the neural network forward pass
        - physics_loss(): the ODE/PDE residual loss
        - compute_residual(): per-point residual for adaptive sampling
        - train_step(): a single training iteration
    """

    def __init__(self, input_size, output_size, hidden_size=64,
                 num_hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)

    @abstractmethod
    def physics_loss(self, t_collocation, **params):
        """Compute mean squared ODE/PDE residual at collocation points."""
        ...

    @abstractmethod
    def compute_residual(self, t_points, **params):
        """Compute per-point residual magnitude (detached) for adaptive sampling."""
        ...

    def train_step(self, optimizer, t_collocation, ic_loss_fn, ic_weight=20.0,
                   **physics_params):
        """
        Execute one training step: forward, loss, backward, step.

        Args:
            optimizer: torch optimizer
            t_collocation: collocation points tensor
            ic_loss_fn: callable returning scalar IC loss
            ic_weight: weight for IC loss
            **physics_params: passed to physics_loss()

        Returns:
            dict with 'total', 'physics', 'ic' loss values
        """
        optimizer.zero_grad()
        loss_phys = self.physics_loss(t_collocation, **physics_params)
        loss_ic = ic_loss_fn()
        total_loss = loss_phys + ic_weight * loss_ic
        total_loss.backward()
        optimizer.step()
        return {
            'total': total_loss.item(),
            'physics': loss_phys.item(),
            'ic': loss_ic.item(),
        }
