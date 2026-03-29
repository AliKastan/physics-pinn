"""PINN for simple pendulum: maps t -> (theta, omega)."""

import torch
from .base_pinn import BasePINN
from ..physics.equations import pendulum_residual


class PINNPendulum(BasePINN):
    """
    Neural network that maps time t -> (theta, omega) for a simple pendulum.

    Architecture: 3 hidden layers x 64 neurons, tanh activation.
    Output: [theta(t), omega(t)] — angular displacement and velocity.
    """

    def __init__(self, hidden_size=64, num_hidden_layers=3):
        super().__init__(
            input_size=1,
            output_size=2,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        )

    def physics_loss(self, t_collocation, g=9.81, L=1.0):
        """Compute mean squared pendulum ODE residual."""
        t_collocation.requires_grad_(True)
        output = self(t_collocation)
        theta = output[:, 0:1]
        omega = output[:, 1:2]

        dtheta_dt = torch.autograd.grad(
            theta, t_collocation,
            grad_outputs=torch.ones_like(theta),
            create_graph=True,
        )[0]
        domega_dt = torch.autograd.grad(
            omega, t_collocation,
            grad_outputs=torch.ones_like(omega),
            create_graph=True,
        )[0]

        res1, res2 = pendulum_residual(dtheta_dt, domega_dt, theta, omega, g, L)
        return torch.mean(res1 ** 2) + torch.mean(res2 ** 2)

    def compute_residual(self, t_points, g=9.81, L=1.0):
        """Per-point residual magnitude (detached) for adaptive sampling."""
        t_points = t_points.clone().requires_grad_(True)
        output = self(t_points)
        theta = output[:, 0:1]
        omega = output[:, 1:2]

        dtheta_dt = torch.autograd.grad(
            theta, t_points, torch.ones_like(theta),
            create_graph=False, retain_graph=True,
        )[0]
        domega_dt = torch.autograd.grad(
            omega, t_points, torch.ones_like(omega), create_graph=False,
        )[0]

        res1, res2 = pendulum_residual(dtheta_dt, domega_dt, theta, omega, g, L)
        return (res1.squeeze() ** 2 + res2.squeeze() ** 2).detach()
