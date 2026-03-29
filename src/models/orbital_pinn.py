"""PINN for two-body orbital mechanics: maps t -> (x, y, vx, vy)."""

import torch
from .base_pinn import BasePINN
from ..physics.equations import orbital_residual


class PINNOrbital(BasePINN):
    """
    Neural network that maps time t -> (x, y, vx, vy) for a two-body orbit.

    Architecture: 4 hidden layers x 128 neurons, tanh activation.
    Wider/deeper than pendulum due to 4 output variables and the
    highly nonlinear 1/r^3 gravitational force.
    """

    def __init__(self, hidden_size=128, num_hidden_layers=4):
        super().__init__(
            input_size=1,
            output_size=4,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        )

    def physics_loss(self, t_collocation, GM=1.0):
        """Compute mean squared gravitational ODE residual."""
        t_collocation.requires_grad_(True)
        output = self(t_collocation)
        x = output[:, 0:1]
        y = output[:, 1:2]
        vx = output[:, 2:3]
        vy = output[:, 3:4]

        ones = torch.ones_like(x)
        dx_dt = torch.autograd.grad(x, t_collocation, ones, create_graph=True)[0]
        dy_dt = torch.autograd.grad(y, t_collocation, ones, create_graph=True)[0]
        dvx_dt = torch.autograd.grad(vx, t_collocation, ones, create_graph=True)[0]
        dvy_dt = torch.autograd.grad(vy, t_collocation, ones, create_graph=True)[0]

        res_x, res_y, res_vx, res_vy = orbital_residual(
            dx_dt, dy_dt, dvx_dt, dvy_dt, x, y, vx, vy, GM
        )
        return (torch.mean(res_x ** 2) + torch.mean(res_y ** 2) +
                torch.mean(res_vx ** 2) + torch.mean(res_vy ** 2))

    def compute_residual(self, t_points, GM=1.0):
        """Per-point residual magnitude (detached) for adaptive sampling."""
        t_points = t_points.clone().requires_grad_(True)
        output = self(t_points)
        x = output[:, 0:1]
        y = output[:, 1:2]
        vx = output[:, 2:3]
        vy = output[:, 3:4]

        ones = torch.ones_like(x)
        dx_dt = torch.autograd.grad(x, t_points, ones, create_graph=False, retain_graph=True)[0]
        dy_dt = torch.autograd.grad(y, t_points, ones, create_graph=False, retain_graph=True)[0]
        dvx_dt = torch.autograd.grad(vx, t_points, ones, create_graph=False, retain_graph=True)[0]
        dvy_dt = torch.autograd.grad(vy, t_points, ones, create_graph=False)[0]

        res_x, res_y, res_vx, res_vy = orbital_residual(
            dx_dt, dy_dt, dvx_dt, dvy_dt, x, y, vx, vy, GM
        )
        return ((res_x ** 2 + res_y ** 2 + res_vx ** 2 + res_vy ** 2)
                .squeeze().detach())
