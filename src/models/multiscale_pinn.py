"""
Multi-scale PINN with Fourier feature encoding.

Standard PINNs suffer from spectral bias — they learn low-frequency
components first and struggle with high-frequency dynamics.  Fourier
feature encoding maps the raw input through sinusoidal functions at
multiple frequencies before feeding into the MLP, enabling the network
to capture high-frequency content from the start.

    input t  ->  [sin(2*pi*sigma_1*t), cos(2*pi*sigma_1*t), ...,
                  sin(2*pi*sigma_K*t), cos(2*pi*sigma_K*t)]
              ->  MLP  ->  output

Reference:
    Tancik et al., "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import numpy as np
from ..models.base_pinn import BasePINN
from ..physics.equations import pendulum_residual, orbital_residual


class FourierFeatureEncoding(nn.Module):
    """
    Maps scalar input t to a vector of Fourier features.

    Args:
        n_frequencies: number of frequency bands K
        sigma_range: (min_freq, max_freq) for log-uniform spacing
        learnable: if True, frequencies are nn.Parameters that get optimised
    """

    def __init__(self, n_frequencies=16, sigma_range=(1.0, 100.0),
                 learnable=False):
        super().__init__()
        # Log-uniform spacing of frequencies
        sigmas = torch.logspace(
            np.log10(sigma_range[0]),
            np.log10(sigma_range[1]),
            n_frequencies,
        )
        if learnable:
            self.sigmas = nn.Parameter(sigmas)
        else:
            self.register_buffer('sigmas', sigmas)

        self.output_dim = 2 * n_frequencies  # sin + cos for each

    def forward(self, t):
        """
        Args:
            t: (N, 1) input
        Returns:
            (N, 2*K) Fourier features
        """
        # t: (N, 1), sigmas: (K,) -> projected: (N, K)
        projected = 2 * np.pi * t * self.sigmas.unsqueeze(0)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=1)


class FourierPendulumPINN(BasePINN):
    """
    Pendulum PINN with Fourier feature encoding on the time input.

    The raw t is first mapped through FourierFeatureEncoding, then the
    resulting high-dimensional vector feeds into the standard MLP.
    """

    def __init__(self, hidden_size=64, num_hidden_layers=3,
                 n_frequencies=16, sigma_range=(1.0, 50.0),
                 learnable_frequencies=False):
        # We skip BasePINN.__init__ because input_size differs
        nn.Module.__init__(self)

        self.encoding = FourierFeatureEncoding(
            n_frequencies=n_frequencies,
            sigma_range=sigma_range,
            learnable=learnable_frequencies,
        )

        enc_dim = self.encoding.output_dim
        layers = [nn.Linear(enc_dim, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        features = self.encoding(t)
        return self.network(features)

    def physics_loss(self, t_collocation, g=9.81, L=1.0):
        t_collocation.requires_grad_(True)
        output = self(t_collocation)
        theta = output[:, 0:1]
        omega = output[:, 1:2]

        dtheta_dt = torch.autograd.grad(
            theta, t_collocation, torch.ones_like(theta),
            create_graph=True)[0]
        domega_dt = torch.autograd.grad(
            omega, t_collocation, torch.ones_like(omega),
            create_graph=True)[0]

        res1, res2 = pendulum_residual(dtheta_dt, domega_dt, theta, omega, g, L)
        return torch.mean(res1 ** 2) + torch.mean(res2 ** 2)

    def compute_residual(self, t_points, g=9.81, L=1.0):
        t_points = t_points.clone().requires_grad_(True)
        output = self(t_points)
        theta = output[:, 0:1]
        omega = output[:, 1:2]

        dtheta_dt = torch.autograd.grad(
            theta, t_points, torch.ones_like(theta),
            create_graph=False, retain_graph=True)[0]
        domega_dt = torch.autograd.grad(
            omega, t_points, torch.ones_like(omega),
            create_graph=False)[0]

        res1, res2 = pendulum_residual(dtheta_dt, domega_dt, theta, omega, g, L)
        return (res1.squeeze() ** 2 + res2.squeeze() ** 2).detach()


class FourierOrbitalPINN(BasePINN):
    """Orbital PINN with Fourier feature encoding."""

    def __init__(self, hidden_size=128, num_hidden_layers=4,
                 n_frequencies=16, sigma_range=(1.0, 50.0),
                 learnable_frequencies=False):
        nn.Module.__init__(self)

        self.encoding = FourierFeatureEncoding(
            n_frequencies=n_frequencies,
            sigma_range=sigma_range,
            learnable=learnable_frequencies,
        )

        enc_dim = self.encoding.output_dim
        layers = [nn.Linear(enc_dim, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(self.encoding(t))

    def physics_loss(self, t_collocation, GM=1.0):
        t_collocation.requires_grad_(True)
        output = self(t_collocation)
        x, y, vx, vy = output[:, 0:1], output[:, 1:2], output[:, 2:3], output[:, 3:4]
        ones = torch.ones_like(x)
        dx = torch.autograd.grad(x, t_collocation, ones, create_graph=True)[0]
        dy = torch.autograd.grad(y, t_collocation, ones, create_graph=True)[0]
        dvx = torch.autograd.grad(vx, t_collocation, ones, create_graph=True)[0]
        dvy = torch.autograd.grad(vy, t_collocation, ones, create_graph=True)[0]
        rx, ry, rvx, rvy = orbital_residual(dx, dy, dvx, dvy, x, y, vx, vy, GM)
        return (torch.mean(rx ** 2) + torch.mean(ry ** 2) +
                torch.mean(rvx ** 2) + torch.mean(rvy ** 2))

    def compute_residual(self, t_points, GM=1.0):
        t_points = t_points.clone().requires_grad_(True)
        output = self(t_points)
        x, y, vx, vy = output[:, 0:1], output[:, 1:2], output[:, 2:3], output[:, 3:4]
        ones = torch.ones_like(x)
        dx = torch.autograd.grad(x, t_points, ones, create_graph=False, retain_graph=True)[0]
        dy = torch.autograd.grad(y, t_points, ones, create_graph=False, retain_graph=True)[0]
        dvx = torch.autograd.grad(vx, t_points, ones, create_graph=False, retain_graph=True)[0]
        dvy = torch.autograd.grad(vy, t_points, ones, create_graph=False)[0]
        rx, ry, rvx, rvy = orbital_residual(dx, dy, dvx, dvy, x, y, vx, vy, GM)
        return ((rx ** 2 + ry ** 2 + rvx ** 2 + rvy ** 2).squeeze().detach())
