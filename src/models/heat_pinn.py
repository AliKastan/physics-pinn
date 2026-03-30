"""
PINN for the 1D heat equation: maps (x, t) -> u(x, t).

PDE:     du/dt = alpha * d²u/dx²
Domain:  x in [0, L_rod], t in [0, t_max]
BCs:     u(0, t) = T_left,  u(L_rod, t) = T_right   (Dirichlet)
IC:      u(x, 0) = f(x)     (user-specified)

The network learns the full spatiotemporal temperature field from the
PDE, boundary conditions, and initial condition alone — no simulation
data required.
"""

import torch
import torch.nn as nn
import numpy as np


class HeatPINN(nn.Module):
    """
    Fully-connected network mapping (x, t) -> scalar u.

    Architecture: input (2) -> 4 hidden layers x 64 tanh -> output (1).
    Deeper than ODE PINNs because the solution lives in a 2D domain and
    we need second-order spatial derivatives (d²u/dx²).
    """

    def __init__(self, hidden_size=64, num_hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        """
        Args:
            x: spatial coordinate (N, 1)
            t: temporal coordinate (N, 1)
        Returns:
            u: temperature field (N, 1)
        """
        return self.network(torch.cat([x, t], dim=1))

    def physics_loss(self, x_int, t_int, alpha=0.01):
        """
        PDE residual: du/dt - alpha * d²u/dx² = 0.

        Args:
            x_int, t_int: interior collocation points (N, 1), requires_grad set here
            alpha: thermal diffusivity

        Returns:
            Scalar mean squared PDE residual
        """
        x_int.requires_grad_(True)
        t_int.requires_grad_(True)
        u = self(x_int, t_int)

        du_dt = torch.autograd.grad(
            u, t_int, grad_outputs=torch.ones_like(u), create_graph=True,
        )[0]
        du_dx = torch.autograd.grad(
            u, x_int, grad_outputs=torch.ones_like(u), create_graph=True,
        )[0]
        d2u_dx2 = torch.autograd.grad(
            du_dx, x_int, grad_outputs=torch.ones_like(du_dx), create_graph=True,
        )[0]

        residual = du_dt - alpha * d2u_dx2
        return torch.mean(residual ** 2)

    def boundary_loss(self, n_bc, t_max, T_left=0.0, T_right=0.0, L_rod=1.0):
        """
        Dirichlet BC loss: u(0, t) = T_left, u(L_rod, t) = T_right.
        """
        t_bc = torch.rand(n_bc, 1) * t_max

        u_left = self(torch.zeros(n_bc, 1), t_bc)
        u_right = self(torch.full((n_bc, 1), L_rod), t_bc)

        loss_left = torch.mean((u_left - T_left) ** 2)
        loss_right = torch.mean((u_right - T_right) ** 2)
        return loss_left + loss_right

    def ic_loss(self, n_ic, ic_fn, L_rod=1.0):
        """
        Initial condition loss: u(x, 0) = ic_fn(x).

        Args:
            n_ic: number of IC collocation points
            ic_fn: callable mapping x tensor (N, 1) -> u tensor (N, 1)
            L_rod: rod length
        """
        x_ic = torch.rand(n_ic, 1) * L_rod
        t_ic = torch.zeros(n_ic, 1)
        u_pred = self(x_ic, t_ic)
        u_true = ic_fn(x_ic)
        return torch.mean((u_pred - u_true) ** 2)

    def compute_residual(self, x_pts, t_pts, alpha=0.01):
        """Per-point PDE residual magnitude (detached) for diagnostics."""
        x_pts = x_pts.clone().requires_grad_(True)
        t_pts = t_pts.clone().requires_grad_(True)
        u = self(x_pts, t_pts)

        du_dt = torch.autograd.grad(
            u, t_pts, torch.ones_like(u),
            create_graph=False, retain_graph=True,
        )[0]
        du_dx = torch.autograd.grad(
            u, x_pts, torch.ones_like(u),
            create_graph=True, retain_graph=True,
        )[0]
        d2u_dx2 = torch.autograd.grad(
            du_dx, x_pts, torch.ones_like(du_dx), create_graph=False,
        )[0]

        residual = du_dt - alpha * d2u_dx2
        return (residual ** 2).squeeze().detach()


# =========================================================================
# Initial condition functions
# =========================================================================

def ic_sine(x, L_rod=1.0):
    """IC: u(x, 0) = sin(pi * x / L_rod)."""
    return torch.sin(np.pi * x / L_rod)


def ic_step(x, L_rod=1.0):
    """IC: step function — 1.0 in the middle third, 0 elsewhere."""
    return ((x > L_rod / 3) & (x < 2 * L_rod / 3)).float()


def ic_gaussian(x, L_rod=1.0):
    """IC: Gaussian pulse centred at L_rod/2."""
    mu = L_rod / 2
    sigma = L_rod / 10
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


IC_FUNCTIONS = {
    'sine': ic_sine,
    'step': ic_step,
    'gaussian': ic_gaussian,
}


# =========================================================================
# Training
# =========================================================================

def train_heat_pinn(alpha=0.01, L_rod=1.0, t_max=1.0,
                    T_left=0.0, T_right=0.0,
                    ic_type='sine',
                    n_interior=2000, n_bc=200, n_ic=200,
                    epochs=8000, lr=1e-3,
                    bc_weight=10.0, ic_weight=10.0,
                    hidden_size=64, num_hidden_layers=4,
                    verbose=True):
    """
    Train a HeatPINN on the 1D heat equation.

    Returns:
        model: trained HeatPINN
        loss_history: per-epoch total loss
    """
    model = HeatPINN(hidden_size=hidden_size,
                     num_hidden_layers=num_hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6,
    )

    ic_fn = lambda x: IC_FUNCTIONS[ic_type](x, L_rod=L_rod)
    loss_history = []

    if verbose:
        print(f"Training HeatPINN  (alpha={alpha}, L={L_rod}, IC={ic_type})")
        print(f"  Domain: x in [0, {L_rod}], t in [0, {t_max}]")
        print(f"  BCs: u(0,t)={T_left}, u({L_rod},t)={T_right}")
        print(f"  Interior pts: {n_interior}, Epochs: {epochs}")
        print("-" * 55)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Interior collocation
        x_int = torch.rand(n_interior, 1) * L_rod
        t_int = torch.rand(n_interior, 1) * t_max
        loss_pde = model.physics_loss(x_int, t_int, alpha=alpha)

        # Boundary conditions
        loss_bc = model.boundary_loss(n_bc, t_max, T_left, T_right, L_rod)

        # Initial condition
        loss_ic = model.ic_loss(n_ic, ic_fn, L_rod)

        total = loss_pde + bc_weight * loss_bc + ic_weight * loss_ic
        total.backward()
        optimizer.step()
        scheduler.step(total.item())
        loss_history.append(total.item())

        if verbose and (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | "
                  f"Loss: {total.item():.6f} | "
                  f"PDE: {loss_pde.item():.6f} | "
                  f"BC: {loss_bc.item():.6f} | "
                  f"IC: {loss_ic.item():.6f}")

    if verbose:
        print("-" * 55)
        print(f"Final loss: {loss_history[-1]:.6f}")

    return model, loss_history


# =========================================================================
# Analytical solution (Fourier series, homogeneous Dirichlet BCs)
# =========================================================================

def heat_analytical(x, t, alpha=0.01, L_rod=1.0, ic_type='sine', n_terms=50):
    """
    Fourier series solution of the 1D heat equation with u(0,t)=u(L,t)=0.

    For IC = sin(pi*x/L), the exact single-mode solution is:
        u(x,t) = exp(-alpha*(pi/L)^2 * t) * sin(pi*x/L)

    For general ICs the series is:
        u(x,t) = sum_n  b_n * exp(-alpha*(n*pi/L)^2 * t) * sin(n*pi*x/L)
    where b_n = (2/L) * integral_0^L f(x)*sin(n*pi*x/L) dx.

    For the step and gaussian ICs we compute b_n numerically.
    """
    if ic_type == 'sine':
        # Exact single-term solution
        return np.exp(-alpha * (np.pi / L_rod) ** 2 * t) * np.sin(np.pi * x / L_rod)

    # General: numerical Fourier coefficients
    n_quad = 1000
    x_quad = np.linspace(0, L_rod, n_quad)
    dx = x_quad[1] - x_quad[0]

    # Evaluate IC
    if ic_type == 'step':
        f_x = ((x_quad > L_rod / 3) & (x_quad < 2 * L_rod / 3)).astype(float)
    elif ic_type == 'gaussian':
        mu = L_rod / 2
        sigma = L_rod / 10
        f_x = np.exp(-((x_quad - mu) ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown IC type: {ic_type}")

    u = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        basis = np.sin(n * np.pi * x_quad / L_rod)
        b_n = (2 / L_rod) * np.trapezoid(f_x * basis, dx=dx)
        decay = np.exp(-alpha * (n * np.pi / L_rod) ** 2 * t)
        u = u + b_n * decay * np.sin(n * np.pi * x / L_rod)

    return u
