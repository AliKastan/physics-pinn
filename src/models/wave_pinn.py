"""
PINN for the 1D wave equation: maps (x, t) -> u(x, t).

PDE:     d²u/dt² = c² * d²u/dx²    (c = wave speed)
Domain:  x in [0, L_string], t in [0, t_max]
BCs:     u(0, t) = 0, u(L_string, t) = 0  (fixed ends)
ICs:     u(x, 0) = f(x)  (initial displacement)
         du/dt(x, 0) = g(x)  (initial velocity, typically 0 = released from rest)

Unlike the heat equation (first-order in time), the wave equation is
second-order in *both* space and time.  This requires two passes of
autograd for each variable.
"""

import torch
import torch.nn as nn
import numpy as np


class WavePINN(nn.Module):
    """
    Fully-connected network mapping (x, t) -> scalar u(x, t).

    Architecture: input (2) -> 4 hidden layers x 64 tanh -> output (1).
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
            u: displacement field (N, 1)
        """
        return self.network(torch.cat([x, t], dim=1))

    def physics_loss(self, x_int, t_int, c=1.0):
        """
        PDE residual: d²u/dt² - c² * d²u/dx² = 0.

        Requires second-order derivatives in both x and t, computed via
        two sequential autograd.grad calls for each variable.
        """
        x_int.requires_grad_(True)
        t_int.requires_grad_(True)
        u = self(x_int, t_int)

        # Second-order temporal derivative
        du_dt = torch.autograd.grad(
            u, t_int, torch.ones_like(u), create_graph=True)[0]
        d2u_dt2 = torch.autograd.grad(
            du_dt, t_int, torch.ones_like(du_dt), create_graph=True)[0]

        # Second-order spatial derivative
        du_dx = torch.autograd.grad(
            u, x_int, torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(
            du_dx, x_int, torch.ones_like(du_dx), create_graph=True)[0]

        residual = d2u_dt2 - c ** 2 * d2u_dx2
        return torch.mean(residual ** 2)

    def boundary_loss(self, n_bc, t_max, L_string=1.0):
        """Dirichlet BC: u(0, t) = 0, u(L_string, t) = 0 (fixed ends)."""
        t_bc = torch.rand(n_bc, 1) * t_max
        u_left = self(torch.zeros(n_bc, 1), t_bc)
        u_right = self(torch.full((n_bc, 1), L_string), t_bc)
        return torch.mean(u_left ** 2) + torch.mean(u_right ** 2)

    def ic_displacement_loss(self, n_ic, ic_fn, L_string=1.0):
        """IC: u(x, 0) = f(x)."""
        x_ic = torch.rand(n_ic, 1) * L_string
        t_ic = torch.zeros(n_ic, 1)
        u_pred = self(x_ic, t_ic)
        u_true = ic_fn(x_ic)
        return torch.mean((u_pred - u_true) ** 2)

    def ic_velocity_loss(self, n_ic, vel_fn, L_string=1.0):
        """
        IC: du/dt(x, 0) = g(x).

        The velocity IC is critical — without it, the PINN could learn a
        travelling wave rather than a standing wave.
        """
        x_ic = torch.rand(n_ic, 1) * L_string
        t_ic = torch.zeros(n_ic, 1)
        t_ic.requires_grad_(True)
        u = self(x_ic, t_ic)
        du_dt = torch.autograd.grad(
            u, t_ic, torch.ones_like(u), create_graph=True)[0]
        v_true = vel_fn(x_ic)
        return torch.mean((du_dt - v_true) ** 2)

    def compute_residual(self, x_pts, t_pts, c=1.0):
        """Per-point PDE residual magnitude (detached)."""
        x_pts = x_pts.clone().requires_grad_(True)
        t_pts = t_pts.clone().requires_grad_(True)
        u = self(x_pts, t_pts)

        du_dt = torch.autograd.grad(
            u, t_pts, torch.ones_like(u),
            create_graph=True, retain_graph=True)[0]
        d2u_dt2 = torch.autograd.grad(
            du_dt, t_pts, torch.ones_like(du_dt),
            create_graph=False, retain_graph=True)[0]
        du_dx = torch.autograd.grad(
            u, x_pts, torch.ones_like(u),
            create_graph=True, retain_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(
            du_dx, x_pts, torch.ones_like(du_dx), create_graph=False)[0]

        residual = d2u_dt2 - c ** 2 * d2u_dx2
        return (residual ** 2).squeeze().detach()


# =========================================================================
# Initial condition functions for the wave equation
# =========================================================================

def wave_ic_sine(x, L_string=1.0):
    """First fundamental mode: sin(pi * x / L)."""
    return torch.sin(np.pi * x / L_string)


def wave_ic_plucked(x, L_string=1.0):
    """
    Plucked string: triangle peaking at the midpoint.
    f(x) = 2x/L  for x <= L/2,   2(L-x)/L  for x > L/2.
    """
    mid = L_string / 2
    left = 2 * x / L_string
    right = 2 * (L_string - x) / L_string
    return torch.where(x <= mid, left, right)


def wave_ic_gaussian(x, L_string=1.0):
    """Gaussian pulse centred at L/2."""
    mu = L_string / 2
    sigma = L_string / 10
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


WAVE_IC_FUNCTIONS = {
    'sine': wave_ic_sine,
    'plucked': wave_ic_plucked,
    'gaussian': wave_ic_gaussian,
}


def zero_velocity(x):
    """Zero initial velocity (released from rest)."""
    return torch.zeros_like(x)


# =========================================================================
# Training
# =========================================================================

def train_wave_pinn(c=1.0, L_string=1.0, t_max=1.0,
                    ic_type='sine',
                    n_interior=2000, n_bc=200, n_ic=200,
                    epochs=8000, lr=1e-3,
                    bc_weight=10.0, ic_weight=10.0,
                    hidden_size=64, num_hidden_layers=4,
                    verbose=True):
    """
    Train a WavePINN on the 1D wave equation.

    Returns:
        model: trained WavePINN
        loss_history: per-epoch total loss
    """
    model = WavePINN(hidden_size=hidden_size,
                     num_hidden_layers=num_hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6,
    )

    ic_fn = lambda x: WAVE_IC_FUNCTIONS[ic_type](x, L_string=L_string)
    loss_history = []

    if verbose:
        print(f"Training WavePINN  (c={c}, L={L_string}, IC={ic_type})")
        print(f"  Domain: x in [0, {L_string}], t in [0, {t_max}]")
        print(f"  BCs: u(0,t)=0, u({L_string},t)=0  (fixed ends)")
        print(f"  Interior pts: {n_interior}, Epochs: {epochs}")
        print("-" * 55)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # PDE residual at interior points
        x_int = torch.rand(n_interior, 1) * L_string
        t_int = torch.rand(n_interior, 1) * t_max
        loss_pde = model.physics_loss(x_int, t_int, c=c)

        # Boundary conditions
        loss_bc = model.boundary_loss(n_bc, t_max, L_string)

        # ICs: displacement + velocity (released from rest)
        loss_ic_u = model.ic_displacement_loss(n_ic, ic_fn, L_string)
        loss_ic_v = model.ic_velocity_loss(n_ic, zero_velocity, L_string)
        loss_ic = loss_ic_u + loss_ic_v

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
# Analytical solution (Fourier series for standing waves)
# =========================================================================

def wave_analytical(x, t, c=1.0, L_string=1.0, ic_type='sine', n_terms=50):
    """
    Fourier series solution of the 1D wave equation with fixed ends
    and zero initial velocity (released from rest).

    General solution:
        u(x,t) = sum_n  b_n * cos(n*pi*c*t/L) * sin(n*pi*x/L)

    where b_n = (2/L) * integral_0^L f(x)*sin(n*pi*x/L) dx.

    For IC = sin(pi*x/L), the exact single-mode solution is:
        u(x,t) = cos(pi*c*t/L) * sin(pi*x/L)
    """
    if ic_type == 'sine':
        return np.cos(np.pi * c * t / L_string) * np.sin(np.pi * x / L_string)

    # General: numerical Fourier coefficients via quadrature
    n_quad = 1000
    x_quad = np.linspace(0, L_string, n_quad)
    dx = x_quad[1] - x_quad[0]

    if ic_type == 'plucked':
        mid = L_string / 2
        f_x = np.where(x_quad <= mid,
                        2 * x_quad / L_string,
                        2 * (L_string - x_quad) / L_string)
    elif ic_type == 'gaussian':
        mu = L_string / 2
        sigma = L_string / 10
        f_x = np.exp(-((x_quad - mu) ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown IC type: {ic_type}")

    u = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        basis = np.sin(n * np.pi * x_quad / L_string)
        b_n = (2 / L_string) * np.trapezoid(f_x * basis, dx=dx)
        u = u + b_n * np.cos(n * np.pi * c * t / L_string) * \
            np.sin(n * np.pi * x / L_string)

    return u


def wave_mode_decomposition(x, t, c=1.0, L_string=1.0, ic_type='sine',
                            n_modes=5):
    """
    Return individual Fourier modes as a list for visualisation.

    Each mode is:  b_n * cos(n*pi*c*t/L) * sin(n*pi*x/L)

    Returns:
        modes: list of (mode_number, b_n, u_mode_array) tuples
    """
    n_quad = 1000
    x_quad = np.linspace(0, L_string, n_quad)
    dx = x_quad[1] - x_quad[0]

    if ic_type == 'sine':
        f_x = np.sin(np.pi * x_quad / L_string)
    elif ic_type == 'plucked':
        mid = L_string / 2
        f_x = np.where(x_quad <= mid,
                        2 * x_quad / L_string,
                        2 * (L_string - x_quad) / L_string)
    elif ic_type == 'gaussian':
        mu = L_string / 2
        sigma = L_string / 10
        f_x = np.exp(-((x_quad - mu) ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown IC type: {ic_type}")

    modes = []
    for n in range(1, n_modes + 1):
        basis = np.sin(n * np.pi * x_quad / L_string)
        b_n = (2 / L_string) * np.trapezoid(f_x * basis, dx=dx)
        u_mode = b_n * np.cos(n * np.pi * c * t / L_string) * \
            np.sin(n * np.pi * x / L_string)
        modes.append((n, b_n, u_mode))

    return modes


def wave_energy(x_arr, u_field, du_dt_field, du_dx_field, c=1.0, dx=None):
    """
    Total mechanical energy of the vibrating string at one time instant.

    E = (1/2) * integral_0^L [ (du/dt)^2 + c^2 * (du/dx)^2 ] dx

    The wave equation conserves this energy exactly.

    Args:
        x_arr: spatial grid (M,)
        u_field: displacement at this time (M,) — unused but kept for API clarity
        du_dt_field: velocity field (M,)
        du_dx_field: strain field (M,)
        c: wave speed
        dx: grid spacing (computed from x_arr if None)

    Returns:
        scalar total energy
    """
    if dx is None:
        dx = x_arr[1] - x_arr[0]
    integrand = du_dt_field ** 2 + c ** 2 * du_dx_field ** 2
    return 0.5 * np.trapezoid(integrand, dx=dx)
