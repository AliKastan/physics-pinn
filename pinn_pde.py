"""
Physics-Informed Neural Networks for Partial Differential Equations
====================================================================

This module extends PINNs from ODEs to PDEs. The key difference: the network
now takes *two* independent variables (x, t) as input instead of one,
and all spatial and temporal derivatives are computed via PyTorch autograd.

PART 1 — 1D Heat Equation:
    PDE:  du/dt = alpha * d²u/dx²     (parabolic, diffusive)
    Domain:  x in [0, 1], t in [0, 1]
    BCs:  u(0, t) = 0,  u(1, t) = 0   (Dirichlet)
    IC:   u(x, 0) = sin(pi * x)
    Exact:  u(x, t) = exp(-alpha * pi^2 * t) * sin(pi * x)

    The heat equation describes how temperature diffuses through a rod.
    The PINN learns the full spatiotemporal solution from the PDE alone.

PART 2 — 1D Wave Equation:
    PDE:  d²u/dt² = c² * d²u/dx²      (hyperbolic, oscillatory)
    Domain:  x in [0, 1], t in [0, 0.5]
    BCs:  u(0, t) = 0,  u(1, t) = 0
    ICs:  u(x, 0) = sin(pi * x),  du/dt(x, 0) = 0
    Exact:  u(x, t) = cos(c * pi * t) * sin(pi * x)

    The wave equation describes vibrations of a string fixed at both ends.
    The zero initial velocity means the string is released from rest.

Why PDEs are Harder than ODEs:
    - The solution lives in a 2D domain (x, t) rather than a 1D line.
    - Boundary conditions must be enforced on the domain edges, not just
      at a single point.
    - Second spatial derivatives (d²u/dx²) require two passes of autograd.
    - The loss landscape has more local minima.
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. Neural Network Architecture
# =============================================================================

class PINNPDE(nn.Module):
    """
    Fully-connected network mapping (x, t) -> u.

    Architecture:
        Input:  (x, t) — 2 neurons
        Hidden: 4 layers x 64 neurons, tanh activation
        Output: u — 1 neuron

    Deeper than the ODE PINNs because the solution lives in 2D and we
    need second-order spatial derivatives (d²u/dx²), which require the
    network to be at least C² smooth. Tanh provides this.
    """

    def __init__(self, hidden_size=64, num_hidden_layers=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        """
        Args:
            x: Spatial coordinate (N, 1)
            t: Temporal coordinate (N, 1)

        Returns:
            u: Solution value (N, 1)
        """
        inp = torch.cat([x, t], dim=1)
        return self.network(inp)


# =============================================================================
# PART 1: 1D Heat Equation
# =============================================================================

def heat_exact(x, t, alpha=0.01):
    """Exact solution: u(x,t) = exp(-alpha * pi^2 * t) * sin(pi * x)."""
    return np.exp(-alpha * np.pi ** 2 * t) * np.sin(np.pi * x)


def train_heat_pinn(alpha=0.01, n_interior=2000, n_bc=200, n_ic=200,
                    epochs=8000, lr=1e-3, bc_weight=10.0, ic_weight=10.0):
    """
    Train a PINN to solve the 1D heat equation.

    Loss = PDE residual + bc_weight * BC loss + ic_weight * IC loss

    The PDE residual at interior collocation points is:
        r = du/dt - alpha * d²u/dx²

    Boundary conditions (u = 0 at x=0 and x=1) and the initial condition
    (u = sin(pi*x) at t=0) are enforced as soft constraints.

    Args:
        alpha: Thermal diffusivity
        n_interior: Collocation points in the domain interior
        n_bc: Collocation points on each boundary (x=0 and x=1)
        n_ic: Collocation points on the initial condition (t=0)
        epochs: Training iterations
        lr: Learning rate
        bc_weight: Weight for boundary condition loss
        ic_weight: Weight for initial condition loss

    Returns:
        model: Trained PINN
        loss_history: Total loss per epoch
    """
    model = PINNPDE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6
    )

    loss_history = []

    print("Training PINN for 1D Heat Equation")
    print(f"  PDE: du/dt = {alpha} * d^2u/dx^2")
    print(f"  Domain: x in [0,1], t in [0,1]")
    print(f"  Interior points: {n_interior}")
    print(f"  Epochs: {epochs}")
    print("-" * 50)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- Interior collocation points ---
        x_int = torch.rand(n_interior, 1)
        t_int = torch.rand(n_interior, 1)
        x_int.requires_grad_(True)
        t_int.requires_grad_(True)

        u = model(x_int, t_int)

        # du/dt
        du_dt = torch.autograd.grad(
            u, t_int, grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        # du/dx
        du_dx = torch.autograd.grad(
            u, x_int, grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        # d²u/dx² (second pass of autograd)
        d2u_dx2 = torch.autograd.grad(
            du_dx, x_int, grad_outputs=torch.ones_like(du_dx),
            create_graph=True
        )[0]

        # PDE residual: du/dt - alpha * d²u/dx²
        residual = du_dt - alpha * d2u_dx2
        loss_pde = torch.mean(residual ** 2)

        # --- Boundary conditions: u(0,t) = 0, u(1,t) = 0 ---
        t_bc = torch.rand(n_bc, 1)

        x_left = torch.zeros(n_bc, 1)
        u_left = model(x_left, t_bc)
        loss_bc_left = torch.mean(u_left ** 2)

        x_right = torch.ones(n_bc, 1)
        u_right = model(x_right, t_bc)
        loss_bc_right = torch.mean(u_right ** 2)

        loss_bc = loss_bc_left + loss_bc_right

        # --- Initial condition: u(x,0) = sin(pi*x) ---
        x_ic = torch.rand(n_ic, 1)
        t_ic = torch.zeros(n_ic, 1)
        u_ic = model(x_ic, t_ic)
        u_ic_true = torch.sin(np.pi * x_ic)
        loss_ic = torch.mean((u_ic - u_ic_true) ** 2)

        # --- Total loss ---
        total_loss = loss_pde + bc_weight * loss_bc + ic_weight * loss_ic
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        loss_history.append(total_loss.item())

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | "
                  f"Loss: {total_loss.item():.6f} | "
                  f"PDE: {loss_pde.item():.6f} | "
                  f"BC: {loss_bc.item():.6f} | "
                  f"IC: {loss_ic.item():.6f}")

    print("-" * 50)
    print(f"Final loss: {loss_history[-1]:.6f}")
    return model, loss_history


# =============================================================================
# PART 2: 1D Wave Equation
# =============================================================================

def wave_exact(x, t, c=1.0):
    """Exact solution: u(x,t) = cos(c * pi * t) * sin(pi * x)."""
    return np.cos(c * np.pi * t) * np.sin(np.pi * x)


def train_wave_pinn(c=1.0, n_interior=2000, n_bc=200, n_ic=200,
                    epochs=8000, lr=1e-3, bc_weight=10.0, ic_weight=10.0):
    """
    Train a PINN to solve the 1D wave equation.

    The PDE residual at interior collocation points is:
        r = d²u/dt² - c² * d²u/dx²

    This requires second-order derivatives in *both* x and t, computed
    via two passes of autograd for each variable.

    Initial conditions enforce both u(x,0) = sin(pi*x) and du/dt(x,0) = 0.
    The velocity IC is critical — without it, the PINN could learn a
    travelling wave rather than a standing wave.

    Args:
        c: Wave speed
        n_interior: Interior collocation points
        n_bc: Boundary collocation points per edge
        n_ic: Initial condition collocation points
        epochs: Training iterations
        lr: Learning rate
        bc_weight: Weight for BC loss
        ic_weight: Weight for IC loss

    Returns:
        model: Trained PINN
        loss_history: Total loss per epoch
    """
    model = PINNPDE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6
    )

    loss_history = []
    t_max = 0.5

    print("Training PINN for 1D Wave Equation")
    print(f"  PDE: d^2u/dt^2 = {c}^2 * d^2u/dx^2")
    print(f"  Domain: x in [0,1], t in [0,{t_max}]")
    print(f"  Interior points: {n_interior}")
    print(f"  Epochs: {epochs}")
    print("-" * 50)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- Interior collocation points ---
        x_int = torch.rand(n_interior, 1)
        t_int = torch.rand(n_interior, 1) * t_max
        x_int.requires_grad_(True)
        t_int.requires_grad_(True)

        u = model(x_int, t_int)

        # du/dt
        du_dt = torch.autograd.grad(
            u, t_int, grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        # d²u/dt²
        d2u_dt2 = torch.autograd.grad(
            du_dt, t_int, grad_outputs=torch.ones_like(du_dt),
            create_graph=True
        )[0]

        # du/dx
        du_dx = torch.autograd.grad(
            u, x_int, grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        # d²u/dx²
        d2u_dx2 = torch.autograd.grad(
            du_dx, x_int, grad_outputs=torch.ones_like(du_dx),
            create_graph=True
        )[0]

        # PDE residual: d²u/dt² - c² * d²u/dx²
        residual = d2u_dt2 - c ** 2 * d2u_dx2
        loss_pde = torch.mean(residual ** 2)

        # --- Boundary conditions: u(0,t) = 0, u(1,t) = 0 ---
        t_bc = torch.rand(n_bc, 1) * t_max

        u_left = model(torch.zeros(n_bc, 1), t_bc)
        u_right = model(torch.ones(n_bc, 1), t_bc)
        loss_bc = torch.mean(u_left ** 2) + torch.mean(u_right ** 2)

        # --- Initial conditions ---
        # IC 1: u(x, 0) = sin(pi * x)
        x_ic = torch.rand(n_ic, 1)
        t_ic_zero = torch.zeros(n_ic, 1)
        u_ic = model(x_ic, t_ic_zero)
        u_ic_true = torch.sin(np.pi * x_ic)
        loss_ic_u = torch.mean((u_ic - u_ic_true) ** 2)

        # IC 2: du/dt(x, 0) = 0
        x_ic2 = torch.rand(n_ic, 1)
        t_ic2 = torch.zeros(n_ic, 1)
        t_ic2.requires_grad_(True)
        u_ic2 = model(x_ic2, t_ic2)
        du_dt_ic = torch.autograd.grad(
            u_ic2, t_ic2, grad_outputs=torch.ones_like(u_ic2),
            create_graph=True
        )[0]
        loss_ic_v = torch.mean(du_dt_ic ** 2)

        loss_ic = loss_ic_u + loss_ic_v

        # --- Total loss ---
        total_loss = loss_pde + bc_weight * loss_bc + ic_weight * loss_ic
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        loss_history.append(total_loss.item())

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | "
                  f"Loss: {total_loss.item():.6f} | "
                  f"PDE: {loss_pde.item():.6f} | "
                  f"BC: {loss_bc.item():.6f} | "
                  f"IC: {loss_ic.item():.6f}")

    print("-" * 50)
    print(f"Final loss: {loss_history[-1]:.6f}")
    return model, loss_history


# =============================================================================
# Visualization
# =============================================================================

def visualize_results(heat_model, wave_model, alpha=0.01, c=1.0):
    """
    Combined 2x3 figure:
        Row 1 (Heat): PINN u(x,t) | Exact u(x,t) | Absolute error
        Row 2 (Wave): PINN u(x,t) | Exact u(x,t) | Absolute error
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('PINNs for Partial Differential Equations',
                 fontsize=15, fontweight='bold')

    # --- Evaluation grids ---
    nx, nt = 100, 100

    # Heat equation: x in [0,1], t in [0,1]
    x_heat = np.linspace(0, 1, nx)
    t_heat = np.linspace(0, 1, nt)
    X_h, T_h = np.meshgrid(x_heat, t_heat)

    heat_model.eval()
    with torch.no_grad():
        x_flat = torch.tensor(X_h.flatten(), dtype=torch.float32).unsqueeze(1)
        t_flat = torch.tensor(T_h.flatten(), dtype=torch.float32).unsqueeze(1)
        u_pinn_h = heat_model(x_flat, t_flat).numpy().reshape(X_h.shape)
    u_exact_h = heat_exact(X_h, T_h, alpha=alpha)
    u_error_h = np.abs(u_pinn_h - u_exact_h)

    # Wave equation: x in [0,1], t in [0,0.5]
    x_wave = np.linspace(0, 1, nx)
    t_wave = np.linspace(0, 0.5, nt)
    X_w, T_w = np.meshgrid(x_wave, t_wave)

    wave_model.eval()
    with torch.no_grad():
        x_flat = torch.tensor(X_w.flatten(), dtype=torch.float32).unsqueeze(1)
        t_flat = torch.tensor(T_w.flatten(), dtype=torch.float32).unsqueeze(1)
        u_pinn_w = wave_model(x_flat, t_flat).numpy().reshape(X_w.shape)
    u_exact_w = wave_exact(X_w, T_w, c=c)
    u_error_w = np.abs(u_pinn_w - u_exact_w)

    # --- Row 1: Heat equation ---
    im1 = axes[0, 0].pcolormesh(X_h, T_h, u_pinn_h, cmap='hot', shading='auto')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('t')
    axes[0, 0].set_title('Heat Eq: PINN u(x,t)')
    fig.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].pcolormesh(X_h, T_h, u_exact_h, cmap='hot', shading='auto')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('t')
    axes[0, 1].set_title('Heat Eq: Exact u(x,t)')
    fig.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].pcolormesh(X_h, T_h, u_error_h, cmap='viridis', shading='auto')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('t')
    axes[0, 2].set_title(f'Heat Eq: |Error| (max: {np.max(u_error_h):.2e})')
    fig.colorbar(im3, ax=axes[0, 2])

    # --- Row 2: Wave equation ---
    vmax_w = max(np.max(np.abs(u_pinn_w)), np.max(np.abs(u_exact_w)))
    im4 = axes[1, 0].pcolormesh(X_w, T_w, u_pinn_w, cmap='RdBu_r',
                                  vmin=-vmax_w, vmax=vmax_w, shading='auto')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('t')
    axes[1, 0].set_title('Wave Eq: PINN u(x,t)')
    fig.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].pcolormesh(X_w, T_w, u_exact_w, cmap='RdBu_r',
                                  vmin=-vmax_w, vmax=vmax_w, shading='auto')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('t')
    axes[1, 1].set_title('Wave Eq: Exact u(x,t)')
    fig.colorbar(im5, ax=axes[1, 1])

    im6 = axes[1, 2].pcolormesh(X_w, T_w, u_error_w, cmap='viridis', shading='auto')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('t')
    axes[1, 2].set_title(f'Wave Eq: |Error| (max: {np.max(u_error_w):.2e})')
    fig.colorbar(im6, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig('pde_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to pde_results.png")

    # --- Summary ---
    print(f"\n{'='*55}")
    print("PDE Solution Summary")
    print(f"{'='*55}")
    print(f"  Heat Equation (alpha={alpha}):")
    print(f"    Max absolute error: {np.max(u_error_h):.2e}")
    print(f"    Mean absolute error: {np.mean(u_error_h):.2e}")
    print(f"  Wave Equation (c={c}):")
    print(f"    Max absolute error: {np.max(u_error_w):.2e}")
    print(f"    Mean absolute error: {np.mean(u_error_w):.2e}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    # Physical parameters
    alpha = 0.01   # thermal diffusivity
    c = 1.0        # wave speed

    # Train heat equation PINN
    heat_model, heat_losses = train_heat_pinn(
        alpha=alpha, n_interior=2000, n_bc=200, n_ic=200,
        epochs=8000, lr=1e-3, bc_weight=10.0, ic_weight=10.0
    )

    print()

    # Train wave equation PINN
    wave_model, wave_losses = train_wave_pinn(
        c=c, n_interior=2000, n_bc=200, n_ic=200,
        epochs=8000, lr=1e-3, bc_weight=10.0, ic_weight=10.0
    )

    # Visualize
    visualize_results(heat_model, wave_model, alpha=alpha, c=c)
