"""
Adaptive vs Uniform Collocation — Orbital PINN Comparison
==========================================================

Trains two orbital PINNs side by side:
  1. Uniform collocation (baseline)
  2. RAR adaptive collocation (concentrates points near perihelion)

Shows that adaptive sampling automatically discovers and focuses on
the hardest part of the orbit — the sharp velocity change at perihelion.

Usage:
    python examples/adaptive_vs_uniform.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.orbital_pinn import PINNOrbital
from src.training.trainer import Trainer
from src.training.losses import orbital_ic_loss
from src.utils.validation import solve_orbit_ode, setup_orbital_ics
from src.utils.metrics import compute_energy_orbital


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup
    eccentricity = 0.5
    GM = 1.0
    x0, y0, vx0, vy0, period = setup_orbital_ics(eccentricity, GM)
    t_max = period  # one full orbit

    epochs = 5000
    n_col = 800

    x0_t = torch.tensor(x0, dtype=torch.float32)
    y0_t = torch.tensor(y0, dtype=torch.float32)
    vx0_t = torch.tensor(vx0, dtype=torch.float32)
    vy0_t = torch.tensor(vy0, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Train uniform
    # ------------------------------------------------------------------
    print("=" * 55)
    print("UNIFORM COLLOCATION")
    print("=" * 55)
    torch.manual_seed(42)
    model_u = PINNOrbital()
    config_u = {
        "epochs": epochs, "lr": 1e-3, "n_collocation": n_col,
        "ic_weight": 50.0, "t_max": t_max,
        "adaptive": False,
    }
    trainer_u = Trainer(model_u, config_u, physics_params={"GM": GM})
    ic_fn_u = lambda: orbital_ic_loss(model_u, x0_t, y0_t, vx0_t, vy0_t)
    losses_u, snaps_u = trainer_u.train(ic_fn_u)

    # ------------------------------------------------------------------
    # Train adaptive (RAR)
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("ADAPTIVE COLLOCATION (RAR, k=1.5)")
    print("=" * 55)
    torch.manual_seed(42)
    model_a = PINNOrbital()
    config_a = {
        "epochs": epochs, "lr": 1e-3, "n_collocation": n_col,
        "ic_weight": 50.0, "t_max": t_max,
        "adaptive": True, "adaptive_interval": 500,
        "adaptive_k": 1.5, "uniform_fraction": 0.2,
    }
    trainer_a = Trainer(model_a, config_a, physics_params={"GM": GM})
    ic_fn_a = lambda: orbital_ic_loss(model_a, x0_t, y0_t, vx0_t, vy0_t)
    losses_a, snaps_a = trainer_a.train(ic_fn_a)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    t_eval = np.linspace(0, t_max, 2000)
    t_ode, x_ode, y_ode, vx_ode, vy_ode = solve_orbit_ode(
        x0, y0, vx0, vy0, (0, t_max), t_eval, GM=GM)

    for m in (model_u, model_a):
        m.eval()
    t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        out_u = model_u(t_tensor).numpy()
        out_a = model_a(t_tensor).numpy()

    err_u = np.sqrt((out_u[:, 0] - x_ode) ** 2 + (out_u[:, 1] - y_ode) ** 2)
    err_a = np.sqrt((out_a[:, 0] - x_ode) ** 2 + (out_a[:, 1] - y_ode) ** 2)

    E_u = compute_energy_orbital(out_u[:, 0], out_u[:, 1], out_u[:, 2], out_u[:, 3], GM)
    E_a = compute_energy_orbital(out_a[:, 0], out_a[:, 1], out_a[:, 2], out_a[:, 3], GM)
    E_true = compute_energy_orbital(x_ode, y_ode, vx_ode, vy_ode, GM)
    E0 = E_true[0]

    # ------------------------------------------------------------------
    # Plot: 2x3
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Adaptive vs Uniform Collocation — Orbital PINN (e={eccentricity})',
        fontsize=14, fontweight='bold',
    )

    # (0,0) Trajectory comparison
    ax = axes[0, 0]
    ax.plot(x_ode, y_ode, 'b-', lw=2, alpha=0.4, label='Ground truth')
    ax.plot(out_u[:, 0], out_u[:, 1], 'r--', lw=1.5, label='Uniform')
    ax.plot(out_a[:, 0], out_a[:, 1], 'g--', lw=1.5, label='Adaptive')
    ax.plot(0, 0, 'ko', ms=8)
    ax.set_aspect('equal')
    ax.set_title('Orbital Trajectory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Position error over time
    ax = axes[0, 1]
    ax.semilogy(t_eval, err_u + 1e-16, 'r-', lw=1.5, label='Uniform')
    ax.semilogy(t_eval, err_a + 1e-16, 'g-', lw=1.5, label='Adaptive')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position Error')
    ax.set_title(f'Mean err: U={np.mean(err_u):.2e}, A={np.mean(err_a):.2e}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,2) Training loss
    ax = axes[0, 2]
    ax.semilogy(losses_u, 'r-', lw=1, alpha=0.7, label='Uniform')
    ax.semilogy(losses_a, 'g-', lw=1, alpha=0.7, label='Adaptive')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Energy conservation
    ax = axes[1, 0]
    dE_u = np.abs((E_u - E0) / (np.abs(E0) + 1e-16))
    dE_a = np.abs((E_a - E0) / (np.abs(E0) + 1e-16))
    ax.semilogy(t_eval, dE_u + 1e-16, 'r-', lw=1.5, label='Uniform')
    ax.semilogy(t_eval, dE_a + 1e-16, 'g-', lw=1.5, label='Adaptive')
    ax.set_xlabel('Time')
    ax.set_ylabel('|dE/E0|')
    ax.set_title('Energy Drift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Collocation point distribution (last snapshot)
    ax = axes[1, 1]
    snap_epochs = sorted(snaps_u.keys())
    last_ep = snap_epochs[-1]
    if last_ep in snaps_u:
        ax.hist(snaps_u[last_ep], bins=50, alpha=0.5, color='red',
                density=True, label='Uniform')
    if last_ep in snaps_a:
        ax.hist(snaps_a[last_ep], bins=50, alpha=0.5, color='green',
                density=True, label='Adaptive')
    # Mark perihelion time (t where x is closest to central body)
    r_ode = np.sqrt(x_ode ** 2 + y_ode ** 2)
    t_peri = t_eval[np.argmin(r_ode)]
    ax.axvline(x=t_peri, color='blue', ls='--', lw=1.5,
               label=f'Perihelion t={t_peri:.2f}')
    ax.set_xlabel('t')
    ax.set_ylabel('Density')
    ax.set_title(f'Collocation Distribution (epoch {last_ep})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) Pointwise residual
    ax = axes[1, 2]
    t_grid = torch.linspace(0, t_max, 1000).unsqueeze(1)
    with torch.no_grad():
        res_u = model_u.compute_residual(t_grid, GM=GM).numpy()
        res_a = model_a.compute_residual(t_grid, GM=GM).numpy()
    t_np = np.linspace(0, t_max, 1000)
    ax.semilogy(t_np, res_u + 1e-16, 'r-', lw=1, alpha=0.7, label='Uniform')
    ax.semilogy(t_np, res_a + 1e-16, 'g-', lw=1, alpha=0.7, label='Adaptive')
    ax.axvline(x=t_peri, color='blue', ls='--', lw=1.5, label='Perihelion')
    ax.set_xlabel('Time')
    ax.set_ylabel('Pointwise Residual')
    ax.set_title('Final Physics Residual')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'adaptive_vs_uniform_orbital.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {out_path}")

    ratio = np.mean(err_u) / (np.mean(err_a) + 1e-16)
    print(f"\n{'='*55}")
    print("Comparison Summary")
    print(f"{'='*55}")
    print(f"  Uniform  mean error: {np.mean(err_u):.6f}")
    print(f"  Adaptive mean error: {np.mean(err_a):.6f}")
    print(f"  Adaptive is {ratio:.1f}x better")
    print(f"\n  The adaptive distribution concentrates points near")
    print(f"  perihelion (t~{t_peri:.2f}) where the 1/r^3 force is strongest.")


if __name__ == '__main__':
    main()
