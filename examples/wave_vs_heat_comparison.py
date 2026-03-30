"""
Wave vs Heat Comparison — Diffusion vs Propagation
===================================================

Trains the same PINN framework on two fundamentally different PDEs:
  - Heat equation (parabolic): solution diffuses and decays
  - Wave equation (hyperbolic): solution propagates and oscillates

Both use the same network architecture, same training loop, same IC
(sine wave). The dramatically different behaviour demonstrates the
framework is genuinely general-purpose.

Usage:
    python examples/wave_vs_heat_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.heat_pinn import train_heat_pinn, heat_analytical
from src.models.wave_pinn import train_wave_pinn, wave_analytical


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    alpha = 0.01
    c = 1.0
    t_max = 1.0

    # ------------------------------------------------------------------
    # Train both models
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PART 1: Heat Equation (diffusion)")
    print("=" * 60)
    heat_model, heat_losses = train_heat_pinn(
        alpha=alpha, t_max=t_max, ic_type='sine',
        epochs=5000, n_interior=2000, n_bc=200, n_ic=200,
    )

    print()
    print("=" * 60)
    print("PART 2: Wave Equation (propagation)")
    print("=" * 60)
    wave_model, wave_losses = train_wave_pinn(
        c=c, t_max=t_max, ic_type='sine',
        epochs=5000, n_interior=2000, n_bc=200, n_ic=200,
    )

    # ------------------------------------------------------------------
    # Evaluate on grids
    # ------------------------------------------------------------------
    nx, nt = 100, 100
    x_arr = np.linspace(0, 1, nx)
    t_arr = np.linspace(0, t_max, nt)
    X, T = np.meshgrid(x_arr, t_arr)

    heat_model.eval()
    wave_model.eval()

    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        u_heat = heat_model(x_flat, t_flat).numpy().reshape(X.shape)
        u_wave = wave_model(x_flat, t_flat).numpy().reshape(X.shape)

    u_heat_exact = heat_analytical(X, T, alpha=alpha, ic_type='sine')
    u_wave_exact = wave_analytical(X, T, c=c, ic_type='sine')

    err_heat = np.abs(u_heat - u_heat_exact)
    err_wave = np.abs(u_wave - u_wave_exact)

    # ------------------------------------------------------------------
    # Plot: 3x3 figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(
        'Heat (Diffusion) vs Wave (Propagation) — Same Framework, Different Physics',
        fontsize=14, fontweight='bold',
    )

    # Row 0: Heatmaps  (PINN | Exact | Error)
    # -- Heat PINN
    im = axes[0, 0].pcolormesh(X, T, u_heat, cmap='hot', shading='auto')
    axes[0, 0].set_title('Heat Eq: PINN')
    axes[0, 0].set_ylabel('t')
    fig.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].pcolormesh(X, T, u_heat_exact, cmap='hot', shading='auto')
    axes[0, 1].set_title('Heat Eq: Exact')
    fig.colorbar(im, ax=axes[0, 1])

    im = axes[0, 2].pcolormesh(X, T, err_heat, cmap='viridis', shading='auto')
    axes[0, 2].set_title(f'Heat |Error| (max {np.max(err_heat):.2e})')
    fig.colorbar(im, ax=axes[0, 2])

    # -- Wave PINN
    vmax = max(np.max(np.abs(u_wave)), np.max(np.abs(u_wave_exact)))
    im = axes[1, 0].pcolormesh(X, T, u_wave, cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, shading='auto')
    axes[1, 0].set_title('Wave Eq: PINN')
    axes[1, 0].set_ylabel('t')
    fig.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].pcolormesh(X, T, u_wave_exact, cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, shading='auto')
    axes[1, 1].set_title('Wave Eq: Exact')
    fig.colorbar(im, ax=axes[1, 1])

    im = axes[1, 2].pcolormesh(X, T, err_wave, cmap='viridis', shading='auto')
    axes[1, 2].set_title(f'Wave |Error| (max {np.max(err_wave):.2e})')
    fig.colorbar(im, ax=axes[1, 2])

    # Row 2: Snapshots side-by-side + Training loss
    t_snaps = [0.0, 0.25, 0.5, 1.0]
    ax = axes[2, 0]
    for ts in t_snaps:
        idx = int(ts / t_max * (nt - 1))
        idx = min(idx, nt - 1)
        ax.plot(x_arr, u_heat[idx, :], label=f't={ts}')
    ax.set_title('Heat: Snapshots (decays)')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    for ts in t_snaps:
        idx = int(ts / t_max * (nt - 1))
        idx = min(idx, nt - 1)
        ax.plot(x_arr, u_wave[idx, :], label=f't={ts}')
    ax.set_title('Wave: Snapshots (oscillates)')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.semilogy(heat_losses, 'r-', lw=1, alpha=0.7, label='Heat')
    ax.semilogy(wave_losses, 'b-', lw=1, alpha=0.7, label='Wave')
    ax.set_title('Training Convergence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for row in axes[:2]:
        for a in row:
            a.set_xlabel('x')

    plt.tight_layout()
    out_path = 'wave_vs_heat_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {out_path}")

    print(f"\n{'='*55}")
    print("Comparison Summary")
    print(f"{'='*55}")
    print(f"  Heat: max |error| = {np.max(err_heat):.2e}, "
          f"mean = {np.mean(err_heat):.2e}")
    print(f"  Wave: max |error| = {np.max(err_wave):.2e}, "
          f"mean = {np.mean(err_wave):.2e}")
    print(f"\n  Key difference:")
    print(f"    Heat solution DECAYS  (parabolic — information diffuses)")
    print(f"    Wave solution OSCILLATES (hyperbolic — information propagates)")


if __name__ == '__main__':
    main()
