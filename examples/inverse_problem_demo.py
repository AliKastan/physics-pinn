"""
Inverse Problem Demo — Infer Physical Parameters from Noisy Data
================================================================

Demonstrates the inverse PINN approach on the pendulum:
  1. Generate synthetic data with known true g=9.81, L=1.0
  2. Add 5% Gaussian noise
  3. Run inverse PINN with wrong initial guesses (g=5.0, L=2.0)
  4. Plot: parameter convergence, trajectory reconstruction, residual

Usage:
    python examples/inverse_problem_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.inverse_pinn import InversePendulumPINN, train_inverse_pendulum
from src.utils.data_generation import generate_noisy_pendulum_data


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # True parameters
    # ------------------------------------------------------------------
    g_true = 9.81
    L_true = 1.0
    theta_0 = np.pi / 4   # 45 degrees
    omega_0 = 0.0
    t_max = 10.0

    # ------------------------------------------------------------------
    # Step 1: Generate noisy data (5% noise)
    # ------------------------------------------------------------------
    print("Generating noisy pendulum data...")
    data = generate_noisy_pendulum_data(
        theta_0, omega_0, t_max, g=g_true, L=L_true,
        n_obs=80, noise_std=0.05,
    )
    print(f"  {len(data['t_obs'])} noisy observations generated.\n")

    # ------------------------------------------------------------------
    # Step 2: Train inverse PINN with wrong initial guesses
    # ------------------------------------------------------------------
    model = InversePendulumPINN(g_init=5.0, L_init=2.0)

    t_obs_t = torch.tensor(data['t_obs'], dtype=torch.float32).unsqueeze(1)
    theta_obs_t = torch.tensor(data['theta_obs'], dtype=torch.float32)

    g_hist, L_hist, loss_hist = train_inverse_pendulum(
        model, theta_0, omega_0, t_max,
        t_obs_t, theta_obs_t,
        epochs=10000, lr_net=1e-3, lr_param=1e-2,
        warmup_epochs=1500,
    )

    # ------------------------------------------------------------------
    # Step 3: Evaluate
    # ------------------------------------------------------------------
    model.eval()
    t_plot = np.linspace(0, t_max, 1000)
    with torch.no_grad():
        out = model(torch.tensor(t_plot, dtype=torch.float32).unsqueeze(1)).numpy()
    theta_pinn = out[:, 0]

    # Compute pointwise physics residual
    t_res = torch.linspace(0, t_max, 500).unsqueeze(1).requires_grad_(True)
    out_res = model(t_res)
    theta_r = out_res[:, 0:1]
    omega_r = out_res[:, 1:2]
    dtheta = torch.autograd.grad(
        theta_r, t_res, torch.ones_like(theta_r),
        create_graph=False, retain_graph=True)[0]
    domega = torch.autograd.grad(
        omega_r, t_res, torch.ones_like(omega_r), create_graph=False)[0]
    res1 = (dtheta - omega_r).detach().numpy().squeeze()
    res2 = (domega + (model.g / model.L) * torch.sin(theta_r)).detach().numpy().squeeze()
    residual = np.sqrt(res1 ** 2 + res2 ** 2)

    # ------------------------------------------------------------------
    # Step 4: Plot — 2x3 figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Inverse PINN — Parameter Inference from Noisy Pendulum Data',
        fontsize=15, fontweight='bold',
    )

    # (0,0) Trajectory reconstruction
    ax = axes[0, 0]
    ax.plot(data['t_dense'], np.degrees(data['theta_true']),
            'b-', lw=2, label='Ground truth', alpha=0.7)
    ax.scatter(data['t_obs'], np.degrees(data['theta_obs']),
               c='gray', s=20, alpha=0.5, label='Noisy obs', zorder=5)
    ax.plot(t_plot, np.degrees(theta_pinn),
            'r--', lw=2, label='PINN reconstruction')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Trajectory Reconstruction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) g convergence
    ax = axes[0, 1]
    ax.plot(g_hist, 'r-', lw=1.5, label=f'Inferred g (final={model.g:.3f})')
    ax.axhline(y=g_true, color='blue', ls='--', lw=2, label=f'True g={g_true}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('g (m/s²)')
    ax.set_title('Gravity Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,2) L convergence
    ax = axes[0, 2]
    ax.plot(L_hist, 'r-', lw=1.5, label=f'Inferred L (final={model.L:.3f})')
    ax.axhline(y=L_true, color='blue', ls='--', lw=2, label=f'True L={L_true}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L (m)')
    ax.set_title('Length Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) g/L ratio convergence
    g_over_L_hist = [g / l for g, l in zip(g_hist, L_hist)]
    ax = axes[1, 0]
    ax.plot(g_over_L_hist, 'r-', lw=1.5,
            label=f'Inferred g/L (final={model.g_over_L:.3f})')
    ax.axhline(y=g_true / L_true, color='blue', ls='--', lw=2,
               label=f'True g/L={g_true / L_true:.2f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('g/L (s⁻²)')
    ax.set_title('g/L Ratio Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Physics residual distribution
    ax = axes[1, 1]
    t_res_np = np.linspace(0, t_max, 500)
    ax.semilogy(t_res_np, residual + 1e-16, 'g-', lw=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|ODE Residual|')
    ax.set_title('Physics Residual Distribution')
    ax.grid(True, alpha=0.3)

    # (1,2) Training loss
    ax = axes[1, 2]
    ax.semilogy(loss_hist, lw=1.5, color='navy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'inverse_problem_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {out_path}")

    # Summary
    g_err = abs(model.g - g_true) / g_true * 100
    L_err = abs(model.L - L_true) / L_true * 100
    gL_err = abs(model.g_over_L - g_true / L_true) / (g_true / L_true) * 100
    print(f"\n{'='*55}")
    print("Parameter Recovery Summary")
    print(f"{'='*55}")
    print(f"  g:   {model.g:.4f}  (true {g_true})   error: {g_err:.2f}%")
    print(f"  L:   {model.L:.4f}  (true {L_true})   error: {L_err:.2f}%")
    print(f"  g/L: {model.g_over_L:.4f}  (true {g_true/L_true:.2f})   error: {gL_err:.2f}%")


if __name__ == '__main__':
    main()
