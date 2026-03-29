"""
HNN vs Standard PINN — Pendulum Comparison
===========================================

Trains both a Hamiltonian Neural Network and a standard PINN on the
simple pendulum problem, then compares:

  1. Trajectory accuracy vs scipy ground truth
  2. Energy conservation over time (HNN should be dramatically better)
  3. Training convergence curves

Reference:
    Greydanus, Dzamba & Cranmer, "Hamiltonian Neural Networks", NeurIPS 2019.

Usage:
    python examples/hnn_vs_pinn_pendulum.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.pendulum_pinn import PINNPendulum
from src.models.hnn import (
    HamiltonianNN, generate_pendulum_data, train_hnn, integrate_hnn,
    compute_hamiltonian,
)
from src.training.trainer import Trainer
from src.training.losses import pendulum_ic_loss
from src.utils.validation import solve_pendulum_ode
from src.utils.metrics import compute_energy_pendulum


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------------
    g = 9.81
    L = 1.0
    m = 1.0
    theta_0 = 1.2   # ~68.8 degrees — large enough to show nonlinear effects
    omega_0 = 0.0
    t_max = 10.0

    # ------------------------------------------------------------------
    # Step 1: Generate HNN training data from true dynamics
    # ------------------------------------------------------------------
    print("Generating training data from true Hamiltonian dynamics...\n")
    q_data, p_data, dqdt_data, dpdt_data, _ = generate_pendulum_data(
        theta_0, omega_0, t_max, g, L, m, n_points=1000,
    )

    # ------------------------------------------------------------------
    # Step 2: Train HNN
    # ------------------------------------------------------------------
    hnn = HamiltonianNN()
    hnn_losses = train_hnn(
        hnn, q_data, p_data, dqdt_data, dpdt_data,
        epochs=5000, lr=1e-3, batch_size=256,
    )

    # ------------------------------------------------------------------
    # Step 3: Train standard PINN for comparison
    # ------------------------------------------------------------------
    print()
    pinn = PINNPendulum()
    trainer = Trainer(
        pinn,
        config={
            "epochs": 5000, "lr": 1e-3, "n_collocation": 500,
            "ic_weight": 20.0, "t_max": t_max,
        },
        physics_params={"g": g, "L": L},
    )
    ic_fn = lambda: pendulum_ic_loss(pinn, theta_0, omega_0)
    pinn_losses, _ = trainer.train(ic_fn, verbose=True)

    # ------------------------------------------------------------------
    # Step 4: Evaluate
    # ------------------------------------------------------------------
    t_eval = np.linspace(0, t_max, 1000)

    # Ground truth
    _, theta_true, omega_true = solve_pendulum_ode(
        theta_0, omega_0, (0, t_max), t_eval, g=g, L=L,
    )
    q_true = theta_true
    p_true = m * L ** 2 * omega_true

    # HNN trajectory
    q0 = theta_0
    p0 = m * L ** 2 * omega_0
    q_hnn, p_hnn = integrate_hnn(hnn, q0, p0, t_eval)
    omega_hnn = p_hnn / (m * L ** 2)

    # PINN trajectory
    pinn.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        pinn_out = pinn(t_tensor).numpy()
    theta_pinn = pinn_out[:, 0]
    omega_pinn = pinn_out[:, 1]
    p_pinn = m * L ** 2 * omega_pinn

    # Energies
    E_true = compute_hamiltonian(q_true, p_true, g, L, m)
    E_hnn  = compute_hamiltonian(q_hnn, p_hnn, g, L, m)
    E_pinn = compute_hamiltonian(theta_pinn, p_pinn, g, L, m)
    E0 = E_true[0]

    dE_true = np.abs((E_true - E0) / (np.abs(E0) + 1e-16))
    dE_hnn  = np.abs((E_hnn  - E0) / (np.abs(E0) + 1e-16))
    dE_pinn = np.abs((E_pinn - E0) / (np.abs(E0) + 1e-16))

    # ------------------------------------------------------------------
    # Step 5: Plot — 2x3 figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        'Hamiltonian Neural Network vs Standard PINN — Energy Conservation',
        fontsize=15, fontweight='bold',
    )

    # (0,0) Trajectory
    ax = axes[0, 0]
    ax.plot(t_eval, np.degrees(theta_true), 'b-', lw=2, label='Ground truth', alpha=0.7)
    ax.plot(t_eval, np.degrees(q_hnn), 'g--', lw=2, label='HNN')
    ax.plot(t_eval, np.degrees(theta_pinn), 'r--', lw=2, label='PINN')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Angular Displacement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Phase portrait
    ax = axes[0, 1]
    ax.plot(np.degrees(q_true), omega_true, 'b-', lw=2, label='Ground truth', alpha=0.7)
    ax.plot(np.degrees(q_hnn), omega_hnn, 'g--', lw=2, label='HNN')
    ax.plot(np.degrees(theta_pinn), omega_pinn, 'r--', lw=2, label='PINN')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('Phase Portrait')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,2) Energy over time
    ax = axes[0, 2]
    ax.plot(t_eval, E_true, 'b-', lw=2, label='Ground truth')
    ax.plot(t_eval, E_hnn, 'g-', lw=2, label='HNN')
    ax.plot(t_eval, E_pinn, 'r-', lw=2, label='PINN')
    ax.axhline(y=E0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy Conservation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Relative energy error
    ax = axes[1, 0]
    ax.semilogy(t_eval, dE_true + 1e-16, 'b-', lw=2, label='RK45 (scipy)')
    ax.semilogy(t_eval, dE_hnn + 1e-16, 'g-', lw=2, label='HNN')
    ax.semilogy(t_eval, dE_pinn + 1e-16, 'r-', lw=2, label='PINN')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|dE / E_0|')
    ax.set_title('Relative Energy Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Training convergence
    ax = axes[1, 1]
    ax.semilogy(hnn_losses, 'g-', lw=1, alpha=0.7, label='HNN')
    ax.semilogy(pinn_losses, 'r-', lw=1, alpha=0.7, label='PINN')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,2) Learned Hamiltonian landscape
    ax = axes[1, 2]
    q_grid = np.linspace(-np.pi, np.pi, 100)
    p_grid = np.linspace(-3, 3, 100)
    Q, P = np.meshgrid(q_grid, p_grid)
    Q_flat = torch.tensor(Q.flatten(), dtype=torch.float32).unsqueeze(1)
    P_flat = torch.tensor(P.flatten(), dtype=torch.float32).unsqueeze(1)
    hnn.eval()
    with torch.no_grad():
        H_pred = hnn(Q_flat, P_flat).numpy().reshape(Q.shape)
    H_true_grid = compute_hamiltonian(Q, P, g, L, m)
    ax.contour(np.degrees(Q), P, H_pred, levels=15, colors='green', alpha=0.7)
    ax.contour(np.degrees(Q), P, H_true_grid, levels=15,
               colors='blue', alpha=0.4, linestyles='dashed')
    ax.set_xlabel('q (degrees)')
    ax.set_ylabel('p (angular momentum)')
    ax.set_title('Learned H(q,p)\n(green=HNN, blue dashed=true)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'hnn_vs_pinn_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {out_path}")

    # Summary
    print(f"\n{'='*55}")
    print("Energy Conservation Comparison")
    print(f"{'='*55}")
    print(f"  Max |dE/E0| — RK45:          {np.max(dE_true):.2e}")
    print(f"  Max |dE/E0| — HNN:           {np.max(dE_hnn):.2e}")
    print(f"  Max |dE/E0| — Standard PINN: {np.max(dE_pinn):.2e}")
    ratio = np.max(dE_pinn) / (np.max(dE_hnn) + 1e-16)
    print(f"\n  HNN is ~{ratio:.0f}x better at energy conservation")


if __name__ == '__main__':
    main()
