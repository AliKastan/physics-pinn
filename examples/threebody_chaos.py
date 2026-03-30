"""
Three-Body Chaos Demonstration
===============================

Shows sensitivity to initial conditions — the hallmark of chaos.
Two simulations start with a perturbation of epsilon=1e-6 in one
coordinate.  The trajectories diverge exponentially.

Also estimates the maximum Lyapunov exponent from the divergence rate.

Usage:
    python examples/threebody_chaos.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from src.models.threebody_pinn import (
    figure_eight_ics, solve_threebody,
    compute_energy, compute_angular_momentum,
)


def main():
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Setup: figure-eight orbit with and without tiny perturbation
    # ------------------------------------------------------------------
    state0, masses, period = figure_eight_ics()
    t_max = 3.0  # about half a period — enough to see divergence

    epsilon = 1e-6
    state0_pert = list(state0)
    state0_pert[0] += epsilon  # perturb x1 by epsilon

    print(f"Figure-eight three-body orbit")
    print(f"  Period ~ {period:.4f}")
    print(f"  Perturbation: epsilon = {epsilon}")
    print(f"  Simulating t in [0, {t_max}]")
    print()

    # ------------------------------------------------------------------
    # Solve both
    # ------------------------------------------------------------------
    print("Solving unperturbed system...")
    t, states = solve_threebody(state0, masses, t_max, n_points=5000)
    print("Solving perturbed system...")
    _, states_pert = solve_threebody(state0_pert, masses, t_max, n_points=5000)

    # ------------------------------------------------------------------
    # Compute divergence
    # ------------------------------------------------------------------
    # Total phase-space distance between the two trajectories
    diff = states - states_pert
    divergence = np.sqrt(np.sum(diff ** 2, axis=1))

    # Estimate Lyapunov exponent from initial exponential growth
    # d(t) ~ epsilon * exp(lambda * t)  =>  log(d/epsilon) ~ lambda * t
    valid = divergence > 1e-14
    log_div = np.log(divergence[valid] / epsilon)
    t_valid = t[valid]
    # Use the first half where growth is roughly exponential
    n_fit = len(t_valid) // 3
    if n_fit > 10:
        coeffs = np.polyfit(t_valid[:n_fit], log_div[:n_fit], 1)
        lyapunov = coeffs[0]
    else:
        lyapunov = float('nan')

    print(f"\nEstimated max Lyapunov exponent: {lyapunov:.3f}")
    if lyapunov > 0:
        print(f"  Positive => CHAOTIC (doubling time ~ {np.log(2)/lyapunov:.3f})")
    print()

    # Conservation quantities
    E_orig = compute_energy(states, masses)
    E_pert = compute_energy(states_pert, masses)
    L_orig = compute_angular_momentum(states, masses)
    L_pert = compute_angular_momentum(states_pert, masses)

    # ------------------------------------------------------------------
    # Plot: 2x3 figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f'Three-Body Chaos: Sensitivity to Initial Conditions '
        f'(epsilon = {epsilon})',
        fontsize=14, fontweight='bold',
    )
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['Body 1', 'Body 2', 'Body 3']

    # (0,0) Unperturbed trajectories
    ax = axes[0, 0]
    for i, (c, lab) in enumerate(zip(colors, labels)):
        ax.plot(states[:, 2 * i], states[:, 2 * i + 1],
                '-', color=c, lw=1.5, label=lab)
        ax.plot(state0[2 * i], state0[2 * i + 1], 'o', color=c, ms=6)
    ax.set_title('Unperturbed Trajectories')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal'); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,1) Perturbed trajectories
    ax = axes[0, 1]
    for i, (c, lab) in enumerate(zip(colors, labels)):
        ax.plot(states_pert[:, 2 * i], states_pert[:, 2 * i + 1],
                '-', color=c, lw=1.5, label=lab)
        ax.plot(state0_pert[2 * i], state0_pert[2 * i + 1], 'o', color=c, ms=6)
    ax.set_title(f'Perturbed (epsilon={epsilon})')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal'); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,2) Phase-space divergence
    ax = axes[0, 2]
    ax.semilogy(t, divergence + 1e-20, 'k-', lw=1.5)
    if not np.isnan(lyapunov):
        t_fit = np.linspace(0, t_valid[n_fit], 200)
        ax.semilogy(t_fit, epsilon * np.exp(lyapunov * t_fit),
                     'r--', lw=1.5, label=f'exp fit: lambda={lyapunov:.2f}')
        ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Phase-space distance')
    ax.set_title('Trajectory Divergence (chaos)')
    ax.grid(True, alpha=0.3)

    # (1,0) Per-body position divergence
    ax = axes[1, 0]
    for i, (c, lab) in enumerate(zip(colors, labels)):
        pos_diff = np.sqrt((states[:, 2*i] - states_pert[:, 2*i]) ** 2 +
                           (states[:, 2*i+1] - states_pert[:, 2*i+1]) ** 2)
        ax.semilogy(t, pos_diff + 1e-20, '-', color=c, lw=1.5, label=lab)
    ax.set_xlabel('Time')
    ax.set_ylabel('Position divergence')
    ax.set_title('Per-Body Divergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Energy conservation (both should be excellent)
    ax = axes[1, 1]
    E0 = E_orig[0]
    dE_orig = np.abs((E_orig - E0) / (np.abs(E0) + 1e-16))
    dE_pert = np.abs((E_pert - E_pert[0]) / (np.abs(E_pert[0]) + 1e-16))
    ax.semilogy(t, dE_orig + 1e-16, 'b-', lw=1.5, label='Unperturbed')
    ax.semilogy(t, dE_pert + 1e-16, 'r-', lw=1.5, label='Perturbed')
    ax.set_xlabel('Time')
    ax.set_ylabel('|dE/E0|')
    ax.set_title('Energy Conservation (both excellent)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) Angular momentum conservation
    ax = axes[1, 2]
    L0 = L_orig[0]
    dL_orig = np.abs((L_orig - L0) / (np.abs(L0) + 1e-16))
    dL_pert = np.abs((L_pert - L_pert[0]) / (np.abs(L_pert[0]) + 1e-16))
    ax.semilogy(t, dL_orig + 1e-16, 'b-', lw=1.5, label='Unperturbed')
    ax.semilogy(t, dL_pert + 1e-16, 'r-', lw=1.5, label='Perturbed')
    ax.set_xlabel('Time')
    ax.set_ylabel('|dL/L0|')
    ax.set_title('Angular Momentum Conservation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'threebody_chaos_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out_path}")

    print(f"\n{'='*55}")
    print("Chaos Summary")
    print(f"{'='*55}")
    print(f"  Max Lyapunov exponent: {lyapunov:.3f}")
    print(f"  Max divergence: {np.max(divergence):.6f}")
    print(f"  Solver energy conservation: {np.max(dE_orig):.2e}")
    print(f"\n  Key insight: both trajectories perfectly conserve energy")
    print(f"  and angular momentum, yet they diverge completely.")
    print(f"  Conservation laws do NOT prevent chaos.")


if __name__ == '__main__':
    main()
