"""Matplotlib visualization helpers for PINN results."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt


def plot_pendulum_results(t_eval, theta_pinn, omega_pinn,
                          theta_ode, omega_ode, t_ode,
                          save_path='pendulum_pinn_results.png'):
    """Compare PINN vs classical solution for the pendulum."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PINN vs Classical ODE Solver — Simple Pendulum',
                 fontsize=14, fontweight='bold')

    # Angular displacement
    ax1 = axes[0, 0]
    ax1.plot(t_ode, np.degrees(theta_ode), 'b-', linewidth=2, label='Classical (RK45)')
    ax1.plot(t_eval, np.degrees(theta_pinn), 'r--', linewidth=2, label='PINN')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Angular Displacement θ(t)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Angular velocity
    ax2 = axes[0, 1]
    ax2.plot(t_ode, omega_ode, 'b-', linewidth=2, label='Classical (RK45)')
    ax2.plot(t_eval, omega_pinn, 'r--', linewidth=2, label='PINN')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Angular Velocity ω(t)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Phase portrait
    ax3 = axes[1, 0]
    ax3.plot(np.degrees(theta_ode), omega_ode, 'b-', linewidth=2, label='Classical')
    ax3.plot(np.degrees(theta_pinn), omega_pinn, 'r--', linewidth=2, label='PINN')
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Angular Velocity (rad/s)')
    ax3.set_title('Phase Portrait (ω vs θ)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Error
    theta_error = np.abs(np.degrees(theta_pinn) - np.degrees(theta_ode))
    omega_error = np.abs(omega_pinn - omega_ode)
    ax4 = axes[1, 1]
    ax4.semilogy(t_eval, theta_error, 'g-', linewidth=2, label='|Δθ| (degrees)')
    ax4.semilogy(t_eval, omega_error, 'm-', linewidth=2, label='|Δω| (rad/s)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('PINN Prediction Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_orbital_results(t_eval, x_pinn, y_pinn, vx_pinn, vy_pinn,
                         x_ode, y_ode, vx_ode, vy_ode, t_ode,
                         E_pinn, E_ode, L_pinn, L_ode,
                         save_path='orbital_pinn_results.png'):
    """Compare PINN vs classical solver for orbital mechanics."""
    E0 = E_ode[0]
    L0 = L_ode[0]
    x0 = x_ode[0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('PINN vs Classical Solver — Two-Body Orbital Mechanics',
                 fontsize=15, fontweight='bold')

    # Orbital trajectory
    ax1 = axes[0, 0]
    ax1.plot(x_ode, y_ode, 'b-', linewidth=2, label='Classical (RK45)')
    ax1.plot(x_pinn, y_pinn, 'r--', linewidth=2, label='PINN')
    ax1.plot(0, 0, 'ko', markersize=10, label='Central body')
    ax1.plot(x0, 0, 'g*', markersize=12, label='Start (perihelion)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Orbital Trajectory')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Position components
    ax2 = axes[0, 1]
    ax2.plot(t_ode, x_ode, 'b-', linewidth=1.5, label='x Classical')
    ax2.plot(t_eval, x_pinn, 'r--', linewidth=1.5, label='x PINN')
    ax2.plot(t_ode, y_ode, 'c-', linewidth=1.5, label='y Classical')
    ax2.plot(t_eval, y_pinn, 'm--', linewidth=1.5, label='y PINN')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position')
    ax2.set_title('Position Components x(t), y(t)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Energy
    ax3 = axes[0, 2]
    ax3.plot(t_ode, E_ode, 'b-', linewidth=2, label='Classical (RK45)')
    ax3.plot(t_eval, E_pinn, 'r-', linewidth=2, label='PINN')
    ax3.axhline(y=E0, color='gray', linestyle=':', alpha=0.5, label=f'True E = {E0:.4f}')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Total Energy (T + V)')
    ax3.set_title('Energy Conservation')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Relative energy error
    dE_ode = np.abs((E_ode - E0) / E0)
    dE_pinn = np.abs((E_pinn - E0) / E0)
    ax4 = axes[1, 0]
    ax4.semilogy(t_ode, dE_ode + 1e-16, 'b-', linewidth=2, label='Classical (RK45)')
    ax4.semilogy(t_eval, dE_pinn + 1e-16, 'r-', linewidth=2, label='PINN')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('|ΔE / E₀|')
    ax4.set_title('Relative Energy Error')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Angular momentum
    ax5 = axes[1, 1]
    ax5.plot(t_ode, L_ode, 'b-', linewidth=2, label='Classical (RK45)')
    ax5.plot(t_eval, L_pinn, 'r-', linewidth=2, label='PINN')
    ax5.axhline(y=L0, color='gray', linestyle=':', alpha=0.5, label=f'True L = {L0:.4f}')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Angular Momentum L')
    ax5.set_title('Angular Momentum Conservation')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Position error
    pos_error = np.sqrt((x_pinn - x_ode) ** 2 + (y_pinn - y_ode) ** 2)
    ax6 = axes[1, 2]
    ax6.semilogy(t_eval, pos_error + 1e-16, 'g-', linewidth=2)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Position Error |Δr|')
    ax6.set_title('PINN Position Error vs Classical')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_training_loss(loss_history, title='PINN Training Loss',
                       save_path='training_loss.png', color='navy'):
    """Plot training loss on a log scale."""
    plt.figure(figsize=(8, 5))
    plt.semilogy(loss_history, linewidth=1.5, color=color)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")
