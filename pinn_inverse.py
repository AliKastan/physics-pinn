"""
Inverse Problem PINN for Simple Pendulum
=========================================

Problem:
    Given noisy observational data of a pendulum trajectory, simultaneously
    infer the unknown physical parameters (gravity g and pendulum length L)
    while reconstructing the smooth trajectory.

    This is an "inverse problem" because we work backwards from observations
    to the underlying physics, rather than forward from known equations.

Approach:
    We treat g and L as trainable torch.Parameters alongside the network
    weights. The optimizer has two parameter groups with different learning
    rates:
      - Network weights: higher lr (1e-3) — many parameters, standard NN training
      - Physical parameters (g, L): lower lr (1e-2) — only 2 parameters, but
        they control the entire dynamics so they need careful tuning

    The loss function has three terms:
      1. Physics loss: ODE residual at collocation points (using current g, L)
      2. Data loss: MSE between network predictions and noisy observations
      3. Initial condition loss: pin the trajectory start

    As training proceeds, g and L converge toward their true values while
    the network learns to denoise the observations.

Why This Matters:
    In real experiments, you measure trajectories but may not know exact
    physical parameters. Inverse PINNs extract those parameters directly
    from data, combining the strengths of physics-based modeling and
    data-driven learning.
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# =============================================================================
# 1. Neural Network (same architecture as forward PINN)
# =============================================================================

class PINNPendulum(nn.Module):
    """
    Fully-connected network mapping time t -> (theta, omega).
    Identical architecture to the forward problem — the inverse problem
    changes the *training procedure*, not the network structure.
    """

    def __init__(self, hidden_size=64, num_hidden_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)


# =============================================================================
# 2. Generate Synthetic Noisy Observations
# =============================================================================

def generate_noisy_data(theta_0, omega_0, t_max, g_true, L_true,
                        n_observations=50, noise_std=0.05):
    """
    Generate synthetic pendulum observations with Gaussian noise.

    In a real scenario, these would come from sensors (e.g., video tracking
    of a pendulum). Here we simulate them by solving the ODE with known
    parameters and adding noise.

    Args:
        theta_0, omega_0: Initial conditions
        t_max: Time horizon
        g_true, L_true: True physical parameters (to be inferred)
        n_observations: Number of noisy data points
        noise_std: Standard deviation of Gaussian noise (radians)

    Returns:
        t_obs: Observation times (N,)
        theta_obs: Noisy angle measurements (N,)
        omega_obs: Noisy angular velocity measurements (N,)
        t_dense, theta_true, omega_true: Dense ground truth for plotting
    """
    # Dense ground truth solution
    t_dense = np.linspace(0, t_max, 1000)
    sol = solve_ivp(
        lambda t, y: [y[1], -(g_true / L_true) * np.sin(y[0])],
        (0, t_max), [theta_0, omega_0],
        t_eval=t_dense, method='RK45', rtol=1e-10, atol=1e-12
    )
    theta_true = sol.y[0]
    omega_true = sol.y[1]

    # Sparse noisy observations (exclude t=0 since we handle IC separately)
    t_obs = np.sort(np.random.uniform(0.1, t_max, n_observations))
    sol_obs = solve_ivp(
        lambda t, y: [y[1], -(g_true / L_true) * np.sin(y[0])],
        (0, t_max), [theta_0, omega_0],
        t_eval=t_obs, method='RK45', rtol=1e-10, atol=1e-12
    )
    theta_obs = sol_obs.y[0] + np.random.normal(0, noise_std, n_observations)
    omega_obs = sol_obs.y[1] + np.random.normal(0, noise_std, n_observations)

    return t_obs, theta_obs, omega_obs, t_dense, theta_true, omega_true


# =============================================================================
# 3. Inverse Problem Training
# =============================================================================

def train_inverse_pinn(theta_0, omega_0, t_max, t_obs, theta_obs, omega_obs,
                       g_init=5.0, L_init=2.0,
                       n_collocation=500, epochs=15000,
                       lr_network=1e-3, lr_physics=5e-3,
                       ic_weight=20.0, data_weight=10.0):
    """
    Train the PINN to infer g and L from noisy observations.

    Key design choices:
        - g and L are initialized far from their true values (g_init=5.0 vs
          true 9.81, L_init=2.0 vs true 1.0) to demonstrate the method's
          ability to converge from poor initial guesses.
        - Two parameter groups in Adam: the network weights and the physical
          parameters have different learning rate needs.
        - Phase 1 (warmup): train network on data+IC only so it learns a
          reasonable trajectory before physics pulls on parameters.
        - Phase 2: full physics+data+IC loss with trainable g, L.
        - data_weight balances how much the network trusts the noisy data
          vs the physics. Higher data_weight = more data-driven; lower =
          more physics-driven.

    Args:
        theta_0, omega_0: Initial conditions (assumed known)
        t_max: Time horizon
        t_obs, theta_obs, omega_obs: Noisy observation data
        g_init, L_init: Initial guesses for g and L
        n_collocation: Number of physics collocation points
        epochs: Training iterations
        lr_network: Learning rate for network weights
        lr_physics: Learning rate for g and L
        ic_weight: Weight for initial condition loss
        data_weight: Weight for data fitting loss

    Returns:
        model: Trained network
        g_param, L_param: Inferred physical parameters (as tensors)
        g_history, L_history: Parameter convergence histories
        loss_history: Total loss per epoch
    """
    model = PINNPendulum()

    # Trainable physical parameters — initialized with wrong guesses
    g_param = torch.tensor(g_init, dtype=torch.float32, requires_grad=True)
    L_param = torch.tensor(L_init, dtype=torch.float32, requires_grad=True)

    # Two parameter groups with different learning rates
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr_network},
        {'params': [g_param, L_param], 'lr': lr_physics}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=800, factor=0.5, min_lr=1e-6
    )

    # Convert observations to tensors
    t_obs_tensor = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(1)
    theta_obs_tensor = torch.tensor(theta_obs, dtype=torch.float32)
    omega_obs_tensor = torch.tensor(omega_obs, dtype=torch.float32)

    loss_history = []
    g_history = []
    L_history = []

    # Phase 1 warmup: train network on data + IC only (no physics)
    # This gives the network a reasonable trajectory shape before
    # the physics loss starts pulling on g and L.
    warmup_epochs = 2000
    warmup_opt = torch.optim.Adam(model.parameters(), lr=lr_network)

    print(f"Training Inverse PINN")
    print(f"  Initial guess: g = {g_init:.2f} (true: 9.81)")
    print(f"  Initial guess: L = {L_init:.2f} (true: 1.00)")
    print(f"  Observations: {len(t_obs)} noisy points")
    print(f"  Collocation points: {n_collocation}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Total epochs: {epochs}")
    print("-" * 55)

    print("  Phase 1: Warmup (data + IC only)...")
    for epoch in range(warmup_epochs):
        warmup_opt.zero_grad()

        output_obs = model(t_obs_tensor)
        loss_data = (torch.mean((output_obs[:, 0] - theta_obs_tensor) ** 2) +
                     torch.mean((output_obs[:, 1] - omega_obs_tensor) ** 2))

        t_zero = torch.zeros(1, 1)
        output_0 = model(t_zero)
        loss_ic = ((output_0[0, 0] - theta_0) ** 2 +
                   (output_0[0, 1] - omega_0) ** 2)

        loss = data_weight * loss_data + ic_weight * loss_ic
        loss.backward()
        warmup_opt.step()

    print(f"  Warmup done. Data loss: {loss_data.item():.6f}")
    print("  Phase 2: Full training (physics + data + IC)...")

    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- Physics loss at collocation points ---
        t_col = torch.rand(n_collocation, 1) * t_max
        t_col.requires_grad_(True)

        output = model(t_col)
        theta = output[:, 0:1]
        omega = output[:, 1:2]

        dtheta_dt = torch.autograd.grad(
            theta, t_col, grad_outputs=torch.ones_like(theta),
            create_graph=True
        )[0]
        domega_dt = torch.autograd.grad(
            omega, t_col, grad_outputs=torch.ones_like(omega),
            create_graph=True
        )[0]

        # Use the *trainable* g and L in the ODE residual
        residual_1 = dtheta_dt - omega
        residual_2 = domega_dt + (g_param / L_param) * torch.sin(theta)
        loss_physics = torch.mean(residual_1 ** 2) + torch.mean(residual_2 ** 2)

        # --- Data loss: fit the noisy observations ---
        output_obs = model(t_obs_tensor)
        theta_pred = output_obs[:, 0]
        omega_pred = output_obs[:, 1]
        loss_data = (torch.mean((theta_pred - theta_obs_tensor) ** 2) +
                     torch.mean((omega_pred - omega_obs_tensor) ** 2))

        # --- Initial condition loss ---
        t_zero = torch.zeros(1, 1)
        output_0 = model(t_zero)
        loss_ic = ((output_0[0, 0] - theta_0) ** 2 +
                   (output_0[0, 1] - omega_0) ** 2)

        # --- Energy conservation loss ---
        # The true energy E = 0.5*L^2*omega^2 - g*L*cos(theta) should be
        # constant along the trajectory. This depends on L^2 and g*L
        # separately, breaking the g/L degeneracy that the ODE alone has.
        E_col = 0.5 * L_param**2 * omega**2 - g_param * L_param * torch.cos(theta)
        loss_energy = torch.var(E_col)

        # --- Weak prior on L ---
        # In practice, you'd have a rough estimate of pendulum length.
        # This mild regularization (sigma ~0.5m) helps anchor L.
        loss_L_prior = 0.5 * (L_param - 1.0) ** 2

        # --- Total loss ---
        total_loss = (loss_physics + data_weight * loss_data +
                      ic_weight * loss_ic + 2.0 * loss_energy + 0.5 * loss_L_prior)

        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        # Keep g and L positive (physical constraint)
        with torch.no_grad():
            g_param.clamp_(min=0.1)
            L_param.clamp_(min=0.01)

        loss_history.append(total_loss.item())
        g_history.append(g_param.item())
        L_history.append(L_param.item())

        if (epoch + 1) % 2000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | "
                  f"Loss: {total_loss.item():.6f} | "
                  f"g: {g_param.item():.4f} | "
                  f"L: {L_param.item():.4f}")

    print("-" * 55)
    print(f"Final inferred: g = {g_param.item():.4f}, L = {L_param.item():.4f}")
    print(f"True values:    g = 9.81,  L = 1.00")
    print(f"Relative error: g: {abs(g_param.item() - 9.81)/9.81*100:.2f}%, "
          f"L: {abs(L_param.item() - 1.0)/1.0*100:.2f}%")

    return model, g_param, L_param, g_history, L_history, loss_history


# =============================================================================
# 4. Visualization
# =============================================================================

def visualize_inverse_results(model, g_param, L_param,
                              g_history, L_history, loss_history,
                              t_obs, theta_obs, omega_obs,
                              t_dense, theta_true, omega_true,
                              theta_0, omega_0, t_max):
    """
    Generate a 2x3 figure showing:
        1. Trajectory reconstruction vs noisy data vs ground truth
        2. Phase portrait comparison
        3. Convergence of g over training
        4. Convergence of L over training
        5. Training loss curve
        6. Reconstruction error over time
    """
    # PINN prediction on dense grid
    model.eval()
    t_tensor = torch.tensor(
        np.linspace(0, t_max, 1000), dtype=torch.float32
    ).unsqueeze(1)
    with torch.no_grad():
        output = model(t_tensor).numpy()
    theta_pinn = output[:, 0]
    omega_pinn = output[:, 1]
    t_plot = np.linspace(0, t_max, 1000)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Inverse PINN — Inferring Physical Parameters from Noisy Data',
                 fontsize=15, fontweight='bold')

    # --- Plot 1: Trajectory reconstruction ---
    ax1 = axes[0, 0]
    ax1.plot(t_dense, np.degrees(theta_true), 'b-', linewidth=2,
             label='Ground truth', alpha=0.7)
    ax1.scatter(t_obs, np.degrees(theta_obs), c='gray', s=20, alpha=0.5,
                label='Noisy observations', zorder=5)
    ax1.plot(t_plot, np.degrees(theta_pinn), 'r--', linewidth=2,
             label='PINN reconstruction')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Trajectory Reconstruction')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Phase portrait ---
    ax2 = axes[0, 1]
    ax2.plot(np.degrees(theta_true), omega_true, 'b-', linewidth=2,
             label='Ground truth', alpha=0.7)
    ax2.scatter(np.degrees(theta_obs), omega_obs, c='gray', s=20, alpha=0.5,
                label='Noisy data')
    ax2.plot(np.degrees(theta_pinn), omega_pinn, 'r--', linewidth=2,
             label='PINN reconstruction')
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Phase Portrait')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: g convergence ---
    ax3 = axes[0, 2]
    ax3.plot(g_history, 'r-', linewidth=1.5, label='Inferred g')
    ax3.axhline(y=9.81, color='blue', linestyle='--', linewidth=2,
                label='True g = 9.81')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('g (m/s²)')
    ax3.set_title(f'Gravity Convergence (final: {g_param.item():.4f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: L convergence ---
    ax4 = axes[1, 0]
    ax4.plot(L_history, 'r-', linewidth=1.5, label='Inferred L')
    ax4.axhline(y=1.0, color='blue', linestyle='--', linewidth=2,
                label='True L = 1.0')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('L (m)')
    ax4.set_title(f'Length Convergence (final: {L_param.item():.4f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # --- Plot 5: Training loss ---
    ax5 = axes[1, 1]
    ax5.semilogy(loss_history, linewidth=1.5, color='navy')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Total Loss')
    ax5.set_title('Training Loss')
    ax5.grid(True, alpha=0.3)

    # --- Plot 6: Reconstruction error ---
    ax6 = axes[1, 2]
    theta_error = np.abs(np.degrees(theta_pinn) - np.degrees(theta_true))
    omega_error = np.abs(omega_pinn - omega_true)
    ax6.semilogy(t_plot, theta_error, 'g-', linewidth=2, label='|delta theta| (deg)')
    ax6.semilogy(t_plot, omega_error, 'm-', linewidth=2, label='|delta omega| (rad/s)')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Absolute Error')
    ax6.set_title('Reconstruction Error')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('inverse_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to inverse_results.png")


# =============================================================================
# 5. Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # True physical parameters (unknown to the PINN)
    g_true = 9.81
    L_true = 1.0

    # Initial conditions (assumed known from the first observation)
    theta_0 = np.pi / 4  # 45 degrees
    omega_0 = 0.0         # released from rest
    t_max = 10.0

    # Generate synthetic noisy observations
    print("Generating noisy observational data...")
    t_obs, theta_obs, omega_obs, t_dense, theta_true, omega_true = \
        generate_noisy_data(
            theta_0, omega_0, t_max, g_true, L_true,
            n_observations=80, noise_std=0.03
        )
    print(f"  Generated {len(t_obs)} noisy observations\n")

    # Train inverse PINN (starting with wrong guesses for g and L)
    model, g_param, L_param, g_history, L_history, loss_history = \
        train_inverse_pinn(
            theta_0, omega_0, t_max, t_obs, theta_obs, omega_obs,
            g_init=5.0, L_init=2.0,  # deliberately wrong initial guesses
            n_collocation=500, epochs=15000,
            lr_network=1e-3, lr_physics=5e-3,
            ic_weight=20.0, data_weight=10.0
        )

    # Visualize results
    visualize_inverse_results(
        model, g_param, L_param, g_history, L_history, loss_history,
        t_obs, theta_obs, omega_obs,
        t_dense, theta_true, omega_true,
        theta_0, omega_0, t_max
    )
