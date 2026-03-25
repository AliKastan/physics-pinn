"""
Hamiltonian Neural Network (HNN) for Simple Pendulum
=====================================================

Background:
    In Hamiltonian mechanics, the state of a system is described by
    generalized coordinates q and conjugate momenta p. The Hamiltonian
    H(q, p) represents the total energy, and the equations of motion are:

        dq/dt =  dH/dp     (Hamilton's first equation)
        dp/dt = -dH/dq     (Hamilton's second equation)

    For a simple pendulum:
        q = theta (angle)
        p = m * L^2 * omega (angular momentum)
        H = p^2 / (2*m*L^2) - m*g*L*cos(q)    (kinetic + potential energy)

    The key insight: if a neural network learns H(q, p), the equations of
    motion derived via autograd *automatically conserve energy* because
    the time derivative of H along any trajectory is:

        dH/dt = (dH/dq)(dq/dt) + (dH/dp)(dp/dt)
              = (dH/dq)(dH/dp) + (dH/dp)(-dH/dq)
              = 0

    This is exact regardless of the network architecture or training quality.
    Energy conservation is built into the structure, not learned.

HNN vs Standard PINN:
    - Standard PINN: Learns theta(t), omega(t) directly. Energy conservation
      is only as good as the training loss — it can drift over long horizons.
    - HNN: Learns H(q, p). Energy is conserved by construction via the
      symplectic structure. The trade-off is that the HNN needs training data
      (or a known Hamiltonian form) rather than just the ODE.

    This file trains both approaches and compares their energy conservation
    over long time horizons to demonstrate the HNN's structural advantage.

Reference:
    Greydanus, S., Dzamba, M., & Cranmer, M. (2019).
    "Hamiltonian Neural Networks." NeurIPS 2019.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# =============================================================================
# 1. Hamiltonian Neural Network
# =============================================================================

class HamiltonianNet(nn.Module):
    """
    A neural network that learns the Hamiltonian H(q, p).

    Input:  (q, p) — generalized coordinate and momentum (2 neurons)
    Hidden: 3 layers x 64 neurons, tanh activation
    Output: H — scalar energy (1 neuron)

    The network does NOT predict the trajectory directly. Instead, we
    derive dq/dt and dp/dt from H using automatic differentiation:
        dq/dt =  dH/dp
        dp/dt = -dH/dq
    """

    def __init__(self, hidden_size=64, num_hidden_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, q, p):
        """
        Compute H(q, p).

        Args:
            q: Generalized coordinate (N, 1)
            p: Conjugate momentum (N, 1)

        Returns:
            H: Hamiltonian value (N, 1)
        """
        x = torch.cat([q, p], dim=1)
        return self.network(x)

    def time_derivatives(self, q, p):
        """
        Compute Hamilton's equations via autograd:
            dq/dt =  dH/dp
            dp/dt = -dH/dq

        Args:
            q, p: State variables with requires_grad=True

        Returns:
            dqdt, dpdt: Time derivatives
        """
        H = self.forward(q, p)

        dH_dq = torch.autograd.grad(
            H, q, grad_outputs=torch.ones_like(H),
            create_graph=True
        )[0]
        dH_dp = torch.autograd.grad(
            H, p, grad_outputs=torch.ones_like(H),
            create_graph=True
        )[0]

        dqdt = dH_dp       # Hamilton's first equation
        dpdt = -dH_dq      # Hamilton's second equation
        return dqdt, dpdt


# =============================================================================
# 2. Standard PINN (for comparison)
# =============================================================================

class PINNPendulum(nn.Module):
    """Standard PINN: maps t -> (theta, omega). Same as pinn_pendulum.py."""

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
# 3. Generate Training Data from True Hamiltonian
# =============================================================================

def generate_pendulum_data(theta_0, omega_0, t_max, g=9.81, L=1.0, m=1.0,
                           n_points=1000):
    """
    Solve the pendulum ODE and return (q, p, dq/dt, dp/dt) pairs.

    The HNN learns from state-derivative pairs: given (q, p), what are
    the correct time derivatives? This is equivalent to learning the
    vector field of the Hamiltonian system.

    Args:
        theta_0, omega_0: Initial conditions
        t_max: Time horizon
        g, L, m: Physical parameters
        n_points: Number of data points

    Returns:
        q, p: State arrays (n_points,)
        dqdt, dpdt: Time derivative arrays (n_points,)
        t_eval: Time array (n_points,)
    """
    t_eval = np.linspace(0, t_max, n_points)

    sol = solve_ivp(
        lambda t, y: [y[1], -(g / L) * np.sin(y[0])],
        (0, t_max), [theta_0, omega_0],
        t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12
    )

    theta = sol.y[0]
    omega = sol.y[1]

    # Convert to Hamiltonian variables
    # q = theta, p = m * L^2 * omega
    q = theta
    p = m * L ** 2 * omega

    # True time derivatives from Hamilton's equations
    # dq/dt = dH/dp = p / (m * L^2)
    # dp/dt = -dH/dq = -m * g * L * sin(q)
    dqdt = p / (m * L ** 2)
    dpdt = -m * g * L * np.sin(q)

    return q, p, dqdt, dpdt, t_eval


# =============================================================================
# 4. HNN Training
# =============================================================================

def train_hnn(q_data, p_data, dqdt_data, dpdt_data,
              epochs=5000, lr=1e-3, batch_size=256):
    """
    Train the HNN to predict correct time derivatives from (q, p) states.

    Loss = MSE between predicted (dq/dt, dp/dt) and true (dq/dt, dp/dt).

    The network learns H(q,p) indirectly: it never sees the Hamiltonian
    value, only the derivatives that H must produce via Hamilton's equations.

    Args:
        q_data, p_data: Training state data
        dqdt_data, dpdt_data: Training derivative data
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Mini-batch size

    Returns:
        model: Trained HNN
        loss_history: Training loss per epoch
    """
    model = HamiltonianNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6
    )

    # Convert to tensors
    q_tensor = torch.tensor(q_data, dtype=torch.float32).unsqueeze(1)
    p_tensor = torch.tensor(p_data, dtype=torch.float32).unsqueeze(1)
    dqdt_tensor = torch.tensor(dqdt_data, dtype=torch.float32).unsqueeze(1)
    dpdt_tensor = torch.tensor(dpdt_data, dtype=torch.float32).unsqueeze(1)

    n_samples = len(q_data)
    loss_history = []

    print(f"Training Hamiltonian Neural Network")
    print(f"  Training samples: {n_samples}")
    print(f"  Epochs: {epochs}")
    print("-" * 50)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Mini-batch sampling
        idx = torch.randint(0, n_samples, (batch_size,))
        q_batch = q_tensor[idx].requires_grad_(True)
        p_batch = p_tensor[idx].requires_grad_(True)

        # Predict derivatives via Hamilton's equations
        dqdt_pred, dpdt_pred = model.time_derivatives(q_batch, p_batch)

        # Loss: how well do the predicted derivatives match the true ones?
        loss = (torch.mean((dqdt_pred - dqdt_tensor[idx]) ** 2) +
                torch.mean((dpdt_pred - dpdt_tensor[idx]) ** 2))

        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        loss_history.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | Loss: {loss.item():.8f}")

    print("-" * 50)
    print(f"Final loss: {loss_history[-1]:.8f}")
    return model, loss_history


# =============================================================================
# 5. Train Standard PINN (for comparison)
# =============================================================================

def train_standard_pinn(theta_0, omega_0, t_max, g=9.81, L=1.0,
                        n_collocation=500, epochs=5000, lr=1e-3,
                        ic_weight=20.0):
    """Train a standard PINN for comparison. Same approach as pinn_pendulum.py."""
    model = PINNPendulum()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6
    )
    loss_history = []

    print(f"Training Standard PINN (for comparison)")
    print(f"  Epochs: {epochs}")
    print("-" * 50)

    for epoch in range(epochs):
        optimizer.zero_grad()

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

        residual_1 = dtheta_dt - omega
        residual_2 = domega_dt + (g / L) * torch.sin(theta)
        loss_physics = torch.mean(residual_1 ** 2) + torch.mean(residual_2 ** 2)

        t_zero = torch.zeros(1, 1)
        output_0 = model(t_zero)
        loss_ic = ((output_0[0, 0] - theta_0) ** 2 +
                   (output_0[0, 1] - omega_0) ** 2)

        total_loss = loss_physics + ic_weight * loss_ic
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        loss_history.append(total_loss.item())

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | Loss: {total_loss.item():.6f}")

    print("-" * 50)
    print(f"Final loss: {loss_history[-1]:.6f}")
    return model, loss_history


# =============================================================================
# 6. Integrate HNN trajectory using learned dynamics
# =============================================================================

def integrate_hnn(hnn_model, q0, p0, t_eval):
    """
    Integrate the HNN's learned dynamics forward in time using RK45.

    The HNN defines a vector field (dq/dt, dp/dt) at every point (q, p).
    We use scipy's ODE solver to follow this vector field from the initial
    state, producing a trajectory.

    Args:
        hnn_model: Trained HamiltonianNet
        q0, p0: Initial state (scalars)
        t_eval: Time points for evaluation

    Returns:
        q_traj, p_traj: Trajectory arrays
    """
    hnn_model.eval()

    def hnn_rhs(t, state):
        q, p = state
        q_t = torch.tensor([[q]], dtype=torch.float32, requires_grad=True)
        p_t = torch.tensor([[p]], dtype=torch.float32, requires_grad=True)
        with torch.enable_grad():
            dqdt, dpdt = hnn_model.time_derivatives(q_t, p_t)
        return [dqdt.item(), dpdt.item()]

    sol = solve_ivp(
        hnn_rhs, (t_eval[0], t_eval[-1]), [q0, p0],
        t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10
    )
    return sol.y[0], sol.y[1]


# =============================================================================
# 7. Compute Energy
# =============================================================================

def compute_energy(q, p, g=9.81, L=1.0, m=1.0):
    """
    True pendulum Hamiltonian: H = p^2/(2mL^2) - mgL*cos(q)

    Args:
        q: Angle array
        p: Angular momentum array
        g, L, m: Physical parameters

    Returns:
        H: Energy array
    """
    kinetic = p ** 2 / (2 * m * L ** 2)
    potential = -m * g * L * np.cos(q)
    return kinetic + potential


# =============================================================================
# 8. Visualization
# =============================================================================

def visualize_hnn_results(hnn_model, pinn_model,
                          theta_0, omega_0, t_max,
                          g=9.81, L=1.0, m=1.0):
    """
    Compare HNN vs Standard PINN with 6 plots:
        1. Trajectory comparison (theta vs t)
        2. Phase portrait comparison
        3. Energy conservation comparison (the key plot)
        4. Relative energy error comparison
        5. HNN learned Hamiltonian landscape
        6. Training loss comparison
    """
    t_eval = np.linspace(0, t_max, 1000)

    # --- Ground truth (scipy) ---
    sol = solve_ivp(
        lambda t, y: [y[1], -(g / L) * np.sin(y[0])],
        (0, t_max), [theta_0, omega_0],
        t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12
    )
    theta_true = sol.y[0]
    omega_true = sol.y[1]
    q_true = theta_true
    p_true = m * L ** 2 * omega_true

    # --- HNN trajectory ---
    q0 = theta_0
    p0 = m * L ** 2 * omega_0
    q_hnn, p_hnn = integrate_hnn(hnn_model, q0, p0, t_eval)
    omega_hnn = p_hnn / (m * L ** 2)

    # --- Standard PINN trajectory ---
    pinn_model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        output = pinn_model(t_tensor).numpy()
    theta_pinn = output[:, 0]
    omega_pinn = output[:, 1]
    p_pinn = m * L ** 2 * omega_pinn

    # --- Compute energies ---
    E_true = compute_energy(q_true, p_true, g, L, m)
    E_hnn = compute_energy(q_hnn, p_hnn, g, L, m)
    E_pinn = compute_energy(theta_pinn, p_pinn, g, L, m)
    E0 = E_true[0]

    dE_true = np.abs((E_true - E0) / (np.abs(E0) + 1e-16))
    dE_hnn = np.abs((E_hnn - E0) / (np.abs(E0) + 1e-16))
    dE_pinn = np.abs((E_pinn - E0) / (np.abs(E0) + 1e-16))

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Hamiltonian Neural Network vs Standard PINN — Energy Conservation',
                 fontsize=15, fontweight='bold')

    # Plot 1: Trajectory
    ax1 = axes[0, 0]
    ax1.plot(t_eval, np.degrees(theta_true), 'b-', linewidth=2,
             label='Ground truth', alpha=0.7)
    ax1.plot(t_eval, np.degrees(q_hnn), 'g--', linewidth=2, label='HNN')
    ax1.plot(t_eval, np.degrees(theta_pinn), 'r--', linewidth=2, label='PINN')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Angular Displacement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Phase portrait
    ax2 = axes[0, 1]
    ax2.plot(np.degrees(q_true), omega_true, 'b-', linewidth=2,
             label='Ground truth', alpha=0.7)
    ax2.plot(np.degrees(q_hnn), omega_hnn, 'g--', linewidth=2, label='HNN')
    ax2.plot(np.degrees(theta_pinn), omega_pinn, 'r--', linewidth=2, label='PINN')
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Phase Portrait')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Energy over time (THE key comparison)
    ax3 = axes[0, 2]
    ax3.plot(t_eval, E_true, 'b-', linewidth=2, label='Ground truth')
    ax3.plot(t_eval, E_hnn, 'g-', linewidth=2, label='HNN')
    ax3.plot(t_eval, E_pinn, 'r-', linewidth=2, label='PINN')
    ax3.axhline(y=E0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Total Energy')
    ax3.set_title('Energy Conservation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Relative energy error
    ax4 = axes[1, 0]
    ax4.semilogy(t_eval, dE_true + 1e-16, 'b-', linewidth=2,
                 label='RK45 (scipy)')
    ax4.semilogy(t_eval, dE_hnn + 1e-16, 'g-', linewidth=2, label='HNN')
    ax4.semilogy(t_eval, dE_pinn + 1e-16, 'r-', linewidth=2, label='PINN')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('|dE / E_0|')
    ax4.set_title('Relative Energy Error')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Learned Hamiltonian landscape
    ax5 = axes[1, 1]
    q_grid = np.linspace(-np.pi, np.pi, 100)
    p_grid = np.linspace(-3, 3, 100)
    Q, P = np.meshgrid(q_grid, p_grid)
    Q_flat = torch.tensor(Q.flatten(), dtype=torch.float32).unsqueeze(1)
    P_flat = torch.tensor(P.flatten(), dtype=torch.float32).unsqueeze(1)

    hnn_model.eval()
    with torch.no_grad():
        H_pred = hnn_model(Q_flat, P_flat).numpy().reshape(Q.shape)

    # True Hamiltonian for comparison
    H_true = P ** 2 / (2 * m * L ** 2) - m * g * L * np.cos(Q)

    contour_pred = ax5.contour(np.degrees(Q), P, H_pred, levels=15,
                               colors='green', alpha=0.7)
    contour_true = ax5.contour(np.degrees(Q), P, H_true, levels=15,
                               colors='blue', alpha=0.4, linestyles='dashed')
    ax5.clabel(contour_pred, inline=True, fontsize=6)
    ax5.set_xlabel('q (degrees)')
    ax5.set_ylabel('p (angular momentum)')
    ax5.set_title('Learned H(q,p) Landscape\n(green=HNN, blue dashed=true)')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary_text = (
        f"Energy Conservation Summary\n"
        f"{'='*40}\n\n"
        f"Initial Energy E_0 = {E0:.4f}\n\n"
        f"Max |dE/E_0|:\n"
        f"  RK45 (scipy): {np.max(dE_true):.2e}\n"
        f"  HNN:          {np.max(dE_hnn):.2e}\n"
        f"  Standard PINN:{np.max(dE_pinn):.2e}\n\n"
        f"Max trajectory error vs truth:\n"
        f"  HNN:  {np.max(np.abs(q_hnn - q_true)):.6f} rad\n"
        f"  PINN: {np.max(np.abs(theta_pinn - theta_true)):.6f} rad\n\n"
        f"HNN advantage: energy conservation\n"
        f"is structural, not learned."
    )
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('hnn_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to hnn_results.png")

    # Print summary
    print(f"\n{'='*55}")
    print("Energy Conservation Comparison")
    print(f"{'='*55}")
    print(f"  Max |dE/E_0| — RK45:          {np.max(dE_true):.2e}")
    print(f"  Max |dE/E_0| — HNN:           {np.max(dE_hnn):.2e}")
    print(f"  Max |dE/E_0| — Standard PINN: {np.max(dE_pinn):.2e}")
    print(f"\n  HNN energy error / PINN energy error = "
          f"{np.max(dE_hnn) / (np.max(dE_pinn) + 1e-16):.4f}")


# =============================================================================
# 9. Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Physical parameters
    g = 9.81
    L = 1.0
    m = 1.0

    # Initial conditions
    theta_0 = np.pi / 4   # 45 degrees
    omega_0 = 0.0          # released from rest
    t_max = 10.0           # 10 seconds

    # Step 1: Generate training data for the HNN
    print("Generating training data from true Hamiltonian dynamics...\n")
    q_data, p_data, dqdt_data, dpdt_data, _ = generate_pendulum_data(
        theta_0, omega_0, t_max, g, L, m, n_points=1000
    )

    # Step 2: Train the HNN
    hnn_model, hnn_losses = train_hnn(
        q_data, p_data, dqdt_data, dpdt_data,
        epochs=5000, lr=1e-3, batch_size=256
    )

    # Step 3: Train the standard PINN for comparison
    print()
    pinn_model, pinn_losses = train_standard_pinn(
        theta_0, omega_0, t_max, g, L,
        n_collocation=500, epochs=5000, lr=1e-3, ic_weight=20.0
    )

    # Step 4: Compare results
    print()
    visualize_hnn_results(
        hnn_model, pinn_model, theta_0, omega_0, t_max, g, L, m
    )
