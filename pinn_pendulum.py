"""
Physics-Informed Neural Network (PINN) for Simple Pendulum Motion
=================================================================

Physics Background:
    A simple pendulum of length L swinging under gravity g obeys
    the nonlinear ODE:

        d²θ/dt² + (g/L) * sin(θ) = 0

    where θ(t) is the angular displacement from vertical.

    This is a second-order ODE. We can decompose it into two first-order
    equations by introducing angular velocity ω = dθ/dt:

        dθ/dt = ω
        dω/dt = -(g/L) * sin(θ)

PINN Approach:
    Instead of discretizing the ODE on a grid, we train a neural network
    N(t) → (θ, ω) to satisfy:
      1. The differential equation (physics loss)
      2. The initial conditions θ(0) = θ₀, ω(0) = ω₀ (IC loss)

    We compute derivatives via PyTorch's automatic differentiation, so
    the network "learns" physics without any simulation data.
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# =============================================================================
# 1. Neural Network Architecture
# =============================================================================

class PINNPendulum(nn.Module):
    """
    A fully-connected neural network that maps time t → (θ, ω).

    Architecture:
        Input:  t (1 neuron)
        Hidden: 3 layers × 64 neurons, tanh activation
        Output: θ and ω (2 neurons)

    Why tanh?
        - Smooth and infinitely differentiable — essential because the physics
          loss requires computing dθ/dt and dω/dt via backpropagation.
        - ReLU has discontinuous second derivatives, which degrades PINN training.
    """

    def __init__(self, hidden_size=64, num_hidden_layers=3):
        super().__init__()

        layers = []
        # Input layer: 1 → hidden_size
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        # Output layer: hidden_size → 2 (θ, ω)
        layers.append(nn.Linear(hidden_size, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, t):
        """
        Forward pass: t → (θ, ω)

        Args:
            t: Tensor of shape (N, 1) — time points

        Returns:
            Tensor of shape (N, 2) — [θ(t), ω(t)] at each time point
        """
        return self.network(t)


# =============================================================================
# 2. Physics-Informed Loss Function
# =============================================================================

def physics_loss(model, t_collocation, g=9.81, L=1.0):
    """
    Compute the residual of the pendulum ODE at collocation points.

    The ODE is:
        dθ/dt = ω           ... (1)  kinematic relation
        dω/dt = -(g/L)sin(θ) ... (2)  Newton's second law for rotation

    We use autograd to compute dθ/dt and dω/dt from the network output,
    then penalize any deviation from equations (1) and (2).

    "Collocation points" are the time values where we enforce the ODE.
    Unlike data-driven learning, these require NO labels — only the equation.

    Args:
        model: The PINN network
        t_collocation: Tensor (N, 1) with requires_grad=True
        g: Gravitational acceleration (m/s²)
        L: Pendulum length (m)

    Returns:
        Scalar loss = mean squared ODE residual
    """
    t_collocation.requires_grad_(True)

    # Forward pass: get θ(t) and ω(t) from the network
    output = model(t_collocation)
    theta = output[:, 0:1]  # angular displacement
    omega = output[:, 1:2]  # angular velocity

    # Compute dθ/dt and dω/dt using automatic differentiation
    # grad_outputs=ones tells autograd to sum gradients across the batch
    dtheta_dt = torch.autograd.grad(
        theta, t_collocation,
        grad_outputs=torch.ones_like(theta),
        create_graph=True  # needed so we can backprop through this gradient
    )[0]

    domega_dt = torch.autograd.grad(
        omega, t_collocation,
        grad_outputs=torch.ones_like(omega),
        create_graph=True
    )[0]

    # ODE residuals:
    #   r1 = dθ/dt - ω              (should be 0)
    #   r2 = dω/dt + (g/L)*sin(θ)   (should be 0)
    residual_1 = dtheta_dt - omega
    residual_2 = domega_dt + (g / L) * torch.sin(theta)

    # Mean squared residual across all collocation points
    loss = torch.mean(residual_1 ** 2) + torch.mean(residual_2 ** 2)
    return loss


def initial_condition_loss(model, theta_0, omega_0):
    """
    Enforce initial conditions: θ(0) = θ₀ and ω(0) = ω₀.

    Without this, the network could learn ANY solution to the ODE.
    The IC pins it to the specific trajectory we want.

    Args:
        model: The PINN network
        theta_0: Initial angle (radians)
        omega_0: Initial angular velocity (rad/s)

    Returns:
        Scalar loss = squared error at t=0
    """
    t_zero = torch.zeros(1, 1)
    output = model(t_zero)
    theta_pred = output[:, 0]
    omega_pred = output[:, 1]

    loss_theta = (theta_pred - theta_0) ** 2
    loss_omega = (omega_pred - omega_0) ** 2
    return loss_theta.squeeze() + loss_omega.squeeze()


# =============================================================================
# 3. Classical ODE Solver (Ground Truth)
# =============================================================================

def solve_pendulum_ode(theta_0, omega_0, t_span, t_eval, g=9.81, L=1.0):
    """
    Solve the pendulum ODE using scipy's Runge-Kutta 4(5) method.

    This serves as our "ground truth" to validate the PINN.

    The system is:
        dy/dt = [ω, -(g/L)*sin(θ)]   where y = [θ, ω]

    Args:
        theta_0: Initial angle (radians)
        omega_0: Initial angular velocity (rad/s)
        t_span: (t_start, t_end)
        t_eval: Array of times to evaluate solution at
        g, L: Physical parameters

    Returns:
        t: Time array
        theta: Angular displacement array
        omega: Angular velocity array
    """
    def pendulum_rhs(t, y):
        theta, omega = y
        return [omega, -(g / L) * np.sin(theta)]

    sol = solve_ivp(
        pendulum_rhs, t_span, [theta_0, omega_0],
        t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12
    )
    return sol.t, sol.y[0], sol.y[1]


# =============================================================================
# 4. Training Loop
# =============================================================================

def train_pinn(theta_0=np.pi / 4, omega_0=0.0, t_max=10.0,
               n_collocation=500, epochs=5000, lr=1e-3,
               g=9.81, L=1.0, ic_weight=20.0):
    """
    Train the PINN to learn pendulum dynamics.

    Strategy:
        Total loss = physics_loss + ic_weight * initial_condition_loss

        - physics_loss enforces the ODE at random collocation points in [0, t_max]
        - ic_loss pins the solution to the correct initial conditions
        - ic_weight > 1 because getting the IC right is critical for a unique solution

    We use Adam optimizer with a learning rate scheduler that reduces lr
    when the loss plateaus — this helps the network refine its solution.

    Args:
        theta_0: Initial angle (rad). Default π/4 (45 degrees)
        omega_0: Initial angular velocity (rad/s). Default 0 (released from rest)
        t_max: Time horizon (seconds)
        n_collocation: Number of collocation points
        epochs: Training iterations
        lr: Initial learning rate
        g, L: Physical parameters
        ic_weight: Weight multiplier for initial condition loss

    Returns:
        model: Trained PINN
        loss_history: List of total loss values per epoch
    """
    model = PINNPendulum()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6
    )

    loss_history = []

    print(f"Training PINN for pendulum motion")
    print(f"  Initial angle: {np.degrees(theta_0):.1f} degrees")
    print(f"  Time horizon:  {t_max:.1f} s")
    print(f"  Collocation points: {n_collocation}")
    print(f"  Epochs: {epochs}")
    print("-" * 50)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Sample collocation points uniformly in [0, t_max]
        # Resampling each epoch helps the network generalize across the domain
        t_col = torch.rand(n_collocation, 1) * t_max

        # Compute losses
        loss_phys = physics_loss(model, t_col, g=g, L=L)
        loss_ic = initial_condition_loss(model, theta_0, omega_0)
        total_loss = loss_phys + ic_weight * loss_ic

        # Backpropagation
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        loss_val = total_loss.item()
        loss_history.append(loss_val)

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | "
                  f"Loss: {loss_val:.6f} | "
                  f"Physics: {loss_phys.item():.6f} | "
                  f"IC: {loss_ic.item():.6f}")

    print("-" * 50)
    print(f"Final loss: {loss_history[-1]:.6f}")
    return model, loss_history


# =============================================================================
# 5. Visualization
# =============================================================================

def visualize_results(model, theta_0, omega_0, t_max, g=9.81, L=1.0):
    """
    Compare PINN predictions against the classical ODE solution.

    Generates three plots:
        1. θ(t) — Angular displacement over time
        2. ω(t) — Angular velocity over time
        3. Phase portrait — ω vs θ (reveals the system's energy conservation)
    """
    # Dense time grid for evaluation
    t_eval = np.linspace(0, t_max, 1000)

    # --- Classical solution (scipy) ---
    t_ode, theta_ode, omega_ode = solve_pendulum_ode(
        theta_0, omega_0, (0, t_max), t_eval, g=g, L=L
    )

    # --- PINN prediction ---
    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        output = model(t_tensor).numpy()
        theta_pinn = output[:, 0]
        omega_pinn = output[:, 1]

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PINN vs Classical ODE Solver — Simple Pendulum',
                 fontsize=14, fontweight='bold')

    # Plot 1: Angular displacement
    ax1 = axes[0, 0]
    ax1.plot(t_ode, np.degrees(theta_ode), 'b-', linewidth=2, label='Classical (RK45)')
    ax1.plot(t_eval, np.degrees(theta_pinn), 'r--', linewidth=2, label='PINN')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Angular Displacement θ(t)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Angular velocity
    ax2 = axes[0, 1]
    ax2.plot(t_ode, omega_ode, 'b-', linewidth=2, label='Classical (RK45)')
    ax2.plot(t_eval, omega_pinn, 'r--', linewidth=2, label='PINN')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Angular Velocity ω(t)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Phase portrait
    # In a conservative system, the phase portrait is a closed curve.
    # Deviations from closure indicate energy non-conservation.
    ax3 = axes[1, 0]
    ax3.plot(np.degrees(theta_ode), omega_ode, 'b-', linewidth=2, label='Classical')
    ax3.plot(np.degrees(theta_pinn), omega_pinn, 'r--', linewidth=2, label='PINN')
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Angular Velocity (rad/s)')
    ax3.set_title('Phase Portrait (ω vs θ)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Absolute error
    # Interpolate ODE solution onto the same grid as PINN
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
    plt.savefig('pendulum_pinn_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to pendulum_pinn_results.png")


def plot_training_loss(loss_history):
    """Plot the training loss curve on a log scale."""
    plt.figure(figsize=(8, 5))
    plt.semilogy(loss_history, linewidth=1.5, color='navy')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('PINN Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pendulum_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to pendulum_training_loss.png")


# =============================================================================
# 6. Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Physical parameters
    g = 9.81    # gravitational acceleration (m/s²)
    L = 1.0     # pendulum length (m)

    # Initial conditions
    theta_0 = np.pi / 4   # 45 degrees — moderate swing
    omega_0 = 0.0          # released from rest

    # Time domain
    t_max = 10.0  # simulate for 10 seconds

    # Train the PINN
    model, loss_history = train_pinn(
        theta_0=theta_0, omega_0=omega_0, t_max=t_max,
        n_collocation=500, epochs=5000, lr=1e-3,
        g=g, L=L, ic_weight=20.0
    )

    # Visualize results
    plot_training_loss(loss_history)
    visualize_results(model, theta_0, omega_0, t_max, g=g, L=L)
