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
# 4. Adaptive Collocation Sampling
# =============================================================================

def compute_pointwise_residual(model, t_points, g=9.81, L=1.0):
    """
    Compute the physics residual at each point *without* reducing to a scalar.

    Returns a (N,) tensor of per-point residual magnitudes (detached).
    Used to build a sampling distribution that concentrates collocation
    points where the ODE is least satisfied.
    """
    t_points = t_points.clone().requires_grad_(True)
    output = model(t_points)
    theta = output[:, 0:1]
    omega = output[:, 1:2]

    dtheta_dt = torch.autograd.grad(
        theta, t_points, torch.ones_like(theta), create_graph=False
    )[0]
    domega_dt = torch.autograd.grad(
        omega, t_points, torch.ones_like(omega), create_graph=False
    )[0]

    r1 = (dtheta_dt - omega).squeeze()
    r2 = (domega_dt + (g / L) * torch.sin(theta)).squeeze()
    return (r1 ** 2 + r2 ** 2).detach()


def sample_adaptive_collocation(model, n_collocation, t_max,
                                g=9.81, L=1.0,
                                n_grid=1000, uniform_fraction=0.2):
    """
    Sample collocation points with probability proportional to residual magnitude.

    Strategy:
        1. Evaluate the residual on a fixed dense grid of n_grid points.
        2. Build a discrete probability distribution: p_i = |r_i| / sum(|r_j|).
        3. Draw (1 - uniform_fraction) * n_collocation points from this distribution.
        4. Draw uniform_fraction * n_collocation points uniformly at random.
        5. Concatenate and return.

    The uniform fraction prevents degenerate clustering — without it, all
    points could pile up in a single high-residual region, leaving the rest
    of the domain unconstrained.

    Args:
        model: Current PINN (used in eval mode, no gradient tracking)
        n_collocation: Total number of collocation points to return
        t_max: Time domain upper bound
        g, L: Physical parameters
        n_grid: Size of the dense evaluation grid
        uniform_fraction: Fraction of points sampled uniformly (default 20%)

    Returns:
        t_col: Tensor (n_collocation, 1)
    """
    # Dense evaluation grid
    t_grid = torch.linspace(0, t_max, n_grid).unsqueeze(1)

    with torch.no_grad():
        residuals = compute_pointwise_residual(model, t_grid, g=g, L=L)

    # Build probability distribution from residual magnitudes
    probs = residuals / (residuals.sum() + 1e-16)

    # Adaptive portion
    n_adaptive = int(n_collocation * (1 - uniform_fraction))
    n_uniform = n_collocation - n_adaptive

    idx = torch.multinomial(probs, n_adaptive, replacement=True)
    t_adaptive = t_grid[idx]

    # Uniform portion
    t_uniform = torch.rand(n_uniform, 1) * t_max

    return torch.cat([t_adaptive, t_uniform], dim=0)


# =============================================================================
# 5. Training Loop
# =============================================================================

def train_pinn(theta_0=np.pi / 4, omega_0=0.0, t_max=10.0,
               n_collocation=500, epochs=5000, lr=1e-3,
               g=9.81, L=1.0, ic_weight=20.0,
               adaptive=False, adaptive_interval=500,
               uniform_fraction=0.2):
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
        adaptive: If True, use residual-based adaptive collocation sampling
        adaptive_interval: Re-evaluate residual distribution every N epochs
        uniform_fraction: Fraction of uniform random points when adaptive=True

    Returns:
        model: Trained PINN
        loss_history: List of total loss values per epoch
        collocation_snapshots: Dict mapping epoch -> sampled t values (for viz)
    """
    model = PINNPendulum()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6
    )

    loss_history = []
    collocation_snapshots = {}

    mode_str = "adaptive" if adaptive else "uniform"
    print(f"Training PINN for pendulum motion ({mode_str} sampling)")
    print(f"  Initial angle: {np.degrees(theta_0):.1f} degrees")
    print(f"  Time horizon:  {t_max:.1f} s")
    print(f"  Collocation points: {n_collocation}")
    print(f"  Epochs: {epochs}")
    if adaptive:
        print(f"  Adaptive interval: every {adaptive_interval} epochs")
        print(f"  Uniform fraction: {uniform_fraction:.0%}")
    print("-" * 50)

    # Cache for adaptive sampling — recomputed every adaptive_interval epochs
    cached_t_col = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- Collocation point sampling ---
        if adaptive and (epoch % adaptive_interval == 0):
            model.eval()
            cached_t_col = sample_adaptive_collocation(
                model, n_collocation, t_max, g=g, L=L,
                uniform_fraction=uniform_fraction
            )
            model.train()
        if adaptive and cached_t_col is not None:
            t_col = cached_t_col.clone()
        else:
            t_col = torch.rand(n_collocation, 1) * t_max

        # Snapshot collocation points at key epochs for visualization
        if epoch in (0, epochs // 2, epochs - 1):
            collocation_snapshots[epoch] = t_col.detach().numpy().flatten()

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
    return model, loss_history, collocation_snapshots


# =============================================================================
# 6. Visualization
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
# 7. Adaptive vs Uniform Comparison
# =============================================================================

def compare_adaptive_vs_uniform(theta_0=np.pi / 4, omega_0=0.0, t_max=10.0,
                                n_collocation=500, epochs=5000, lr=1e-3,
                                g=9.81, L=1.0, ic_weight=20.0):
    """
    Train two PINNs — one with uniform sampling, one with adaptive — and
    compare their performance side by side.

    Generates a 2x3 figure saved as adaptive_collocation_results.png:
        Row 1: (1) L2 error vs ground truth, (2) physics residual, (3) collocation dist
        Row 2: same three for collocation distribution snapshots at 3 epochs
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Train uniform
    print("=== UNIFORM SAMPLING ===")
    model_u, loss_u, snaps_u = train_pinn(
        theta_0, omega_0, t_max, n_collocation, epochs, lr,
        g, L, ic_weight, adaptive=False
    )

    torch.manual_seed(42)
    np.random.seed(42)

    # Train adaptive
    print("\n=== ADAPTIVE SAMPLING ===")
    model_a, loss_a, snaps_a = train_pinn(
        theta_0, omega_0, t_max, n_collocation, epochs, lr,
        g, L, ic_weight, adaptive=True
    )

    # Ground truth
    t_eval = np.linspace(0, t_max, 1000)
    _, theta_true, omega_true = solve_pendulum_ode(
        theta_0, omega_0, (0, t_max), t_eval, g=g, L=L
    )

    # Predictions
    for m in (model_u, model_a):
        m.eval()
    t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        out_u = model_u(t_tensor).numpy()
        out_a = model_a(t_tensor).numpy()

    err_u = np.sqrt((out_u[:, 0] - theta_true)**2 + (out_u[:, 1] - omega_true)**2)
    err_a = np.sqrt((out_a[:, 0] - theta_true)**2 + (out_a[:, 1] - omega_true)**2)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Adaptive vs Uniform Collocation Sampling — Pendulum PINN',
                 fontsize=15, fontweight='bold')

    # (0,0) L2 error vs ground truth over time
    ax = axes[0, 0]
    ax.semilogy(t_eval, err_u + 1e-16, 'b-', linewidth=2, label='Uniform')
    ax.semilogy(t_eval, err_a + 1e-16, 'r-', linewidth=2, label='Adaptive')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('L2 Error vs Ground Truth')
    ax.set_title(f'Final L2: Uniform={np.mean(err_u):.2e}, '
                 f'Adaptive={np.mean(err_a):.2e}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Physics residual (training loss) over epochs
    ax = axes[0, 1]
    ax.semilogy(loss_u, 'b-', linewidth=1, alpha=0.7, label='Uniform')
    ax.semilogy(loss_a, 'r-', linewidth=1, alpha=0.7, label='Adaptive')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Loss Over Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,2) Pointwise residual at final epoch
    t_grid = torch.linspace(0, t_max, 1000).unsqueeze(1)
    with torch.no_grad():
        res_u = compute_pointwise_residual(model_u, t_grid, g=g, L=L).numpy()
        res_a = compute_pointwise_residual(model_a, t_grid, g=g, L=L).numpy()
    ax = axes[0, 2]
    ax.semilogy(t_eval, res_u + 1e-16, 'b-', linewidth=1.5, label='Uniform')
    ax.semilogy(t_eval, res_a + 1e-16, 'r-', linewidth=1.5, label='Adaptive')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointwise Residual')
    ax.set_title('Physics Residual Distribution (Final)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: collocation point distribution snapshots
    snap_epochs = sorted(snaps_u.keys())
    for col, ep in enumerate(snap_epochs):
        ax = axes[1, col]
        if ep in snaps_u:
            ax.hist(snaps_u[ep], bins=50, alpha=0.5, color='blue',
                    density=True, label='Uniform')
        if ep in snaps_a:
            ax.hist(snaps_a[ep], bins=50, alpha=0.5, color='red',
                    density=True, label='Adaptive')
        ax.set_xlabel('t')
        ax.set_ylabel('Density')
        ax.set_title(f'Collocation Distribution (Epoch {ep})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('adaptive_collocation_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to adaptive_collocation_results.png")

    print(f"\n{'='*55}")
    print("Adaptive vs Uniform Summary")
    print(f"{'='*55}")
    print(f"  Uniform  — mean L2 error: {np.mean(err_u):.6f}, "
          f"final loss: {loss_u[-1]:.6f}")
    print(f"  Adaptive — mean L2 error: {np.mean(err_a):.6f}, "
          f"final loss: {loss_a[-1]:.6f}")
    ratio = np.mean(err_u) / (np.mean(err_a) + 1e-16)
    print(f"  Adaptive improvement: {ratio:.2f}x lower mean L2 error")


# =============================================================================
# 8. Main Entry Point
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

    # Train the PINN (default: uniform sampling)
    model, loss_history, _ = train_pinn(
        theta_0=theta_0, omega_0=omega_0, t_max=t_max,
        n_collocation=500, epochs=5000, lr=1e-3,
        g=g, L=L, ic_weight=20.0
    )

    # Visualize results
    plot_training_loss(loss_history)
    visualize_results(model, theta_0, omega_0, t_max, g=g, L=L)

    # Run adaptive vs uniform comparison
    compare_adaptive_vs_uniform(
        theta_0=theta_0, omega_0=omega_0, t_max=t_max,
        n_collocation=500, epochs=5000, g=g, L=L
    )
