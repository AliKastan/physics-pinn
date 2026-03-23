"""
Physics-Informed Neural Network (PINN) for Two-Body Orbital Mechanics
=====================================================================

Physics Background:
    A body of mass m orbiting a central mass M under Newtonian gravity
    obeys:

        d²x/dt² = -GM * x / r³
        d²y/dt² = -GM * y / r³

    where r = sqrt(x² + y²) is the distance from the central body,
    and GM is the gravitational parameter (G * M_central).

    This is equivalent to four first-order ODEs by introducing velocities
    vx = dx/dt, vy = dy/dt:

        dx/dt  = vx
        dy/dt  = vy
        dvx/dt = -GM * x / r³
        dvy/dt = -GM * y / r³

Energy Conservation:
    The total mechanical energy is conserved in a two-body system:

        E = T + V = ½m(vx² + vy²) - GMm/r

    For a bound orbit (ellipse), E < 0. The PINN's ability to conserve
    energy — without being explicitly told to — is a strong test of
    whether it has truly learned the physics.

Angular Momentum Conservation:
    L = m(x*vy - y*vx) is also conserved (Kepler's second law).
    We track this as an additional validation metric.

PINN Approach:
    The network N(t) → (x, y, vx, vy) is trained to satisfy:
      1. The gravitational ODEs at collocation points (physics loss)
      2. The initial conditions (IC loss)
    No simulation data is used — only the equations of motion.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# =============================================================================
# 1. Neural Network Architecture
# =============================================================================

class PINNOrbital(nn.Module):
    """
    Fully-connected network mapping time t → (x, y, vx, vy).

    Architecture:
        Input:  t (1 neuron)
        Hidden: 4 layers × 128 neurons, tanh activation
        Output: x, y, vx, vy (4 neurons)

    Wider and deeper than the pendulum network because:
        - 4 output variables instead of 2
        - Orbital dynamics have sharper features (perihelion passage)
        - The 1/r³ gravitational force is highly nonlinear near the central body
    """

    def __init__(self, hidden_size=128, num_hidden_layers=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.Tanh())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_size, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        """
        Args:
            t: Tensor of shape (N, 1)
        Returns:
            Tensor of shape (N, 4) — [x, y, vx, vy]
        """
        return self.network(t)


# =============================================================================
# 2. Physics-Informed Loss Function
# =============================================================================

def physics_loss(model, t_col, GM=1.0):
    """
    Enforce Newton's law of gravitation at collocation points.

    The four ODE residuals are:
        r1: dx/dt  - vx                 = 0   (definition of velocity)
        r2: dy/dt  - vy                 = 0
        r3: dvx/dt + GM * x / r³        = 0   (gravitational acceleration, x)
        r4: dvy/dt + GM * y / r³        = 0   (gravitational acceleration, y)

    The 1/r³ term makes this loss challenging: near perihelion, r is small
    and the force is large, creating steep gradients that the network must
    capture accurately.

    Args:
        model: The PINN network
        t_col: Collocation time points (N, 1), requires_grad will be set
        GM: Gravitational parameter (G * M_central)

    Returns:
        Scalar mean squared ODE residual
    """
    t_col.requires_grad_(True)
    output = model(t_col)

    x  = output[:, 0:1]
    y  = output[:, 1:2]
    vx = output[:, 2:3]
    vy = output[:, 3:4]

    # Compute time derivatives via autograd
    dx_dt = torch.autograd.grad(
        x, t_col, grad_outputs=torch.ones_like(x), create_graph=True
    )[0]
    dy_dt = torch.autograd.grad(
        y, t_col, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]
    dvx_dt = torch.autograd.grad(
        vx, t_col, grad_outputs=torch.ones_like(vx), create_graph=True
    )[0]
    dvy_dt = torch.autograd.grad(
        vy, t_col, grad_outputs=torch.ones_like(vy), create_graph=True
    )[0]

    # Distance from central body
    # Add small epsilon to avoid division by zero if the network
    # briefly predicts r ≈ 0 during early training
    r = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
    r_cubed = r ** 3

    # ODE residuals
    res_x  = dx_dt - vx
    res_y  = dy_dt - vy
    res_vx = dvx_dt + GM * x / r_cubed
    res_vy = dvy_dt + GM * y / r_cubed

    loss = (torch.mean(res_x ** 2) + torch.mean(res_y ** 2) +
            torch.mean(res_vx ** 2) + torch.mean(res_vy ** 2))
    return loss


def initial_condition_loss(model, x0, y0, vx0, vy0):
    """
    Enforce initial conditions at t = 0.

    For an orbit, the initial position and velocity completely determine
    the trajectory. Getting these right is essential.

    Args:
        model: The PINN network
        x0, y0: Initial position
        vx0, vy0: Initial velocity

    Returns:
        Scalar IC loss
    """
    t_zero = torch.zeros(1, 1)
    output = model(t_zero)

    loss = ((output[0, 0] - x0) ** 2 +
            (output[0, 1] - y0) ** 2 +
            (output[0, 2] - vx0) ** 2 +
            (output[0, 3] - vy0) ** 2)
    return loss


# =============================================================================
# 3. Classical ODE Solver (Ground Truth)
# =============================================================================

def solve_orbit_ode(x0, y0, vx0, vy0, t_span, t_eval, GM=1.0):
    """
    Solve the two-body problem with scipy's RK45 (high accuracy).

    State vector: y = [x, y, vx, vy]

    Args:
        x0, y0, vx0, vy0: Initial conditions
        t_span: (t_start, t_end)
        t_eval: Times to evaluate at
        GM: Gravitational parameter

    Returns:
        t, x, y, vx, vy arrays
    """
    def gravity_rhs(t, state):
        x, y, vx, vy = state
        r = np.sqrt(x ** 2 + y ** 2)
        r3 = r ** 3
        ax = -GM * x / r3
        ay = -GM * y / r3
        return [vx, vy, ax, ay]

    sol = solve_ivp(
        gravity_rhs, t_span, [x0, y0, vx0, vy0],
        t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-14
    )
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


# =============================================================================
# 4. Energy & Angular Momentum Calculations
# =============================================================================

def compute_energy(x, y, vx, vy, GM=1.0, m=1.0):
    """
    Total mechanical energy: E = T + V

    Kinetic energy:   T = ½ m (vx² + vy²)
    Potential energy:  V = -GMm / r

    For a Keplerian orbit, E is constant along the trajectory.
    Any drift in E indicates numerical error (classical solver) or
    failure to learn the physics (PINN).

    Args:
        x, y, vx, vy: Position and velocity arrays
        GM: Gravitational parameter
        m: Orbiting body mass (set to 1 for specific energy)

    Returns:
        E: Total energy array (same length as inputs)
    """
    r = np.sqrt(x ** 2 + y ** 2)
    kinetic = 0.5 * m * (vx ** 2 + vy ** 2)
    potential = -GM * m / r
    return kinetic + potential


def compute_angular_momentum(x, y, vx, vy, m=1.0):
    """
    Angular momentum: L = m (x * vy - y * vx)

    Conservation of L is Kepler's second law — the orbiting body
    sweeps equal areas in equal times. This is a consequence of the
    central force being radial (no torque).

    Args:
        x, y, vx, vy: Position and velocity arrays
        m: Orbiting body mass

    Returns:
        L: Angular momentum array
    """
    return m * (x * vy - y * vx)


# =============================================================================
# 5. Training
# =============================================================================

def setup_orbital_ics(eccentricity=0.5, GM=1.0):
    """
    Set up initial conditions for an elliptical orbit.

    We start at perihelion (closest approach), where the velocity is
    purely tangential. For an orbit with semi-major axis a and
    eccentricity e:

        r_perihelion = a(1 - e)
        v_perihelion = sqrt(GM/a * (1+e)/(1-e))   (vis-viva equation)

    We choose a = 1 for simplicity, giving:
        x0 = 1 - e,  y0 = 0
        vx0 = 0,  vy0 = sqrt(GM * (1+e) / (1-e))

    The orbital period is T = 2π * sqrt(a³/GM) = 2π for a=1, GM=1.

    Args:
        eccentricity: Orbital eccentricity (0 = circle, <1 = ellipse)
        GM: Gravitational parameter

    Returns:
        x0, y0, vx0, vy0, period
    """
    a = 1.0  # semi-major axis
    e = eccentricity

    # Start at perihelion (closest point to central body)
    x0 = a * (1.0 - e)
    y0 = 0.0
    vx0 = 0.0
    # Vis-viva equation at perihelion gives the tangential velocity
    vy0 = np.sqrt(GM / a * (1.0 + e) / (1.0 - e))

    # Kepler's third law: T² = 4π²a³/(GM)
    period = 2.0 * np.pi * np.sqrt(a ** 3 / GM)

    return x0, y0, vx0, vy0, period


def train_pinn_orbital(eccentricity=0.3, GM=1.0, n_orbits=1.0,
                       n_collocation=800, epochs=8000, lr=1e-3,
                       ic_weight=50.0):
    """
    Train the orbital PINN.

    Key differences from the pendulum PINN:
        - Higher IC weight (50 vs 20): orbital dynamics are more sensitive
          to initial conditions — a small IC error compounds over the orbit
        - More collocation points (800 vs 500): the orbit covers a 2D domain
        - More epochs (8000 vs 5000): the 1/r³ nonlinearity is harder to learn
        - Time normalization: we normalize t to [0, 1] internally so the
          network inputs stay in a well-conditioned range

    Args:
        eccentricity: Orbital eccentricity (0 < e < 1)
        GM: Gravitational parameter
        n_orbits: Number of orbital periods to simulate
        n_collocation: Number of collocation points per epoch
        epochs: Number of training epochs
        lr: Initial learning rate
        ic_weight: Weight for initial condition loss

    Returns:
        model, loss_history, ics, t_max
    """
    x0, y0, vx0, vy0, period = setup_orbital_ics(eccentricity, GM)
    t_max = n_orbits * period

    print(f"Training PINN for orbital mechanics")
    print(f"  Eccentricity: {eccentricity}")
    print(f"  Orbital period: {period:.4f}")
    print(f"  Simulation time: {t_max:.4f} ({n_orbits} orbit(s))")
    print(f"  Perihelion: r = {x0:.4f}, v = {vy0:.4f}")
    print(f"  Collocation points: {n_collocation}")
    print(f"  Epochs: {epochs}")
    print("-" * 55)

    model = PINNOrbital()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6
    )

    # Convert ICs to tensors
    x0_t = torch.tensor(x0, dtype=torch.float32)
    y0_t = torch.tensor(y0, dtype=torch.float32)
    vx0_t = torch.tensor(vx0, dtype=torch.float32)
    vy0_t = torch.tensor(vy0, dtype=torch.float32)

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Random collocation points in [0, t_max]
        t_col = torch.rand(n_collocation, 1) * t_max

        loss_phys = physics_loss(model, t_col, GM=GM)
        loss_ic = initial_condition_loss(model, x0_t, y0_t, vx0_t, vy0_t)
        total_loss = loss_phys + ic_weight * loss_ic

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

    print("-" * 55)
    print(f"Final loss: {loss_history[-1]:.6f}")

    ics = (x0, y0, vx0, vy0)
    return model, loss_history, ics, t_max


# =============================================================================
# 6. Visualization
# =============================================================================

def visualize_orbital_results(model, ics, t_max, GM=1.0):
    """
    Compare PINN vs classical solver with 5 plots:
        1. Orbital trajectory (x-y plane)
        2. x(t) and y(t) over time
        3. Total energy over time (energy conservation test)
        4. Angular momentum over time (Kepler's second law test)
        5. Position error over time
    """
    x0, y0, vx0, vy0 = ics
    t_eval = np.linspace(0, t_max, 2000)

    # --- Classical solution ---
    t_ode, x_ode, y_ode, vx_ode, vy_ode = solve_orbit_ode(
        x0, y0, vx0, vy0, (0, t_max), t_eval, GM=GM
    )

    # --- PINN prediction ---
    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        output = model(t_tensor).numpy()
        x_pinn  = output[:, 0]
        y_pinn  = output[:, 1]
        vx_pinn = output[:, 2]
        vy_pinn = output[:, 3]

    # --- Compute conserved quantities ---
    E_ode  = compute_energy(x_ode, y_ode, vx_ode, vy_ode, GM)
    E_pinn = compute_energy(x_pinn, y_pinn, vx_pinn, vy_pinn, GM)

    L_ode  = compute_angular_momentum(x_ode, y_ode, vx_ode, vy_ode)
    L_pinn = compute_angular_momentum(x_pinn, y_pinn, vx_pinn, vy_pinn)

    # Initial values for relative drift calculation
    E0 = E_ode[0]
    L0 = L_ode[0]

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('PINN vs Classical Solver — Two-Body Orbital Mechanics',
                 fontsize=15, fontweight='bold')

    # ---- Plot 1: Orbital trajectory in x-y plane ----
    ax1 = axes[0, 0]
    ax1.plot(x_ode, y_ode, 'b-', linewidth=2, label='Classical (RK45)')
    ax1.plot(x_pinn, y_pinn, 'r--', linewidth=2, label='PINN')
    ax1.plot(0, 0, 'ko', markersize=10, label='Central body')
    ax1.plot(x0, y0, 'g*', markersize=12, label='Start (perihelion)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Orbital Trajectory')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # ---- Plot 2: Position components over time ----
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

    # ---- Plot 3: Total energy over time ----
    # This is the key physics test: energy should be constant.
    # Any drift reveals how well each method respects the conservation law.
    ax3 = axes[0, 2]
    ax3.plot(t_ode, E_ode, 'b-', linewidth=2, label='Classical (RK45)')
    ax3.plot(t_eval, E_pinn, 'r-', linewidth=2, label='PINN')
    ax3.axhline(y=E0, color='gray', linestyle=':', alpha=0.5, label=f'True E = {E0:.4f}')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Total Energy (T + V)')
    ax3.set_title('Energy Conservation')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- Plot 4: Relative energy error ----
    # |ΔE/E₀| shows the fractional energy drift — a dimensionless
    # measure that lets us compare methods fairly.
    ax4 = axes[1, 0]
    dE_ode = np.abs((E_ode - E0) / E0)
    dE_pinn = np.abs((E_pinn - E0) / E0)
    ax4.semilogy(t_ode, dE_ode + 1e-16, 'b-', linewidth=2, label='Classical (RK45)')
    ax4.semilogy(t_eval, dE_pinn + 1e-16, 'r-', linewidth=2, label='PINN')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('|ΔE / E₀|')
    ax4.set_title('Relative Energy Error')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ---- Plot 5: Angular momentum over time ----
    # L = x*vy - y*vx should be constant (Kepler's second law).
    ax5 = axes[1, 1]
    ax5.plot(t_ode, L_ode, 'b-', linewidth=2, label='Classical (RK45)')
    ax5.plot(t_eval, L_pinn, 'r-', linewidth=2, label='PINN')
    ax5.axhline(y=L0, color='gray', linestyle=':', alpha=0.5, label=f'True L = {L0:.4f}')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Angular Momentum L')
    ax5.set_title('Angular Momentum Conservation')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ---- Plot 6: Position error ----
    pos_error = np.sqrt((x_pinn - x_ode) ** 2 + (y_pinn - y_ode) ** 2)
    ax6 = axes[1, 2]
    ax6.semilogy(t_eval, pos_error + 1e-16, 'g-', linewidth=2)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Position Error |Δr|')
    ax6.set_title('PINN Position Error vs Classical')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('orbital_pinn_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to orbital_pinn_results.png")

    # --- Print summary statistics ---
    print(f"\n{'='*55}")
    print("Conservation Summary")
    print(f"{'='*55}")
    print(f"  Initial energy E₀ = {E0:.6f}")
    print(f"  Classical max |ΔE/E₀| = {np.max(dE_ode):.2e}")
    print(f"  PINN max |ΔE/E₀|     = {np.max(dE_pinn):.2e}")
    print(f"  Initial angular momentum L₀ = {L0:.6f}")
    dL_ode = np.abs((L_ode - L0) / L0)
    dL_pinn = np.abs((L_pinn - L0) / L0)
    print(f"  Classical max |ΔL/L₀| = {np.max(dL_ode):.2e}")
    print(f"  PINN max |ΔL/L₀|     = {np.max(dL_pinn):.2e}")
    print(f"  Max position error    = {np.max(pos_error):.6f}")


def plot_training_loss(loss_history):
    """Plot the orbital PINN training loss."""
    plt.figure(figsize=(8, 5))
    plt.semilogy(loss_history, linewidth=1.5, color='darkred')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Orbital PINN Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('orbital_training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to orbital_training_loss.png")


# =============================================================================
# 7. Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Gravitational parameter (normalized units: G*M = 1)
    # This gives orbital period T = 2π for semi-major axis a = 1
    GM = 1.0

    # Eccentricity controls the orbit shape:
    #   e = 0.0 → perfect circle
    #   e = 0.3 → mild ellipse (good starting point for PINN)
    #   e = 0.7 → highly eccentric (harder — large velocity at perihelion)
    eccentricity = 0.3

    # Train for one full orbit
    model, loss_history, ics, t_max = train_pinn_orbital(
        eccentricity=eccentricity,
        GM=GM,
        n_orbits=1.0,
        n_collocation=800,
        epochs=8000,
        lr=1e-3,
        ic_weight=50.0
    )

    # Visualize
    plot_training_loss(loss_history)
    visualize_orbital_results(model, ics, t_max, GM=GM)
