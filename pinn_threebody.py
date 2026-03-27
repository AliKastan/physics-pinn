"""
Physics-Informed Neural Network (PINN) for the Gravitational Three-Body Problem
=================================================================================

Physics Background:
    The three-body problem is one of the oldest unsolved problems in physics.
    Three masses interact via Newtonian gravity:

        m_i * d²r_i/dt² = sum_{j != i} G * m_i * m_j * (r_j - r_i) / |r_j - r_i|³

    Unlike the two-body problem, there is NO general closed-form solution.
    Trajectories can be chaotic — tiny perturbations lead to wildly different
    outcomes. This makes it an ideal test for PINNs: can a neural network
    approximate a solution that respects the equations of motion?

    We use a figure-eight orbit (Chenciner & Montgomery, 2000) as the initial
    condition — one of the few known periodic three-body solutions.

Conserved Quantities:
    - Total energy: E = T + V (kinetic + gravitational potential)
    - Total linear momentum: P = sum(m_i * v_i) = const
    - Total angular momentum: L = sum(m_i * r_i x v_i) = const

    Tracking how well the PINN preserves these gives a direct measure
    of solution quality.

PINN Approach:
    The network N(t) -> (x1,y1, x2,y2, x3,y3, vx1,vy1, vx2,vy2, vx3,vy3)
    maps time to all positions and velocities. The physics loss enforces
    Newton's law of gravitation for all three pairs of interactions.

Reference:
    Chenciner, A. & Montgomery, R. (2000). "A remarkable periodic solution
    of the three-body problem in the case of equal masses."
    Annals of Mathematics, 152(3), 881-901.
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

class PINNThreeBody(nn.Module):
    """
    Fully-connected network mapping time t -> 12 state variables:
        (x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3)

    Architecture:
        Input:  t (1 neuron)
        Hidden: 4 layers x 128 neurons, tanh activation
        Output: 12 state variables

    Larger than the two-body network because:
        - 12 outputs (3 bodies x 4 state variables)
        - Three pairwise gravitational interactions create complex dynamics
        - The figure-eight orbit has sharp curvature at the crossing points
    """

    def __init__(self, hidden_size=128, num_hidden_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 12))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)


# =============================================================================
# 2. Figure-Eight Initial Conditions
# =============================================================================

def figure_eight_ics(G=1.0):
    """
    Initial conditions for the figure-eight three-body orbit.

    Three equal masses follow a planar figure-eight path. This is one
    of the few known periodic solutions to the three-body problem.

    The initial conditions are from Moore (1993) / Chenciner & Montgomery (2000):
        Body 1 starts at (-1, 0) moving along the figure-eight
        Body 2 starts at (1, 0) moving symmetrically
        Body 3 starts at (0, 0) moving to close the loop

    All masses are set to 1.0 and G = 1.0 for normalized units.
    The period is approximately T = 6.3259.

    Returns:
        state0: [x1,y1, x2,y2, x3,y3, vx1,vy1, vx2,vy2, vx3,vy3]
        masses: [m1, m2, m3]
        period: Approximate orbital period
    """
    masses = [1.0, 1.0, 1.0]

    # Positions
    x1, y1 = -0.97000436, 0.24308753
    x2, y2 = 0.97000436, -0.24308753
    x3, y3 = 0.0, 0.0

    # Velocities (from the specific figure-eight solution)
    vx3, vy3 = -0.93240737, -0.86473146
    vx1, vy1 = -vx3 / 2.0, -vy3 / 2.0
    vx2, vy2 = -vx3 / 2.0, -vy3 / 2.0

    state0 = [x1, y1, x2, y2, x3, y3,
              vx1, vy1, vx2, vy2, vx3, vy3]

    period = 6.3259

    return state0, masses, period


# =============================================================================
# 3. Physics: Gravitational Forces
# =============================================================================

def three_body_rhs(t, state, masses, G=1.0):
    """
    Right-hand side of the three-body equations of motion.

    For each body i, the acceleration is the sum of gravitational
    pulls from the other two bodies:

        a_i = sum_{j != i} G * m_j * (r_j - r_i) / |r_j - r_i|^3

    Args:
        t: Time (unused but required by solve_ivp)
        state: [x1,y1, x2,y2, x3,y3, vx1,vy1, vx2,vy2, vx3,vy3]
        masses: [m1, m2, m3]
        G: Gravitational constant

    Returns:
        dstate/dt: Time derivatives of all state variables
    """
    x1, y1, x2, y2, x3, y3 = state[0:6]
    vx1, vy1, vx2, vy2, vx3, vy3 = state[6:12]
    m1, m2, m3 = masses

    # Pairwise distances with gravitational softening (epsilon=1e-3)
    # to prevent singularities when bodies approach closely
    eps = 1e-3
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + eps**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2 + eps**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2 + eps**2)

    # Accelerations on body 1 (from bodies 2 and 3)
    ax1 = G * m2 * (x2 - x1) / r12**3 + G * m3 * (x3 - x1) / r13**3
    ay1 = G * m2 * (y2 - y1) / r12**3 + G * m3 * (y3 - y1) / r13**3

    # Accelerations on body 2 (from bodies 1 and 3)
    ax2 = G * m1 * (x1 - x2) / r12**3 + G * m3 * (x3 - x2) / r23**3
    ay2 = G * m1 * (y1 - y2) / r12**3 + G * m3 * (y3 - y2) / r23**3

    # Accelerations on body 3 (from bodies 1 and 2)
    ax3 = G * m1 * (x1 - x3) / r13**3 + G * m2 * (x2 - x3) / r23**3
    ay3 = G * m1 * (y1 - y3) / r13**3 + G * m2 * (y2 - y3) / r23**3

    return [vx1, vy1, vx2, vy2, vx3, vy3,
            ax1, ay1, ax2, ay2, ax3, ay3]


# =============================================================================
# 4. Physics-Informed Loss
# =============================================================================

def physics_loss(model, t_col, masses, G=1.0):
    """
    Enforce Newton's law of gravitation for all three bodies.

    For each body i, we compute:
        dx_i/dt - vx_i = 0        (kinematic)
        dy_i/dt - vy_i = 0
        dvx_i/dt - ax_i = 0       (Newton's second law)
        dvy_i/dt - ay_i = 0

    where a_i comes from gravitational interactions with the other two.
    """
    t_col.requires_grad_(True)
    output = model(t_col)

    # Unpack positions and velocities
    x1, y1 = output[:, 0:1], output[:, 1:2]
    x2, y2 = output[:, 2:3], output[:, 3:4]
    x3, y3 = output[:, 4:5], output[:, 5:6]
    vx1, vy1 = output[:, 6:7], output[:, 7:8]
    vx2, vy2 = output[:, 8:9], output[:, 9:10]
    vx3, vy3 = output[:, 10:11], output[:, 11:12]

    ones = torch.ones_like(x1)

    # Time derivatives via autograd
    dx1 = torch.autograd.grad(x1, t_col, ones, create_graph=True)[0]
    dy1 = torch.autograd.grad(y1, t_col, ones, create_graph=True)[0]
    dx2 = torch.autograd.grad(x2, t_col, ones, create_graph=True)[0]
    dy2 = torch.autograd.grad(y2, t_col, ones, create_graph=True)[0]
    dx3 = torch.autograd.grad(x3, t_col, ones, create_graph=True)[0]
    dy3 = torch.autograd.grad(y3, t_col, ones, create_graph=True)[0]
    dvx1 = torch.autograd.grad(vx1, t_col, ones, create_graph=True)[0]
    dvy1 = torch.autograd.grad(vy1, t_col, ones, create_graph=True)[0]
    dvx2 = torch.autograd.grad(vx2, t_col, ones, create_graph=True)[0]
    dvy2 = torch.autograd.grad(vy2, t_col, ones, create_graph=True)[0]
    dvx3 = torch.autograd.grad(vx3, t_col, ones, create_graph=True)[0]
    dvy3 = torch.autograd.grad(vy3, t_col, ones, create_graph=True)[0]

    m1, m2, m3 = masses

    # Pairwise distances with gravitational softening (epsilon=1e-3)
    # Same softening used in the classical solver for consistent comparison
    eps = 1e-3
    r12 = torch.sqrt((x2 - x1)**2 + (y2 - y1)**2 + eps**2)
    r13 = torch.sqrt((x3 - x1)**2 + (y3 - y1)**2 + eps**2)
    r23 = torch.sqrt((x3 - x2)**2 + (y3 - y2)**2 + eps**2)

    # Gravitational accelerations
    ax1 = G * m2 * (x2 - x1) / r12**3 + G * m3 * (x3 - x1) / r13**3
    ay1 = G * m2 * (y2 - y1) / r12**3 + G * m3 * (y3 - y1) / r13**3
    ax2 = G * m1 * (x1 - x2) / r12**3 + G * m3 * (x3 - x2) / r23**3
    ay2 = G * m1 * (y1 - y2) / r12**3 + G * m3 * (y3 - y2) / r23**3
    ax3 = G * m1 * (x1 - x3) / r13**3 + G * m2 * (x2 - x3) / r23**3
    ay3 = G * m1 * (y1 - y3) / r13**3 + G * m2 * (y2 - y3) / r23**3

    # Kinematic residuals
    loss = (torch.mean((dx1 - vx1)**2) + torch.mean((dy1 - vy1)**2) +
            torch.mean((dx2 - vx2)**2) + torch.mean((dy2 - vy2)**2) +
            torch.mean((dx3 - vx3)**2) + torch.mean((dy3 - vy3)**2))

    # Dynamical residuals (Newton's second law)
    loss += (torch.mean((dvx1 - ax1)**2) + torch.mean((dvy1 - ay1)**2) +
             torch.mean((dvx2 - ax2)**2) + torch.mean((dvy2 - ay2)**2) +
             torch.mean((dvx3 - ax3)**2) + torch.mean((dvy3 - ay3)**2))

    return loss


def initial_condition_loss(model, state0):
    """Enforce initial conditions at t=0 for all 12 state variables."""
    t_zero = torch.zeros(1, 1)
    output = model(t_zero).squeeze()
    state0_tensor = torch.tensor(state0, dtype=torch.float32)
    return torch.sum((output - state0_tensor) ** 2)


# =============================================================================
# 5. Conservation Quantities
# =============================================================================

def compute_energy(states, masses, G=1.0):
    """
    Total energy E = T + V for the three-body system.

    Kinetic: T = sum_i 0.5 * m_i * |v_i|^2
    Potential: V = -sum_{i<j} G * m_i * m_j / |r_i - r_j|
    """
    x1, y1, x2, y2, x3, y3 = [states[:, i] for i in range(6)]
    vx1, vy1, vx2, vy2, vx3, vy3 = [states[:, i] for i in range(6, 12)]
    m1, m2, m3 = masses

    T = (0.5 * m1 * (vx1**2 + vy1**2) +
         0.5 * m2 * (vx2**2 + vy2**2) +
         0.5 * m3 * (vx3**2 + vy3**2))

    eps = 1e-3
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + eps**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2 + eps**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2 + eps**2)

    V = -G * (m1 * m2 / r12 + m1 * m3 / r13 + m2 * m3 / r23)

    return T + V


def compute_angular_momentum(states, masses):
    """Total angular momentum L = sum_i m_i * (x_i * vy_i - y_i * vx_i)."""
    x1, y1, x2, y2, x3, y3 = [states[:, i] for i in range(6)]
    vx1, vy1, vx2, vy2, vx3, vy3 = [states[:, i] for i in range(6, 12)]
    m1, m2, m3 = masses

    L = (m1 * (x1 * vy1 - y1 * vx1) +
         m2 * (x2 * vy2 - y2 * vx2) +
         m3 * (x3 * vy3 - y3 * vx3))
    return L


# =============================================================================
# 6. Training
# =============================================================================

def train_threebody_pinn(state0, masses, t_max, G=1.0,
                         n_collocation=1000, epochs=10000, lr=1e-3,
                         ic_weight=100.0):
    """
    Train PINN for the three-body problem.

    The three-body problem is significantly harder than two-body:
        - 12 state variables (vs 4)
        - Three pairwise 1/r^3 interactions
        - Chaotic sensitivity to errors
        - High IC weight (100) needed to anchor the trajectory

    We simulate for a fraction of one period to keep the problem tractable.
    """
    model = PINNThreeBody()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6
    )

    loss_history = []

    print(f"Training PINN for Three-Body Problem (figure-eight orbit)")
    print(f"  Time horizon: {t_max:.4f}")
    print(f"  Collocation points: {n_collocation}")
    print(f"  Epochs: {epochs}")
    print("-" * 55)

    for epoch in range(epochs):
        optimizer.zero_grad()

        t_col = torch.rand(n_collocation, 1) * t_max
        loss_phys = physics_loss(model, t_col, masses, G)
        loss_ic = initial_condition_loss(model, state0)
        total_loss = loss_phys + ic_weight * loss_ic

        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        loss_history.append(total_loss.item())

        if (epoch + 1) % 2000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | "
                  f"Loss: {total_loss.item():.6f} | "
                  f"Physics: {loss_phys.item():.6f} | "
                  f"IC: {loss_ic.item():.6f}")

    print("-" * 55)
    print(f"Final loss: {loss_history[-1]:.6f}")
    return model, loss_history


# =============================================================================
# 7. Visualization
# =============================================================================

def visualize_threebody_results(model, state0, masses, t_max, G=1.0):
    """
    Compare PINN approximation vs classical solver with 4 plots:
        1. Trajectories in x-y plane (PINN vs RK45)
        2. Total energy drift over time
        3. Position error per body (shows chaos boundary)
        4. Angular momentum conservation
    """
    t_eval = np.linspace(0, t_max, 2000)

    # --- Classical solution ---
    sol = solve_ivp(
        three_body_rhs, (0, t_max), state0,
        args=(masses, G), t_eval=t_eval,
        method='RK45', rtol=1e-12, atol=1e-14
    )
    states_ode = sol.y.T  # shape (N, 12)

    # --- PINN prediction ---
    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        states_pinn = model(t_tensor).numpy()  # shape (N, 12)

    # --- Conservation quantities ---
    E_ode = compute_energy(states_ode, masses, G)
    E_pinn = compute_energy(states_pinn, masses, G)
    E0 = E_ode[0]

    L_ode = compute_angular_momentum(states_ode, masses)
    L_pinn = compute_angular_momentum(states_pinn, masses)
    L0 = L_ode[0]

    # --- Detect divergence: where total position error first exceeds a threshold ---
    total_pos_err = np.zeros(len(t_eval))
    for i in range(3):
        total_pos_err += np.sqrt(
            (states_pinn[:, 2*i] - states_ode[:, 2*i])**2 +
            (states_pinn[:, 2*i+1] - states_ode[:, 2*i+1])**2)
    diverge_threshold = 0.1
    diverge_idx = np.argmax(total_pos_err > diverge_threshold)
    if total_pos_err[diverge_idx] <= diverge_threshold:
        diverge_idx = None  # never diverged
    diverge_time = t_eval[diverge_idx] if diverge_idx is not None else None

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('PINN for Three-Body Problem (Figure-Eight Orbit)',
                 fontsize=15, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['Body 1', 'Body 2', 'Body 3']

    # Plot 1: Orbital trajectories (x-y plane)
    ax1 = axes[0, 0]
    for i, (c, lab) in enumerate(zip(colors, labels)):
        ax1.plot(states_ode[:, 2*i], states_ode[:, 2*i+1],
                 '-', color=c, linewidth=2, alpha=0.4, label=f'{lab} (RK45)')
        ax1.plot(states_pinn[:, 2*i], states_pinn[:, 2*i+1],
                 '--', color=c, linewidth=2, label=f'{lab} (PINN)')
        ax1.plot(state0[2*i], state0[2*i+1], 'o', color=c, markersize=8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Orbital Trajectories')
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Total energy drift |dE/E0| over time
    ax2 = axes[0, 1]
    dE_ode = np.abs((E_ode - E0) / (np.abs(E0) + 1e-16))
    dE_pinn = np.abs((E_pinn - E0) / (np.abs(E0) + 1e-16))
    ax2.semilogy(t_eval, dE_ode + 1e-16, 'b-', linewidth=2,
                 label='Classical (RK45)')
    ax2.semilogy(t_eval, dE_pinn + 1e-16, 'r-', linewidth=2, label='PINN')
    if diverge_time is not None:
        ax2.axvline(x=diverge_time, color='gray', linestyle='--', alpha=0.6)
        ax2.annotate(
            f'PINN diverges\nt = {diverge_time:.3f}',
            xy=(diverge_time, dE_pinn[diverge_idx]),
            xytext=(diverge_time + 0.05 * t_max, 1e-1),
            fontsize=9, color='gray',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))
    ax2.set_xlabel('Time')
    ax2.set_ylabel('|dE / E_0|')
    ax2.set_title('Relative Energy Drift')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Position error per body (chaos boundary)
    ax3 = axes[1, 0]
    for i, (c, lab) in enumerate(zip(colors, labels)):
        pos_err = np.sqrt((states_pinn[:, 2*i] - states_ode[:, 2*i])**2 +
                          (states_pinn[:, 2*i+1] - states_ode[:, 2*i+1])**2)
        ax3.semilogy(t_eval, pos_err + 1e-16, '-', color=c, linewidth=1.5,
                     label=lab)
    if diverge_time is not None:
        ax3.axvline(x=diverge_time, color='gray', linestyle='--', alpha=0.6,
                    label=f'Divergence (t={diverge_time:.3f})')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Position Error |dr|')
    ax3.set_title('PINN Error vs Classical — Chaos Boundary')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Energy conservation (absolute values)
    ax4 = axes[1, 1]
    ax4.plot(t_eval, E_ode, 'b-', linewidth=2, label='Classical (RK45)')
    ax4.plot(t_eval, E_pinn, 'r-', linewidth=2, label='PINN')
    ax4.axhline(y=E0, color='gray', linestyle=':', alpha=0.5,
                label=f'True E = {E0:.4f}')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Total Energy')
    ax4.set_title('Energy Conservation')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Add note about chaos if divergence detected
    if diverge_time is not None:
        fig.text(
            0.5, 0.01,
            "Note: The three-body problem is chaotic — small PINN errors "
            "grow exponentially. The divergence point marks where the PINN's "
            "approximate dynamics depart from the true trajectory. "
            "Showing this chaos boundary is itself a result.",
            ha='center', fontsize=9, style='italic', color='#555555',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.7))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('threebody_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to threebody_results.png")

    # Print summary
    print(f"\n{'='*55}")
    print("Three-Body Conservation Summary")
    print(f"{'='*55}")
    print(f"  Initial energy E0 = {E0:.6f}")
    print(f"  Classical max |dE/E0| = {np.max(dE_ode):.2e}")
    print(f"  PINN max |dE/E0|     = {np.max(dE_pinn):.2e}")
    print(f"  Initial angular momentum L0 = {L0:.6f}")
    if diverge_time is not None:
        print(f"  PINN diverges from RK45 at t = {diverge_time:.3f}")

    for i, lab in enumerate(labels):
        pos_err = np.sqrt((states_pinn[:, 2*i] - states_ode[:, 2*i])**2 +
                          (states_pinn[:, 2*i+1] - states_ode[:, 2*i+1])**2)
        print(f"  {lab} max position error = {np.max(pos_err):.6f}")


# =============================================================================
# 8. Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    G = 1.0
    state0, masses, period = figure_eight_ics(G)

    # Simulate for t in [0, 1] — a fraction of the period (T ~ 6.33).
    # The three-body problem is chaotic, so even this short horizon
    # is a meaningful test. The PINN will likely diverge before t=1,
    # and showing that chaos boundary is itself a result.
    t_max = 1.0

    print(f"Figure-eight three-body orbit")
    print(f"  Period: {period:.4f}")
    print(f"  Simulating: t in [0, {t_max}]")
    print()

    model, loss_history = train_threebody_pinn(
        state0, masses, t_max, G=G,
        n_collocation=1000, epochs=10000, lr=1e-3,
        ic_weight=100.0
    )

    visualize_threebody_results(model, state0, masses, t_max, G=G)
