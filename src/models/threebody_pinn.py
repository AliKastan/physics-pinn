"""
PINN for the gravitational three-body problem.

The three-body problem has no general closed-form solution and exhibits
chaotic dynamics — tiny perturbations grow exponentially.  A PINN
approximates the solution by learning a neural-network trajectory that
satisfies Newton's law of gravitation for all three pairwise interactions.

State vector (12 components):
    (x1,y1, x2,y2, x3,y3, vx1,vy1, vx2,vy2, vx3,vy3)

Physics residual — 12 first-order ODEs:
    dx_i/dt  = vx_i                            (kinematic)
    dy_i/dt  = vy_i
    dvx_i/dt = G * sum_{j!=i} m_j*(xj-xi)/rij^3  (Newton's law)
    dvy_i/dt = G * sum_{j!=i} m_j*(yj-yi)/rij^3

Conserved quantities tracked for validation:
    - Total energy  E = T + V
    - Angular momentum  L = sum m_i * (x_i*vy_i - y_i*vx_i)
    - Centre of mass  (should be constant if total momentum is zero)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp


# =========================================================================
# Network
# =========================================================================

class ThreeBodyPINN(nn.Module):
    """
    Maps t -> 12 state variables for three gravitating bodies.

    Architecture: 5 hidden layers x 256 neurons, tanh.
    Larger than the two-body PINN because 12 coupled outputs with
    three pairwise 1/r^3 interactions.
    """

    def __init__(self, hidden_size=256, num_hidden_layers=5):
        super().__init__()
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 12))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)


# =========================================================================
# Famous preset configurations
# =========================================================================

def figure_eight_ics(G=1.0):
    """
    Figure-eight orbit (Chenciner & Montgomery 2000).
    Three equal masses trace a single figure-eight curve.
    Period ~ 6.3259.
    """
    masses = [1.0, 1.0, 1.0]
    x1, y1 = -0.97000436, 0.24308753
    x2, y2 = 0.97000436, -0.24308753
    x3, y3 = 0.0, 0.0
    vx3, vy3 = -0.93240737, -0.86473146
    vx1, vy1 = -vx3 / 2.0, -vy3 / 2.0
    vx2, vy2 = -vx3 / 2.0, -vy3 / 2.0
    state0 = [x1, y1, x2, y2, x3, y3,
              vx1, vy1, vx2, vy2, vx3, vy3]
    return state0, masses, 6.3259


def lagrange_triangle_ics(G=1.0):
    """
    Lagrange equilateral triangle — three equal masses at vertices
    of a rotating equilateral triangle.  A stable periodic solution.
    """
    masses = [1.0, 1.0, 1.0]
    M_total = sum(masses)
    # Radius of circumscribed circle for unit-side equilateral triangle
    R = 1.0 / np.sqrt(3)
    angles = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3]
    # Angular velocity for circular orbits: omega^2 * R = G*M/(2R)^2 * sqrt(3)
    # For equal masses on equilateral triangle: omega = sqrt(G*M_total / R^3) * correction
    omega = np.sqrt(G * M_total / (R ** 3 * np.sqrt(3)))

    state0 = []
    vels = []
    for a in angles:
        x, y = R * np.cos(a), R * np.sin(a)
        state0.extend([x, y])
        vels.extend([-omega * y, omega * x])
    state0.extend(vels)

    period = 2 * np.pi / omega
    return state0, masses, period


def pythagorean_ics(G=1.0):
    """
    Pythagorean three-body problem — masses 3, 4, 5 at vertices of a
    right triangle with sides 3, 4, 5.  All bodies start at rest.
    This configuration leads to dramatic chaotic dynamics.
    """
    masses = [3.0, 4.0, 5.0]
    # Right triangle with legs 3 and 4
    state0 = [
        1.0, 3.0,    # body 1 (mass 3)
        -2.0, -1.0,  # body 2 (mass 4)
        1.0, -1.0,   # body 3 (mass 5)
        0.0, 0.0,    # v1 = 0 (at rest)
        0.0, 0.0,    # v2 = 0
        0.0, 0.0,    # v3 = 0
    ]
    return state0, masses, 25.0  # approximate interaction time


def sun_earth_moon_ics(G=1.0):
    """
    Hierarchical Sun-Earth-Moon (approximate, normalized).
    Sun is very massive at origin, Earth orbits circularly,
    Moon orbits Earth.
    """
    m_sun = 100.0
    m_earth = 1.0
    m_moon = 0.01
    masses = [m_sun, m_earth, m_moon]

    # Sun at origin, at rest
    x_s, y_s, vx_s, vy_s = 0.0, 0.0, 0.0, 0.0
    # Earth in circular orbit at r=1
    r_e = 1.0
    v_e = np.sqrt(G * m_sun / r_e)
    x_e, y_e, vx_e, vy_e = r_e, 0.0, 0.0, v_e
    # Moon orbiting Earth at r=0.05
    r_m = 0.05
    v_m = np.sqrt(G * m_earth / r_m) + v_e
    x_m, y_m = r_e + r_m, 0.0
    vx_m, vy_m = 0.0, v_m

    state0 = [x_s, y_s, x_e, y_e, x_m, y_m,
              vx_s, vy_s, vx_e, vy_e, vx_m, vy_m]
    period = 2 * np.pi * np.sqrt(r_e ** 3 / (G * m_sun))
    return state0, masses, period


PRESETS = {
    'figure_eight': figure_eight_ics,
    'lagrange_triangle': lagrange_triangle_ics,
    'pythagorean': pythagorean_ics,
    'sun_earth_moon': sun_earth_moon_ics,
}


# =========================================================================
# Classical solver (ground truth)
# =========================================================================

def three_body_rhs(t, state, masses, G=1.0, eps=1e-3):
    """RHS for scipy solve_ivp. Gravitational softening eps prevents singularities."""
    x1, y1, x2, y2, x3, y3 = state[0:6]
    vx1, vy1, vx2, vy2, vx3, vy3 = state[6:12]
    m1, m2, m3 = masses

    r12 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + eps ** 2)
    r13 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2 + eps ** 2)
    r23 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2 + eps ** 2)

    ax1 = G * m2 * (x2 - x1) / r12 ** 3 + G * m3 * (x3 - x1) / r13 ** 3
    ay1 = G * m2 * (y2 - y1) / r12 ** 3 + G * m3 * (y3 - y1) / r13 ** 3
    ax2 = G * m1 * (x1 - x2) / r12 ** 3 + G * m3 * (x3 - x2) / r23 ** 3
    ay2 = G * m1 * (y1 - y2) / r12 ** 3 + G * m3 * (y3 - y2) / r23 ** 3
    ax3 = G * m1 * (x1 - x3) / r13 ** 3 + G * m2 * (x2 - x3) / r23 ** 3
    ay3 = G * m1 * (y1 - y3) / r13 ** 3 + G * m2 * (y2 - y3) / r23 ** 3

    return [vx1, vy1, vx2, vy2, vx3, vy3,
            ax1, ay1, ax2, ay2, ax3, ay3]


def solve_threebody(state0, masses, t_max, G=1.0, n_points=2000):
    """Solve with DOP853 (8th-order Dormand-Prince) at tight tolerances."""
    t_eval = np.linspace(0, t_max, n_points)
    sol = solve_ivp(
        three_body_rhs, (0, t_max), state0,
        args=(masses, G), t_eval=t_eval,
        method='DOP853', rtol=1e-12, atol=1e-14,
    )
    return sol.t, sol.y.T  # (n_points, 12)


# =========================================================================
# Conservation quantities (numpy, for evaluation)
# =========================================================================

def compute_energy(states, masses, G=1.0, eps=1e-3):
    """Total energy E = T + V."""
    x1, y1 = states[:, 0], states[:, 1]
    x2, y2 = states[:, 2], states[:, 3]
    x3, y3 = states[:, 4], states[:, 5]
    vx1, vy1 = states[:, 6], states[:, 7]
    vx2, vy2 = states[:, 8], states[:, 9]
    vx3, vy3 = states[:, 10], states[:, 11]
    m1, m2, m3 = masses

    T = (0.5 * m1 * (vx1 ** 2 + vy1 ** 2) +
         0.5 * m2 * (vx2 ** 2 + vy2 ** 2) +
         0.5 * m3 * (vx3 ** 2 + vy3 ** 2))

    r12 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + eps ** 2)
    r13 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2 + eps ** 2)
    r23 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2 + eps ** 2)

    V = -G * (m1 * m2 / r12 + m1 * m3 / r13 + m2 * m3 / r23)
    return T + V


def compute_angular_momentum(states, masses):
    """Total angular momentum L = sum m_i*(x_i*vy_i - y_i*vx_i)."""
    m1, m2, m3 = masses
    L = (m1 * (states[:, 0] * states[:, 7] - states[:, 1] * states[:, 6]) +
         m2 * (states[:, 2] * states[:, 9] - states[:, 3] * states[:, 8]) +
         m3 * (states[:, 4] * states[:, 11] - states[:, 5] * states[:, 10]))
    return L


def compute_center_of_mass(states, masses):
    """Centre of mass position (should be constant if P=0)."""
    m1, m2, m3 = masses
    M = m1 + m2 + m3
    cx = (m1 * states[:, 0] + m2 * states[:, 2] + m3 * states[:, 4]) / M
    cy = (m1 * states[:, 1] + m2 * states[:, 3] + m3 * states[:, 5]) / M
    return cx, cy


# =========================================================================
# Physics-informed loss
# =========================================================================

def physics_loss(model, t_col, masses, G=1.0, eps=1e-3):
    """Enforce Newton's law of gravitation for all 12 first-order ODEs."""
    t_col.requires_grad_(True)
    output = model(t_col)

    x1, y1 = output[:, 0:1], output[:, 1:2]
    x2, y2 = output[:, 2:3], output[:, 3:4]
    x3, y3 = output[:, 4:5], output[:, 5:6]
    vx1, vy1 = output[:, 6:7], output[:, 7:8]
    vx2, vy2 = output[:, 8:9], output[:, 9:10]
    vx3, vy3 = output[:, 10:11], output[:, 11:12]

    ones = torch.ones_like(x1)

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

    r12 = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + eps ** 2)
    r13 = torch.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2 + eps ** 2)
    r23 = torch.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2 + eps ** 2)

    ax1 = G * m2 * (x2 - x1) / r12 ** 3 + G * m3 * (x3 - x1) / r13 ** 3
    ay1 = G * m2 * (y2 - y1) / r12 ** 3 + G * m3 * (y3 - y1) / r13 ** 3
    ax2 = G * m1 * (x1 - x2) / r12 ** 3 + G * m3 * (x3 - x2) / r23 ** 3
    ay2 = G * m1 * (y1 - y2) / r12 ** 3 + G * m3 * (y3 - y2) / r23 ** 3
    ax3 = G * m1 * (x1 - x3) / r13 ** 3 + G * m2 * (x2 - x3) / r23 ** 3
    ay3 = G * m1 * (y1 - y3) / r13 ** 3 + G * m2 * (y2 - y3) / r23 ** 3

    # Kinematic residuals
    loss = (torch.mean((dx1 - vx1) ** 2) + torch.mean((dy1 - vy1) ** 2) +
            torch.mean((dx2 - vx2) ** 2) + torch.mean((dy2 - vy2) ** 2) +
            torch.mean((dx3 - vx3) ** 2) + torch.mean((dy3 - vy3) ** 2))
    # Dynamic residuals
    loss = loss + (
        torch.mean((dvx1 - ax1) ** 2) + torch.mean((dvy1 - ay1) ** 2) +
        torch.mean((dvx2 - ax2) ** 2) + torch.mean((dvy2 - ay2) ** 2) +
        torch.mean((dvx3 - ax3) ** 2) + torch.mean((dvy3 - ay3) ** 2))
    return loss


def ic_loss(model, state0):
    """Enforce initial conditions at t=0 for all 12 variables."""
    t_zero = torch.zeros(1, 1)
    output = model(t_zero).squeeze()
    state0_t = torch.tensor(state0, dtype=torch.float32)
    return torch.sum((output - state0_t) ** 2)


def conservation_loss(model, t_col, masses, G=1.0, eps=1e-3):
    """
    Penalise energy and angular-momentum drift along the trajectory.
    E(t) should equal E(0); L(t) should equal L(0).
    """
    with torch.no_grad():
        output = model(t_col)
    states_np = output.numpy()

    E = compute_energy(states_np, masses, G, eps)
    L = compute_angular_momentum(states_np, masses)
    E0, L0 = E[0], L[0]

    loss_E = np.mean((E - E0) ** 2)
    loss_L = np.mean((L - L0) ** 2)
    return torch.tensor(loss_E + loss_L, dtype=torch.float32)


# =========================================================================
# Training
# =========================================================================

def train_threebody(state0, masses, t_max, G=1.0,
                    n_collocation=1000, epochs=10000, lr=1e-3,
                    ic_weight=100.0, conservation_weight=1.0,
                    hidden_size=256, num_hidden_layers=5,
                    verbose=True):
    """
    Train the three-body PINN.

    Loss = physics + ic_weight * IC + conservation_weight * (energy + angmom)

    Returns:
        model, loss_history
    """
    model = ThreeBodyPINN(hidden_size=hidden_size,
                          num_hidden_layers=num_hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6,
    )
    loss_history = []

    if verbose:
        print(f"Training ThreeBodyPINN  ({hidden_size}x{num_hidden_layers})")
        print(f"  Time horizon: {t_max:.4f}")
        print(f"  Collocation: {n_collocation}, Epochs: {epochs}")
        print(f"  Masses: {masses}")
        print("-" * 55)

    for epoch in range(epochs):
        optimizer.zero_grad()

        t_col = torch.rand(n_collocation, 1) * t_max
        loss_phys = physics_loss(model, t_col, masses, G)
        loss_ic = ic_loss(model, state0)

        total = loss_phys + ic_weight * loss_ic

        # Conservation penalty (every 100 epochs — expensive)
        if conservation_weight > 0 and (epoch + 1) % 100 == 0:
            t_eval_cons = torch.linspace(0, t_max, 50).unsqueeze(1)
            loss_cons = conservation_loss(model, t_eval_cons, masses, G)
            total = total + conservation_weight * loss_cons

        total.backward()
        optimizer.step()
        scheduler.step(total.item())
        loss_history.append(total.item())

        if verbose and (epoch + 1) % 2000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | "
                  f"Loss: {total.item():.6f} | "
                  f"Phys: {loss_phys.item():.6f} | "
                  f"IC: {loss_ic.item():.6f}")

    if verbose:
        print("-" * 55)
        print(f"Final loss: {loss_history[-1]:.6f}")

    return model, loss_history
