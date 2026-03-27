"""
Inverse Problem Solving with Physics-Informed Neural Networks
=============================================================

PART 1 — Pendulum Parameter Inference:
    Given noisy observations of theta(t), infer the ratio g/L by treating
    it as a trainable torch.nn.Parameter. The PINN simultaneously denoises
    the trajectory and recovers the unknown physics.

    Loss = physics residual (using trainable g/L) + data fitting + IC

PART 2 — Orbital Parameter Inference:
    Given noisy (x, y) position observations of an orbit, infer the
    gravitational parameter GM as a trainable torch.nn.Parameter.

Why This Works:
    In the forward PINN, physical parameters are fixed constants in the
    ODE residual. In the inverse problem, we promote them to trainable
    variables. The optimizer adjusts them alongside the network weights
    until the physics residual and data-fitting loss are both small —
    this can only happen when the parameters match reality.

    The key insight: the physics loss constrains the *form* of the solution
    (it must satisfy the ODE), while the data loss constrains the *specific*
    trajectory. Together they pin down both the trajectory and the unknown
    parameters.
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# =============================================================================
# 1. Neural Network Architectures
# =============================================================================

class PINNPendulum(nn.Module):
    """Maps time t -> (theta, omega). Same architecture as pinn_pendulum.py."""

    def __init__(self, hidden_size=64, num_hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)


class PINNOrbital(nn.Module):
    """Maps time t -> (x, y, vx, vy). Same architecture as pinn_orbital.py."""

    def __init__(self, hidden_size=128, num_hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)


# =============================================================================
# PART 1: Pendulum — Infer g/L from noisy theta observations
# =============================================================================

def generate_pendulum_data(theta_0, omega_0, t_max, g_true=9.81, L_true=1.0,
                           n_obs=50, noise_std=0.05):
    """
    Generate noisy pendulum observations by solving the ODE with true
    parameters and adding Gaussian noise to theta.

    Returns:
        t_obs, theta_obs: Noisy observation times and angles
        t_dense, theta_true, omega_true: Dense ground truth for plotting
    """
    # Dense ground truth
    t_dense = np.linspace(0, t_max, 1000)
    sol = solve_ivp(
        lambda t, y: [y[1], -(g_true / L_true) * np.sin(y[0])],
        (0, t_max), [theta_0, omega_0],
        t_eval=t_dense, method='RK45', rtol=1e-10, atol=1e-12
    )
    theta_true = sol.y[0]
    omega_true = sol.y[1]

    # Sparse noisy observations (avoid t=0, handled by IC loss)
    t_obs = np.sort(np.random.uniform(0.1, t_max, n_obs))
    sol_obs = solve_ivp(
        lambda t, y: [y[1], -(g_true / L_true) * np.sin(y[0])],
        (0, t_max), [theta_0, omega_0],
        t_eval=t_obs, method='RK45', rtol=1e-10, atol=1e-12
    )
    theta_obs = sol_obs.y[0] + np.random.normal(0, noise_std, n_obs)

    return t_obs, theta_obs, t_dense, theta_true, omega_true


def train_inverse_pendulum(theta_0, omega_0, t_max, t_obs, theta_obs,
                           g_over_L_init=5.0,
                           n_collocation=500, epochs=10000,
                           lr_net=1e-3, lr_param=1e-2,
                           ic_weight=20.0, data_weight=10.0):
    """
    Train a PINN where g/L is a single trainable parameter.

    The pendulum ODE only depends on the ratio g/L, not on g and L
    separately:
        dtheta/dt = omega
        domega/dt = -(g/L) * sin(theta)

    So we can identify g/L directly from trajectory data.

    Strategy:
        - Phase 1 (warmup): fit network to data + IC only, so it learns
          a reasonable trajectory shape before physics kicks in.
        - Phase 2: full loss = physics + data + IC. The physics residual
          uses the trainable g_over_L, which the optimizer drives toward
          the true value.

    Returns:
        model, g_over_L_param, g_over_L_history, loss_history
    """
    model = PINNPendulum()

    # Trainable parameter: g/L, initialized far from truth (9.81)
    g_over_L = nn.Parameter(torch.tensor(g_over_L_init, dtype=torch.float32))

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr_net},
        {'params': [g_over_L], 'lr': lr_param}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6
    )

    t_obs_t = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(1)
    theta_obs_t = torch.tensor(theta_obs, dtype=torch.float32)

    loss_history = []
    g_over_L_history = []

    # Phase 1: warmup — data + IC only
    warmup_epochs = 1500
    warmup_opt = torch.optim.Adam(model.parameters(), lr=lr_net)

    print("PART 1: Pendulum Inverse Problem")
    print(f"  Initial guess: g/L = {g_over_L_init:.2f} (true: 9.81)")
    print(f"  Observations: {len(t_obs)} noisy points (sigma={theta_obs.std():.3f})")
    print(f"  Warmup: {warmup_epochs} epochs | Main: {epochs} epochs")
    print("-" * 55)

    for ep in range(warmup_epochs):
        warmup_opt.zero_grad()
        out = model(t_obs_t)
        loss = data_weight * torch.mean((out[:, 0] - theta_obs_t) ** 2)
        t0 = torch.zeros(1, 1)
        out0 = model(t0)
        loss = loss + ic_weight * ((out0[0, 0] - theta_0)**2 +
                                   (out0[0, 1] - omega_0)**2)
        loss.backward()
        warmup_opt.step()

    print(f"  Warmup done.")

    # Phase 2: full training
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Physics loss at collocation points
        t_col = torch.rand(n_collocation, 1) * t_max
        t_col.requires_grad_(True)

        out = model(t_col)
        theta = out[:, 0:1]
        omega = out[:, 1:2]

        dtheta_dt = torch.autograd.grad(
            theta, t_col, torch.ones_like(theta), create_graph=True)[0]
        domega_dt = torch.autograd.grad(
            omega, t_col, torch.ones_like(omega), create_graph=True)[0]

        res1 = dtheta_dt - omega
        res2 = domega_dt + g_over_L * torch.sin(theta)
        loss_phys = torch.mean(res1**2) + torch.mean(res2**2)

        # Data loss
        out_obs = model(t_obs_t)
        loss_data = torch.mean((out_obs[:, 0] - theta_obs_t) ** 2)

        # IC loss
        t0 = torch.zeros(1, 1)
        out0 = model(t0)
        loss_ic = (out0[0, 0] - theta_0)**2 + (out0[0, 1] - omega_0)**2

        total = loss_phys + data_weight * loss_data + ic_weight * loss_ic
        total.backward()
        optimizer.step()
        scheduler.step(total.item())

        # Keep g/L positive
        with torch.no_grad():
            g_over_L.clamp_(min=0.1)

        loss_history.append(total.item())
        g_over_L_history.append(g_over_L.item())

        if (epoch + 1) % 2000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | "
                  f"Loss: {total.item():.6f} | "
                  f"g/L: {g_over_L.item():.4f}")

    print("-" * 55)
    g_final = g_over_L.item()
    print(f"  Recovered g/L = {g_final:.4f} (true: 9.81)")
    print(f"  Relative error: {abs(g_final - 9.81)/9.81*100:.2f}%")

    return model, g_over_L, g_over_L_history, loss_history


# =============================================================================
# PART 2: Orbital — Infer GM from noisy (x, y) observations
# =============================================================================

def generate_orbital_data(ecc=0.5, GM_true=1.0, n_obs=80, noise_std=0.05):
    """
    Generate noisy orbital position observations.

    Returns:
        t_obs, x_obs, y_obs: Noisy observation data
        t_dense, x_true, y_true, vx_true, vy_true: Dense ground truth
        ics: (x0, y0, vx0, vy0)
        t_max: Simulation time (one orbital period)
    """
    a = 1.0
    x0 = a * (1.0 - ecc)
    y0 = 0.0
    vx0 = 0.0
    vy0 = np.sqrt(GM_true / a * (1.0 + ecc) / (1.0 - ecc))
    period = 2.0 * np.pi * np.sqrt(a**3 / GM_true)
    t_max = period  # one full orbit

    def rhs(t, s):
        x, y, vx, vy = s
        r3 = (x**2 + y**2)**1.5
        return [vx, vy, -GM_true * x / r3, -GM_true * y / r3]

    # Dense ground truth
    t_dense = np.linspace(0, t_max, 1500)
    sol = solve_ivp(rhs, (0, t_max), [x0, y0, vx0, vy0],
                    t_eval=t_dense, method='RK45', rtol=1e-12, atol=1e-14)
    x_true, y_true = sol.y[0], sol.y[1]
    vx_true, vy_true = sol.y[2], sol.y[3]

    # Sparse noisy position observations
    t_obs = np.sort(np.random.uniform(0.05 * t_max, 0.95 * t_max, n_obs))
    sol_obs = solve_ivp(rhs, (0, t_max), [x0, y0, vx0, vy0],
                        t_eval=t_obs, method='RK45', rtol=1e-12, atol=1e-14)
    x_obs = sol_obs.y[0] + np.random.normal(0, noise_std, n_obs)
    y_obs = sol_obs.y[1] + np.random.normal(0, noise_std, n_obs)

    ics = (x0, y0, vx0, vy0)
    return t_obs, x_obs, y_obs, t_dense, x_true, y_true, vx_true, vy_true, ics, t_max


def train_inverse_orbital(ics, t_max, t_obs, x_obs, y_obs,
                          GM_init=0.5,
                          n_collocation=600, epochs=10000,
                          lr_net=1e-3, lr_param=5e-3,
                          ic_weight=50.0, data_weight=10.0):
    """
    Train an orbital PINN where GM is a trainable parameter.

    Returns:
        model, GM_param, GM_history, loss_history
    """
    x0, y0, vx0, vy0 = ics

    model = PINNOrbital()
    GM_param = nn.Parameter(torch.tensor(GM_init, dtype=torch.float32))

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr_net},
        {'params': [GM_param], 'lr': lr_param}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6
    )

    t_obs_t = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(1)
    x_obs_t = torch.tensor(x_obs, dtype=torch.float32)
    y_obs_t = torch.tensor(y_obs, dtype=torch.float32)

    x0_t = torch.tensor(x0, dtype=torch.float32)
    y0_t = torch.tensor(y0, dtype=torch.float32)
    vx0_t = torch.tensor(vx0, dtype=torch.float32)
    vy0_t = torch.tensor(vy0, dtype=torch.float32)

    loss_history = []
    GM_history = []

    # Phase 1: warmup — data + IC only
    warmup_epochs = 2000
    warmup_opt = torch.optim.Adam(model.parameters(), lr=lr_net)

    print("\nPART 2: Orbital Inverse Problem")
    print(f"  Initial guess: GM = {GM_init:.2f} (true: 1.00)")
    print(f"  Observations: {len(t_obs)} noisy (x,y) points")
    print(f"  Warmup: {warmup_epochs} epochs | Main: {epochs} epochs")
    print("-" * 55)

    for ep in range(warmup_epochs):
        warmup_opt.zero_grad()
        out = model(t_obs_t)
        loss = data_weight * (torch.mean((out[:, 0] - x_obs_t)**2) +
                              torch.mean((out[:, 1] - y_obs_t)**2))
        t0 = torch.zeros(1, 1)
        out0 = model(t0)
        loss = loss + ic_weight * ((out0[0, 0] - x0_t)**2 +
                                   (out0[0, 1] - y0_t)**2 +
                                   (out0[0, 2] - vx0_t)**2 +
                                   (out0[0, 3] - vy0_t)**2)
        loss.backward()
        warmup_opt.step()

    print(f"  Warmup done.")

    # Phase 2: full training
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Physics loss
        t_col = torch.rand(n_collocation, 1) * t_max
        t_col.requires_grad_(True)

        out = model(t_col)
        x, y, vx, vy = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

        dx = torch.autograd.grad(x, t_col, torch.ones_like(x), create_graph=True)[0]
        dy = torch.autograd.grad(y, t_col, torch.ones_like(y), create_graph=True)[0]
        dvx = torch.autograd.grad(vx, t_col, torch.ones_like(vx), create_graph=True)[0]
        dvy = torch.autograd.grad(vy, t_col, torch.ones_like(vy), create_graph=True)[0]

        r = torch.sqrt(x**2 + y**2 + 1e-8)
        r3 = r**3

        loss_phys = (torch.mean((dx - vx)**2) + torch.mean((dy - vy)**2) +
                     torch.mean((dvx + GM_param * x / r3)**2) +
                     torch.mean((dvy + GM_param * y / r3)**2))

        # Data loss (position only)
        out_obs = model(t_obs_t)
        loss_data = (torch.mean((out_obs[:, 0] - x_obs_t)**2) +
                     torch.mean((out_obs[:, 1] - y_obs_t)**2))

        # IC loss
        t0 = torch.zeros(1, 1)
        out0 = model(t0)
        loss_ic = ((out0[0, 0] - x0_t)**2 + (out0[0, 1] - y0_t)**2 +
                   (out0[0, 2] - vx0_t)**2 + (out0[0, 3] - vy0_t)**2)

        total = loss_phys + data_weight * loss_data + ic_weight * loss_ic
        total.backward()
        optimizer.step()
        scheduler.step(total.item())

        with torch.no_grad():
            GM_param.clamp_(min=0.01)

        loss_history.append(total.item())
        GM_history.append(GM_param.item())

        if (epoch + 1) % 2000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | "
                  f"Loss: {total.item():.6f} | "
                  f"GM: {GM_param.item():.4f}")

    print("-" * 55)
    gm_final = GM_param.item()
    print(f"  Recovered GM = {gm_final:.4f} (true: 1.00)")
    print(f"  Relative error: {abs(gm_final - 1.0)/1.0*100:.2f}%")

    return model, GM_param, GM_history, loss_history


# =============================================================================
# Visualization
# =============================================================================

def visualize_results(
    # Pendulum results
    pend_model, g_over_L_param, g_over_L_history, pend_losses,
    t_obs_p, theta_obs_p, t_dense_p, theta_true_p, omega_true_p,
    theta_0, omega_0, t_max_p,
    # Orbital results
    orb_model, GM_param, GM_history, orb_losses,
    t_obs_o, x_obs_o, y_obs_o,
    t_dense_o, x_true_o, y_true_o, ics_o, t_max_o
):
    """
    Combined 2x3 figure:
        Row 1 (Pendulum): noisy data vs PINN vs truth | g/L convergence | loss
        Row 2 (Orbital):  noisy orbit vs PINN vs truth | GM convergence | loss
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Inverse PINNs — Inferring Physical Parameters from Noisy Data',
                 fontsize=15, fontweight='bold')

    # --- Pendulum: trajectory reconstruction ---
    pend_model.eval()
    t_plot = np.linspace(0, t_max_p, 1000)
    with torch.no_grad():
        out = pend_model(
            torch.tensor(t_plot, dtype=torch.float32).unsqueeze(1)
        ).numpy()
    theta_pinn = out[:, 0]

    ax1 = axes[0, 0]
    ax1.plot(t_dense_p, np.degrees(theta_true_p), 'b-', linewidth=2,
             label='Ground truth', alpha=0.7)
    ax1.scatter(t_obs_p, np.degrees(theta_obs_p), c='gray', s=20, alpha=0.5,
                label='Noisy observations', zorder=5)
    ax1.plot(t_plot, np.degrees(theta_pinn), 'r--', linewidth=2,
             label='PINN reconstruction')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Pendulum: Trajectory Reconstruction')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Pendulum: g/L convergence ---
    ax2 = axes[0, 1]
    ax2.plot(g_over_L_history, 'r-', linewidth=1.5, label='Inferred g/L')
    ax2.axhline(y=9.81, color='blue', linestyle='--', linewidth=2,
                label='True g/L = 9.81')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('g/L (s$^{-2}$)')
    ax2.set_title(f'g/L Convergence (final: {g_over_L_param.item():.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Pendulum: training loss ---
    ax3 = axes[0, 2]
    ax3.semilogy(pend_losses, linewidth=1.5, color='navy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Total Loss')
    ax3.set_title('Pendulum Training Loss')
    ax3.grid(True, alpha=0.3)

    # --- Orbital: trajectory reconstruction ---
    orb_model.eval()
    t_plot_o = np.linspace(0, t_max_o, 1500)
    with torch.no_grad():
        out_o = orb_model(
            torch.tensor(t_plot_o, dtype=torch.float32).unsqueeze(1)
        ).numpy()
    x_pinn, y_pinn = out_o[:, 0], out_o[:, 1]

    ax4 = axes[1, 0]
    ax4.plot(x_true_o, y_true_o, 'b-', linewidth=2,
             label='Ground truth', alpha=0.7)
    ax4.scatter(x_obs_o, y_obs_o, c='gray', s=20, alpha=0.5,
                label='Noisy observations', zorder=5)
    ax4.plot(x_pinn, y_pinn, 'r--', linewidth=2, label='PINN reconstruction')
    ax4.plot(0, 0, 'ko', markersize=8, label='Central body')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Orbital: Trajectory Reconstruction')
    ax4.legend(fontsize=8)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # --- Orbital: GM convergence ---
    ax5 = axes[1, 1]
    ax5.plot(GM_history, 'r-', linewidth=1.5, label='Inferred GM')
    ax5.axhline(y=1.0, color='blue', linestyle='--', linewidth=2,
                label='True GM = 1.0')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('GM')
    ax5.set_title(f'GM Convergence (final: {GM_param.item():.4f})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # --- Orbital: training loss ---
    ax6 = axes[1, 2]
    ax6.semilogy(orb_losses, linewidth=1.5, color='darkred')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Total Loss')
    ax6.set_title('Orbital Training Loss')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('inverse_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to inverse_results.png")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- PART 1: Pendulum ----
    theta_0 = np.pi / 4
    omega_0 = 0.0
    t_max_p = 10.0

    print("Generating noisy pendulum data...")
    t_obs_p, theta_obs_p, t_dense_p, theta_true_p, omega_true_p = \
        generate_pendulum_data(
            theta_0, omega_0, t_max_p,
            g_true=9.81, L_true=1.0,
            n_obs=80, noise_std=0.05
        )
    print(f"  {len(t_obs_p)} observations generated.\n")

    pend_model, g_over_L, g_over_L_hist, pend_losses = \
        train_inverse_pendulum(
            theta_0, omega_0, t_max_p, t_obs_p, theta_obs_p,
            g_over_L_init=5.0, epochs=10000
        )

    # ---- PART 2: Orbital ----
    print("\nGenerating noisy orbital data...")
    (t_obs_o, x_obs_o, y_obs_o,
     t_dense_o, x_true_o, y_true_o, vx_true_o, vy_true_o,
     ics_o, t_max_o) = generate_orbital_data(
        ecc=0.5, GM_true=1.0, n_obs=80, noise_std=0.05
    )
    print(f"  {len(t_obs_o)} observations generated.\n")

    orb_model, GM_param, GM_hist, orb_losses = \
        train_inverse_orbital(
            ics_o, t_max_o, t_obs_o, x_obs_o, y_obs_o,
            GM_init=0.5, epochs=10000
        )

    # ---- Visualization ----
    visualize_results(
        pend_model, g_over_L, g_over_L_hist, pend_losses,
        t_obs_p, theta_obs_p, t_dense_p, theta_true_p, omega_true_p,
        theta_0, omega_0, t_max_p,
        orb_model, GM_param, GM_hist, orb_losses,
        t_obs_o, x_obs_o, y_obs_o,
        t_dense_o, x_true_o, y_true_o, ics_o, t_max_o
    )
