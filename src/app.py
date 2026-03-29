"""
Streamlit Web App — Physics-Informed Neural Networks
=====================================================
Interactive comparison of PINN predictions vs classical ODE solvers
for pendulum motion and orbital mechanics.

Now imports models and utilities from the src package.
"""

import sys
import os

# Add parent directory to path so we can import the src package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

from src.models import PINNPendulum, PINNOrbital
from src.physics.equations import pendulum_residual, orbital_residual
from src.physics.constants import GRAVITY, PENDULUM_LENGTH, GM_DEFAULT
from src.utils.validation import solve_pendulum_ode, solve_orbit_ode, setup_orbital_ics
from src.utils.metrics import (
    compute_energy_pendulum, compute_energy_orbital,
    compute_angular_momentum,
)
from src.training.losses import pendulum_ic_loss, orbital_ic_loss

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Physics-Informed Neural Networks",
    page_icon="",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load default configs
# ---------------------------------------------------------------------------
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def load_config(name):
    path = os.path.join(CONFIG_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


PENDULUM_CFG = load_config("pendulum_default.yaml")
ORBITAL_CFG = load_config("orbital_default.yaml")

# ---------------------------------------------------------------------------
# Color constants (matching original app)
# ---------------------------------------------------------------------------
C_CLASSICAL = "#636EFA"
C_PINN = "#EF553B"
C_ERROR = "#00CC96"
C_ACCENT = "#AB63FA"


# ===========================================================================
# Models not yet in the package (kept self-contained for now)
# ===========================================================================

class HamiltonianNet(nn.Module):
    """Learns scalar Hamiltonian H(q, p); EoMs derived via autograd."""
    def __init__(self, hidden_size=64, num_hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, q, p):
        return self.network(torch.cat([q, p], dim=1))

    def time_derivatives(self, q, p):
        H = self.forward(q, p)
        dH_dq = torch.autograd.grad(
            H, q, grad_outputs=torch.ones_like(H), create_graph=True)[0]
        dH_dp = torch.autograd.grad(
            H, p, grad_outputs=torch.ones_like(H), create_graph=True)[0]
        return dH_dp, -dH_dq


class PINNPDE(nn.Module):
    """Maps (x, t) -> u for PDE solving."""
    def __init__(self, hidden_size=64, num_hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.network(torch.cat([x, t], dim=1))


class PINNThreeBody(nn.Module):
    """Maps t -> 12 state variables for the three-body problem."""
    def __init__(self, hidden_size=128, num_hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 12))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)


# ===========================================================================
# Training helpers
# ===========================================================================

def _pendulum_pointwise_residual(model, t_points, g, L):
    """Per-point ODE residual for adaptive sampling (no graph needed)."""
    t_points = t_points.clone().requires_grad_(True)
    out = model(t_points)
    theta, omega = out[:, 0:1], out[:, 1:2]
    dtheta = torch.autograd.grad(
        theta, t_points, torch.ones_like(theta), create_graph=False)[0]
    domega = torch.autograd.grad(
        omega, t_points, torch.ones_like(omega), create_graph=False)[0]
    r1, r2 = pendulum_residual(dtheta, domega, theta, omega, g, L)
    return (r1.squeeze() ** 2 + r2.squeeze() ** 2).detach()


def _orbital_pointwise_residual(model, t_points, GM):
    """Per-point ODE residual for orbital adaptive sampling."""
    t_points = t_points.clone().requires_grad_(True)
    out = model(t_points)
    x, y, vx, vy = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
    ones = torch.ones_like(x)
    dx = torch.autograd.grad(x, t_points, ones, create_graph=False)[0]
    dy = torch.autograd.grad(y, t_points, ones, create_graph=False)[0]
    dvx = torch.autograd.grad(vx, t_points, ones, create_graph=False)[0]
    dvy = torch.autograd.grad(vy, t_points, ones, create_graph=False)[0]
    rx, ry, rvx, rvy = orbital_residual(dx, dy, dvx, dvy, x, y, vx, vy, GM)
    return (rx.squeeze()**2 + ry.squeeze()**2 +
            rvx.squeeze()**2 + rvy.squeeze()**2).detach()


def _sample_adaptive_1d(model, residual_fn, n_col, t_max, n_grid=1000,
                        uniform_frac=0.2):
    """Sample collocation points proportional to residual on a 1D domain."""
    t_grid = torch.linspace(0, t_max, n_grid).unsqueeze(1)
    with torch.no_grad():
        res = residual_fn(model, t_grid)
    probs = res / (res.sum() + 1e-16)
    n_a = int(n_col * (1 - uniform_frac))
    idx = torch.multinomial(probs, n_a, replacement=True)
    t_a = t_grid[idx]
    t_u = torch.rand(n_col - n_a, 1) * t_max
    return torch.cat([t_a, t_u], dim=0)


def train_pendulum(theta_0, omega_0, t_max, g, L,
                   n_col=400, epochs=3000, lr=1e-3, ic_w=20.0,
                   adaptive=False):
    """Train a pendulum PINN and return model + loss history."""
    cfg = PENDULUM_CFG.get("network", {})
    model = PINNPendulum(
        hidden_size=cfg.get("hidden_size", 64),
        num_hidden_layers=cfg.get("num_hidden_layers", 3),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=300, factor=0.5, min_lr=1e-6)
    losses = []
    label = "adaptive" if adaptive else "uniform"
    progress = st.progress(0, text=f"Training pendulum PINN ({label})...")

    cached_t_col = None

    for ep in range(epochs):
        opt.zero_grad()

        if adaptive and (ep % 500 == 0):
            model.eval()
            cached_t_col = _sample_adaptive_1d(
                model,
                lambda m, t: _pendulum_pointwise_residual(m, t, g, L),
                n_col, t_max)
            model.train()
        if adaptive and cached_t_col is not None:
            t_col = cached_t_col.clone().requires_grad_(True)
        else:
            t_col = torch.rand(n_col, 1) * t_max
            t_col.requires_grad_(True)

        out = model(t_col)
        theta, omega = out[:, 0:1], out[:, 1:2]
        dtheta = torch.autograd.grad(
            theta, t_col, torch.ones_like(theta), create_graph=True)[0]
        domega = torch.autograd.grad(
            omega, t_col, torch.ones_like(omega), create_graph=True)[0]

        r1, r2 = pendulum_residual(dtheta, domega, theta, omega, g, L)
        phys = torch.mean(r1**2) + torch.mean(r2**2)

        ic = pendulum_ic_loss(model, theta_0, omega_0)

        loss = phys + ic_w * ic
        loss.backward()
        opt.step()
        sched.step(loss.item())
        losses.append(loss.item())

        if (ep + 1) % 100 == 0:
            progress.progress(
                (ep + 1) / epochs,
                text=f"Epoch {ep+1}/{epochs}  |  Loss: {loss.item():.6f}")

    progress.empty()
    return model, losses


def train_orbital(x0, y0, vx0, vy0, t_max, GM,
                  n_col=600, epochs=5000, lr=1e-3, ic_w=50.0,
                  adaptive=False):
    """Train an orbital PINN and return model + loss history."""
    cfg = ORBITAL_CFG.get("network", {})
    model = PINNOrbital(
        hidden_size=cfg.get("hidden_size", 128),
        num_hidden_layers=cfg.get("num_hidden_layers", 4),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=300, factor=0.5, min_lr=1e-6)
    losses = []
    label = "adaptive" if adaptive else "uniform"
    progress = st.progress(0, text=f"Training orbital PINN ({label})...")

    x0_t = torch.tensor(x0, dtype=torch.float32)
    y0_t = torch.tensor(y0, dtype=torch.float32)
    vx0_t = torch.tensor(vx0, dtype=torch.float32)
    vy0_t = torch.tensor(vy0, dtype=torch.float32)

    cached_t_col = None

    for ep in range(epochs):
        opt.zero_grad()

        if adaptive and (ep % 500 == 0):
            model.eval()
            cached_t_col = _sample_adaptive_1d(
                model,
                lambda m, t: _orbital_pointwise_residual(m, t, GM),
                n_col, t_max)
            model.train()
        if adaptive and cached_t_col is not None:
            t_col = cached_t_col.clone().requires_grad_(True)
        else:
            t_col = torch.rand(n_col, 1) * t_max
            t_col.requires_grad_(True)

        out = model(t_col)
        x, y, vx, vy = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        ones = torch.ones_like(x)
        dx = torch.autograd.grad(x, t_col, ones, create_graph=True)[0]
        dy = torch.autograd.grad(y, t_col, ones, create_graph=True)[0]
        dvx = torch.autograd.grad(vx, t_col, ones, create_graph=True)[0]
        dvy = torch.autograd.grad(vy, t_col, ones, create_graph=True)[0]

        rx, ry, rvx, rvy = orbital_residual(dx, dy, dvx, dvy, x, y, vx, vy, GM)
        phys = (torch.mean(rx**2) + torch.mean(ry**2) +
                torch.mean(rvx**2) + torch.mean(rvy**2))

        ic = orbital_ic_loss(model, x0_t, y0_t, vx0_t, vy0_t)

        loss = phys + ic_w * ic
        loss.backward()
        opt.step()
        sched.step(loss.item())
        losses.append(loss.item())

        if (ep + 1) % 100 == 0:
            progress.progress(
                (ep + 1) / epochs,
                text=f"Epoch {ep+1}/{epochs}  |  Loss: {loss.item():.6f}")

    progress.empty()
    return model, losses


# ===========================================================================
# Sidebar navigation
# ===========================================================================

st.sidebar.title("PINN Explorer")
mode = st.sidebar.radio(
    "Select System",
    ["Pendulum", "Orbital Mechanics"],
    index=0,
)

# ===========================================================================
# Pendulum mode
# ===========================================================================

if mode == "Pendulum":
    st.title("Simple Pendulum PINN")
    st.markdown("Train a neural network to solve the pendulum ODE using only physics constraints.")

    col1, col2 = st.columns(2)
    with col1:
        L = st.slider("Pendulum length L (m)", 0.5, 3.0, float(PENDULUM_CFG.get("physics", {}).get("L", 1.0)), key="pend_L")
        theta_deg = st.slider("Initial angle (degrees)", 5, 85,
                               int(PENDULUM_CFG.get("initial_conditions", {}).get("theta_0_deg", 45)), key="pend_theta")
    with col2:
        t_max = st.slider("Time span (s)", 2, 20,
                           int(PENDULUM_CFG.get("training", {}).get("t_max", 10)), key="pend_tmax")
        epochs = st.select_slider("Training epochs",
                                   options=[1000, 2000, 3000, 5000, 8000],
                                   value=int(PENDULUM_CFG.get("training", {}).get("epochs", 3000)), key="pend_epochs")

    adaptive = st.checkbox("Use Adaptive Sampling", value=False, key="pend_adaptive")

    g = GRAVITY
    theta_0 = np.radians(theta_deg)
    omega_0 = 0.0

    if st.button("Train & Compare", key="pend_train"):
        torch.manual_seed(42)
        model, loss_history = train_pendulum(
            theta_0, omega_0, t_max, g, L,
            epochs=epochs, adaptive=adaptive)

        t_eval = np.linspace(0, t_max, 1000)
        t_ode, theta_ode, omega_ode = solve_pendulum_ode(
            theta_0, omega_0, (0, t_max), t_eval, g=g, L=L)

        model.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
            output = model(t_tensor).numpy()
        theta_pinn = output[:, 0]
        omega_pinn = output[:, 1]

        # Metrics
        max_err = np.max(np.abs(np.degrees(theta_pinn - theta_ode)))
        E_pinn = compute_energy_pendulum(theta_pinn, omega_pinn, g, L)
        E_drift = np.max(np.abs((E_pinn - E_pinn[0]) / (np.abs(E_pinn[0]) + 1e-16)))

        m1, m2, m3 = st.columns(3)
        m1.metric("Final Loss", f"{loss_history[-1]:.6f}")
        m2.metric("Max Angle Error", f"{max_err:.2f}°")
        m3.metric("Energy Drift", f"{E_drift:.4f}")

        # Plots
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["θ(t)", "ω(t)", "Phase Portrait", "Error"])

        fig.add_trace(go.Scatter(x=t_ode, y=np.degrees(theta_ode), mode='lines',
                                  name='Classical', line=dict(color=C_CLASSICAL)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_eval, y=np.degrees(theta_pinn), mode='lines',
                                  name='PINN', line=dict(color=C_PINN, dash='dash')), row=1, col=1)

        fig.add_trace(go.Scatter(x=t_ode, y=omega_ode, mode='lines',
                                  name='Classical', line=dict(color=C_CLASSICAL), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=t_eval, y=omega_pinn, mode='lines',
                                  name='PINN', line=dict(color=C_PINN, dash='dash'), showlegend=False), row=1, col=2)

        fig.add_trace(go.Scatter(x=np.degrees(theta_ode), y=omega_ode, mode='lines',
                                  name='Classical', line=dict(color=C_CLASSICAL), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=np.degrees(theta_pinn), y=omega_pinn, mode='lines',
                                  name='PINN', line=dict(color=C_PINN, dash='dash'), showlegend=False), row=2, col=1)

        theta_err = np.abs(np.degrees(theta_pinn - theta_ode))
        omega_err = np.abs(omega_pinn - omega_ode)
        fig.add_trace(go.Scatter(x=t_eval, y=theta_err, mode='lines',
                                  name='|Δθ|', line=dict(color=C_ERROR)), row=2, col=2)
        fig.add_trace(go.Scatter(x=t_eval, y=omega_err, mode='lines',
                                  name='|Δω|', line=dict(color=C_ACCENT)), row=2, col=2)

        fig.update_layout(height=700, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Training Loss"):
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=loss_history, mode='lines',
                                           line=dict(color='navy')))
            fig_loss.update_layout(yaxis_type="log", height=350,
                                    xaxis_title="Epoch", yaxis_title="Loss",
                                    template="plotly_white")
            st.plotly_chart(fig_loss, use_container_width=True)


# ===========================================================================
# Orbital mode
# ===========================================================================

elif mode == "Orbital Mechanics":
    st.title("Two-Body Orbital PINN")
    st.markdown("Train a neural network to solve Kepler's two-body problem.")

    col1, col2 = st.columns(2)
    with col1:
        eccentricity = st.slider("Eccentricity", 0.0, 0.85,
                                  float(ORBITAL_CFG.get("physics", {}).get("eccentricity", 0.3)), key="orb_e")
        n_orbits = st.slider("Number of orbits", 0.5, 2.0,
                              float(ORBITAL_CFG.get("physics", {}).get("n_orbits", 1.0)), key="orb_n")
    with col2:
        epochs = st.select_slider("Training epochs",
                                   options=[3000, 5000, 8000, 10000],
                                   value=int(ORBITAL_CFG.get("training", {}).get("epochs", 5000)), key="orb_epochs")
    adaptive = st.checkbox("Use Adaptive Sampling", value=False, key="orb_adaptive")

    GM = GM_DEFAULT
    x0, y0, vx0, vy0, period = setup_orbital_ics(eccentricity, GM)
    t_max = n_orbits * period

    if st.button("Train & Compare", key="orb_train"):
        torch.manual_seed(42)
        model, loss_history = train_orbital(
            x0, y0, vx0, vy0, t_max, GM,
            epochs=epochs, adaptive=adaptive)

        t_eval = np.linspace(0, t_max, 2000)
        t_ode, x_ode, y_ode, vx_ode, vy_ode = solve_orbit_ode(
            x0, y0, vx0, vy0, (0, t_max), t_eval, GM=GM)

        model.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
            output = model(t_tensor).numpy()
        x_p, y_p, vx_p, vy_p = output[:, 0], output[:, 1], output[:, 2], output[:, 3]

        E_ode = compute_energy_orbital(x_ode, y_ode, vx_ode, vy_ode, GM)
        E_pinn = compute_energy_orbital(x_p, y_p, vx_p, vy_p, GM)
        L_ode = compute_angular_momentum(x_ode, y_ode, vx_ode, vy_ode)
        L_pinn = compute_angular_momentum(x_p, y_p, vx_p, vy_p)

        E0 = E_ode[0]
        L0 = L_ode[0]
        pos_err = np.sqrt((x_p - x_ode)**2 + (y_p - y_ode)**2)

        m1, m2, m3 = st.columns(3)
        m1.metric("Final Loss", f"{loss_history[-1]:.6f}")
        m2.metric("Max |ΔE/E₀|", f"{np.max(np.abs((E_pinn-E0)/E0)):.4f}")
        m3.metric("Max |Δr|", f"{np.max(pos_err):.4f}")

        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=["Trajectory", "x(t), y(t)", "Energy",
                                            "|ΔE/E₀|", "Angular Momentum", "|Δr|"])

        fig.add_trace(go.Scatter(x=x_ode, y=y_ode, mode='lines',
                                  name='Classical', line=dict(color=C_CLASSICAL)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_p, y=y_p, mode='lines',
                                  name='PINN', line=dict(color=C_PINN, dash='dash')), row=1, col=1)

        fig.add_trace(go.Scatter(x=t_ode, y=x_ode, mode='lines',
                                  name='x Classical', line=dict(color=C_CLASSICAL), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=t_eval, y=x_p, mode='lines',
                                  name='x PINN', line=dict(color=C_PINN, dash='dash'), showlegend=False), row=1, col=2)

        fig.add_trace(go.Scatter(x=t_ode, y=E_ode, mode='lines',
                                  name='Classical', line=dict(color=C_CLASSICAL), showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=t_eval, y=E_pinn, mode='lines',
                                  name='PINN', line=dict(color=C_PINN), showlegend=False), row=1, col=3)

        dE_pinn = np.abs((E_pinn - E0) / E0)
        fig.add_trace(go.Scatter(x=t_eval, y=dE_pinn, mode='lines',
                                  line=dict(color=C_PINN), showlegend=False), row=2, col=1)

        fig.add_trace(go.Scatter(x=t_ode, y=L_ode, mode='lines',
                                  line=dict(color=C_CLASSICAL), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=t_eval, y=L_pinn, mode='lines',
                                  line=dict(color=C_PINN), showlegend=False), row=2, col=2)

        fig.add_trace(go.Scatter(x=t_eval, y=pos_err, mode='lines',
                                  line=dict(color=C_ERROR), showlegend=False), row=2, col=3)

        fig.update_layout(height=700, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Training Loss"):
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=loss_history, mode='lines',
                                           line=dict(color='darkred')))
            fig_loss.update_layout(yaxis_type="log", height=350,
                                    xaxis_title="Epoch", yaxis_title="Loss",
                                    template="plotly_white")
            st.plotly_chart(fig_loss, use_container_width=True)
