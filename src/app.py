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

from src.models import PINNPendulum, PINNOrbital, HamiltonianNN, HeatPINN, WavePINN, ThreeBodyPINN
from src.models import InversePendulumPINN, InverseOrbitalPINN
from src.models.hnn import generate_pendulum_data, train_hnn, integrate_hnn, compute_hamiltonian
from src.models.inverse_pinn import train_inverse_pendulum, train_inverse_orbital
from src.models.heat_pinn import (
    train_heat_pinn, heat_analytical, IC_FUNCTIONS,
)
from src.models.wave_pinn import (
    train_wave_pinn, wave_analytical, WAVE_IC_FUNCTIONS,
    wave_mode_decomposition, wave_energy, WavePINN as WavePINNClass,
)
from src.physics.equations import pendulum_residual, orbital_residual
from src.physics.constants import GRAVITY, PENDULUM_LENGTH, GM_DEFAULT
from src.utils.validation import solve_pendulum_ode, solve_orbit_ode, setup_orbital_ics
from src.utils.data_generation import generate_noisy_pendulum_data, generate_noisy_orbital_data
from src.utils.metrics import (
    compute_energy_pendulum, compute_energy_orbital,
    compute_angular_momentum, compute_hamiltonian_pendulum,
)
from src.training.losses import pendulum_ic_loss, orbital_ic_loss
from src.benchmarks.benchmark_runner import BenchmarkRunner
from src.models.threebody_pinn import (
    ThreeBodyPINN as ThreeBodyPINNClass, PRESETS as TB_PRESETS,
    train_threebody, solve_threebody,
    compute_energy as tb_energy, compute_angular_momentum as tb_angmom,
    physics_loss as tb_physics_loss, ic_loss as tb_ic_loss,
)

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


def train_hnn_pendulum(theta_0, omega_0, t_max, g, L, m=1.0,
                       epochs=5000, lr=1e-3, batch_size=256):
    """Train an HNN on pendulum data and return model + loss history."""
    progress = st.progress(0, text="Generating HNN training data...")
    q_data, p_data, dqdt_data, dpdt_data, _ = generate_pendulum_data(
        theta_0, omega_0, t_max, g, L, m, n_points=1000,
    )
    progress.progress(0.05, text="Training HNN...")

    model = HamiltonianNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6,
    )

    q_t = torch.tensor(q_data, dtype=torch.float32).unsqueeze(1)
    p_t = torch.tensor(p_data, dtype=torch.float32).unsqueeze(1)
    dqdt_t = torch.tensor(dqdt_data, dtype=torch.float32).unsqueeze(1)
    dpdt_t = torch.tensor(dpdt_data, dtype=torch.float32).unsqueeze(1)
    n_samples = len(q_data)
    losses = []

    for ep in range(epochs):
        optimizer.zero_grad()
        idx = torch.randint(0, n_samples, (batch_size,))
        q_batch = q_t[idx].requires_grad_(True)
        p_batch = p_t[idx].requires_grad_(True)
        dqdt_pred, dpdt_pred = model.time_derivative(None, q_batch, p_batch)
        loss = (torch.mean((dqdt_pred - dqdt_t[idx]) ** 2) +
                torch.mean((dpdt_pred - dpdt_t[idx]) ** 2))
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        losses.append(loss.item())

        if (ep + 1) % 100 == 0:
            progress.progress(
                (ep + 1) / epochs,
                text=f"HNN Epoch {ep+1}/{epochs}  |  Loss: {loss.item():.8f}")

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
    ["Pendulum", "Orbital Mechanics", "Inverse Problem",
     "Heat Equation", "Wave Equation", "Three-Body Problem", "Benchmarks"],
    index=0,
)

# ===========================================================================
# Pendulum mode
# ===========================================================================

C_HNN = "#00CC96"  # green for HNN traces

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
    hnn_mode = st.checkbox(
        "HNN Mode — also train a Hamiltonian Neural Network for comparison",
        value=False, key="pend_hnn",
        help="Trains an HNN alongside the PINN. The HNN conserves energy by construction (Greydanus et al., NeurIPS 2019).",
    )

    g = GRAVITY
    m_phys = 1.0
    theta_0 = np.radians(theta_deg)
    omega_0 = 0.0

    if st.button("Train & Compare", key="pend_train"):
        torch.manual_seed(42)

        # ---- Train PINN ----
        pinn_model, pinn_loss_history = train_pendulum(
            theta_0, omega_0, t_max, g, L,
            epochs=epochs, adaptive=adaptive)

        # ---- Optionally train HNN ----
        hnn_model = None
        hnn_loss_history = None
        if hnn_mode:
            torch.manual_seed(42)
            hnn_model, hnn_loss_history = train_hnn_pendulum(
                theta_0, omega_0, t_max, g, L, m_phys, epochs=epochs)

        # ---- Ground truth ----
        t_eval = np.linspace(0, t_max, 1000)
        t_ode, theta_ode, omega_ode = solve_pendulum_ode(
            theta_0, omega_0, (0, t_max), t_eval, g=g, L=L)

        # ---- PINN evaluation ----
        pinn_model.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
            output = pinn_model(t_tensor).numpy()
        theta_pinn = output[:, 0]
        omega_pinn = output[:, 1]

        # ---- HNN evaluation ----
        q_hnn = p_hnn = omega_hnn = None
        if hnn_model is not None:
            q0 = theta_0
            p0 = m_phys * L ** 2 * omega_0
            q_hnn, p_hnn = integrate_hnn(hnn_model, q0, p0, t_eval)
            omega_hnn = p_hnn / (m_phys * L ** 2)

        # ---- Metrics ----
        max_err = np.max(np.abs(np.degrees(theta_pinn - theta_ode)))
        E_pinn = compute_energy_pendulum(theta_pinn, omega_pinn, g, L, m_phys)
        E_drift_pinn = np.max(np.abs((E_pinn - E_pinn[0]) / (np.abs(E_pinn[0]) + 1e-16)))

        if hnn_mode and q_hnn is not None:
            E_hnn = compute_hamiltonian_pendulum(q_hnn, p_hnn, g, L, m_phys)
            E_drift_hnn = np.max(np.abs((E_hnn - E_hnn[0]) / (np.abs(E_hnn[0]) + 1e-16)))
            max_err_hnn = np.max(np.abs(np.degrees(q_hnn - theta_ode)))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PINN Final Loss", f"{pinn_loss_history[-1]:.6f}")
            m2.metric("PINN Energy Drift", f"{E_drift_pinn:.4f}")
            m3.metric("HNN Energy Drift", f"{E_drift_hnn:.6f}")
            m4.metric("HNN Advantage",
                       f"{E_drift_pinn / (E_drift_hnn + 1e-16):.0f}x")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Final Loss", f"{pinn_loss_history[-1]:.6f}")
            m2.metric("Max Angle Error", f"{max_err:.2f}")
            m3.metric("Energy Drift", f"{E_drift_pinn:.4f}")

        # ---- Plots ----
        if hnn_mode and q_hnn is not None:
            # 2x3 comparison layout
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[
                    "Angular Displacement", "Phase Portrait",
                    "Energy Conservation",
                    "Relative Energy Error", "Training Convergence",
                    "Learned H(q,p) Landscape",
                ],
            )

            # (1,1) Trajectory
            fig.add_trace(go.Scatter(x=t_ode, y=np.degrees(theta_ode), mode='lines',
                                      name='Classical', line=dict(color=C_CLASSICAL)), row=1, col=1)
            fig.add_trace(go.Scatter(x=t_eval, y=np.degrees(theta_pinn), mode='lines',
                                      name='PINN', line=dict(color=C_PINN, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=t_eval, y=np.degrees(q_hnn), mode='lines',
                                      name='HNN', line=dict(color=C_HNN, dash='dot')), row=1, col=1)

            # (1,2) Phase portrait
            fig.add_trace(go.Scatter(x=np.degrees(theta_ode), y=omega_ode, mode='lines',
                                      name='Classical', line=dict(color=C_CLASSICAL), showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=np.degrees(theta_pinn), y=omega_pinn, mode='lines',
                                      name='PINN', line=dict(color=C_PINN, dash='dash'), showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=np.degrees(q_hnn), y=omega_hnn, mode='lines',
                                      name='HNN', line=dict(color=C_HNN, dash='dot'), showlegend=False), row=1, col=2)

            # (1,3) Energy conservation
            E_true = compute_energy_pendulum(theta_ode, omega_ode, g, L, m_phys)
            fig.add_trace(go.Scatter(x=t_eval, y=E_true, mode='lines',
                                      name='Classical', line=dict(color=C_CLASSICAL), showlegend=False), row=1, col=3)
            fig.add_trace(go.Scatter(x=t_eval, y=E_pinn, mode='lines',
                                      name='PINN', line=dict(color=C_PINN), showlegend=False), row=1, col=3)
            fig.add_trace(go.Scatter(x=t_eval, y=E_hnn, mode='lines',
                                      name='HNN', line=dict(color=C_HNN), showlegend=False), row=1, col=3)

            # (2,1) Relative energy error
            E0 = E_true[0]
            dE_pinn = np.abs((E_pinn - E0) / (np.abs(E0) + 1e-16))
            dE_hnn_arr = np.abs((E_hnn - E0) / (np.abs(E0) + 1e-16))
            fig.add_trace(go.Scatter(x=t_eval, y=dE_pinn, mode='lines',
                                      name='PINN', line=dict(color=C_PINN), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=t_eval, y=dE_hnn_arr, mode='lines',
                                      name='HNN', line=dict(color=C_HNN), showlegend=False), row=2, col=1)
            fig.update_yaxes(type="log", row=2, col=1)

            # (2,2) Training convergence
            fig.add_trace(go.Scatter(y=pinn_loss_history, mode='lines',
                                      name='PINN', line=dict(color=C_PINN), showlegend=False), row=2, col=2)
            fig.add_trace(go.Scatter(y=hnn_loss_history, mode='lines',
                                      name='HNN', line=dict(color=C_HNN), showlegend=False), row=2, col=2)
            fig.update_yaxes(type="log", row=2, col=2)

            # (2,3) Learned Hamiltonian landscape
            q_grid = np.linspace(-np.pi, np.pi, 80)
            p_grid = np.linspace(-3, 3, 80)
            Q, P = np.meshgrid(q_grid, p_grid)
            Q_flat = torch.tensor(Q.flatten(), dtype=torch.float32).unsqueeze(1)
            P_flat = torch.tensor(P.flatten(), dtype=torch.float32).unsqueeze(1)
            hnn_model.eval()
            with torch.no_grad():
                H_pred = hnn_model(Q_flat, P_flat).numpy().reshape(Q.shape)
            fig.add_trace(go.Contour(
                x=np.degrees(q_grid), y=p_grid, z=H_pred,
                colorscale='Greens', showscale=False,
                contours=dict(showlabels=True),
                name='Learned H',
            ), row=2, col=3)

            fig.update_layout(height=800, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Standard 2x2 PINN-only layout
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
            fig_loss.add_trace(go.Scatter(y=pinn_loss_history, mode='lines',
                                           name='PINN', line=dict(color='navy')))
            if hnn_loss_history is not None:
                fig_loss.add_trace(go.Scatter(y=hnn_loss_history, mode='lines',
                                               name='HNN', line=dict(color=C_HNN)))
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


# ===========================================================================
# Inverse Problem mode
# ===========================================================================

elif mode == "Inverse Problem":
    st.title("Inverse PINN — Parameter Estimation from Noisy Data")
    st.markdown(
        "Given noisy observations of a physical system, simultaneously "
        "reconstruct the trajectory and infer unknown physical parameters."
    )

    inv_system = st.radio(
        "System", ["Pendulum (g, L)", "Orbital (GM)"],
        key="inv_system", horizontal=True,
    )

    if inv_system == "Pendulum (g, L)":
        st.subheader("Pendulum: Infer g and L")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**True parameters** (used to generate data)")
            g_true = st.slider("True g (m/s²)", 5.0, 15.0, 9.81, key="inv_g_true")
            L_true = st.slider("True L (m)", 0.5, 3.0, 1.0, key="inv_L_true")
            theta_deg = st.slider("Initial angle (deg)", 10, 80, 45, key="inv_theta")
            noise_pct = st.slider("Noise level (%)", 1, 20, 5, key="inv_noise")
        with col2:
            st.markdown("**Initial guesses** (wrong on purpose)")
            g_init = st.slider("g guess (m/s²)", 1.0, 20.0, 5.0, key="inv_g_init")
            L_init = st.slider("L guess (m)", 0.2, 5.0, 2.0, key="inv_L_init")
            n_obs = st.slider("Observations", 20, 200, 80, key="inv_nobs")
            epochs = st.select_slider("Training epochs",
                                       options=[5000, 8000, 10000, 15000],
                                       value=10000, key="inv_epochs")

        theta_0 = np.radians(theta_deg)
        omega_0 = 0.0
        t_max = 10.0
        noise_std = noise_pct / 100.0 * theta_0  # relative to initial angle

        if st.button("Generate Data & Train", key="inv_pend_train"):
            torch.manual_seed(42)
            np.random.seed(42)

            # Generate data
            progress = st.progress(0, text="Generating noisy observations...")
            data = generate_noisy_pendulum_data(
                theta_0, omega_0, t_max, g=g_true, L=L_true,
                n_obs=n_obs, noise_std=noise_std,
            )
            progress.progress(0.05, text="Warming up network...")

            # Train
            model = InversePendulumPINN(g_init=g_init, L_init=L_init)
            t_obs_t = torch.tensor(data['t_obs'], dtype=torch.float32).unsqueeze(1)
            theta_obs_t = torch.tensor(data['theta_obs'], dtype=torch.float32)

            # Warmup phase
            warmup_opt = torch.optim.Adam(model.network.parameters(), lr=1e-3)
            for ep in range(1500):
                warmup_opt.zero_grad()
                out = model(t_obs_t)
                loss = 10.0 * torch.mean((out[:, 0] - theta_obs_t) ** 2)
                t0 = torch.zeros(1, 1)
                out0 = model(t0)
                loss = loss + 20.0 * ((out0[0, 0] - theta_0) ** 2 +
                                      (out0[0, 1] - omega_0) ** 2)
                loss.backward()
                warmup_opt.step()
            progress.progress(0.15, text="Training with physics constraints...")

            # Main training
            optimizer = torch.optim.Adam([
                {'params': model.network.parameters(), 'lr': 1e-3},
                {'params': [model.g_param, model.L_param], 'lr': 1e-2},
            ])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=500, factor=0.5, min_lr=1e-6,
            )

            loss_history = []
            g_history = []
            L_history = []

            for epoch in range(epochs):
                optimizer.zero_grad()
                t_col = torch.rand(500, 1) * t_max
                loss_phys = model.physics_loss(t_col)
                out_obs = model(t_obs_t)
                loss_data = torch.mean((out_obs[:, 0] - theta_obs_t) ** 2)
                t0 = torch.zeros(1, 1)
                out0 = model(t0)
                loss_ic = (out0[0, 0] - theta_0) ** 2 + (out0[0, 1] - omega_0) ** 2
                total = loss_phys + 10.0 * loss_data + 20.0 * loss_ic
                total.backward()
                optimizer.step()
                scheduler.step(total.item())
                model.clamp_parameters()
                loss_history.append(total.item())
                g_history.append(model.g)
                L_history.append(model.L)

                if (epoch + 1) % 200 == 0:
                    progress.progress(
                        0.15 + 0.85 * (epoch + 1) / epochs,
                        text=f"Epoch {epoch+1}/{epochs} | "
                             f"g={model.g:.3f} L={model.L:.3f} "
                             f"g/L={model.g_over_L:.3f} | "
                             f"Loss: {total.item():.6f}")

            progress.empty()

            # ---- Metrics ----
            g_err = abs(model.g - g_true) / g_true * 100
            L_err = abs(model.L - L_true) / L_true * 100
            gL_err = abs(model.g_over_L - g_true / L_true) / (g_true / L_true) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Inferred g", f"{model.g:.3f}", f"{g_err:.1f}% error")
            m2.metric("Inferred L", f"{model.L:.3f}", f"{L_err:.1f}% error")
            m3.metric("g/L ratio", f"{model.g_over_L:.3f}", f"{gL_err:.1f}% error")
            m4.metric("Final Loss", f"{loss_history[-1]:.6f}")

            # ---- PINN reconstruction ----
            model.eval()
            t_plot = np.linspace(0, t_max, 1000)
            with torch.no_grad():
                out = model(torch.tensor(t_plot, dtype=torch.float32).unsqueeze(1)).numpy()
            theta_pinn = out[:, 0]

            # ---- Plots: 2x3 ----
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[
                    "Trajectory Reconstruction", "g Convergence", "L Convergence",
                    "g/L Ratio Convergence", "Training Loss", "True vs Inferred",
                ],
            )

            # (1,1) Trajectory
            fig.add_trace(go.Scatter(
                x=data['t_dense'], y=np.degrees(data['theta_true']),
                mode='lines', name='Ground truth',
                line=dict(color=C_CLASSICAL),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=data['t_obs'], y=np.degrees(data['theta_obs']),
                mode='markers', name='Noisy obs',
                marker=dict(color='gray', size=4, opacity=0.5),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=t_plot, y=np.degrees(theta_pinn),
                mode='lines', name='PINN reconstruction',
                line=dict(color=C_PINN, dash='dash'),
            ), row=1, col=1)

            # (1,2) g convergence
            fig.add_trace(go.Scatter(
                y=g_history, mode='lines', name='Inferred g',
                line=dict(color=C_PINN), showlegend=False,
            ), row=1, col=2)
            fig.add_hline(y=g_true, line_dash="dash", line_color=C_CLASSICAL,
                          row=1, col=2)

            # (1,3) L convergence
            fig.add_trace(go.Scatter(
                y=L_history, mode='lines', name='Inferred L',
                line=dict(color=C_PINN), showlegend=False,
            ), row=1, col=3)
            fig.add_hline(y=L_true, line_dash="dash", line_color=C_CLASSICAL,
                          row=1, col=3)

            # (2,1) g/L ratio
            g_over_L_hist = [g / l for g, l in zip(g_history, L_history)]
            fig.add_trace(go.Scatter(
                y=g_over_L_hist, mode='lines', name='Inferred g/L',
                line=dict(color=C_PINN), showlegend=False,
            ), row=2, col=1)
            fig.add_hline(y=g_true / L_true, line_dash="dash",
                          line_color=C_CLASSICAL, row=2, col=1)

            # (2,2) Training loss
            fig.add_trace(go.Scatter(
                y=loss_history, mode='lines', name='Loss',
                line=dict(color='navy'), showlegend=False,
            ), row=2, col=2)
            fig.update_yaxes(type="log", row=2, col=2)

            # (2,3) Parameter comparison bar chart
            fig.add_trace(go.Bar(
                x=['g', 'L', 'g/L'],
                y=[g_true, L_true, g_true / L_true],
                name='True', marker_color=C_CLASSICAL,
            ), row=2, col=3)
            fig.add_trace(go.Bar(
                x=['g', 'L', 'g/L'],
                y=[model.g, model.L, model.g_over_L],
                name='Inferred', marker_color=C_PINN,
            ), row=2, col=3)

            fig.update_layout(height=750, template="plotly_white",
                              barmode='group')
            st.plotly_chart(fig, use_container_width=True)

    else:  # Orbital (GM)
        st.subheader("Orbital: Infer GM")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**True parameters**")
            GM_true = st.slider("True GM", 0.5, 3.0, 1.0, key="inv_GM_true")
            ecc = st.slider("Eccentricity", 0.0, 0.7, 0.3, key="inv_ecc")
            noise_pct_o = st.slider("Noise level (%)", 1, 15, 5, key="inv_noise_o")
        with col2:
            st.markdown("**Initial guess**")
            GM_init = st.slider("GM guess", 0.1, 5.0, 0.5, key="inv_GM_init")
            n_obs_o = st.slider("Observations", 30, 200, 80, key="inv_nobs_o")
            epochs_o = st.select_slider("Training epochs",
                                         options=[5000, 8000, 10000, 15000],
                                         value=10000, key="inv_epochs_o")

        if st.button("Generate Data & Train", key="inv_orb_train"):
            torch.manual_seed(42)
            np.random.seed(42)

            progress = st.progress(0, text="Generating noisy orbital observations...")
            noise_std_o = noise_pct_o / 100.0
            data = generate_noisy_orbital_data(
                eccentricity=ecc, GM=GM_true,
                n_obs=n_obs_o, noise_std=noise_std_o,
            )
            progress.progress(0.05, text="Warming up network...")

            model = InverseOrbitalPINN(GM_init=GM_init)
            ics = data['ics']
            t_max_o = data['t_max']

            t_obs_t = torch.tensor(data['t_obs'], dtype=torch.float32).unsqueeze(1)
            x_obs_t = torch.tensor(data['x_obs'], dtype=torch.float32)
            y_obs_t = torch.tensor(data['y_obs'], dtype=torch.float32)

            x0_t = torch.tensor(ics[0], dtype=torch.float32)
            y0_t = torch.tensor(ics[1], dtype=torch.float32)
            vx0_t = torch.tensor(ics[2], dtype=torch.float32)
            vy0_t = torch.tensor(ics[3], dtype=torch.float32)

            # Warmup
            warmup_opt = torch.optim.Adam(model.network.parameters(), lr=1e-3)
            for ep in range(2000):
                warmup_opt.zero_grad()
                out = model(t_obs_t)
                loss = 10.0 * (torch.mean((out[:, 0] - x_obs_t) ** 2) +
                               torch.mean((out[:, 1] - y_obs_t) ** 2))
                t0 = torch.zeros(1, 1)
                out0 = model(t0)
                loss = loss + 50.0 * ((out0[0, 0] - x0_t) ** 2 +
                                      (out0[0, 1] - y0_t) ** 2 +
                                      (out0[0, 2] - vx0_t) ** 2 +
                                      (out0[0, 3] - vy0_t) ** 2)
                loss.backward()
                warmup_opt.step()
            progress.progress(0.15, text="Training with physics constraints...")

            # Main training
            optimizer = torch.optim.Adam([
                {'params': model.network.parameters(), 'lr': 1e-3},
                {'params': [model.GM_param], 'lr': 5e-3},
            ])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=500, factor=0.5, min_lr=1e-6,
            )

            loss_history = []
            GM_history = []

            for epoch in range(epochs_o):
                optimizer.zero_grad()
                t_col = torch.rand(600, 1) * t_max_o
                loss_phys = model.physics_loss(t_col)
                out_obs = model(t_obs_t)
                loss_data = (torch.mean((out_obs[:, 0] - x_obs_t) ** 2) +
                             torch.mean((out_obs[:, 1] - y_obs_t) ** 2))
                t0 = torch.zeros(1, 1)
                out0 = model(t0)
                loss_ic = ((out0[0, 0] - x0_t) ** 2 + (out0[0, 1] - y0_t) ** 2 +
                           (out0[0, 2] - vx0_t) ** 2 + (out0[0, 3] - vy0_t) ** 2)
                total = loss_phys + 10.0 * loss_data + 50.0 * loss_ic
                total.backward()
                optimizer.step()
                scheduler.step(total.item())
                model.clamp_parameters()
                loss_history.append(total.item())
                GM_history.append(model.GM)

                if (epoch + 1) % 200 == 0:
                    progress.progress(
                        0.15 + 0.85 * (epoch + 1) / epochs_o,
                        text=f"Epoch {epoch+1}/{epochs_o} | "
                             f"GM={model.GM:.4f} | "
                             f"Loss: {total.item():.6f}")

            progress.empty()

            GM_err = abs(model.GM - GM_true) / GM_true * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Inferred GM", f"{model.GM:.4f}",
                       f"{GM_err:.1f}% error")
            m2.metric("True GM", f"{GM_true:.4f}")
            m3.metric("Final Loss", f"{loss_history[-1]:.6f}")

            # PINN reconstruction
            model.eval()
            t_plot = np.linspace(0, t_max_o, 1500)
            with torch.no_grad():
                out = model(torch.tensor(t_plot, dtype=torch.float32).unsqueeze(1)).numpy()
            x_pinn, y_pinn = out[:, 0], out[:, 1]

            # Plots: 1x3
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=[
                    "Orbit Reconstruction", "GM Convergence", "Training Loss",
                ],
            )

            fig.add_trace(go.Scatter(
                x=data['x_true'], y=data['y_true'],
                mode='lines', name='Ground truth',
                line=dict(color=C_CLASSICAL),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=data['x_obs'], y=data['y_obs'],
                mode='markers', name='Noisy obs',
                marker=dict(color='gray', size=4, opacity=0.5),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=x_pinn, y=y_pinn,
                mode='lines', name='PINN reconstruction',
                line=dict(color=C_PINN, dash='dash'),
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                y=GM_history, mode='lines', name='Inferred GM',
                line=dict(color=C_PINN), showlegend=False,
            ), row=1, col=2)
            fig.add_hline(y=GM_true, line_dash="dash",
                          line_color=C_CLASSICAL, row=1, col=2)

            fig.add_trace(go.Scatter(
                y=loss_history, mode='lines',
                line=dict(color='navy'), showlegend=False,
            ), row=1, col=3)
            fig.update_yaxes(type="log", row=1, col=3)

            fig.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# Heat Equation mode
# ===========================================================================

elif mode == "Heat Equation":
    st.title("1D Heat Equation PINN")
    st.markdown(
        r"Solve $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$ "
        "on a rod with specified boundary and initial conditions."
    )

    HEAT_CFG = load_config("heat_default.yaml")
    heat_phys = HEAT_CFG.get("physics", {})
    heat_train = HEAT_CFG.get("training", {})

    col1, col2 = st.columns(2)
    with col1:
        alpha_val = st.slider(
            "Thermal diffusivity (alpha)", 0.001, 0.1,
            float(heat_phys.get("alpha", 0.01)),
            format="%.3f", key="heat_alpha")
        L_rod = st.slider("Rod length L", 0.5, 2.0,
                           float(heat_phys.get("L_rod", 1.0)), key="heat_L")
        ic_type = st.selectbox(
            "Initial condition",
            ["sine", "step", "gaussian"], key="heat_ic")
    with col2:
        T_left = st.slider("Left BC: u(0, t)", -1.0, 1.0,
                             float(heat_phys.get("T_left", 0.0)), key="heat_Tl")
        T_right = st.slider("Right BC: u(L, t)", -1.0, 1.0,
                              float(heat_phys.get("T_right", 0.0)), key="heat_Tr")
        epochs_h = st.select_slider(
            "Training epochs",
            options=[3000, 5000, 8000, 10000],
            value=int(heat_train.get("epochs", 8000)), key="heat_epochs")

    t_max_h = 1.0

    if st.button("Train & Solve", key="heat_train"):
        torch.manual_seed(42)

        # ---- Train ----
        progress_h = st.progress(0, text="Training Heat Equation PINN...")

        model_h = HeatPINN()
        opt_h = torch.optim.Adam(model_h.parameters(), lr=1e-3)
        sched_h = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_h, patience=500, factor=0.5, min_lr=1e-6)
        ic_fn_h = lambda x: IC_FUNCTIONS[ic_type](x, L_rod=L_rod)
        loss_hist_h = []

        for ep in range(epochs_h):
            opt_h.zero_grad()
            x_int = torch.rand(2000, 1) * L_rod
            t_int = torch.rand(2000, 1) * t_max_h
            l_pde = model_h.physics_loss(x_int, t_int, alpha=alpha_val)
            l_bc = model_h.boundary_loss(200, t_max_h, T_left, T_right, L_rod)
            l_ic = model_h.ic_loss(200, ic_fn_h, L_rod)
            total = l_pde + 10.0 * l_bc + 10.0 * l_ic
            total.backward()
            opt_h.step()
            sched_h.step(total.item())
            loss_hist_h.append(total.item())
            if (ep + 1) % 200 == 0:
                progress_h.progress(
                    (ep + 1) / epochs_h,
                    text=f"Epoch {ep+1}/{epochs_h} | Loss: {total.item():.6f}")

        progress_h.empty()

        # ---- Evaluate on grid ----
        nx, nt = 100, 100
        x_arr = np.linspace(0, L_rod, nx)
        t_arr = np.linspace(0, t_max_h, nt)
        X, T_grid = np.meshgrid(x_arr, t_arr)

        model_h.eval()
        with torch.no_grad():
            x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
            t_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32).unsqueeze(1)
            u_pinn = model_h(x_flat, t_flat).numpy().reshape(X.shape)

        # Analytical (only exact for sine IC with homogeneous BCs)
        has_analytical = (ic_type in ('sine', 'step', 'gaussian')
                          and T_left == 0.0 and T_right == 0.0)
        if has_analytical:
            u_exact = heat_analytical(X, T_grid, alpha=alpha_val,
                                       L_rod=L_rod, ic_type=ic_type)
            u_error = np.abs(u_pinn - u_exact)
        else:
            u_exact = None
            u_error = None

        # ---- Metrics ----
        m1, m2, m3 = st.columns(3)
        m1.metric("Final Loss", f"{loss_hist_h[-1]:.6f}")
        if u_error is not None:
            m2.metric("Max |Error|", f"{np.max(u_error):.2e}")
            m3.metric("Mean |Error|", f"{np.mean(u_error):.2e}")

        # ---- Heatmap: u(x,t) ----
        st.subheader("Temperature Field u(x, t)")
        fig_heat = go.Figure(data=go.Heatmap(
            x=x_arr, y=t_arr, z=u_pinn,
            colorscale='Hot', colorbar=dict(title='u'),
        ))
        fig_heat.update_layout(
            xaxis_title='x', yaxis_title='t',
            height=450, template="plotly_white",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ---- Comparison at time snapshots ----
        if has_analytical:
            st.subheader("PINN vs Analytical at Time Snapshots")
            t_snaps = [0.0, 0.1, 0.5, 1.0]
            fig_snap = make_subplots(
                rows=1, cols=len(t_snaps),
                subplot_titles=[f"t = {ts}" for ts in t_snaps],
            )
            for i, ts in enumerate(t_snaps):
                t_idx = int(ts / t_max_h * (nt - 1))
                t_idx = min(t_idx, nt - 1)
                fig_snap.add_trace(go.Scatter(
                    x=x_arr, y=u_exact[t_idx, :],
                    mode='lines', name='Analytical' if i == 0 else None,
                    line=dict(color=C_CLASSICAL),
                    showlegend=(i == 0),
                ), row=1, col=i + 1)
                fig_snap.add_trace(go.Scatter(
                    x=x_arr, y=u_pinn[t_idx, :],
                    mode='lines', name='PINN' if i == 0 else None,
                    line=dict(color=C_PINN, dash='dash'),
                    showlegend=(i == 0),
                ), row=1, col=i + 1)
            fig_snap.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig_snap, use_container_width=True)

            # ---- Error heatmap ----
            st.subheader("Error: |u_PINN - u_analytical|")
            fig_err = go.Figure(data=go.Heatmap(
                x=x_arr, y=t_arr, z=u_error,
                colorscale='Viridis', colorbar=dict(title='|error|'),
            ))
            fig_err.update_layout(
                xaxis_title='x', yaxis_title='t',
                height=400, template="plotly_white",
            )
            st.plotly_chart(fig_err, use_container_width=True)

        # ---- Training loss ----
        with st.expander("Training Loss"):
            fig_loss_h = go.Figure()
            fig_loss_h.add_trace(go.Scatter(
                y=loss_hist_h, mode='lines', line=dict(color='navy')))
            fig_loss_h.update_layout(
                yaxis_type="log", height=350,
                xaxis_title="Epoch", yaxis_title="Loss",
                template="plotly_white")
            st.plotly_chart(fig_loss_h, use_container_width=True)


# ===========================================================================
# Wave Equation mode
# ===========================================================================

elif mode == "Wave Equation":
    st.title("1D Wave Equation PINN")
    st.markdown(
        r"Solve $\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$ "
        "on a vibrating string with fixed ends."
    )

    WAVE_CFG = load_config("wave_default.yaml")
    wave_phys = WAVE_CFG.get("physics", {})
    wave_train = WAVE_CFG.get("training", {})

    col1, col2 = st.columns(2)
    with col1:
        c_val = st.slider("Wave speed c", 0.5, 3.0,
                           float(wave_phys.get("c", 1.0)), key="wave_c")
        L_string = st.slider("String length L", 0.5, 2.0,
                              float(wave_phys.get("L_string", 1.0)), key="wave_L")
        ic_type_w = st.selectbox(
            "Initial shape",
            ["sine", "plucked", "gaussian"], key="wave_ic")
    with col2:
        t_max_w = st.slider("Time span", 0.5, 2.0,
                              float(wave_train.get("t_max", 1.0)), key="wave_tmax")
        epochs_w = st.select_slider(
            "Training epochs",
            options=[3000, 5000, 8000, 10000],
            value=int(wave_train.get("epochs", 8000)), key="wave_epochs")

    if st.button("Train & Solve", key="wave_train"):
        torch.manual_seed(42)

        # ---- Train ----
        progress_w = st.progress(0, text="Training Wave Equation PINN...")
        from src.models.wave_pinn import zero_velocity

        model_w = WavePINNClass()
        opt_w = torch.optim.Adam(model_w.parameters(), lr=1e-3)
        sched_w = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_w, patience=500, factor=0.5, min_lr=1e-6)
        ic_fn_w = lambda x: WAVE_IC_FUNCTIONS[ic_type_w](x, L_string=L_string)
        loss_hist_w = []

        for ep in range(epochs_w):
            opt_w.zero_grad()
            x_int = torch.rand(2000, 1) * L_string
            t_int = torch.rand(2000, 1) * t_max_w
            l_pde = model_w.physics_loss(x_int, t_int, c=c_val)
            l_bc = model_w.boundary_loss(200, t_max_w, L_string)
            l_ic_u = model_w.ic_displacement_loss(200, ic_fn_w, L_string)
            l_ic_v = model_w.ic_velocity_loss(200, zero_velocity, L_string)
            total = l_pde + 10.0 * l_bc + 10.0 * (l_ic_u + l_ic_v)
            total.backward()
            opt_w.step()
            sched_w.step(total.item())
            loss_hist_w.append(total.item())
            if (ep + 1) % 200 == 0:
                progress_w.progress(
                    (ep + 1) / epochs_w,
                    text=f"Epoch {ep+1}/{epochs_w} | Loss: {total.item():.6f}")

        progress_w.empty()

        # ---- Evaluate ----
        nx, nt = 100, 100
        x_arr = np.linspace(0, L_string, nx)
        t_arr = np.linspace(0, t_max_w, nt)
        X, T_grid = np.meshgrid(x_arr, t_arr)

        model_w.eval()
        with torch.no_grad():
            x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
            t_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32).unsqueeze(1)
            u_pinn_w = model_w(x_flat, t_flat).numpy().reshape(X.shape)

        has_analytical_w = (ic_type_w in ('sine', 'plucked', 'gaussian'))
        if has_analytical_w:
            u_exact_w = wave_analytical(X, T_grid, c=c_val,
                                         L_string=L_string, ic_type=ic_type_w)
            u_error_w = np.abs(u_pinn_w - u_exact_w)
        else:
            u_exact_w = None
            u_error_w = None

        # ---- Metrics ----
        m1, m2, m3 = st.columns(3)
        m1.metric("Final Loss", f"{loss_hist_w[-1]:.6f}")
        if u_error_w is not None:
            m2.metric("Max |Error|", f"{np.max(u_error_w):.2e}")
            m3.metric("Mean |Error|", f"{np.mean(u_error_w):.2e}")

        # ---- Wave animation (heatmap) ----
        st.subheader("Displacement Field u(x, t)")
        vmax_w = np.max(np.abs(u_pinn_w))
        fig_wave = go.Figure(data=go.Heatmap(
            x=x_arr, y=t_arr, z=u_pinn_w,
            colorscale='RdBu_r', zmid=0,
            zmin=-vmax_w, zmax=vmax_w,
            colorbar=dict(title='u'),
        ))
        fig_wave.update_layout(
            xaxis_title='x', yaxis_title='t',
            height=450, template="plotly_white",
        )
        st.plotly_chart(fig_wave, use_container_width=True)

        # ---- Comparison at snapshots ----
        if has_analytical_w:
            st.subheader("PINN vs Analytical at Time Snapshots")
            t_snaps_w = [0.0, t_max_w * 0.25, t_max_w * 0.5, t_max_w]
            fig_snap_w = make_subplots(
                rows=1, cols=len(t_snaps_w),
                subplot_titles=[f"t = {ts:.2f}" for ts in t_snaps_w],
            )
            for i, ts in enumerate(t_snaps_w):
                t_idx = int(ts / t_max_w * (nt - 1))
                t_idx = min(t_idx, nt - 1)
                fig_snap_w.add_trace(go.Scatter(
                    x=x_arr, y=u_exact_w[t_idx, :],
                    mode='lines', name='Analytical' if i == 0 else None,
                    line=dict(color=C_CLASSICAL), showlegend=(i == 0),
                ), row=1, col=i + 1)
                fig_snap_w.add_trace(go.Scatter(
                    x=x_arr, y=u_pinn_w[t_idx, :],
                    mode='lines', name='PINN' if i == 0 else None,
                    line=dict(color=C_PINN, dash='dash'), showlegend=(i == 0),
                ), row=1, col=i + 1)
            fig_snap_w.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig_snap_w, use_container_width=True)

        # ---- Mode decomposition ----
        st.subheader("Fourier Mode Decomposition")
        t_mode = t_max_w * 0.1
        modes = wave_mode_decomposition(
            x_arr, np.full_like(x_arr, t_mode),
            c=c_val, L_string=L_string, ic_type=ic_type_w, n_modes=5,
        )
        fig_modes = go.Figure()
        mode_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
        u_superposition = np.zeros_like(x_arr)
        for (n, b_n, u_mode), color in zip(modes, mode_colors):
            u_superposition += u_mode
            fig_modes.add_trace(go.Scatter(
                x=x_arr, y=u_mode, mode='lines',
                name=f'Mode {n} (b={b_n:.3f})',
                line=dict(color=color, dash='dot', width=1),
            ))
        fig_modes.add_trace(go.Scatter(
            x=x_arr, y=u_superposition, mode='lines',
            name='Superposition', line=dict(color='black', width=2),
        ))
        fig_modes.update_layout(
            title=f'First 5 modes at t = {t_mode:.2f}',
            xaxis_title='x', yaxis_title='u',
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_modes, use_container_width=True)

        # ---- Energy conservation ----
        st.subheader("Energy Conservation")
        energies = []
        for ti in range(nt):
            t_val = t_arr[ti]
            x_e = torch.tensor(x_arr, dtype=torch.float32).unsqueeze(1)
            t_e = torch.full((nx, 1), t_val)
            t_e.requires_grad_(True)
            x_e_g = x_e.clone().requires_grad_(True)
            u_e = model_w(x_e_g, t_e)
            du_dt_e = torch.autograd.grad(
                u_e, t_e, torch.ones_like(u_e),
                create_graph=False, retain_graph=True)[0]
            du_dx_e = torch.autograd.grad(
                u_e, x_e_g, torch.ones_like(u_e), create_graph=False)[0]
            du_dt_np = du_dt_e.detach().numpy().squeeze()
            du_dx_np = du_dx_e.detach().numpy().squeeze()
            E = wave_energy(x_arr, None, du_dt_np, du_dx_np, c=c_val)
            energies.append(E)

        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(
            x=t_arr, y=energies, mode='lines',
            line=dict(color=C_CLASSICAL),
        ))
        E0 = energies[0]
        fig_energy.add_hline(y=E0, line_dash="dash", line_color="gray",
                              annotation_text=f"E(0) = {E0:.4f}")
        max_drift = max(abs(e - E0) for e in energies) / (abs(E0) + 1e-16)
        fig_energy.update_layout(
            title=f'Total Energy (max drift: {max_drift:.2%})',
            xaxis_title='t', yaxis_title='Energy',
            height=350, template="plotly_white",
        )
        st.plotly_chart(fig_energy, use_container_width=True)

        # ---- Error heatmap ----
        if u_error_w is not None:
            with st.expander("Error Heatmap"):
                fig_err_w = go.Figure(data=go.Heatmap(
                    x=x_arr, y=t_arr, z=u_error_w,
                    colorscale='Viridis', colorbar=dict(title='|error|'),
                ))
                fig_err_w.update_layout(
                    xaxis_title='x', yaxis_title='t',
                    height=400, template="plotly_white",
                )
                st.plotly_chart(fig_err_w, use_container_width=True)

        # ---- Training loss ----
        with st.expander("Training Loss"):
            fig_loss_w = go.Figure()
            fig_loss_w.add_trace(go.Scatter(
                y=loss_hist_w, mode='lines', line=dict(color='navy')))
            fig_loss_w.update_layout(
                yaxis_type="log", height=350,
                xaxis_title="Epoch", yaxis_title="Loss",
                template="plotly_white")
            st.plotly_chart(fig_loss_w, use_container_width=True)


# ===========================================================================
# Three-Body Problem mode
# ===========================================================================

elif mode == "Three-Body Problem":
    st.title("Gravitational Three-Body Problem")
    st.markdown(
        "No closed-form solution exists. The PINN learns an approximate "
        "trajectory that satisfies Newton's law of gravitation for all "
        "three pairwise interactions."
    )

    TB_CFG = load_config("threebody_default.yaml")

    col1, col2 = st.columns(2)
    with col1:
        preset_name = st.selectbox(
            "Configuration preset",
            list(TB_PRESETS.keys()),
            format_func=lambda s: s.replace('_', ' ').title(),
            key="tb_preset",
        )
        t_max_tb = st.slider("Time horizon", 0.2, 3.0, 1.0, key="tb_tmax")
    with col2:
        epochs_tb = st.select_slider(
            "Training epochs",
            options=[3000, 5000, 8000, 10000, 15000],
            value=10000, key="tb_epochs")
        net_size = st.select_slider(
            "Network size",
            options=[64, 128, 256],
            value=128, key="tb_net")

    state0_tb, masses_tb, period_tb = TB_PRESETS[preset_name]()
    st.caption(f"Masses: {masses_tb}  |  Approximate period: {period_tb:.3f}")

    if st.button("Train & Simulate", key="tb_train"):
        torch.manual_seed(42)

        # ---- Classical ground truth ----
        progress_tb = st.progress(0, text="Computing classical solution (DOP853)...")
        t_gt, states_gt = solve_threebody(state0_tb, masses_tb, t_max_tb, n_points=2000)

        # ---- Train PINN ----
        progress_tb.progress(0.05, text="Training Three-Body PINN...")

        model_tb = ThreeBodyPINNClass(hidden_size=net_size, num_hidden_layers=5)
        opt_tb = torch.optim.Adam(model_tb.parameters(), lr=1e-3)
        sched_tb = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_tb, patience=500, factor=0.5, min_lr=1e-6)
        loss_hist_tb = []

        for ep in range(epochs_tb):
            opt_tb.zero_grad()
            t_col = torch.rand(1000, 1) * t_max_tb
            l_phys = tb_physics_loss(model_tb, t_col, masses_tb)
            l_ic = tb_ic_loss(model_tb, state0_tb)
            total = l_phys + 100.0 * l_ic
            total.backward()
            opt_tb.step()
            sched_tb.step(total.item())
            loss_hist_tb.append(total.item())
            if (ep + 1) % 200 == 0:
                progress_tb.progress(
                    0.05 + 0.90 * (ep + 1) / epochs_tb,
                    text=f"Epoch {ep+1}/{epochs_tb} | Loss: {total.item():.6f}")

        progress_tb.empty()

        # ---- Evaluate PINN ----
        model_tb.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t_gt, dtype=torch.float32).unsqueeze(1)
            states_pinn = model_tb(t_tensor).numpy()

        # ---- Conservation ----
        E_gt = tb_energy(states_gt, masses_tb)
        E_pinn = tb_energy(states_pinn, masses_tb)
        L_gt = tb_angmom(states_gt, masses_tb)
        L_pinn = tb_angmom(states_pinn, masses_tb)
        E0 = E_gt[0]
        L0 = L_gt[0]

        # ---- Metrics ----
        dE_pinn = np.max(np.abs((E_pinn - E0) / (np.abs(E0) + 1e-16)))
        total_pos_err = np.zeros(len(t_gt))
        for i in range(3):
            total_pos_err += np.sqrt(
                (states_pinn[:, 2*i] - states_gt[:, 2*i]) ** 2 +
                (states_pinn[:, 2*i+1] - states_gt[:, 2*i+1]) ** 2)

        m1_c, m2_c, m3_c = st.columns(3)
        m1_c.metric("Final Loss", f"{loss_hist_tb[-1]:.6f}")
        m2_c.metric("PINN max |dE/E0|", f"{dE_pinn:.4f}")
        m3_c.metric("Max pos error", f"{np.max(total_pos_err):.4f}")

        # ---- Colours for bodies ----
        body_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        body_labels = ['Body 1', 'Body 2', 'Body 3']

        # ---- Trajectory plot ----
        st.subheader("Orbital Trajectories")
        fig_traj = go.Figure()
        for i, (clr, lab) in enumerate(zip(body_colors, body_labels)):
            fig_traj.add_trace(go.Scatter(
                x=states_gt[:, 2*i], y=states_gt[:, 2*i+1],
                mode='lines', name=f'{lab} (DOP853)',
                line=dict(color=clr, width=2), opacity=0.4,
            ))
            fig_traj.add_trace(go.Scatter(
                x=states_pinn[:, 2*i], y=states_pinn[:, 2*i+1],
                mode='lines', name=f'{lab} (PINN)',
                line=dict(color=clr, width=2, dash='dash'),
            ))
            fig_traj.add_trace(go.Scatter(
                x=[state0_tb[2*i]], y=[state0_tb[2*i+1]],
                mode='markers', name=f'{lab} start',
                marker=dict(color=clr, size=10, symbol='circle'),
                showlegend=False,
            ))
        fig_traj.update_layout(
            xaxis_title='x', yaxis_title='y',
            height=550, template="plotly_white",
            yaxis=dict(scaleanchor="x"),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

        # ---- Energy + Angular momentum ----
        st.subheader("Conservation Laws")
        fig_cons = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Energy Conservation", "Angular Momentum"],
        )
        fig_cons.add_trace(go.Scatter(
            x=t_gt, y=E_gt, mode='lines', name='DOP853',
            line=dict(color=C_CLASSICAL),
        ), row=1, col=1)
        fig_cons.add_trace(go.Scatter(
            x=t_gt, y=E_pinn, mode='lines', name='PINN',
            line=dict(color=C_PINN, dash='dash'),
        ), row=1, col=1)
        fig_cons.add_trace(go.Scatter(
            x=t_gt, y=L_gt, mode='lines', name='DOP853',
            line=dict(color=C_CLASSICAL), showlegend=False,
        ), row=1, col=2)
        fig_cons.add_trace(go.Scatter(
            x=t_gt, y=L_pinn, mode='lines', name='PINN',
            line=dict(color=C_PINN, dash='dash'), showlegend=False,
        ), row=1, col=2)
        fig_cons.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_cons, use_container_width=True)

        # ---- Per-body position error ----
        st.subheader("PINN Position Error vs Classical Solver")
        fig_err_tb = go.Figure()
        for i, (clr, lab) in enumerate(zip(body_colors, body_labels)):
            pos_err_i = np.sqrt(
                (states_pinn[:, 2*i] - states_gt[:, 2*i]) ** 2 +
                (states_pinn[:, 2*i+1] - states_gt[:, 2*i+1]) ** 2)
            fig_err_tb.add_trace(go.Scatter(
                x=t_gt, y=pos_err_i, mode='lines',
                name=lab, line=dict(color=clr),
            ))
        fig_err_tb.update_layout(
            xaxis_title='Time', yaxis_title='|dr|', yaxis_type='log',
            height=350, template="plotly_white",
        )
        st.plotly_chart(fig_err_tb, use_container_width=True)

        with st.expander("Training Loss"):
            fig_loss_tb = go.Figure()
            fig_loss_tb.add_trace(go.Scatter(
                y=loss_hist_tb, mode='lines', line=dict(color='navy')))
            fig_loss_tb.update_layout(
                yaxis_type="log", height=350,
                xaxis_title="Epoch", yaxis_title="Loss",
                template="plotly_white")
            st.plotly_chart(fig_loss_tb, use_container_width=True)


# ===========================================================================
# Benchmarks mode
# ===========================================================================

elif mode == "Benchmarks":
    st.title("PINN Benchmarks Dashboard")
    st.markdown(
        "Run systematic benchmarks across all problems and methods. "
        "Compare accuracy, energy conservation, and training efficiency."
    )

    bench_mode = st.radio(
        "Benchmark mode",
        ["Quick (CI-level, ~1 min)", "Load saved results"],
        key="bench_mode", horizontal=True,
    )

    if bench_mode == "Quick (CI-level, ~1 min)":
        problem = st.selectbox(
            "Problem",
            ["pendulum", "orbital", "heat", "wave", "all"],
            key="bench_problem",
        )

        if st.button("Run Benchmark", key="bench_run"):
            torch.manual_seed(42)
            runner = BenchmarkRunner(mode="quick")
            progress_b = st.progress(0, text="Running benchmarks...")

            if problem == "all":
                steps = ["pendulum", "orbital", "heat", "wave"]
                for i, p in enumerate(steps):
                    progress_b.progress(
                        (i) / len(steps),
                        text=f"Benchmarking {p}...")
                    getattr(runner, f"run_{p}")()
                    runner.results[p] = getattr(runner, f"run_{p}")()
                progress_b.progress(1.0, text="Done!")
            else:
                progress_b.progress(0.1, text=f"Benchmarking {problem}...")
                runner.results[problem] = getattr(runner, f"run_{problem}")()
                progress_b.progress(1.0, text="Done!")

            progress_b.empty()
            path = runner.save_results()
            st.success(f"Results saved to {path}")

            # Display results table
            st.subheader("Results")
            st.markdown(runner.generate_markdown_table())

            # Interactive comparison chart
            st.subheader("Method Comparison")
            for prob_name, methods in runner.results.items():
                method_names = list(methods.keys())
                if len(method_names) < 2:
                    continue

                # Radar chart for multi-method problems
                metrics_to_plot = []
                for m in method_names:
                    d = methods[m]
                    l2 = d.get("l2_rel_theta", d.get("l2_rel_pos",
                         d.get("l2_rel_error", 0)))
                    ed = d.get("energy_drift", 0)
                    wt = d.get("wall_time_s", 0)
                    eff = d.get("epochs_to_0.01", d.get("epochs", 0))
                    metrics_to_plot.append({
                        "method": m,
                        "Accuracy (1-L2)": max(0, 1 - min(l2, 1)),
                        "Energy Cons.": max(0, 1 - min(ed, 1)),
                        "Speed (1/time)": 1 / (wt + 0.1),
                        "Efficiency": 1 / (eff + 1),
                    })

                categories = ["Accuracy (1-L2)", "Energy Cons.",
                               "Speed (1/time)", "Efficiency"]

                fig_radar = go.Figure()
                radar_colors = [C_CLASSICAL, C_PINN, C_ERROR, C_ACCENT]
                for i, mp in enumerate(metrics_to_plot):
                    vals = [mp[c] for c in categories]
                    # Normalise to [0, 1] for radar
                    max_vals = [max(m2[c] for m2 in metrics_to_plot)
                                for c in categories]
                    vals_norm = [v / (mv + 1e-16) for v, mv in zip(vals, max_vals)]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals_norm + [vals_norm[0]],
                        theta=categories + [categories[0]],
                        fill='toself', name=mp["method"],
                        line=dict(color=radar_colors[i % len(radar_colors)]),
                        opacity=0.6,
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
                    title=f"{prob_name.title()}: Method Comparison",
                    height=450, template="plotly_white",
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Loss bar chart
            st.subheader("Final Loss Comparison")
            probs = []
            meths = []
            losses_bar = []
            for prob_name, methods in runner.results.items():
                for m_name, metrics in methods.items():
                    probs.append(prob_name)
                    meths.append(m_name)
                    losses_bar.append(metrics["final_loss"])

            fig_bar = go.Figure()
            for m_name in sorted(set(meths)):
                mask = [i for i, m in enumerate(meths) if m == m_name]
                fig_bar.add_trace(go.Bar(
                    x=[probs[i] for i in mask],
                    y=[losses_bar[i] for i in mask],
                    name=m_name,
                ))
            fig_bar.update_layout(
                barmode='group', yaxis_type='log',
                yaxis_title='Final Loss',
                height=400, template='plotly_white',
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        # Load saved results
        import glob
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "benchmarks", "results")
        json_files = glob.glob(os.path.join(results_dir, "*.json"))
        if not json_files:
            st.warning("No saved benchmark results found. Run a benchmark first.")
        else:
            selected = st.selectbox(
                "Select results file",
                [os.path.basename(f) for f in json_files],
                key="bench_file",
            )
            if selected:
                import json
                with open(os.path.join(results_dir, selected)) as f:
                    saved = json.load(f)
                runner = BenchmarkRunner(mode="quick")
                runner.results = saved
                st.subheader("Results")
                st.markdown(runner.generate_markdown_table())

                st.subheader("Raw JSON")
                st.json(saved)
