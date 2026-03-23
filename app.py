"""
Streamlit Web App — Physics-Informed Neural Networks
=====================================================
Interactive comparison of PINN predictions vs classical ODE solvers
for pendulum motion and orbital mechanics.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Physics-Informed Neural Networks",
    page_icon="",
    layout="wide",
)


# ===========================================================================
# PINN Models (self-contained so the app has no external imports)
# ===========================================================================

class PINNPendulum(nn.Module):
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
    def __init__(self, hidden_size=128, num_hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)


# ===========================================================================
# Training helpers
# ===========================================================================

def train_pendulum(theta_0, omega_0, t_max, g, L,
                   n_col=400, epochs=3000, lr=1e-3, ic_w=20.0):
    """Train a pendulum PINN and return model + loss history."""
    model = PINNPendulum()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=300, factor=0.5, min_lr=1e-6)
    losses = []
    progress = st.progress(0, text="Training pendulum PINN...")

    for ep in range(epochs):
        opt.zero_grad()
        t_col = torch.rand(n_col, 1) * t_max
        t_col.requires_grad_(True)

        out = model(t_col)
        theta, omega = out[:, 0:1], out[:, 1:2]

        dtheta = torch.autograd.grad(
            theta, t_col, torch.ones_like(theta), create_graph=True)[0]
        domega = torch.autograd.grad(
            omega, t_col, torch.ones_like(omega), create_graph=True)[0]

        phys = torch.mean((dtheta - omega)**2) + \
               torch.mean((domega + (g / L) * torch.sin(theta))**2)

        t0 = torch.zeros(1, 1)
        out0 = model(t0)
        ic = (out0[0, 0] - theta_0)**2 + (out0[0, 1] - omega_0)**2

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


def train_orbital(ecc, GM, n_orbits, n_col=600, epochs=5000,
                  lr=1e-3, ic_w=50.0):
    """Train an orbital PINN and return model + loss history + ICs + t_max."""
    a = 1.0
    x0 = a * (1.0 - ecc)
    y0 = 0.0
    vx0 = 0.0
    vy0 = float(np.sqrt(GM / a * (1.0 + ecc) / (1.0 - ecc)))
    period = 2.0 * np.pi * np.sqrt(a**3 / GM)
    t_max = n_orbits * period

    model = PINNOrbital()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=300, factor=0.5, min_lr=1e-6)
    losses = []
    progress = st.progress(0, text="Training orbital PINN...")

    x0_t = torch.tensor(x0, dtype=torch.float32)
    y0_t = torch.tensor(y0, dtype=torch.float32)
    vx0_t = torch.tensor(vx0, dtype=torch.float32)
    vy0_t = torch.tensor(vy0, dtype=torch.float32)

    for ep in range(epochs):
        opt.zero_grad()
        t_col = torch.rand(n_col, 1) * t_max
        t_col.requires_grad_(True)

        out = model(t_col)
        x, y, vx, vy = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

        dx = torch.autograd.grad(x, t_col, torch.ones_like(x), create_graph=True)[0]
        dy = torch.autograd.grad(y, t_col, torch.ones_like(y), create_graph=True)[0]
        dvx = torch.autograd.grad(vx, t_col, torch.ones_like(vx), create_graph=True)[0]
        dvy = torch.autograd.grad(vy, t_col, torch.ones_like(vy), create_graph=True)[0]

        r = torch.sqrt(x**2 + y**2 + 1e-8)
        r3 = r**3

        phys = (torch.mean((dx - vx)**2) + torch.mean((dy - vy)**2) +
                torch.mean((dvx + GM * x / r3)**2) +
                torch.mean((dvy + GM * y / r3)**2))

        t0 = torch.zeros(1, 1)
        out0 = model(t0)
        ic = ((out0[0, 0] - x0_t)**2 + (out0[0, 1] - y0_t)**2 +
              (out0[0, 2] - vx0_t)**2 + (out0[0, 3] - vy0_t)**2)

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
    ics = (x0, y0, vx0, vy0)
    return model, losses, ics, t_max, period


# ===========================================================================
# Classical solvers
# ===========================================================================

def solve_pendulum(theta_0, omega_0, t_max, g, L, n_pts=1000):
    t_eval = np.linspace(0, t_max, n_pts)
    sol = solve_ivp(
        lambda t, s: [s[1], -(g / L) * np.sin(s[0])],
        (0, t_max), [theta_0, omega_0],
        t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[0], sol.y[1]


def solve_orbit(x0, y0, vx0, vy0, t_max, GM, n_pts=1500):
    t_eval = np.linspace(0, t_max, n_pts)
    def rhs(t, s):
        x, y, vx, vy = s
        r3 = (x**2 + y**2)**1.5
        return [vx, vy, -GM * x / r3, -GM * y / r3]
    sol = solve_ivp(rhs, (0, t_max), [x0, y0, vx0, vy0],
                    t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-14)
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


# ===========================================================================
# Plotly color palette
# ===========================================================================
C_CLASSICAL = "#636EFA"   # blue
C_PINN      = "#EF553B"   # red
C_ERROR     = "#00CC96"   # green
C_ACCENT    = "#AB63FA"   # purple


# ===========================================================================
# Sidebar
# ===========================================================================

st.sidebar.title("PINN Parameters")

sim_tab = st.sidebar.radio("Simulation", ["Pendulum", "Orbital Mechanics"],
                           index=0)

st.sidebar.markdown("---")

if sim_tab == "Pendulum":
    st.sidebar.subheader("Pendulum Settings")
    p_length = st.sidebar.slider("Pendulum length L (m)", 0.5, 3.0, 1.0, 0.1)
    p_angle  = st.sidebar.slider("Initial angle (degrees)", 5, 85, 45, 5)
    p_tmax   = st.sidebar.slider("Time span (s)", 2.0, 20.0, 10.0, 1.0)
    p_epochs = st.sidebar.select_slider(
        "Training epochs", options=[1000, 2000, 3000, 5000, 8000], value=3000)
    st.sidebar.caption(
        "More epochs = better accuracy but longer training. "
        "3000 is a good balance.")
else:
    st.sidebar.subheader("Orbital Settings")
    o_ecc    = st.sidebar.slider("Eccentricity", 0.0, 0.85, 0.3, 0.05)
    o_orbits = st.sidebar.slider("Number of orbits", 0.5, 2.0, 1.0, 0.25)
    o_epochs = st.sidebar.select_slider(
        "Training epochs", options=[2000, 3000, 5000, 8000, 12000], value=5000)
    st.sidebar.caption(
        "Higher eccentricity is harder for the PINN — "
        "the velocity spike at perihelion is steep.")

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Train & Compare", type="primary", use_container_width=True)


# ===========================================================================
# Main content
# ===========================================================================

st.title("Physics-Informed Neural Networks")
st.markdown("*Teaching neural networks the laws of physics*")

# ---------------------------------------------------------------------------
# Pendulum tab
# ---------------------------------------------------------------------------
if sim_tab == "Pendulum":
    st.header("Simple Pendulum")

    with st.expander("The Physics", expanded=False):
        st.markdown(r"""
**Governing equation** (nonlinear pendulum):

$$\frac{d^2\theta}{dt^2} + \frac{g}{L}\sin\theta = 0$$

This describes a mass on a rigid rod of length $L$ swinging under gravity $g$.
We decompose it into two first-order ODEs:

$$\frac{d\theta}{dt} = \omega, \qquad \frac{d\omega}{dt} = -\frac{g}{L}\sin\theta$$

**Why not the small-angle approximation?** The linearized version
$\ddot\theta + (g/L)\theta = 0$ gives a simple sine wave, but the *real*
pendulum has an amplitude-dependent period. The PINN learns the full
nonlinear dynamics — no shortcuts.

**Energy**: The total mechanical energy
$E = \tfrac{1}{2}mL^2\omega^2 - mgL\cos\theta$ is conserved.
How well the PINN preserves this is a key quality metric.
        """)

    if run_btn:
        theta_0 = np.radians(p_angle)
        omega_0 = 0.0
        g = 9.81

        # Train
        model, losses = train_pendulum(
            theta_0, omega_0, p_tmax, g, p_length,
            epochs=p_epochs)

        # Evaluate
        t_eval = np.linspace(0, p_tmax, 1000)
        t_ode, th_ode, om_ode = solve_pendulum(theta_0, omega_0, p_tmax, g, p_length)

        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)).numpy()
        th_pinn, om_pinn = out[:, 0], out[:, 1]

        # --- Metrics ---
        E_ode  = 0.5 * p_length**2 * om_ode**2 - g * p_length * np.cos(th_ode)
        E_pinn = 0.5 * p_length**2 * om_pinn**2 - g * p_length * np.cos(th_pinn)
        E0 = E_ode[0]
        dE_ode  = np.abs((E_ode - E0) / (np.abs(E0) + 1e-12))
        dE_pinn = np.abs((E_pinn - E0) / (np.abs(E0) + 1e-12))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Loss", f"{losses[-1]:.2e}")
        col2.metric("Max Angle Error",
                     f"{np.max(np.abs(np.degrees(th_pinn - th_ode))):.2f} deg")
        col3.metric("Classical Max |dE/E|", f"{np.max(dE_ode):.2e}")
        col4.metric("PINN Max |dE/E|", f"{np.max(dE_pinn):.2e}")

        # --- Plot 1: Angular displacement ---
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=t_ode, y=np.degrees(th_ode), mode='lines',
            name='Classical (RK45)', line=dict(color=C_CLASSICAL, width=2.5)))
        fig1.add_trace(go.Scatter(
            x=t_eval, y=np.degrees(th_pinn), mode='lines',
            name='PINN', line=dict(color=C_PINN, width=2.5, dash='dash')))
        fig1.update_layout(
            title="Angular Displacement over Time",
            xaxis_title="Time (s)", yaxis_title="Angle (degrees)",
            template="plotly_white", height=420)
        st.plotly_chart(fig1, use_container_width=True)

        # --- Row of two charts ---
        c1, c2 = st.columns(2)

        # Phase portrait
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=np.degrees(th_ode), y=om_ode, mode='lines',
            name='Classical', line=dict(color=C_CLASSICAL, width=2)))
        fig2.add_trace(go.Scatter(
            x=np.degrees(th_pinn), y=om_pinn, mode='lines',
            name='PINN', line=dict(color=C_PINN, width=2, dash='dash')))
        fig2.update_layout(
            title="Phase Portrait",
            xaxis_title="Angle (degrees)", yaxis_title="Angular Velocity (rad/s)",
            template="plotly_white", height=400)
        c1.plotly_chart(fig2, use_container_width=True)

        # Energy
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=t_ode, y=E_ode, mode='lines',
            name='Classical', line=dict(color=C_CLASSICAL, width=2)))
        fig3.add_trace(go.Scatter(
            x=t_eval, y=E_pinn, mode='lines',
            name='PINN', line=dict(color=C_PINN, width=2)))
        fig3.add_hline(y=E0, line_dash="dot", line_color="gray",
                       annotation_text=f"True E = {E0:.4f}")
        fig3.update_layout(
            title="Energy Conservation",
            xaxis_title="Time (s)", yaxis_title="Total Energy",
            template="plotly_white", height=400)
        c2.plotly_chart(fig3, use_container_width=True)

        # --- Training loss ---
        with st.expander("Training Loss Curve"):
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=losses, mode='lines',
                line=dict(color=C_ACCENT, width=1.5)))
            fig_loss.update_layout(
                xaxis_title="Epoch", yaxis_title="Loss",
                yaxis_type="log", template="plotly_white", height=350)
            st.plotly_chart(fig_loss, use_container_width=True)

    else:
        st.info("Adjust parameters in the sidebar, then click **Train & Compare**.")


# ---------------------------------------------------------------------------
# Orbital tab
# ---------------------------------------------------------------------------
elif sim_tab == "Orbital Mechanics":
    st.header("Two-Body Orbital Mechanics")

    with st.expander("The Physics", expanded=False):
        st.markdown(r"""
**Newton's law of gravitation** gives the equations of motion for a
body orbiting a central mass:

$$\frac{d^2 x}{dt^2} = -\frac{GM\,x}{r^3}, \qquad
  \frac{d^2 y}{dt^2} = -\frac{GM\,y}{r^3}$$

where $r = \sqrt{x^2 + y^2}$ and $GM$ is the gravitational parameter.

**Conserved quantities** (the gold standard for testing any solver):

| Quantity | Formula | Physical meaning |
|----------|---------|-----------------|
| **Energy** | $E = \tfrac{1}{2}(v_x^2 + v_y^2) - \tfrac{GM}{r}$ | Total mechanical energy (constant for Kepler orbits) |
| **Angular momentum** | $L = x\,v_y - y\,v_x$ | Kepler's second law — equal areas in equal times |

The PINN is never told about energy or angular momentum. If it learns
to conserve them, it has truly internalized the physics.

**Orbit shape**: Eccentricity $e$ controls the ellipse.
$e = 0$ is a circle; as $e \to 1$ the orbit becomes extremely elongated
with a fast perihelion passage that is hard for the PINN to capture.
        """)

    if run_btn:
        GM = 1.0
        model, losses, ics, t_max, period = train_orbital(
            o_ecc, GM, o_orbits, epochs=o_epochs)

        x0, y0, vx0, vy0 = ics
        t_ode, x_o, y_o, vx_o, vy_o = solve_orbit(x0, y0, vx0, vy0, t_max, GM)

        model.eval()
        t_eval = np.linspace(0, t_max, 1500)
        with torch.no_grad():
            out = model(torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)).numpy()
        x_p, y_p, vx_p, vy_p = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

        # Conserved quantities
        def energy(x, y, vx, vy):
            return 0.5 * (vx**2 + vy**2) - GM / np.sqrt(x**2 + y**2)
        def ang_mom(x, y, vx, vy):
            return x * vy - y * vx

        E_o, E_p = energy(x_o, y_o, vx_o, vy_o), energy(x_p, y_p, vx_p, vy_p)
        L_o, L_p = ang_mom(x_o, y_o, vx_o, vy_o), ang_mom(x_p, y_p, vx_p, vy_p)
        E0, L0 = E_o[0], L_o[0]
        dE_o = np.abs((E_o - E0) / (np.abs(E0) + 1e-16))
        dE_p = np.abs((E_p - E0) / (np.abs(E0) + 1e-16))
        pos_err = np.sqrt((x_p - x_o)**2 + (y_p - y_o)**2)

        # --- Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Loss", f"{losses[-1]:.2e}")
        col2.metric("Max Position Error", f"{np.max(pos_err):.4f}")
        col3.metric("PINN Max |dE/E|", f"{np.max(dE_p):.2e}")
        col4.metric("Orbital Period", f"{period:.3f}")

        # --- Plot 1: Orbit trajectory ---
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=x_o, y=y_o, mode='lines',
            name='Classical (RK45)', line=dict(color=C_CLASSICAL, width=2.5)))
        fig1.add_trace(go.Scatter(
            x=x_p, y=y_p, mode='lines',
            name='PINN', line=dict(color=C_PINN, width=2.5, dash='dash')))
        fig1.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            name='Central body',
            marker=dict(size=12, color='black', symbol='circle')))
        fig1.add_trace(go.Scatter(
            x=[x0], y=[y0], mode='markers',
            name='Start (perihelion)',
            marker=dict(size=10, color='green', symbol='star')))
        fig1.update_layout(
            title="Orbital Trajectory (x-y plane)",
            xaxis_title="x", yaxis_title="y",
            template="plotly_white", height=500,
            yaxis_scaleanchor="x", yaxis_scaleratio=1)
        st.plotly_chart(fig1, use_container_width=True)

        # --- Row: Energy + Angular momentum ---
        c1, c2 = st.columns(2)

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=t_ode, y=E_o, mode='lines',
            name='Classical', line=dict(color=C_CLASSICAL, width=2)))
        fig_e.add_trace(go.Scatter(
            x=t_eval, y=E_p, mode='lines',
            name='PINN', line=dict(color=C_PINN, width=2)))
        fig_e.add_hline(y=E0, line_dash="dot", line_color="gray",
                        annotation_text=f"True E = {E0:.4f}")
        fig_e.update_layout(
            title="Energy Conservation",
            xaxis_title="Time", yaxis_title="Total Energy (T + V)",
            template="plotly_white", height=400)
        c1.plotly_chart(fig_e, use_container_width=True)

        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(
            x=t_ode, y=L_o, mode='lines',
            name='Classical', line=dict(color=C_CLASSICAL, width=2)))
        fig_l.add_trace(go.Scatter(
            x=t_eval, y=L_p, mode='lines',
            name='PINN', line=dict(color=C_PINN, width=2)))
        fig_l.add_hline(y=L0, line_dash="dot", line_color="gray",
                        annotation_text=f"True L = {L0:.4f}")
        fig_l.update_layout(
            title="Angular Momentum Conservation",
            xaxis_title="Time", yaxis_title="Angular Momentum L",
            template="plotly_white", height=400)
        c2.plotly_chart(fig_l, use_container_width=True)

        # --- Row: Relative energy error + Position error ---
        c3, c4 = st.columns(2)

        fig_de = go.Figure()
        fig_de.add_trace(go.Scatter(
            x=t_ode, y=dE_o + 1e-16, mode='lines',
            name='Classical', line=dict(color=C_CLASSICAL, width=2)))
        fig_de.add_trace(go.Scatter(
            x=t_eval, y=dE_p + 1e-16, mode='lines',
            name='PINN', line=dict(color=C_PINN, width=2)))
        fig_de.update_layout(
            title="Relative Energy Error |dE/E|",
            xaxis_title="Time", yaxis_title="|dE / E_0|",
            yaxis_type="log", template="plotly_white", height=400)
        c3.plotly_chart(fig_de, use_container_width=True)

        fig_pe = go.Figure()
        fig_pe.add_trace(go.Scatter(
            x=t_eval, y=pos_err + 1e-16, mode='lines',
            line=dict(color=C_ERROR, width=2)))
        fig_pe.update_layout(
            title="PINN Position Error vs Classical",
            xaxis_title="Time", yaxis_title="|delta r|",
            yaxis_type="log", template="plotly_white", height=400)
        c4.plotly_chart(fig_pe, use_container_width=True)

        # --- Training loss ---
        with st.expander("Training Loss Curve"):
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=losses, mode='lines',
                line=dict(color=C_ACCENT, width=1.5)))
            fig_loss.update_layout(
                xaxis_title="Epoch", yaxis_title="Loss",
                yaxis_type="log", template="plotly_white", height=350)
            st.plotly_chart(fig_loss, use_container_width=True)

    else:
        st.info("Adjust parameters in the sidebar, then click **Train & Compare**.")


# ===========================================================================
# "How it works" section (always visible at the bottom)
# ===========================================================================

st.markdown("---")
st.header("How PINNs Work")

st.markdown("""
A **Physics-Informed Neural Network** is a regular neural network with a
twist: instead of learning purely from data, it also learns from the
*equations of physics*.

Here's the idea in four steps:

**1. The network guesses a solution.**
Give the network a time value *t* and it outputs the physical quantities
(angle, position, velocity, etc.). At first these are random nonsense.

**2. We check the physics.**
Using automatic differentiation (the same engine that trains any neural
network), we compute derivatives of the network's output. Then we plug
those derivatives into the governing equation. If the equation isn't
satisfied, the "physics residual" is large.

**3. We check the initial conditions.**
The network must also match the starting state of the system. A correct
equation with wrong starting conditions gives a wrong trajectory.

**4. We minimize both errors.**
The total loss is:

> **Loss = Physics residual + Weight x Initial condition error**

An ordinary optimizer (Adam) adjusts the network weights until both terms
are small. The result is a smooth, differentiable function that satisfies
the differential equation *everywhere* in the time domain, not just at
discrete grid points.

**Why is this interesting?**
- No simulation data needed — just the equation.
- The solution is a continuous function, not a table of numbers.
- It generalizes to PDEs, inverse problems, and noisy data.
- Conservation laws (energy, momentum) emerge naturally if the network
  truly learns the physics.
""")

st.caption(
    "Built with PyTorch, Streamlit, and Plotly. "
    "PINN training runs entirely in your browser session (CPU).")
