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
        return dH_dp, -dH_dq  # dq/dt, dp/dt


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
    r = ((dtheta - omega)**2 + (domega + (g/L)*torch.sin(theta))**2).squeeze()
    return r.detach()


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
    model = PINNPendulum()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=300, factor=0.5, min_lr=1e-6)
    losses = []
    label = "adaptive" if adaptive else "uniform"
    progress = st.progress(0, text=f"Training pendulum PINN ({label})...")

    cached_t_col = None

    for ep in range(epochs):
        opt.zero_grad()

        # Adaptive resampling every 500 epochs
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
    r = torch.sqrt(x**2 + y**2 + 1e-8)
    r3 = r**3
    res = ((dx-vx)**2 + (dy-vy)**2 +
           (dvx + GM*x/r3)**2 + (dvy + GM*y/r3)**2).squeeze()
    return res.detach()


def train_orbital(ecc, GM, n_orbits, n_col=600, epochs=5000,
                  lr=1e-3, ic_w=50.0, adaptive=False):
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
    label = "adaptive" if adaptive else "uniform"
    progress = st.progress(0, text=f"Training orbital PINN ({label})...")

    x0_t = torch.tensor(x0, dtype=torch.float32)
    y0_t = torch.tensor(y0, dtype=torch.float32)
    vx0_t = torch.tensor(vx0, dtype=torch.float32)
    vy0_t = torch.tensor(vy0, dtype=torch.float32)

    cached_t_col = None

    for ep in range(epochs):
        opt.zero_grad()

        # Adaptive resampling every 500 epochs
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


def train_hnn_pendulum(theta_0, omega_0, t_max, g, L,
                       n_data=1000, epochs=3000, lr=1e-3, batch_size=256):
    """Train an HNN for the pendulum and return model + loss history."""
    # Generate training data from ground-truth ODE
    t_data = np.linspace(0, t_max, n_data)
    sol = solve_ivp(
        lambda t, y: [y[1], -(g / L) * np.sin(y[0])],
        (0, t_max), [theta_0, omega_0],
        t_eval=t_data, method='RK45', rtol=1e-10, atol=1e-12)
    q_np, p_np = sol.y[0], sol.y[1]  # q=theta, p=omega (unit mass, L=1 absorbed)
    dq_np = p_np                      # dq/dt = omega
    dp_np = -(g / L) * np.sin(q_np)  # dp/dt = -dH/dq

    q_t = torch.tensor(q_np, dtype=torch.float32).unsqueeze(1)
    p_t = torch.tensor(p_np, dtype=torch.float32).unsqueeze(1)
    dq_t = torch.tensor(dq_np, dtype=torch.float32).unsqueeze(1)
    dp_t = torch.tensor(dp_np, dtype=torch.float32).unsqueeze(1)

    model = HamiltonianNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=300, factor=0.5, min_lr=1e-6)
    losses = []
    n = len(q_np)
    progress = st.progress(0, text="Training HNN...")

    for ep in range(epochs):
        opt.zero_grad()
        idx = torch.randint(0, n, (batch_size,))
        q_b = q_t[idx].requires_grad_(True)
        p_b = p_t[idx].requires_grad_(True)

        dq_pred, dp_pred = model.time_derivatives(q_b, p_b)
        loss = (torch.mean((dq_pred - dq_t[idx])**2) +
                torch.mean((dp_pred - dp_t[idx])**2))

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


def integrate_hnn(hnn_model, q0, p0, t_eval):
    """Integrate HNN dynamics forward using RK45."""
    hnn_model.eval()

    def rhs(t, state):
        q, p = state
        q_t = torch.tensor([[q]], dtype=torch.float32, requires_grad=True)
        p_t = torch.tensor([[p]], dtype=torch.float32, requires_grad=True)
        with torch.enable_grad():
            dq, dp = hnn_model.time_derivatives(q_t, p_t)
        return [dq.item(), dp.item()]

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [q0, p0],
                    t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    return sol.y[0], sol.y[1]


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

sim_tab = st.sidebar.radio(
    "Simulation",
    ["Pendulum", "HNN Pendulum", "Orbital Mechanics", "Inverse Problem",
     "PDE Solver", "Three-Body"],
    index=0)

st.sidebar.markdown("---")

if sim_tab == "Pendulum":
    st.sidebar.subheader("Pendulum Settings")
    p_length = st.sidebar.slider("Pendulum length L (m)", 0.5, 3.0, 1.0, 0.1)
    p_angle  = st.sidebar.slider("Initial angle (degrees)", 5, 85, 45, 5)
    p_tmax   = st.sidebar.slider("Time span (s)", 2.0, 20.0, 10.0, 1.0)
    p_epochs = st.sidebar.select_slider(
        "Training epochs", options=[1000, 2000, 3000, 5000, 8000], value=3000)
    p_adaptive = st.sidebar.checkbox(
        "Adaptive Sampling", value=False, key="p_adaptive",
        help="Concentrate collocation points where the physics residual is highest. "
             "Re-evaluates every 500 epochs. 20% of points remain uniform.")
    st.sidebar.caption(
        "More epochs = better accuracy but longer training. "
        "3000 is a good balance.")
elif sim_tab == "HNN Pendulum":
    st.sidebar.subheader("HNN Pendulum Settings")
    h_length = st.sidebar.slider("Pendulum length L (m)", 0.5, 3.0, 1.0, 0.1,
                                  key="h_length")
    h_angle  = st.sidebar.slider("Initial angle (degrees)", 5, 85, 69, 5,
                                  key="h_angle")
    h_tmax   = st.sidebar.slider("Time span (s)", 2.0, 20.0, 10.0, 1.0,
                                  key="h_tmax")
    h_epochs = st.sidebar.select_slider(
        "Training epochs", options=[1000, 2000, 3000, 5000, 8000], value=3000,
        key="h_epochs")
    st.sidebar.caption(
        "The HNN learns H(q,p) and derives dynamics via autograd. "
        "Energy conservation is structural, not learned.")
elif sim_tab == "Inverse Problem":
    st.sidebar.subheader("Inverse Problem Settings")
    inv_mode = st.sidebar.radio("System", ["Pendulum (g/L)", "Orbital (GM)"],
                                 key="inv_mode")
    inv_noise = st.sidebar.slider("Noise level (sigma)", 0.01, 0.20, 0.05, 0.01,
                                   key="inv_noise")
    if inv_mode == "Pendulum (g/L)":
        inv_g_over_L_init = st.sidebar.slider(
            "Initial guess for g/L", 1.0, 20.0, 5.0, 0.5, key="inv_gl_init")
        inv_angle = st.sidebar.slider("Initial angle (degrees)", 10, 85, 45, 5,
                                       key="inv_angle")
    else:
        inv_GM_init = st.sidebar.slider(
            "Initial guess for GM", 0.1, 3.0, 0.5, 0.1, key="inv_gm_init")
        inv_ecc = st.sidebar.slider("Eccentricity", 0.1, 0.7, 0.5, 0.05,
                                     key="inv_ecc")
    inv_epochs = st.sidebar.select_slider(
        "Training epochs",
        options=[3000, 5000, 8000, 10000, 15000], value=8000,
        key="inv_epochs")
    st.sidebar.caption(
        "Inverse PINNs infer unknown physical parameters from noisy data. "
        "More noise requires more epochs.")
elif sim_tab == "PDE Solver":
    st.sidebar.subheader("PDE Solver Settings")
    pde_mode = st.sidebar.radio("Equation", ["Heat Equation", "Wave Equation"],
                                 key="pde_mode")
    if pde_mode == "Heat Equation":
        pde_alpha = st.sidebar.slider(
            "Thermal diffusivity (alpha)", 0.001, 0.1, 0.01, 0.001,
            format="%.3f", key="pde_alpha")
    else:
        pde_c = st.sidebar.slider(
            "Wave speed (c)", 0.5, 3.0, 1.0, 0.1, key="pde_c")
    pde_epochs = st.sidebar.select_slider(
        "Training epochs",
        options=[3000, 5000, 8000, 12000, 15000], value=8000,
        key="pde_epochs")
    st.sidebar.caption(
        "PDE PINNs take (x, t) as input and learn u(x, t) by enforcing "
        "the PDE, boundary conditions, and initial conditions.")
elif sim_tab == "Three-Body":
    st.sidebar.subheader("Three-Body Settings")
    tb_tmax = st.sidebar.slider("Time horizon", 0.5, 2.0, 1.0, 0.1,
                                 key="tb_tmax")
    tb_epochs = st.sidebar.select_slider(
        "Training epochs",
        options=[5000, 8000, 10000, 15000, 20000], value=10000,
        key="tb_epochs")
    st.sidebar.caption(
        "The three-body problem is chaotic — small errors grow "
        "exponentially. The PINN will likely diverge from the true "
        "trajectory. Showing this chaos boundary is itself a result.")
else:
    st.sidebar.subheader("Orbital Settings")
    o_ecc    = st.sidebar.slider("Eccentricity", 0.0, 0.85, 0.3, 0.05)
    o_orbits = st.sidebar.slider("Number of orbits", 0.5, 2.0, 1.0, 0.25)
    o_epochs = st.sidebar.select_slider(
        "Training epochs", options=[2000, 3000, 5000, 8000, 12000], value=5000)
    o_adaptive = st.sidebar.checkbox(
        "Adaptive Sampling", value=False, key="o_adaptive",
        help="Concentrate collocation points where the physics residual is highest. "
             "Re-evaluates every 500 epochs. 20% of points remain uniform.")
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
            epochs=p_epochs, adaptive=p_adaptive)

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
# HNN Pendulum tab
# ---------------------------------------------------------------------------
elif sim_tab == "HNN Pendulum":
    st.header("Hamiltonian Neural Network — Pendulum")

    with st.expander("The Physics", expanded=False):
        st.markdown(r"""
**Hamiltonian mechanics** reformulates Newton's laws using a scalar energy
function $H(q, p)$. For the simple pendulum with $q = \theta$, $p = \omega$:

$$H(q, p) = \frac{1}{2}p^2 - \frac{g}{L}\cos q$$

Hamilton's equations of motion:

$$\frac{dq}{dt} = \frac{\partial H}{\partial p}, \qquad
  \frac{dp}{dt} = -\frac{\partial H}{\partial q}$$

**HNN approach:** Instead of learning $\theta(t)$ directly (like a PINN),
we train a neural network to approximate the scalar $H(q, p)$. The equations
of motion are then derived via automatic differentiation of the learned $H$.

**Why this is powerful:** The symplectic structure *guarantees* that
$dH/dt = 0$ along any trajectory — energy is conserved **by construction**,
regardless of how well the network is trained. This is a structural
advantage over standard PINNs, where energy conservation depends on
training quality.

**Reference:** Greydanus, Dzamba & Cranmer, *Hamiltonian Neural Networks*,
NeurIPS 2019.
        """)

    if run_btn:
        theta_0 = np.radians(h_angle)
        omega_0 = 0.0
        g = 9.81

        # Train HNN
        hnn_model, hnn_losses = train_hnn_pendulum(
            theta_0, omega_0, h_tmax, g, h_length, epochs=h_epochs)

        # Also train a standard PINN for comparison
        pinn_model, pinn_losses = train_pendulum(
            theta_0, omega_0, h_tmax, g, h_length, epochs=h_epochs)

        # Evaluate
        t_eval = np.linspace(0, h_tmax, 1000)
        t_ode, th_ode, om_ode = solve_pendulum(
            theta_0, omega_0, h_tmax, g, h_length)

        # HNN trajectory
        th_hnn, om_hnn = integrate_hnn(hnn_model, theta_0, omega_0, t_eval)

        # PINN trajectory
        pinn_model.eval()
        with torch.no_grad():
            out = pinn_model(
                torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
            ).numpy()
        th_pinn, om_pinn = out[:, 0], out[:, 1]

        # Energy calculations (specific energy: H = p^2/2 - (g/L)*cos(q))
        def pend_energy(th, om):
            return 0.5 * om**2 - (g / h_length) * np.cos(th)

        E_ode = pend_energy(th_ode, om_ode)
        E_hnn = pend_energy(th_hnn, om_hnn)
        E_pinn = pend_energy(th_pinn, om_pinn)
        E0 = E_ode[0]
        dE_ode = np.abs((E_ode - E0) / (np.abs(E0) + 1e-16))
        dE_hnn = np.abs((E_hnn - E0) / (np.abs(E0) + 1e-16))
        dE_pinn = np.abs((E_pinn - E0) / (np.abs(E0) + 1e-16))

        # --- Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("HNN Final Loss", f"{hnn_losses[-1]:.2e}")
        col2.metric("PINN Final Loss", f"{pinn_losses[-1]:.2e}")
        col3.metric("HNN Max |dE/E|", f"{np.max(dE_hnn):.2e}")
        col4.metric("PINN Max |dE/E|", f"{np.max(dE_pinn):.2e}")

        # --- Plot 1: Theta prediction vs ground truth ---
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=t_ode, y=np.degrees(th_ode), mode='lines',
            name='Ground Truth (RK45)',
            line=dict(color=C_CLASSICAL, width=2.5)))
        fig1.add_trace(go.Scatter(
            x=t_eval, y=np.degrees(th_hnn), mode='lines',
            name='HNN',
            line=dict(color=C_ERROR, width=2.5, dash='dash')))
        fig1.add_trace(go.Scatter(
            x=t_eval, y=np.degrees(th_pinn), mode='lines',
            name='PINN',
            line=dict(color=C_PINN, width=2.5, dash='dot')))
        fig1.update_layout(
            title="Angular Displacement over Time",
            xaxis_title="Time (s)", yaxis_title="Angle (degrees)",
            template="plotly_white", height=420)
        st.plotly_chart(fig1, use_container_width=True)

        # --- Row: Energy drift + Phase portrait ---
        c1, c2 = st.columns(2)

        # Energy drift |dE/E_0|
        fig_de = go.Figure()
        fig_de.add_trace(go.Scatter(
            x=t_ode, y=dE_ode + 1e-16, mode='lines',
            name='Ground Truth (RK45)',
            line=dict(color=C_CLASSICAL, width=2)))
        fig_de.add_trace(go.Scatter(
            x=t_eval, y=dE_hnn + 1e-16, mode='lines',
            name='HNN',
            line=dict(color=C_ERROR, width=2)))
        fig_de.add_trace(go.Scatter(
            x=t_eval, y=dE_pinn + 1e-16, mode='lines',
            name='PINN',
            line=dict(color=C_PINN, width=2)))
        fig_de.update_layout(
            title="Relative Energy Drift |dE/E_0|",
            xaxis_title="Time (s)", yaxis_title="|dE / E_0|",
            yaxis_type="log", template="plotly_white", height=400)
        c1.plotly_chart(fig_de, use_container_width=True)

        # Phase portrait
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(
            x=np.degrees(th_ode), y=om_ode, mode='lines',
            name='Ground Truth',
            line=dict(color=C_CLASSICAL, width=2)))
        fig_ph.add_trace(go.Scatter(
            x=np.degrees(th_hnn), y=om_hnn, mode='lines',
            name='HNN',
            line=dict(color=C_ERROR, width=2, dash='dash')))
        fig_ph.add_trace(go.Scatter(
            x=np.degrees(th_pinn), y=om_pinn, mode='lines',
            name='PINN',
            line=dict(color=C_PINN, width=2, dash='dot')))
        fig_ph.update_layout(
            title="Phase Portrait",
            xaxis_title="Angle (degrees)",
            yaxis_title="Angular Velocity (rad/s)",
            template="plotly_white", height=400)
        c2.plotly_chart(fig_ph, use_container_width=True)

        # --- Row: Energy conservation + Training loss ---
        c3, c4 = st.columns(2)

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=t_ode, y=E_ode, mode='lines',
            name='Ground Truth', line=dict(color=C_CLASSICAL, width=2)))
        fig_e.add_trace(go.Scatter(
            x=t_eval, y=E_hnn, mode='lines',
            name='HNN', line=dict(color=C_ERROR, width=2)))
        fig_e.add_trace(go.Scatter(
            x=t_eval, y=E_pinn, mode='lines',
            name='PINN', line=dict(color=C_PINN, width=2)))
        fig_e.add_hline(y=E0, line_dash="dot", line_color="gray",
                        annotation_text=f"True E = {E0:.4f}")
        fig_e.update_layout(
            title="Energy Conservation",
            xaxis_title="Time (s)", yaxis_title="Total Energy",
            template="plotly_white", height=400)
        c3.plotly_chart(fig_e, use_container_width=True)

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=hnn_losses, mode='lines', name='HNN',
            line=dict(color=C_ERROR, width=1.5)))
        fig_loss.add_trace(go.Scatter(
            y=pinn_losses, mode='lines', name='PINN',
            line=dict(color=C_PINN, width=1.5)))
        fig_loss.update_layout(
            title="Training Loss Curves",
            xaxis_title="Epoch", yaxis_title="Loss",
            yaxis_type="log", template="plotly_white", height=400)
        c4.plotly_chart(fig_loss, use_container_width=True)

    else:
        st.info("Adjust parameters in the sidebar, then click **Train & Compare**.")


# ---------------------------------------------------------------------------
# Inverse Problem tab
# ---------------------------------------------------------------------------
elif sim_tab == "Inverse Problem":
    st.header("Inverse Problem — Parameter Inference")

    with st.expander("The Physics", expanded=False):
        st.markdown(r"""
**Inverse problems** work backwards from observations to unknown physics.

In the **forward PINN**, physical parameters ($g/L$, $GM$) are known constants
in the ODE residual. In the **inverse PINN**, we promote them to trainable
`torch.nn.Parameter` variables. The optimizer adjusts them alongside the
network weights until the physics residual *and* the data-fitting loss are
both small — this can only happen when the parameters match reality.

**Pendulum:** We observe noisy $\theta(t)$ and infer $g/L$. The ODE
$\ddot\theta + (g/L)\sin\theta = 0$ depends only on the ratio, so a single
parameter suffices.

**Orbital:** We observe noisy $(x, y)$ positions and infer $GM$.
The gravitational ODE $\ddot{\mathbf{r}} = -GM\,\mathbf{r}/r^3$ is
parameterized by $GM$ alone.

**Training strategy:**
1. *Warmup* — fit the network to data + IC only (no physics), so it learns
   a reasonable trajectory shape first.
2. *Full training* — physics + data + IC loss. The physics residual uses the
   trainable parameter, which the optimizer drives toward the true value.
        """)

    if run_btn:
        if inv_mode == "Pendulum (g/L)":
            theta_0 = np.radians(inv_angle)
            omega_0 = 0.0
            t_max = 10.0
            g_true, L_true = 9.81, 1.0

            # Generate noisy data
            np.random.seed(42)
            t_obs_np = np.sort(np.random.uniform(0.1, t_max, 80))
            sol_obs = solve_ivp(
                lambda t, s: [s[1], -(g_true / L_true) * np.sin(s[0])],
                (0, t_max), [theta_0, omega_0],
                t_eval=t_obs_np, method='RK45', rtol=1e-10, atol=1e-12)
            theta_obs_np = sol_obs.y[0] + np.random.normal(0, inv_noise, 80)

            # Ground truth dense
            t_dense = np.linspace(0, t_max, 1000)
            sol_gt = solve_ivp(
                lambda t, s: [s[1], -(g_true / L_true) * np.sin(s[0])],
                (0, t_max), [theta_0, omega_0],
                t_eval=t_dense, method='RK45', rtol=1e-10, atol=1e-12)
            theta_gt = sol_gt.y[0]

            # Train inverse PINN
            model = PINNPendulum()
            g_over_L = nn.Parameter(
                torch.tensor(inv_g_over_L_init, dtype=torch.float32))

            opt = torch.optim.Adam([
                {'params': model.parameters(), 'lr': 1e-3},
                {'params': [g_over_L], 'lr': 1e-2}
            ])
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=500, factor=0.5, min_lr=1e-6)

            t_obs_t = torch.tensor(t_obs_np, dtype=torch.float32).unsqueeze(1)
            theta_obs_t = torch.tensor(theta_obs_np, dtype=torch.float32)

            losses, gl_hist = [], []
            progress = st.progress(0, text="Training inverse PINN...")

            # Warmup: data + IC only
            warmup_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for ep in range(1500):
                warmup_opt.zero_grad()
                out = model(t_obs_t)
                loss = 10.0 * torch.mean((out[:, 0] - theta_obs_t)**2)
                t0 = torch.zeros(1, 1)
                out0 = model(t0)
                loss = loss + 20.0 * ((out0[0, 0] - theta_0)**2 +
                                      (out0[0, 1] - omega_0)**2)
                loss.backward()
                warmup_opt.step()

            # Main training
            for ep in range(inv_epochs):
                opt.zero_grad()

                t_col = torch.rand(500, 1) * t_max
                t_col.requires_grad_(True)
                out = model(t_col)
                theta_c, omega_c = out[:, 0:1], out[:, 1:2]
                dth = torch.autograd.grad(
                    theta_c, t_col, torch.ones_like(theta_c),
                    create_graph=True)[0]
                dom = torch.autograd.grad(
                    omega_c, t_col, torch.ones_like(omega_c),
                    create_graph=True)[0]
                phys = (torch.mean((dth - omega_c)**2) +
                        torch.mean((dom + g_over_L * torch.sin(theta_c))**2))

                out_obs = model(t_obs_t)
                data_l = torch.mean((out_obs[:, 0] - theta_obs_t)**2)

                t0 = torch.zeros(1, 1)
                out0 = model(t0)
                ic_l = ((out0[0, 0] - theta_0)**2 +
                        (out0[0, 1] - omega_0)**2)

                total = phys + 10.0 * data_l + 20.0 * ic_l
                total.backward()
                opt.step()
                sched.step(total.item())

                with torch.no_grad():
                    g_over_L.clamp_(min=0.1)

                losses.append(total.item())
                gl_hist.append(g_over_L.item())

                if (ep + 1) % 100 == 0:
                    progress.progress(
                        (ep + 1) / inv_epochs,
                        text=f"Epoch {ep+1}/{inv_epochs}  |  "
                             f"g/L: {g_over_L.item():.4f}  |  "
                             f"Loss: {total.item():.6f}")

            progress.empty()

            # PINN reconstruction
            model.eval()
            with torch.no_grad():
                out_p = model(
                    torch.tensor(t_dense, dtype=torch.float32).unsqueeze(1)
                ).numpy()
            theta_pinn = out_p[:, 0]

            # --- Metrics ---
            gl_final = g_over_L.item()
            col1, col2, col3 = st.columns(3)
            col1.metric("Recovered g/L", f"{gl_final:.4f}")
            col2.metric("True g/L", "9.8100")
            col3.metric("Relative Error",
                         f"{abs(gl_final - 9.81)/9.81*100:.2f}%")

            # --- Plot: noisy data vs PINN vs truth ---
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=t_dense, y=np.degrees(theta_gt), mode='lines',
                name='Ground Truth',
                line=dict(color=C_CLASSICAL, width=2.5)))
            fig1.add_trace(go.Scatter(
                x=t_obs_np, y=np.degrees(theta_obs_np), mode='markers',
                name='Noisy Observations',
                marker=dict(color='gray', size=5, opacity=0.5)))
            fig1.add_trace(go.Scatter(
                x=t_dense, y=np.degrees(theta_pinn), mode='lines',
                name='PINN Reconstruction',
                line=dict(color=C_PINN, width=2.5, dash='dash')))
            fig1.update_layout(
                title="Trajectory Reconstruction",
                xaxis_title="Time (s)", yaxis_title="Angle (degrees)",
                template="plotly_white", height=420)
            st.plotly_chart(fig1, use_container_width=True)

            # --- Row: g/L convergence + training loss ---
            c1, c2 = st.columns(2)

            fig_gl = go.Figure()
            fig_gl.add_trace(go.Scatter(
                y=gl_hist, mode='lines', name='Inferred g/L',
                line=dict(color=C_PINN, width=2)))
            fig_gl.add_hline(y=9.81, line_dash="dash", line_color=C_CLASSICAL,
                             annotation_text="True g/L = 9.81")
            fig_gl.update_layout(
                title="g/L Convergence over Epochs",
                xaxis_title="Epoch", yaxis_title="g/L",
                template="plotly_white", height=400)
            c1.plotly_chart(fig_gl, use_container_width=True)

            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=losses, mode='lines',
                line=dict(color=C_ACCENT, width=1.5)))
            fig_loss.update_layout(
                title="Training Loss",
                xaxis_title="Epoch", yaxis_title="Loss",
                yaxis_type="log", template="plotly_white", height=400)
            c2.plotly_chart(fig_loss, use_container_width=True)

        else:
            # --- Orbital inverse problem ---
            GM_true = 1.0
            ecc = inv_ecc
            a = 1.0
            x0 = a * (1.0 - ecc)
            y0, vx0 = 0.0, 0.0
            vy0 = float(np.sqrt(GM_true / a * (1.0 + ecc) / (1.0 - ecc)))
            period = 2.0 * np.pi * np.sqrt(a**3 / GM_true)
            t_max = period

            np.random.seed(42)

            def orb_rhs(t, s):
                x, y, vx, vy = s
                r3 = (x**2 + y**2)**1.5
                return [vx, vy, -GM_true * x / r3, -GM_true * y / r3]

            # Noisy observations
            t_obs_np = np.sort(np.random.uniform(
                0.05 * t_max, 0.95 * t_max, 80))
            sol_obs = solve_ivp(orb_rhs, (0, t_max), [x0, y0, vx0, vy0],
                                t_eval=t_obs_np, method='RK45',
                                rtol=1e-12, atol=1e-14)
            x_obs_np = sol_obs.y[0] + np.random.normal(0, inv_noise, 80)
            y_obs_np = sol_obs.y[1] + np.random.normal(0, inv_noise, 80)

            # Ground truth
            t_dense = np.linspace(0, t_max, 1500)
            sol_gt = solve_ivp(orb_rhs, (0, t_max), [x0, y0, vx0, vy0],
                               t_eval=t_dense, method='RK45',
                               rtol=1e-12, atol=1e-14)
            x_gt, y_gt = sol_gt.y[0], sol_gt.y[1]

            # Train
            model = PINNOrbital()
            GM_param = nn.Parameter(
                torch.tensor(inv_GM_init, dtype=torch.float32))

            opt = torch.optim.Adam([
                {'params': model.parameters(), 'lr': 1e-3},
                {'params': [GM_param], 'lr': 5e-3}
            ])
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=500, factor=0.5, min_lr=1e-6)

            t_obs_t = torch.tensor(t_obs_np, dtype=torch.float32).unsqueeze(1)
            x_obs_t = torch.tensor(x_obs_np, dtype=torch.float32)
            y_obs_t = torch.tensor(y_obs_np, dtype=torch.float32)
            x0_t = torch.tensor(x0, dtype=torch.float32)
            y0_t = torch.tensor(y0, dtype=torch.float32)
            vx0_t = torch.tensor(vx0, dtype=torch.float32)
            vy0_t = torch.tensor(vy0, dtype=torch.float32)

            losses, gm_hist = [], []
            progress = st.progress(0, text="Training inverse orbital PINN...")

            # Warmup
            warmup_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for ep in range(2000):
                warmup_opt.zero_grad()
                out = model(t_obs_t)
                loss = 10.0 * (torch.mean((out[:, 0] - x_obs_t)**2) +
                               torch.mean((out[:, 1] - y_obs_t)**2))
                t0 = torch.zeros(1, 1)
                out0 = model(t0)
                loss = loss + 50.0 * ((out0[0, 0] - x0_t)**2 +
                                      (out0[0, 1] - y0_t)**2 +
                                      (out0[0, 2] - vx0_t)**2 +
                                      (out0[0, 3] - vy0_t)**2)
                loss.backward()
                warmup_opt.step()

            # Main training
            for ep in range(inv_epochs):
                opt.zero_grad()

                t_col = torch.rand(600, 1) * t_max
                t_col.requires_grad_(True)
                out = model(t_col)
                x, y = out[:, 0:1], out[:, 1:2]
                vx, vy = out[:, 2:3], out[:, 3:4]

                dx = torch.autograd.grad(
                    x, t_col, torch.ones_like(x), create_graph=True)[0]
                dy = torch.autograd.grad(
                    y, t_col, torch.ones_like(y), create_graph=True)[0]
                dvx = torch.autograd.grad(
                    vx, t_col, torch.ones_like(vx), create_graph=True)[0]
                dvy = torch.autograd.grad(
                    vy, t_col, torch.ones_like(vy), create_graph=True)[0]

                r = torch.sqrt(x**2 + y**2 + 1e-8)
                r3 = r**3

                phys = (torch.mean((dx - vx)**2) +
                        torch.mean((dy - vy)**2) +
                        torch.mean((dvx + GM_param * x / r3)**2) +
                        torch.mean((dvy + GM_param * y / r3)**2))

                out_obs = model(t_obs_t)
                data_l = (torch.mean((out_obs[:, 0] - x_obs_t)**2) +
                          torch.mean((out_obs[:, 1] - y_obs_t)**2))

                t0 = torch.zeros(1, 1)
                out0 = model(t0)
                ic_l = ((out0[0, 0] - x0_t)**2 + (out0[0, 1] - y0_t)**2 +
                        (out0[0, 2] - vx0_t)**2 + (out0[0, 3] - vy0_t)**2)

                total = phys + 10.0 * data_l + 50.0 * ic_l
                total.backward()
                opt.step()
                sched.step(total.item())

                with torch.no_grad():
                    GM_param.clamp_(min=0.01)

                losses.append(total.item())
                gm_hist.append(GM_param.item())

                if (ep + 1) % 100 == 0:
                    progress.progress(
                        (ep + 1) / inv_epochs,
                        text=f"Epoch {ep+1}/{inv_epochs}  |  "
                             f"GM: {GM_param.item():.4f}  |  "
                             f"Loss: {total.item():.6f}")

            progress.empty()

            # PINN reconstruction
            model.eval()
            with torch.no_grad():
                out_p = model(
                    torch.tensor(t_dense, dtype=torch.float32).unsqueeze(1)
                ).numpy()
            x_pinn, y_pinn = out_p[:, 0], out_p[:, 1]

            # Metrics
            gm_final = GM_param.item()
            col1, col2, col3 = st.columns(3)
            col1.metric("Recovered GM", f"{gm_final:.4f}")
            col2.metric("True GM", "1.0000")
            col3.metric("Relative Error",
                         f"{abs(gm_final - 1.0)/1.0*100:.2f}%")

            # Orbit plot
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=x_gt, y=y_gt, mode='lines', name='Ground Truth',
                line=dict(color=C_CLASSICAL, width=2.5)))
            fig1.add_trace(go.Scatter(
                x=x_obs_np, y=y_obs_np, mode='markers',
                name='Noisy Observations',
                marker=dict(color='gray', size=5, opacity=0.5)))
            fig1.add_trace(go.Scatter(
                x=x_pinn, y=y_pinn, mode='lines',
                name='PINN Reconstruction',
                line=dict(color=C_PINN, width=2.5, dash='dash')))
            fig1.add_trace(go.Scatter(
                x=[0], y=[0], mode='markers', name='Central body',
                marker=dict(size=10, color='black')))
            fig1.update_layout(
                title="Orbital Trajectory Reconstruction",
                xaxis_title="x", yaxis_title="y",
                template="plotly_white", height=500,
                yaxis_scaleanchor="x", yaxis_scaleratio=1)
            st.plotly_chart(fig1, use_container_width=True)

            # GM convergence + loss
            c1, c2 = st.columns(2)

            fig_gm = go.Figure()
            fig_gm.add_trace(go.Scatter(
                y=gm_hist, mode='lines', name='Inferred GM',
                line=dict(color=C_PINN, width=2)))
            fig_gm.add_hline(y=1.0, line_dash="dash",
                             line_color=C_CLASSICAL,
                             annotation_text="True GM = 1.0")
            fig_gm.update_layout(
                title="GM Convergence over Epochs",
                xaxis_title="Epoch", yaxis_title="GM",
                template="plotly_white", height=400)
            c1.plotly_chart(fig_gm, use_container_width=True)

            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=losses, mode='lines',
                line=dict(color=C_ACCENT, width=1.5)))
            fig_loss.update_layout(
                title="Training Loss",
                xaxis_title="Epoch", yaxis_title="Loss",
                yaxis_type="log", template="plotly_white", height=400)
            c2.plotly_chart(fig_loss, use_container_width=True)

    else:
        st.info("Adjust parameters in the sidebar, then click **Train & Compare**.")


# ---------------------------------------------------------------------------
# PDE Solver tab
# ---------------------------------------------------------------------------
elif sim_tab == "PDE Solver":
    st.header("PDE Solver — Heat & Wave Equations")

    with st.expander("The Physics", expanded=False):
        st.markdown(r"""
**1D Heat Equation** (parabolic — diffusion):

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

Domain $x \in [0,1]$, $t \in [0,1]$. Boundary conditions $u(0,t) = u(1,t) = 0$.
Initial condition $u(x,0) = \sin(\pi x)$.
Exact solution: $u(x,t) = e^{-\alpha \pi^2 t} \sin(\pi x)$.

**1D Wave Equation** (hyperbolic — oscillation):

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

Domain $x \in [0,1]$, $t \in [0,0.5]$. BCs $u(0,t) = u(1,t) = 0$.
ICs $u(x,0) = \sin(\pi x)$, $\partial u/\partial t(x,0) = 0$.
Exact solution: $u(x,t) = \cos(c\pi t) \sin(\pi x)$.

The PINN takes $(x, t)$ as input and outputs $u$. All derivatives
($\partial u/\partial t$, $\partial^2 u/\partial x^2$, etc.) are computed
via PyTorch autograd.
        """)

    if run_btn:
        if pde_mode == "Heat Equation":
            alpha_val = pde_alpha

            model = PINNPDE()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=500, factor=0.5, min_lr=1e-6)
            losses = []
            progress = st.progress(0, text="Training heat equation PINN...")

            for ep in range(pde_epochs):
                opt.zero_grad()

                # Interior
                x_int = torch.rand(2000, 1, requires_grad=True)
                t_int = torch.rand(2000, 1, requires_grad=True)
                u = model(x_int, t_int)

                du_dt = torch.autograd.grad(
                    u, t_int, torch.ones_like(u), create_graph=True)[0]
                du_dx = torch.autograd.grad(
                    u, x_int, torch.ones_like(u), create_graph=True)[0]
                d2u_dx2 = torch.autograd.grad(
                    du_dx, x_int, torch.ones_like(du_dx), create_graph=True)[0]

                loss_pde = torch.mean((du_dt - alpha_val * d2u_dx2) ** 2)

                # BCs
                t_bc = torch.rand(200, 1)
                loss_bc = (torch.mean(model(torch.zeros(200, 1), t_bc) ** 2) +
                           torch.mean(model(torch.ones(200, 1), t_bc) ** 2))

                # IC
                x_ic = torch.rand(200, 1)
                u_ic = model(x_ic, torch.zeros(200, 1))
                loss_ic = torch.mean(
                    (u_ic - torch.sin(np.pi * x_ic)) ** 2)

                total = loss_pde + 10.0 * loss_bc + 10.0 * loss_ic
                total.backward()
                opt.step()
                sched.step(total.item())
                losses.append(total.item())

                if (ep + 1) % 100 == 0:
                    progress.progress(
                        (ep + 1) / pde_epochs,
                        text=f"Epoch {ep+1}/{pde_epochs}  |  "
                             f"Loss: {total.item():.6f}")

            progress.empty()

            # Evaluate on grid
            nx, nt = 100, 100
            x_np = np.linspace(0, 1, nx)
            t_np = np.linspace(0, 1, nt)
            X, T = np.meshgrid(x_np, t_np)

            model.eval()
            with torch.no_grad():
                xf = torch.tensor(
                    X.flatten(), dtype=torch.float32).unsqueeze(1)
                tf = torch.tensor(
                    T.flatten(), dtype=torch.float32).unsqueeze(1)
                U_pinn = model(xf, tf).numpy().reshape(X.shape)

            U_exact = (np.exp(-alpha_val * np.pi**2 * T) *
                       np.sin(np.pi * X))
            U_err = np.abs(U_pinn - U_exact)

            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Loss", f"{losses[-1]:.2e}")
            col2.metric("Max |Error|", f"{np.max(U_err):.2e}")
            col3.metric("Mean |Error|", f"{np.mean(U_err):.2e}")

            # Heatmaps
            c1, c2, c3 = st.columns(3)

            fig_p = go.Figure(data=go.Heatmap(
                z=U_pinn, x=x_np, y=t_np,
                colorscale='Hot', colorbar=dict(title='u')))
            fig_p.update_layout(
                title="PINN u(x,t)",
                xaxis_title="x", yaxis_title="t",
                template="plotly_white", height=400)
            c1.plotly_chart(fig_p, use_container_width=True)

            fig_e = go.Figure(data=go.Heatmap(
                z=U_exact, x=x_np, y=t_np,
                colorscale='Hot', colorbar=dict(title='u')))
            fig_e.update_layout(
                title="Exact u(x,t)",
                xaxis_title="x", yaxis_title="t",
                template="plotly_white", height=400)
            c2.plotly_chart(fig_e, use_container_width=True)

            fig_err = go.Figure(data=go.Heatmap(
                z=U_err, x=x_np, y=t_np,
                colorscale='Viridis', colorbar=dict(title='|err|')))
            fig_err.update_layout(
                title=f"Absolute Error (max: {np.max(U_err):.2e})",
                xaxis_title="x", yaxis_title="t",
                template="plotly_white", height=400)
            c3.plotly_chart(fig_err, use_container_width=True)

            # Training loss
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
            # Wave equation
            c_val = pde_c
            t_max_w = 0.5

            model = PINNPDE()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=500, factor=0.5, min_lr=1e-6)
            losses = []
            progress = st.progress(0, text="Training wave equation PINN...")

            for ep in range(pde_epochs):
                opt.zero_grad()

                # Interior
                x_int = torch.rand(2000, 1, requires_grad=True)
                t_int = torch.rand(2000, 1, requires_grad=True) * t_max_w
                u = model(x_int, t_int)

                du_dt = torch.autograd.grad(
                    u, t_int, torch.ones_like(u), create_graph=True)[0]
                d2u_dt2 = torch.autograd.grad(
                    du_dt, t_int, torch.ones_like(du_dt),
                    create_graph=True)[0]
                du_dx = torch.autograd.grad(
                    u, x_int, torch.ones_like(u), create_graph=True)[0]
                d2u_dx2 = torch.autograd.grad(
                    du_dx, x_int, torch.ones_like(du_dx),
                    create_graph=True)[0]

                loss_pde = torch.mean(
                    (d2u_dt2 - c_val**2 * d2u_dx2) ** 2)

                # BCs
                t_bc = torch.rand(200, 1) * t_max_w
                loss_bc = (
                    torch.mean(model(torch.zeros(200, 1), t_bc) ** 2) +
                    torch.mean(model(torch.ones(200, 1), t_bc) ** 2))

                # IC: u(x,0) = sin(pi*x)
                x_ic = torch.rand(200, 1)
                u_ic = model(x_ic, torch.zeros(200, 1))
                loss_ic_u = torch.mean(
                    (u_ic - torch.sin(np.pi * x_ic)) ** 2)

                # IC: du/dt(x,0) = 0
                x_ic2 = torch.rand(200, 1)
                t_ic2 = torch.zeros(200, 1, requires_grad=True)
                u_ic2 = model(x_ic2, t_ic2)
                du_dt_ic = torch.autograd.grad(
                    u_ic2, t_ic2, torch.ones_like(u_ic2),
                    create_graph=True)[0]
                loss_ic_v = torch.mean(du_dt_ic ** 2)

                total = (loss_pde + 10.0 * loss_bc +
                         10.0 * (loss_ic_u + loss_ic_v))
                total.backward()
                opt.step()
                sched.step(total.item())
                losses.append(total.item())

                if (ep + 1) % 100 == 0:
                    progress.progress(
                        (ep + 1) / pde_epochs,
                        text=f"Epoch {ep+1}/{pde_epochs}  |  "
                             f"Loss: {total.item():.6f}")

            progress.empty()

            # Evaluate on grid
            nx, nt = 100, 100
            x_np = np.linspace(0, 1, nx)
            t_np = np.linspace(0, t_max_w, nt)
            X, T = np.meshgrid(x_np, t_np)

            model.eval()
            with torch.no_grad():
                xf = torch.tensor(
                    X.flatten(), dtype=torch.float32).unsqueeze(1)
                tf = torch.tensor(
                    T.flatten(), dtype=torch.float32).unsqueeze(1)
                U_pinn = model(xf, tf).numpy().reshape(X.shape)

            U_exact = (np.cos(c_val * np.pi * T) *
                       np.sin(np.pi * X))
            U_err = np.abs(U_pinn - U_exact)

            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Loss", f"{losses[-1]:.2e}")
            col2.metric("Max |Error|", f"{np.max(U_err):.2e}")
            col3.metric("Mean |Error|", f"{np.mean(U_err):.2e}")

            # Heatmaps — use diverging colorscale for wave (has negatives)
            vmax = max(np.max(np.abs(U_pinn)), np.max(np.abs(U_exact)))
            c1, c2, c3 = st.columns(3)

            fig_p = go.Figure(data=go.Heatmap(
                z=U_pinn, x=x_np, y=t_np,
                colorscale='RdBu_r', zmid=0, zmin=-vmax, zmax=vmax,
                colorbar=dict(title='u')))
            fig_p.update_layout(
                title="PINN u(x,t)",
                xaxis_title="x", yaxis_title="t",
                template="plotly_white", height=400)
            c1.plotly_chart(fig_p, use_container_width=True)

            fig_e = go.Figure(data=go.Heatmap(
                z=U_exact, x=x_np, y=t_np,
                colorscale='RdBu_r', zmid=0, zmin=-vmax, zmax=vmax,
                colorbar=dict(title='u')))
            fig_e.update_layout(
                title="Exact u(x,t)",
                xaxis_title="x", yaxis_title="t",
                template="plotly_white", height=400)
            c2.plotly_chart(fig_e, use_container_width=True)

            fig_err = go.Figure(data=go.Heatmap(
                z=U_err, x=x_np, y=t_np,
                colorscale='Viridis',
                colorbar=dict(title='|err|')))
            fig_err.update_layout(
                title=f"Absolute Error (max: {np.max(U_err):.2e})",
                xaxis_title="x", yaxis_title="t",
                template="plotly_white", height=400)
            c3.plotly_chart(fig_err, use_container_width=True)

            # Training loss
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
# Three-Body tab
# ---------------------------------------------------------------------------
elif sim_tab == "Three-Body":
    st.header("Three-Body Problem — Figure-Eight Orbit")

    with st.expander("The Physics", expanded=False):
        st.markdown(r"""
**The three-body problem** is one of the oldest unsolved problems in physics.
Three masses interact via Newtonian gravity:

$$m_i \frac{d^2 \mathbf{r}_i}{dt^2} = \sum_{j \neq i}
  \frac{G\,m_i\,m_j\,(\mathbf{r}_j - \mathbf{r}_i)}{|\mathbf{r}_j - \mathbf{r}_i|^3}$$

Unlike the two-body problem, there is **no general closed-form solution**.
Trajectories can be *chaotic* — tiny perturbations lead to wildly different
outcomes.

**Figure-eight orbit** (Chenciner & Montgomery, 2000): Three equal masses
chase each other along a figure-eight path. This is one of very few known
periodic three-body solutions.

**Why PINNs struggle here:** The chaotic sensitivity means any small error
in the PINN's approximation grows *exponentially* over time. The point
where the PINN diverges from the classical solver reveals the **chaos
boundary** — and showing that boundary is itself a meaningful result.

We add gravitational softening ($\epsilon = 10^{-3}$) to prevent
singularities when bodies approach closely.
        """)

    if run_btn:
        # Figure-eight initial conditions
        masses = [1.0, 1.0, 1.0]
        x1_0, y1_0 = -0.97000436, 0.24308753
        x2_0, y2_0 = 0.97000436, -0.24308753
        x3_0, y3_0 = 0.0, 0.0
        vx3_0, vy3_0 = -0.93240737, -0.86473146
        vx1_0, vy1_0 = -vx3_0 / 2.0, -vy3_0 / 2.0
        vx2_0, vy2_0 = -vx3_0 / 2.0, -vy3_0 / 2.0
        state0 = [x1_0, y1_0, x2_0, y2_0, x3_0, y3_0,
                  vx1_0, vy1_0, vx2_0, vy2_0, vx3_0, vy3_0]
        state0_t = torch.tensor(state0, dtype=torch.float32)

        G_val = 1.0
        eps = 1e-3
        t_max = tb_tmax

        # Train
        model = PINNThreeBody()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=500, factor=0.5, min_lr=1e-6)
        losses = []
        progress = st.progress(0, text="Training three-body PINN...")

        m1, m2, m3 = masses

        for ep in range(tb_epochs):
            opt.zero_grad()

            t_col = torch.rand(1000, 1) * t_max
            t_col.requires_grad_(True)
            out = model(t_col)

            x1, y1 = out[:, 0:1], out[:, 1:2]
            x2, y2 = out[:, 2:3], out[:, 3:4]
            x3, y3 = out[:, 4:5], out[:, 5:6]
            vx1, vy1 = out[:, 6:7], out[:, 7:8]
            vx2, vy2 = out[:, 8:9], out[:, 9:10]
            vx3, vy3 = out[:, 10:11], out[:, 11:12]

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

            r12 = torch.sqrt((x2-x1)**2 + (y2-y1)**2 + eps**2)
            r13 = torch.sqrt((x3-x1)**2 + (y3-y1)**2 + eps**2)
            r23 = torch.sqrt((x3-x2)**2 + (y3-y2)**2 + eps**2)

            ax1 = G_val*(m2*(x2-x1)/r12**3 + m3*(x3-x1)/r13**3)
            ay1 = G_val*(m2*(y2-y1)/r12**3 + m3*(y3-y1)/r13**3)
            ax2 = G_val*(m1*(x1-x2)/r12**3 + m3*(x3-x2)/r23**3)
            ay2 = G_val*(m1*(y1-y2)/r12**3 + m3*(y3-y2)/r23**3)
            ax3 = G_val*(m1*(x1-x3)/r13**3 + m2*(x2-x3)/r23**3)
            ay3 = G_val*(m1*(y1-y3)/r13**3 + m2*(y2-y3)/r23**3)

            phys = (torch.mean((dx1-vx1)**2) + torch.mean((dy1-vy1)**2) +
                    torch.mean((dx2-vx2)**2) + torch.mean((dy2-vy2)**2) +
                    torch.mean((dx3-vx3)**2) + torch.mean((dy3-vy3)**2) +
                    torch.mean((dvx1-ax1)**2) + torch.mean((dvy1-ay1)**2) +
                    torch.mean((dvx2-ax2)**2) + torch.mean((dvy2-ay2)**2) +
                    torch.mean((dvx3-ax3)**2) + torch.mean((dvy3-ay3)**2))

            t0 = torch.zeros(1, 1)
            out0 = model(t0).squeeze()
            ic = torch.sum((out0 - state0_t)**2)

            total = phys + 100.0 * ic
            total.backward()
            opt.step()
            sched.step(total.item())
            losses.append(total.item())

            if (ep + 1) % 100 == 0:
                progress.progress(
                    (ep + 1) / tb_epochs,
                    text=f"Epoch {ep+1}/{tb_epochs}  |  "
                         f"Loss: {total.item():.6f}")

        progress.empty()

        # Classical solution
        def tb_rhs(t, state):
            x1, y1, x2, y2, x3, y3 = state[0:6]
            vx1, vy1, vx2, vy2, vx3, vy3 = state[6:12]
            r12 = np.sqrt((x2-x1)**2+(y2-y1)**2+eps**2)
            r13 = np.sqrt((x3-x1)**2+(y3-y1)**2+eps**2)
            r23 = np.sqrt((x3-x2)**2+(y3-y2)**2+eps**2)
            ax1 = G_val*(m2*(x2-x1)/r12**3+m3*(x3-x1)/r13**3)
            ay1 = G_val*(m2*(y2-y1)/r12**3+m3*(y3-y1)/r13**3)
            ax2 = G_val*(m1*(x1-x2)/r12**3+m3*(x3-x2)/r23**3)
            ay2 = G_val*(m1*(y1-y2)/r12**3+m3*(y3-y2)/r23**3)
            ax3 = G_val*(m1*(x1-x3)/r13**3+m2*(x2-x3)/r23**3)
            ay3 = G_val*(m1*(y1-y3)/r13**3+m2*(y2-y3)/r23**3)
            return [vx1,vy1,vx2,vy2,vx3,vy3,ax1,ay1,ax2,ay2,ax3,ay3]

        t_eval = np.linspace(0, t_max, 1500)
        sol = solve_ivp(tb_rhs, (0, t_max), state0,
                        t_eval=t_eval, method='RK45',
                        rtol=1e-12, atol=1e-14)
        st_ode = sol.y.T

        model.eval()
        with torch.no_grad():
            st_pinn = model(
                torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
            ).numpy()

        # Energy
        def tb_energy(s):
            x1,y1,x2,y2,x3,y3 = [s[:,i] for i in range(6)]
            vx1,vy1,vx2,vy2,vx3,vy3 = [s[:,i] for i in range(6,12)]
            T = 0.5*(vx1**2+vy1**2+vx2**2+vy2**2+vx3**2+vy3**2)
            r12 = np.sqrt((x2-x1)**2+(y2-y1)**2+eps**2)
            r13 = np.sqrt((x3-x1)**2+(y3-y1)**2+eps**2)
            r23 = np.sqrt((x3-x2)**2+(y3-y2)**2+eps**2)
            V = -(1.0/r12 + 1.0/r13 + 1.0/r23)
            return T + V

        E_ode = tb_energy(st_ode)
        E_pinn = tb_energy(st_pinn)
        E0 = E_ode[0]
        dE_ode = np.abs((E_ode - E0) / (np.abs(E0) + 1e-16))
        dE_pinn = np.abs((E_pinn - E0) / (np.abs(E0) + 1e-16))

        # Detect divergence
        tot_err = np.zeros(len(t_eval))
        for i in range(3):
            tot_err += np.sqrt(
                (st_pinn[:,2*i]-st_ode[:,2*i])**2 +
                (st_pinn[:,2*i+1]-st_ode[:,2*i+1])**2)
        div_idx = np.argmax(tot_err > 0.1)
        div_time = t_eval[div_idx] if tot_err[div_idx] > 0.1 else None

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Loss", f"{losses[-1]:.2e}")
        col2.metric("PINN Max |dE/E|", f"{np.max(dE_pinn):.2e}")
        if div_time is not None:
            col3.metric("Divergence Time", f"t = {div_time:.3f}")
        else:
            col3.metric("Max Position Error", f"{np.max(tot_err):.4f}")

        # Trajectory plot
        body_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        body_names = ["Body 1", "Body 2", "Body 3"]

        fig1 = go.Figure()
        for i, (bc, bn) in enumerate(zip(body_colors, body_names)):
            fig1.add_trace(go.Scatter(
                x=st_ode[:, 2*i], y=st_ode[:, 2*i+1], mode='lines',
                name=f'{bn} (RK45)',
                line=dict(color=bc, width=2), opacity=0.4))
            fig1.add_trace(go.Scatter(
                x=st_pinn[:, 2*i], y=st_pinn[:, 2*i+1], mode='lines',
                name=f'{bn} (PINN)',
                line=dict(color=bc, width=2, dash='dash')))
            fig1.add_trace(go.Scatter(
                x=[state0[2*i]], y=[state0[2*i+1]], mode='markers',
                name=f'{bn} start',
                marker=dict(color=bc, size=8), showlegend=False))
        fig1.update_layout(
            title="Figure-Eight Trajectories — PINN vs Classical",
            xaxis_title="x", yaxis_title="y",
            template="plotly_white", height=500,
            yaxis_scaleanchor="x", yaxis_scaleratio=1)
        st.plotly_chart(fig1, use_container_width=True)

        # Energy drift + position error
        c1, c2 = st.columns(2)

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=t_eval, y=dE_ode + 1e-16, mode='lines',
            name='Classical (RK45)',
            line=dict(color=C_CLASSICAL, width=2)))
        fig_e.add_trace(go.Scatter(
            x=t_eval, y=dE_pinn + 1e-16, mode='lines',
            name='PINN', line=dict(color=C_PINN, width=2)))
        if div_time is not None:
            fig_e.add_vline(x=div_time, line_dash="dash",
                            line_color="gray",
                            annotation_text=f"Diverges t={div_time:.3f}")
        fig_e.update_layout(
            title="Relative Energy Drift |dE/E_0|",
            xaxis_title="Time", yaxis_title="|dE / E_0|",
            yaxis_type="log", template="plotly_white", height=400)
        c1.plotly_chart(fig_e, use_container_width=True)

        fig_pe = go.Figure()
        for i, (bc, bn) in enumerate(zip(body_colors, body_names)):
            pe = np.sqrt((st_pinn[:,2*i]-st_ode[:,2*i])**2 +
                         (st_pinn[:,2*i+1]-st_ode[:,2*i+1])**2)
            fig_pe.add_trace(go.Scatter(
                x=t_eval, y=pe + 1e-16, mode='lines',
                name=bn, line=dict(color=bc, width=2)))
        if div_time is not None:
            fig_pe.add_vline(x=div_time, line_dash="dash",
                             line_color="gray",
                             annotation_text="Chaos boundary")
        fig_pe.update_layout(
            title="PINN Position Error vs Classical",
            xaxis_title="Time", yaxis_title="|dr|",
            yaxis_type="log", template="plotly_white", height=400)
        c2.plotly_chart(fig_pe, use_container_width=True)

        if div_time is not None:
            st.info(
                f"**Chaos boundary detected at t = {div_time:.3f}.** "
                "The three-body problem is chaotic — small PINN errors "
                "grow exponentially, causing the learned trajectory to "
                "diverge from the classical solution. This divergence "
                "point is itself a meaningful result: it reveals the "
                "practical time horizon over which the PINN approximation "
                "remains faithful to the true dynamics.")

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
            o_ecc, GM, o_orbits, epochs=o_epochs,
            adaptive=o_adaptive)

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
