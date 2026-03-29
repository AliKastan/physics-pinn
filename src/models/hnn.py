"""
Hamiltonian Neural Network (HNN) for energy-conserving dynamics.

A HNN learns the scalar Hamiltonian H(q, p) and derives equations of
motion via Hamilton's equations using autograd:

    dq/dt =  dH/dp
    dp/dt = -dH/dq

Energy conservation is structural: dH/dt = (dH/dq)(dH/dp) + (dH/dp)(-dH/dq) = 0
exactly, regardless of network accuracy.  The trade-off versus a standard
PINN is that the HNN requires training data (state-derivative pairs)
rather than only the governing ODE.

Pendulum specialization:
    q = theta                       (angle)
    p = m * L^2 * omega             (angular momentum)
    H = p^2 / (2*m*L^2) - m*g*L*cos(q)

Reference:
    Greydanus, Dzamba & Cranmer, "Hamiltonian Neural Networks", NeurIPS 2019.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp


class HamiltonianNN(nn.Module):
    """
    MLP that maps (q, p) -> scalar Hamiltonian H.

    Architecture: input (2) -> hidden 3x64 tanh -> output (1).
    Time derivatives are obtained from H via Hamilton's equations,
    not predicted directly.
    """

    def __init__(self, hidden_size=64, num_hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, q, p):
        """
        Compute H(q, p).

        Args:
            q: generalized coordinate (N, 1)
            p: conjugate momentum (N, 1)

        Returns:
            H: scalar Hamiltonian (N, 1)
        """
        return self.network(torch.cat([q, p], dim=1))

    def time_derivative(self, t, q, p):
        """
        Hamilton's equations via autograd:
            dq/dt =  dH/dp
            dp/dt = -dH/dq

        The ``t`` argument is accepted for ODE-solver compatibility but is
        unused — the Hamiltonian is autonomous.

        Args:
            t: time (unused, for solver API compatibility)
            q, p: state tensors with requires_grad=True, each (N, 1)

        Returns:
            (dq_dt, dp_dt) tuple of (N, 1) tensors
        """
        H = self.forward(q, p)
        dH_dq = torch.autograd.grad(
            H, q, grad_outputs=torch.ones_like(H), create_graph=True,
        )[0]
        dH_dp = torch.autograd.grad(
            H, p, grad_outputs=torch.ones_like(H), create_graph=True,
        )[0]
        return dH_dp, -dH_dq

    # Convenience alias used by the old standalone script and app
    time_derivatives = time_derivative


def generate_pendulum_data(theta_0, omega_0, t_max, g=9.81, L=1.0, m=1.0,
                           n_points=1000):
    """
    Solve the pendulum ODE and return (q, p, dq/dt, dp/dt) training pairs.

    The HNN learns from these state-derivative pairs: given (q, p), predict
    the correct time derivatives.  This is equivalent to learning the vector
    field of the Hamiltonian system.

    Returns:
        q, p, dqdt, dpdt: numpy arrays of shape (n_points,)
        t_eval: time array (n_points,)
    """
    t_eval = np.linspace(0, t_max, n_points)

    sol = solve_ivp(
        lambda t, y: [y[1], -(g / L) * np.sin(y[0])],
        (0, t_max), [theta_0, omega_0],
        t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12,
    )

    theta = sol.y[0]
    omega = sol.y[1]

    q = theta
    p = m * L ** 2 * omega

    # True time derivatives from Hamilton's equations
    dqdt = p / (m * L ** 2)
    dpdt = -m * g * L * np.sin(q)

    return q, p, dqdt, dpdt, t_eval


def train_hnn(model, q_data, p_data, dqdt_data, dpdt_data,
              epochs=5000, lr=1e-3, batch_size=256, verbose=True):
    """
    Train the HNN to match true time derivatives.

    Loss = MSE(predicted dq/dt, true dq/dt) + MSE(predicted dp/dt, true dp/dt)

    The network never sees H directly — only the derivatives that H must
    produce via Hamilton's equations.

    Args:
        model: HamiltonianNN instance
        q_data, p_data: training states (numpy 1-D)
        dqdt_data, dpdt_data: training derivatives (numpy 1-D)
        epochs, lr, batch_size: training hyperparameters
        verbose: print progress

    Returns:
        loss_history: list of per-epoch losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6,
    )

    q_t = torch.tensor(q_data, dtype=torch.float32).unsqueeze(1)
    p_t = torch.tensor(p_data, dtype=torch.float32).unsqueeze(1)
    dqdt_t = torch.tensor(dqdt_data, dtype=torch.float32).unsqueeze(1)
    dpdt_t = torch.tensor(dpdt_data, dtype=torch.float32).unsqueeze(1)

    n_samples = len(q_data)
    loss_history = []

    if verbose:
        print("Training Hamiltonian Neural Network")
        print(f"  Training samples: {n_samples}")
        print(f"  Epochs: {epochs}")
        print("-" * 50)

    for epoch in range(epochs):
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
        loss_history.append(loss.item())

        if verbose and (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | Loss: {loss.item():.8f}")

    if verbose:
        print("-" * 50)
        print(f"Final loss: {loss_history[-1]:.8f}")

    return loss_history


def integrate_hnn(model, q0, p0, t_eval):
    """
    Integrate the HNN's learned dynamics forward using scipy RK45.

    The HNN defines a vector field (dq/dt, dp/dt) at every (q, p).
    We follow this field from (q0, p0) to produce a trajectory.

    Args:
        model: trained HamiltonianNN
        q0, p0: initial state scalars
        t_eval: array of evaluation times

    Returns:
        q_traj, p_traj: numpy arrays of shape (len(t_eval),)
    """
    model.eval()

    def hnn_rhs(t, state):
        q, p = state
        q_t = torch.tensor([[q]], dtype=torch.float32, requires_grad=True)
        p_t = torch.tensor([[p]], dtype=torch.float32, requires_grad=True)
        with torch.enable_grad():
            dqdt, dpdt = model.time_derivative(None, q_t, p_t)
        return [dqdt.item(), dpdt.item()]

    sol = solve_ivp(
        hnn_rhs, (t_eval[0], t_eval[-1]), [q0, p0],
        t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10,
    )
    return sol.y[0], sol.y[1]


def compute_hamiltonian(q, p, g=9.81, L=1.0, m=1.0):
    """
    True pendulum Hamiltonian: H = p^2/(2mL^2) - mgL*cos(q).

    Works on numpy arrays.
    """
    return p ** 2 / (2 * m * L ** 2) - m * g * L * np.cos(q)
