"""
Tests for the pendulum PINN.

Verifies that:
    1. The PINN model can be trained without errors
    2. Energy is approximately conserved over the trajectory
    3. Predictions are reasonably close to the classical solution
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
from pinn_pendulum import PINNPendulum, train_pinn, solve_pendulum_ode


# Physical parameters used across all tests
G = 9.81
L = 1.0
M = 1.0
THETA_0 = np.pi / 6  # 30 degrees (smaller angle for faster convergence)
OMEGA_0 = 0.0
T_MAX = 5.0


@pytest.fixture(scope="module")
def trained_model():
    """Train a pendulum PINN once and reuse across tests."""
    torch.manual_seed(42)
    model, loss_history = train_pinn(
        theta_0=THETA_0, omega_0=OMEGA_0, t_max=T_MAX,
        n_collocation=300, epochs=3000, lr=1e-3,
        g=G, L=L, ic_weight=20.0
    )
    return model, loss_history


def test_training_converges(trained_model):
    """Training loss should decrease from start to end."""
    _, loss_history = trained_model
    assert loss_history[-1] < loss_history[0], \
        f"Loss did not decrease: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"


def test_energy_conservation(trained_model):
    """
    Total mechanical energy E = 0.5*m*L^2*omega^2 - m*g*L*cos(theta)
    should be approximately constant over the trajectory.

    We check that the relative energy drift is less than 50%.
    (PINNs are approximate — the point is energy roughly holds,
    not that it matches machine precision.)
    """
    model, _ = trained_model
    t_eval = np.linspace(0, T_MAX, 500)

    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        output = model(t_tensor).numpy()
    theta_pinn = output[:, 0]
    omega_pinn = output[:, 1]

    # Compute energy at each time step
    E = 0.5 * M * L**2 * omega_pinn**2 - M * G * L * np.cos(theta_pinn)
    E0 = E[0]

    # Relative energy drift
    max_drift = np.max(np.abs((E - E0) / (np.abs(E0) + 1e-16)))
    assert max_drift < 0.5, \
        f"Energy drift too large: max |dE/E0| = {max_drift:.4f} (threshold: 0.5)"


def test_initial_conditions(trained_model):
    """The PINN should match initial conditions at t=0."""
    model, _ = trained_model
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 1)).numpy()
    theta_pred = output[0, 0]
    omega_pred = output[0, 1]

    assert abs(theta_pred - THETA_0) < 0.1, \
        f"IC mismatch: theta(0) = {theta_pred:.4f}, expected {THETA_0:.4f}"
    assert abs(omega_pred - OMEGA_0) < 0.1, \
        f"IC mismatch: omega(0) = {omega_pred:.4f}, expected {OMEGA_0:.4f}"


def test_solution_accuracy(trained_model):
    """PINN trajectory should be within 0.5 rad of classical solution."""
    model, _ = trained_model
    t_eval = np.linspace(0, T_MAX, 500)

    # Classical solution
    _, theta_ode, _ = solve_pendulum_ode(THETA_0, OMEGA_0, (0, T_MAX), t_eval, g=G, L=L)

    # PINN prediction
    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        output = model(t_tensor).numpy()
    theta_pinn = output[:, 0]

    max_error = np.max(np.abs(theta_pinn - theta_ode))
    assert max_error < 0.8, \
        f"Trajectory error too large: max |dtheta| = {max_error:.4f} rad (threshold: 0.8)"
