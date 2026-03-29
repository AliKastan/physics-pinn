"""
Tests for the pendulum PINN.

Verifies:
    1. Model forward pass shapes are correct
    2. Physics residual is non-zero before training
    3. Training reduces loss over 100 epochs
    4. Energy is approximately conserved
    5. Initial conditions are matched
    6. Solution is reasonably accurate
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
from src.models.pendulum_pinn import PINNPendulum
from src.training.trainer import Trainer
from src.training.losses import pendulum_ic_loss
from src.utils.validation import solve_pendulum_ode
from src.utils.metrics import compute_energy_pendulum


G = 9.81
L = 1.0
M = 1.0
THETA_0 = np.pi / 6  # 30 degrees
OMEGA_0 = 0.0
T_MAX = 5.0


@pytest.fixture(scope="module")
def trained_model():
    """Train a pendulum PINN once and reuse across tests."""
    torch.manual_seed(42)
    model = PINNPendulum()
    config = {
        "epochs": 3000,
        "lr": 1e-3,
        "n_collocation": 300,
        "ic_weight": 20.0,
        "t_max": T_MAX,
    }
    trainer = Trainer(model, config, physics_params={"g": G, "L": L})
    ic_fn = lambda: pendulum_ic_loss(model, THETA_0, OMEGA_0)
    loss_history, _ = trainer.train(ic_fn, verbose=False)
    return model, loss_history


def test_forward_pass_shape():
    """Model forward pass should produce correct output shape."""
    model = PINNPendulum()
    t = torch.rand(50, 1)
    output = model(t)
    assert output.shape == (50, 2), f"Expected (50, 2), got {output.shape}"


def test_residual_nonzero_before_training():
    """Physics residual should be non-zero for an untrained network."""
    torch.manual_seed(0)
    model = PINNPendulum()
    t = torch.rand(100, 1)
    residual = model.compute_residual(t, g=G, L=L)
    assert residual.sum().item() > 0, "Residual should be non-zero before training"


def test_training_reduces_loss():
    """Training should reduce loss over 100 epochs."""
    torch.manual_seed(42)
    model = PINNPendulum()
    config = {
        "epochs": 100,
        "lr": 1e-3,
        "n_collocation": 200,
        "ic_weight": 20.0,
        "t_max": T_MAX,
    }
    trainer = Trainer(model, config, physics_params={"g": G, "L": L})
    ic_fn = lambda: pendulum_ic_loss(model, THETA_0, OMEGA_0)
    loss_history, _ = trainer.train(ic_fn, verbose=False)
    assert loss_history[-1] < loss_history[0], \
        f"Loss did not decrease: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"


def test_training_converges(trained_model):
    """Training loss should decrease from start to end."""
    _, loss_history = trained_model
    assert loss_history[-1] < loss_history[0], \
        f"Loss did not decrease: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"


def test_energy_conservation(trained_model):
    """Energy should be approximately constant over the trajectory."""
    model, _ = trained_model
    t_eval = np.linspace(0, T_MAX, 500)

    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        output = model(t_tensor).numpy()
    theta_pinn = output[:, 0]
    omega_pinn = output[:, 1]

    E = compute_energy_pendulum(theta_pinn, omega_pinn, G, L, M)
    E0 = E[0]
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
    """PINN trajectory should be within 0.8 rad of classical solution."""
    model, _ = trained_model
    t_eval = np.linspace(0, T_MAX, 500)
    _, theta_ode, _ = solve_pendulum_ode(THETA_0, OMEGA_0, (0, T_MAX), t_eval, g=G, L=L)

    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        output = model(t_tensor).numpy()
    theta_pinn = output[:, 0]

    max_error = np.max(np.abs(theta_pinn - theta_ode))
    assert max_error < 0.8, \
        f"Trajectory error too large: max |dtheta| = {max_error:.4f} rad (threshold: 0.8)"
