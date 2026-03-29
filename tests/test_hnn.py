"""
Tests for the Hamiltonian Neural Network module.

Verifies:
    1. Forward pass shape is correct (q, p) -> scalar H
    2. time_derivative returns correct shapes
    3. Physics residual (derivative mismatch) is non-zero before training
    4. Training reduces loss over 100 epochs
    5. Energy is well-conserved after full training
    6. Learned dynamics match true derivatives
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest

from src.models.hnn import (
    HamiltonianNN, generate_pendulum_data, train_hnn,
    integrate_hnn, compute_hamiltonian,
)
from src.training.losses import hamiltonian_loss, energy_conservation_loss


G = 9.81
L = 1.0
M = 1.0
THETA_0 = np.pi / 4
OMEGA_0 = 0.0
T_MAX = 5.0


@pytest.fixture(scope="module")
def training_data():
    """Generate pendulum training data once."""
    q, p, dqdt, dpdt, t = generate_pendulum_data(
        THETA_0, OMEGA_0, T_MAX, G, L, M, n_points=500,
    )
    return q, p, dqdt, dpdt, t


@pytest.fixture(scope="module")
def trained_hnn(training_data):
    """Train an HNN once and reuse across tests."""
    q, p, dqdt, dpdt, _ = training_data
    torch.manual_seed(42)
    model = HamiltonianNN()
    losses = train_hnn(model, q, p, dqdt, dpdt,
                       epochs=3000, lr=1e-3, batch_size=128, verbose=False)
    return model, losses


# ---- Shape tests ----

def test_forward_shape():
    """H(q, p) should return (N, 1) for (N, 1) inputs."""
    model = HamiltonianNN()
    q = torch.randn(50, 1)
    p = torch.randn(50, 1)
    H = model(q, p)
    assert H.shape == (50, 1), f"Expected (50, 1), got {H.shape}"


def test_time_derivative_shape():
    """time_derivative should return two (N, 1) tensors."""
    model = HamiltonianNN()
    q = torch.randn(50, 1, requires_grad=True)
    p = torch.randn(50, 1, requires_grad=True)
    dqdt, dpdt = model.time_derivative(None, q, p)
    assert dqdt.shape == (50, 1), f"dqdt shape: {dqdt.shape}"
    assert dpdt.shape == (50, 1), f"dpdt shape: {dpdt.shape}"


# ---- Pre-training test ----

def test_derivative_mismatch_before_training(training_data):
    """Predicted derivatives should differ from true ones before training."""
    q, p, dqdt_true, dpdt_true, _ = training_data
    torch.manual_seed(0)
    model = HamiltonianNN()

    q_t = torch.tensor(q[:50], dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    p_t = torch.tensor(p[:50], dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    dqdt_t = torch.tensor(dqdt_true[:50], dtype=torch.float32).unsqueeze(1)
    dpdt_t = torch.tensor(dpdt_true[:50], dtype=torch.float32).unsqueeze(1)

    loss = hamiltonian_loss(model, q_t, p_t, dqdt_t, dpdt_t)
    assert loss.item() > 1e-3, "Loss should be large before training"


# ---- Training tests ----

def test_training_reduces_loss_100_epochs(training_data):
    """Loss should decrease over 100 epochs."""
    q, p, dqdt, dpdt, _ = training_data
    torch.manual_seed(42)
    model = HamiltonianNN()
    losses = train_hnn(model, q, p, dqdt, dpdt,
                       epochs=100, lr=1e-3, batch_size=128, verbose=False)
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"


def test_training_converges(trained_hnn):
    """Full training should reduce loss substantially."""
    _, losses = trained_hnn
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    assert losses[-1] < 1e-3, \
        f"Final loss too high: {losses[-1]:.6f}"


# ---- Post-training physics tests ----

def test_energy_conservation(trained_hnn):
    """
    HNN trajectory should conserve energy much better than 10% drift.
    """
    model, _ = trained_hnn
    q0 = THETA_0
    p0 = M * L ** 2 * OMEGA_0
    t_eval = np.linspace(0, T_MAX, 500)

    q_traj, p_traj = integrate_hnn(model, q0, p0, t_eval)
    E = compute_hamiltonian(q_traj, p_traj, G, L, M)
    E0 = E[0]
    max_drift = np.max(np.abs((E - E0) / (np.abs(E0) + 1e-16)))
    assert max_drift < 0.10, \
        f"Energy drift too large for HNN: {max_drift:.4f} (threshold: 0.10)"


def test_learned_derivatives_accuracy(trained_hnn, training_data):
    """Predicted derivatives should closely match true derivatives after training."""
    model, _ = trained_hnn
    q, p, dqdt_true, dpdt_true, _ = training_data

    q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(1).requires_grad_(True)

    dqdt_pred, dpdt_pred = model.time_derivative(None, q_t, p_t)
    dqdt_pred = dqdt_pred.detach().numpy().squeeze()
    dpdt_pred = dpdt_pred.detach().numpy().squeeze()

    dqdt_err = np.mean((dqdt_pred - dqdt_true) ** 2)
    dpdt_err = np.mean((dpdt_pred - dpdt_true) ** 2)
    assert dqdt_err < 0.01, f"dq/dt MSE too high: {dqdt_err:.6f}"
    assert dpdt_err < 0.01, f"dp/dt MSE too high: {dpdt_err:.6f}"


def test_energy_conservation_loss_function(trained_hnn, training_data):
    """energy_conservation_loss should be small for a trajectory from the trained HNN."""
    model, _ = trained_hnn
    q, p, _, _, _ = training_data

    q_t = torch.tensor(q[:100], dtype=torch.float32).unsqueeze(1)
    p_t = torch.tensor(p[:100], dtype=torch.float32).unsqueeze(1)

    ec_loss = energy_conservation_loss(model, q_t, p_t)
    assert ec_loss.item() < 1.0, \
        f"Energy conservation loss too high: {ec_loss.item():.4f}"
