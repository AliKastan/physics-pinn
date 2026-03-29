"""
Tests for the orbital mechanics PINN.

Verifies:
    1. Model forward pass shapes are correct
    2. Physics residual is non-zero before training
    3. Training reduces loss over 100 epochs
    4. Angular momentum is approximately conserved
    5. Energy is approximately conserved
    6. Initial conditions are matched
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
from src.models.orbital_pinn import PINNOrbital
from src.training.trainer import Trainer
from src.training.losses import orbital_ic_loss
from src.utils.validation import setup_orbital_ics
from src.utils.metrics import compute_energy_orbital, compute_angular_momentum


@pytest.fixture(scope="module")
def trained_model():
    """Train an orbital PINN once and reuse across tests."""
    torch.manual_seed(42)
    GM = 1.0
    x0, y0, vx0, vy0, period = setup_orbital_ics(0.2, GM)
    t_max = 0.5 * period

    model = PINNOrbital()
    config = {
        "epochs": 4000,
        "lr": 1e-3,
        "n_collocation": 500,
        "ic_weight": 50.0,
        "t_max": t_max,
    }
    x0_t = torch.tensor(x0, dtype=torch.float32)
    y0_t = torch.tensor(y0, dtype=torch.float32)
    vx0_t = torch.tensor(vx0, dtype=torch.float32)
    vy0_t = torch.tensor(vy0, dtype=torch.float32)

    trainer = Trainer(model, config, physics_params={"GM": GM})
    ic_fn = lambda: orbital_ic_loss(model, x0_t, y0_t, vx0_t, vy0_t)
    loss_history, _ = trainer.train(ic_fn, verbose=False)

    ics = (x0, y0, vx0, vy0)
    return model, loss_history, ics, t_max


def test_forward_pass_shape():
    """Model forward pass should produce correct output shape."""
    model = PINNOrbital()
    t = torch.rand(50, 1)
    output = model(t)
    assert output.shape == (50, 4), f"Expected (50, 4), got {output.shape}"


def test_residual_nonzero_before_training():
    """Physics residual should be non-zero for an untrained network."""
    torch.manual_seed(0)
    model = PINNOrbital()
    t = torch.rand(100, 1)
    residual = model.compute_residual(t, GM=1.0)
    assert residual.sum().item() > 0, "Residual should be non-zero before training"


def test_training_reduces_loss():
    """Training should reduce loss over 100 epochs."""
    torch.manual_seed(42)
    GM = 1.0
    x0, y0, vx0, vy0, period = setup_orbital_ics(0.2, GM)
    t_max = 0.5 * period

    model = PINNOrbital()
    config = {
        "epochs": 100,
        "lr": 1e-3,
        "n_collocation": 300,
        "ic_weight": 50.0,
        "t_max": t_max,
    }
    x0_t = torch.tensor(x0, dtype=torch.float32)
    y0_t = torch.tensor(y0, dtype=torch.float32)
    vx0_t = torch.tensor(vx0, dtype=torch.float32)
    vy0_t = torch.tensor(vy0, dtype=torch.float32)

    trainer = Trainer(model, config, physics_params={"GM": GM})
    ic_fn = lambda: orbital_ic_loss(model, x0_t, y0_t, vx0_t, vy0_t)
    loss_history, _ = trainer.train(ic_fn, verbose=False)
    assert loss_history[-1] < loss_history[0], \
        f"Loss did not decrease: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"


def test_training_converges(trained_model):
    """Training loss should decrease."""
    _, loss_history, _, _ = trained_model
    assert loss_history[-1] < loss_history[0], \
        f"Loss did not decrease: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"


def test_angular_momentum_conservation(trained_model):
    """Angular momentum should be approximately conserved."""
    model, _, ics, t_max = trained_model
    x0, y0, vx0, vy0 = ics
    t_eval = np.linspace(0, t_max, 500)

    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        output = model(t_tensor).numpy()
    x, y, vx, vy = output[:, 0], output[:, 1], output[:, 2], output[:, 3]

    L = compute_angular_momentum(x, y, vx, vy)
    L0 = x0 * vy0 - y0 * vx0
    max_drift = np.max(np.abs((L - L0) / (np.abs(L0) + 1e-16)))
    assert max_drift < 0.8, \
        f"Angular momentum drift too large: max |dL/L0| = {max_drift:.4f} (threshold: 0.8)"


def test_energy_conservation(trained_model):
    """Energy should be approximately conserved."""
    model, _, ics, t_max = trained_model
    t_eval = np.linspace(0, t_max, 500)

    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        output = model(t_tensor).numpy()
    x, y, vx, vy = output[:, 0], output[:, 1], output[:, 2], output[:, 3]

    E = compute_energy_orbital(x, y, vx, vy, GM=1.0)
    E0 = E[0]
    max_drift = np.max(np.abs((E - E0) / (np.abs(E0) + 1e-16)))
    assert max_drift < 1.0, \
        f"Energy drift too large: max |dE/E0| = {max_drift:.4f} (threshold: 1.0)"


def test_initial_conditions(trained_model):
    """PINN should approximately match initial conditions."""
    model, _, ics, _ = trained_model
    x0, y0, vx0, vy0 = ics

    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 1)).numpy().squeeze()

    assert abs(output[0] - x0) < 0.2, f"x(0) mismatch: {output[0]:.4f} vs {x0:.4f}"
    assert abs(output[1] - y0) < 0.2, f"y(0) mismatch: {output[1]:.4f} vs {y0:.4f}"
    assert abs(output[2] - vx0) < 0.2, f"vx(0) mismatch: {output[2]:.4f} vs {vx0:.4f}"
    assert abs(output[3] - vy0) < 0.5, f"vy(0) mismatch: {output[3]:.4f} vs {vy0:.4f}"
