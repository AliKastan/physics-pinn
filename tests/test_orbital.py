"""
Tests for the orbital mechanics PINN.

Verifies that:
    1. The PINN model can be trained without errors
    2. Angular momentum is approximately conserved
    3. Energy is approximately conserved
    4. Initial conditions are matched
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
from pinn_orbital import (
    PINNOrbital, train_pinn_orbital, setup_orbital_ics,
    compute_energy, compute_angular_momentum
)


@pytest.fixture(scope="module")
def trained_model():
    """Train an orbital PINN once and reuse across tests."""
    torch.manual_seed(42)
    model, loss_history, ics, t_max = train_pinn_orbital(
        eccentricity=0.2,  # mild eccentricity for easier training
        GM=1.0,
        n_orbits=0.5,  # half orbit for faster test
        n_collocation=500,
        epochs=4000,
        lr=1e-3,
        ic_weight=50.0
    )
    return model, loss_history, ics, t_max


def test_training_converges(trained_model):
    """Training loss should decrease."""
    _, loss_history, _, _ = trained_model
    assert loss_history[-1] < loss_history[0], \
        f"Loss did not decrease: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"


def test_angular_momentum_conservation(trained_model):
    """
    Angular momentum L = x*vy - y*vx should be approximately conserved.

    For a central force, angular momentum is exactly conserved.
    The PINN should preserve it to within 80% relative drift.
    """
    model, _, ics, t_max = trained_model
    x0, y0, vx0, vy0 = ics
    t_eval = np.linspace(0, t_max, 500)

    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        output = model(t_tensor).numpy()
    x, y, vx, vy = output[:, 0], output[:, 1], output[:, 2], output[:, 3]

    L = compute_angular_momentum(x, y, vx, vy)
    L0 = x0 * vy0 - y0 * vx0  # true initial angular momentum

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

    E = compute_energy(x, y, vx, vy, GM=1.0)
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
