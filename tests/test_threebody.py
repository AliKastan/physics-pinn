"""
Tests for the three-body problem PINN.

Verifies:
    1. Forward pass shape (t -> 12 outputs)
    2. All preset configurations produce valid initial conditions
    3. Classical solver runs without error
    4. Conservation quantities are computed correctly
    5. Training reduces loss over 100 epochs
    6. Trained model approximately satisfies initial conditions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest

from src.models.threebody_pinn import (
    ThreeBodyPINN,
    figure_eight_ics, lagrange_triangle_ics, pythagorean_ics,
    sun_earth_moon_ics, PRESETS,
    solve_threebody, compute_energy, compute_angular_momentum,
    compute_center_of_mass,
    physics_loss, ic_loss, train_threebody,
)


# =========================================================================
# Shape tests
# =========================================================================

def test_forward_shape():
    model = ThreeBodyPINN()
    t = torch.rand(50, 1)
    out = model(t)
    assert out.shape == (50, 12), f"Expected (50, 12), got {out.shape}"


def test_forward_shape_custom_arch():
    model = ThreeBodyPINN(hidden_size=64, num_hidden_layers=3)
    t = torch.rand(20, 1)
    out = model(t)
    assert out.shape == (20, 12)


# =========================================================================
# Preset tests
# =========================================================================

def test_all_presets_return_valid_state():
    for name, fn in PRESETS.items():
        state0, masses, period = fn()
        assert len(state0) == 12, f"{name}: state should have 12 elements"
        assert len(masses) == 3, f"{name}: should have 3 masses"
        assert period > 0, f"{name}: period should be positive"
        assert all(m > 0 for m in masses), f"{name}: masses should be positive"


def test_figure_eight_zero_momentum():
    """Figure-eight has zero total momentum (by construction)."""
    state0, masses, _ = figure_eight_ics()
    m1, m2, m3 = masses
    px = m1 * state0[6] + m2 * state0[8] + m3 * state0[10]
    py = m1 * state0[7] + m2 * state0[9] + m3 * state0[11]
    assert abs(px) < 1e-10, f"Px should be ~0, got {px}"
    assert abs(py) < 1e-10, f"Py should be ~0, got {py}"


# =========================================================================
# Classical solver tests
# =========================================================================

def test_classical_solver_runs():
    """DOP853 solver should complete without errors."""
    state0, masses, _ = figure_eight_ics()
    t, states = solve_threebody(state0, masses, t_max=0.5, n_points=100)
    assert states.shape == (100, 12)
    assert t.shape == (100,)


def test_classical_energy_conserved():
    """Energy should be conserved to <1e-6 by the classical solver."""
    state0, masses, _ = figure_eight_ics()
    _, states = solve_threebody(state0, masses, t_max=1.0, n_points=500)
    E = compute_energy(states, masses)
    E0 = E[0]
    max_drift = np.max(np.abs((E - E0) / (np.abs(E0) + 1e-16)))
    assert max_drift < 1e-6, f"Classical energy drift: {max_drift:.2e}"


# =========================================================================
# Conservation quantity tests
# =========================================================================

def test_angular_momentum_shape():
    state0, masses, _ = figure_eight_ics()
    _, states = solve_threebody(state0, masses, t_max=0.5, n_points=50)
    L = compute_angular_momentum(states, masses)
    assert L.shape == (50,)


def test_center_of_mass_shape():
    state0, masses, _ = figure_eight_ics()
    _, states = solve_threebody(state0, masses, t_max=0.5, n_points=50)
    cx, cy = compute_center_of_mass(states, masses)
    assert cx.shape == (50,)
    assert cy.shape == (50,)


# =========================================================================
# Physics loss test
# =========================================================================

def test_physics_loss_nonzero():
    """Untrained network should have non-zero physics residual."""
    torch.manual_seed(0)
    model = ThreeBodyPINN(hidden_size=64, num_hidden_layers=3)
    state0, masses, _ = figure_eight_ics()
    t = torch.rand(50, 1)
    loss = physics_loss(model, t, masses)
    assert loss.item() > 0


# =========================================================================
# Training tests
# =========================================================================

def test_training_reduces_loss_100_epochs():
    """Loss should decrease over 100 epochs."""
    torch.manual_seed(42)
    state0, masses, _ = figure_eight_ics()
    model, losses = train_threebody(
        state0, masses, t_max=0.5,
        n_collocation=200, epochs=100, lr=1e-3,
        ic_weight=100.0, conservation_weight=0.0,
        hidden_size=64, num_hidden_layers=3,
        verbose=False,
    )
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


@pytest.fixture(scope="module")
def trained_model():
    """Train a three-body PINN once for convergence tests."""
    torch.manual_seed(42)
    state0, masses, _ = figure_eight_ics()
    model, losses = train_threebody(
        state0, masses, t_max=0.5,
        n_collocation=500, epochs=5000, lr=1e-3,
        ic_weight=100.0, conservation_weight=0.5,
        hidden_size=128, num_hidden_layers=4,
        verbose=False,
    )
    return model, losses, state0, masses


def test_training_converges(trained_model):
    _, losses, _, _ = trained_model
    assert losses[-1] < losses[0]


def test_initial_conditions_satisfied(trained_model):
    """PINN should approximately match ICs at t=0."""
    model, _, state0, _ = trained_model
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1)).numpy().squeeze()
    state0_np = np.array(state0)
    max_err = np.max(np.abs(out - state0_np))
    assert max_err < 0.5, \
        f"IC error too large: max |error| = {max_err:.4f} (threshold: 0.5)"
