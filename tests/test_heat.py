"""
Tests for the 1D heat equation PINN.

Verifies:
    1. Forward pass shape is correct: (x, t) -> u
    2. Physics residual is non-zero before training
    3. Analytical solution is correct for sine IC
    4. Training reduces loss over 100 epochs
    5. Trained model matches analytical solution at key time snapshots
    6. Boundary conditions are approximately satisfied
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest

from src.models.heat_pinn import (
    HeatPINN, train_heat_pinn, heat_analytical,
    ic_sine, ic_step, ic_gaussian, IC_FUNCTIONS,
)
from src.physics.equations import heat_equation_residual


# =========================================================================
# Shape and basic tests
# =========================================================================

def test_forward_shape():
    """HeatPINN should map (N, 1) x (N, 1) -> (N, 1)."""
    model = HeatPINN()
    x = torch.rand(50, 1)
    t = torch.rand(50, 1)
    u = model(x, t)
    assert u.shape == (50, 1), f"Expected (50, 1), got {u.shape}"


def test_residual_nonzero_before_training():
    """PDE residual should be non-zero for an untrained network."""
    torch.manual_seed(0)
    model = HeatPINN()
    x = torch.rand(100, 1)
    t = torch.rand(100, 1)
    residual = model.compute_residual(x, t, alpha=0.01)
    assert residual.sum().item() > 0, "Residual should be non-zero"


def test_heat_equation_residual_function():
    """The residual function should compute du/dt - alpha*d²u/dx² correctly."""
    # For a known exact solution, the residual should be close to zero
    du_dt = torch.tensor([[1.0], [2.0]])
    d2u_dx2 = torch.tensor([[100.0], [200.0]])
    alpha = 0.01
    res = heat_equation_residual(du_dt, d2u_dx2, alpha)
    # du/dt - alpha*d²u/dx² = 1 - 0.01*100 = 0, 2 - 0.01*200 = 0
    assert torch.allclose(res, torch.zeros(2, 1), atol=1e-6)


# =========================================================================
# IC function tests
# =========================================================================

def test_ic_functions_exist():
    """All three IC types should be available."""
    assert 'sine' in IC_FUNCTIONS
    assert 'step' in IC_FUNCTIONS
    assert 'gaussian' in IC_FUNCTIONS


def test_ic_sine_shape():
    x = torch.rand(50, 1)
    u = ic_sine(x)
    assert u.shape == (50, 1)


def test_ic_sine_boundary_values():
    """sin(pi*0) = 0, sin(pi*1) = 0."""
    u0 = ic_sine(torch.zeros(1, 1))
    u1 = ic_sine(torch.ones(1, 1))
    assert abs(u0.item()) < 1e-6
    assert abs(u1.item()) < 1e-6


# =========================================================================
# Analytical solution tests
# =========================================================================

def test_analytical_sine_ic_at_t0():
    """At t=0, analytical solution should equal sin(pi*x)."""
    x = np.linspace(0, 1, 100)
    t = np.zeros_like(x)
    u = heat_analytical(x, t, alpha=0.01, ic_type='sine')
    expected = np.sin(np.pi * x)
    assert np.allclose(u, expected, atol=1e-10)


def test_analytical_sine_decays():
    """The sine IC solution should decay exponentially."""
    x = np.array([0.5])
    u0 = heat_analytical(x, np.array([0.0]), alpha=0.01, ic_type='sine')
    u1 = heat_analytical(x, np.array([1.0]), alpha=0.01, ic_type='sine')
    assert u1[0] < u0[0], "Solution should decay over time"


def test_analytical_satisfies_bc():
    """Analytical solution should be zero at boundaries for all t."""
    t = np.linspace(0, 1, 50)
    u_left = heat_analytical(np.zeros_like(t), t, alpha=0.01, ic_type='sine')
    u_right = heat_analytical(np.ones_like(t), t, alpha=0.01, ic_type='sine')
    assert np.allclose(u_left, 0, atol=1e-10)
    assert np.allclose(u_right, 0, atol=1e-10)


# =========================================================================
# Training tests
# =========================================================================

def test_training_reduces_loss_100_epochs():
    """Training should reduce loss over 100 epochs."""
    torch.manual_seed(42)
    model, losses = train_heat_pinn(
        alpha=0.01, epochs=100, n_interior=500, n_bc=50, n_ic=50,
        verbose=False,
    )
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# =========================================================================
# Full training convergence (module-scoped fixture)
# =========================================================================

@pytest.fixture(scope="module")
def trained_heat_model():
    """Train heat PINN once and reuse across tests."""
    torch.manual_seed(42)
    model, losses = train_heat_pinn(
        alpha=0.01, L_rod=1.0, t_max=1.0,
        ic_type='sine',
        epochs=5000, n_interior=2000, n_bc=200, n_ic=200,
        verbose=False,
    )
    return model, losses


def test_training_converges(trained_heat_model):
    """Training loss should decrease substantially."""
    _, losses = trained_heat_model
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.01, f"Final loss too high: {losses[-1]:.6f}"


def test_matches_analytical_at_snapshots(trained_heat_model):
    """PINN should match analytical solution within 0.1 at t=0.1, 0.5, 1.0."""
    model, _ = trained_heat_model
    model.eval()

    x = np.linspace(0, 1, 100)
    for t_val in [0.1, 0.5, 1.0]:
        t = np.full_like(x, t_val)
        u_exact = heat_analytical(x, t, alpha=0.01, ic_type='sine')

        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        t_t = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            u_pinn = model(x_t, t_t).numpy().squeeze()

        max_err = np.max(np.abs(u_pinn - u_exact))
        assert max_err < 0.1, \
            f"Max error at t={t_val}: {max_err:.4f} (threshold: 0.1)"


def test_boundary_conditions_satisfied(trained_heat_model):
    """u(0, t) and u(1, t) should be close to zero."""
    model, _ = trained_heat_model
    model.eval()

    t = torch.linspace(0, 1, 50).unsqueeze(1)
    with torch.no_grad():
        u_left = model(torch.zeros(50, 1), t).numpy().squeeze()
        u_right = model(torch.ones(50, 1), t).numpy().squeeze()

    assert np.max(np.abs(u_left)) < 0.05, \
        f"Left BC violated: max |u(0,t)| = {np.max(np.abs(u_left)):.4f}"
    assert np.max(np.abs(u_right)) < 0.05, \
        f"Right BC violated: max |u(1,t)| = {np.max(np.abs(u_right)):.4f}"
