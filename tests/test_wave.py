"""
Tests for the 1D wave equation PINN.

Verifies:
    1. Forward pass shape
    2. Physics residual is non-zero before training
    3. wave_equation_residual function correctness
    4. IC functions produce correct shapes and boundary values
    5. Analytical solution correctness for sine IC
    6. Training reduces loss over 100 epochs
    7. Trained model matches analytical solution at time snapshots
    8. Boundary conditions satisfied
    9. Mode decomposition returns correct number of modes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest

from src.models.wave_pinn import (
    WavePINN, train_wave_pinn, wave_analytical,
    wave_ic_sine, wave_ic_plucked, wave_ic_gaussian,
    WAVE_IC_FUNCTIONS, wave_mode_decomposition, wave_energy,
)
from src.physics.equations import wave_equation_residual


# =========================================================================
# Shape and basic tests
# =========================================================================

def test_forward_shape():
    """WavePINN should map (N,1) x (N,1) -> (N,1)."""
    model = WavePINN()
    x = torch.rand(50, 1)
    t = torch.rand(50, 1)
    u = model(x, t)
    assert u.shape == (50, 1), f"Expected (50, 1), got {u.shape}"


def test_residual_nonzero_before_training():
    """PDE residual should be non-zero for an untrained network."""
    torch.manual_seed(0)
    model = WavePINN()
    x = torch.rand(100, 1)
    t = torch.rand(100, 1)
    residual = model.compute_residual(x, t, c=1.0)
    assert residual.sum().item() > 0


def test_wave_equation_residual_function():
    """Residual should be zero when d²u/dt² == c² * d²u/dx²."""
    d2u_dt2 = torch.tensor([[4.0], [9.0]])
    d2u_dx2 = torch.tensor([[4.0], [9.0]])
    c = 1.0
    res = wave_equation_residual(d2u_dt2, d2u_dx2, c)
    assert torch.allclose(res, torch.zeros(2, 1), atol=1e-6)

    # With c=2: d²u/dt² - 4*d²u/dx² should not be zero for the same inputs
    res2 = wave_equation_residual(d2u_dt2, d2u_dx2, c=2.0)
    assert not torch.allclose(res2, torch.zeros(2, 1), atol=1e-6)


# =========================================================================
# IC function tests
# =========================================================================

def test_ic_functions_exist():
    assert 'sine' in WAVE_IC_FUNCTIONS
    assert 'plucked' in WAVE_IC_FUNCTIONS
    assert 'gaussian' in WAVE_IC_FUNCTIONS


def test_ic_sine_boundary_zero():
    """sin(pi*0/L) = 0, sin(pi*L/L) = 0."""
    assert abs(wave_ic_sine(torch.zeros(1, 1)).item()) < 1e-6
    assert abs(wave_ic_sine(torch.ones(1, 1)).item()) < 1e-6


def test_ic_plucked_shape_and_peak():
    """Plucked string should peak at midpoint with value 1."""
    x = torch.tensor([[0.5]])
    assert abs(wave_ic_plucked(x).item() - 1.0) < 1e-6
    # Boundary values should be 0
    assert abs(wave_ic_plucked(torch.zeros(1, 1)).item()) < 1e-6
    assert abs(wave_ic_plucked(torch.ones(1, 1)).item()) < 1e-6


# =========================================================================
# Analytical solution tests
# =========================================================================

def test_analytical_sine_at_t0():
    """At t=0, solution should be sin(pi*x/L)."""
    x = np.linspace(0, 1, 100)
    t = np.zeros_like(x)
    u = wave_analytical(x, t, c=1.0, ic_type='sine')
    expected = np.sin(np.pi * x)
    assert np.allclose(u, expected, atol=1e-10)


def test_analytical_sine_oscillates():
    """The sine-mode solution should oscillate, not decay."""
    x = np.array([0.5])
    u0 = wave_analytical(x, np.array([0.0]), c=1.0, ic_type='sine')
    u_half = wave_analytical(x, np.array([0.5]), c=1.0, ic_type='sine')
    u_full = wave_analytical(x, np.array([1.0]), c=1.0, ic_type='sine')
    # At t=0.5: cos(pi*0.5) = 0  -> u should be ~0
    assert abs(u_half[0]) < 1e-10
    # At t=1.0: cos(pi*1.0) = -1  -> u should be -sin(pi*0.5) = -1
    assert abs(u_full[0] - (-1.0)) < 1e-10


def test_analytical_satisfies_bcs():
    """u(0,t) and u(L,t) should be zero for all t."""
    t = np.linspace(0, 2, 50)
    u_left = wave_analytical(np.zeros_like(t), t, c=1.0, ic_type='sine')
    u_right = wave_analytical(np.ones_like(t), t, c=1.0, ic_type='sine')
    assert np.allclose(u_left, 0, atol=1e-10)
    assert np.allclose(u_right, 0, atol=1e-10)


# =========================================================================
# Mode decomposition
# =========================================================================

def test_mode_decomposition_count():
    """Should return exactly n_modes modes."""
    x = np.linspace(0, 1, 50)
    t = np.zeros_like(x)
    modes = wave_mode_decomposition(x, t, ic_type='plucked', n_modes=5)
    assert len(modes) == 5
    for n, b_n, u_mode in modes:
        assert u_mode.shape == x.shape


def test_mode_superposition_equals_analytical():
    """Sum of modes should match full analytical solution."""
    x = np.linspace(0, 1, 100)
    t_val = 0.3
    t = np.full_like(x, t_val)
    u_full = wave_analytical(x, t, c=1.0, ic_type='plucked', n_terms=20)
    modes = wave_mode_decomposition(x, t, c=1.0, ic_type='plucked', n_modes=20)
    u_sum = sum(m[2] for m in modes)
    assert np.allclose(u_full, u_sum, atol=1e-6)


# =========================================================================
# Energy
# =========================================================================

def test_wave_energy_positive():
    """Energy of a non-trivial state should be positive."""
    x = np.linspace(0, 1, 100)
    dx = x[1] - x[0]
    du_dt = np.sin(np.pi * x)
    du_dx = np.pi * np.cos(np.pi * x)
    E = wave_energy(x, np.zeros_like(x), du_dt, du_dx, c=1.0, dx=dx)
    assert E > 0


# =========================================================================
# Training tests
# =========================================================================

def test_training_reduces_loss_100_epochs():
    """Loss should decrease over 100 epochs."""
    torch.manual_seed(42)
    _, losses = train_wave_pinn(
        c=1.0, epochs=100, n_interior=500, n_bc=50, n_ic=50,
        verbose=False,
    )
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


@pytest.fixture(scope="module")
def trained_wave_model():
    """Train wave PINN once and reuse."""
    torch.manual_seed(42)
    model, losses = train_wave_pinn(
        c=1.0, L_string=1.0, t_max=0.5, ic_type='sine',
        epochs=5000, n_interior=2000, n_bc=200, n_ic=200,
        verbose=False,
    )
    return model, losses


def test_training_converges(trained_wave_model):
    _, losses = trained_wave_model
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.01, f"Final loss too high: {losses[-1]:.6f}"


def test_matches_analytical_at_snapshots(trained_wave_model):
    """PINN should match analytical within 0.15 at key time snapshots."""
    model, _ = trained_wave_model
    model.eval()
    x = np.linspace(0, 1, 100)
    for t_val in [0.0, 0.1, 0.25, 0.5]:
        t = np.full_like(x, t_val)
        u_exact = wave_analytical(x, t, c=1.0, ic_type='sine')
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        t_t = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            u_pinn = model(x_t, t_t).numpy().squeeze()
        max_err = np.max(np.abs(u_pinn - u_exact))
        assert max_err < 0.15, \
            f"Max error at t={t_val}: {max_err:.4f} (threshold: 0.15)"


def test_boundary_conditions_satisfied(trained_wave_model):
    model, _ = trained_wave_model
    model.eval()
    t = torch.linspace(0, 0.5, 50).unsqueeze(1)
    with torch.no_grad():
        u_left = model(torch.zeros(50, 1), t).numpy().squeeze()
        u_right = model(torch.ones(50, 1), t).numpy().squeeze()
    assert np.max(np.abs(u_left)) < 0.05, \
        f"Left BC violated: max |u(0,t)| = {np.max(np.abs(u_left)):.4f}"
    assert np.max(np.abs(u_right)) < 0.05, \
        f"Right BC violated: max |u(1,t)| = {np.max(np.abs(u_right)):.4f}"
