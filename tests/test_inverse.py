"""
Tests for the inverse problem PINN.

Verifies that:
    1. The inverse PINN can be trained without errors
    2. Inferred g is within 5% of the true value (9.81)
    3. Inferred L is within 5% of the true value (1.0)
    4. The g/L ratio converges to within 2% of true value
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
from pinn_inverse import PINNPendulum, generate_noisy_data, train_inverse_pinn


# True values
G_TRUE = 9.81
L_TRUE = 1.0
THETA_0 = np.pi / 4
OMEGA_0 = 0.0
T_MAX = 10.0


@pytest.fixture(scope="module")
def trained_inverse():
    """Train inverse PINN once and reuse across tests."""
    torch.manual_seed(42)
    np.random.seed(42)

    t_obs, theta_obs, omega_obs, _, _, _ = generate_noisy_data(
        THETA_0, OMEGA_0, T_MAX, G_TRUE, L_TRUE,
        n_observations=80, noise_std=0.03
    )

    model, g_param, L_param, g_history, L_history, loss_history = \
        train_inverse_pinn(
            THETA_0, OMEGA_0, T_MAX, t_obs, theta_obs, omega_obs,
            g_init=5.0, L_init=2.0,
            n_collocation=500, epochs=15000,
            lr_network=1e-3, lr_physics=5e-3,
            ic_weight=20.0, data_weight=10.0
        )
    return model, g_param, L_param, g_history, L_history, loss_history


def test_training_converges(trained_inverse):
    """Training loss should decrease from start to end."""
    _, _, _, _, _, loss_history = trained_inverse
    # Compare average of first 100 epochs vs last 100 epochs
    early_loss = np.mean(loss_history[:100])
    late_loss = np.mean(loss_history[-100:])
    assert late_loss < early_loss, \
        f"Loss did not decrease: {early_loss:.4f} -> {late_loss:.4f}"


def test_g_within_5_percent(trained_inverse):
    """Inferred gravity should be within 5% of true value."""
    _, g_param, _, _, _, _ = trained_inverse
    g_inferred = g_param.item()
    relative_error = abs(g_inferred - G_TRUE) / G_TRUE
    assert relative_error < 0.05, \
        f"g error too large: {g_inferred:.4f} vs {G_TRUE} ({relative_error*100:.1f}%)"


def test_L_within_5_percent(trained_inverse):
    """Inferred pendulum length should be within 5% of true value."""
    _, _, L_param, _, _, _ = trained_inverse
    L_inferred = L_param.item()
    relative_error = abs(L_inferred - L_TRUE) / L_TRUE
    assert relative_error < 0.05, \
        f"L error too large: {L_inferred:.4f} vs {L_TRUE} ({relative_error*100:.1f}%)"


def test_gL_ratio_within_2_percent(trained_inverse):
    """
    The ratio g/L is directly identifiable from the pendulum ODE.
    It should converge more accurately than g or L individually.
    """
    _, g_param, L_param, _, _, _ = trained_inverse
    ratio_inferred = g_param.item() / L_param.item()
    ratio_true = G_TRUE / L_TRUE
    relative_error = abs(ratio_inferred - ratio_true) / ratio_true
    assert relative_error < 0.02, \
        f"g/L ratio error: {ratio_inferred:.4f} vs {ratio_true} ({relative_error*100:.1f}%)"


def test_parameters_converge_monotonically(trained_inverse):
    """
    g should generally increase toward 9.81 and L should generally
    decrease toward 1.0 from their initial values.
    Check that the final values are closer to truth than initial values.
    """
    _, g_param, L_param, g_history, L_history, _ = trained_inverse

    g_init_error = abs(g_history[0] - G_TRUE)
    g_final_error = abs(g_param.item() - G_TRUE)
    assert g_final_error < g_init_error, \
        "g did not converge toward true value"

    L_init_error = abs(L_history[0] - L_TRUE)
    L_final_error = abs(L_param.item() - L_TRUE)
    assert L_final_error < L_init_error, \
        "L did not converge toward true value"
