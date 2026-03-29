"""
Tests for physics equation residual functions.

Verifies that residual functions correctly compute zero residual
for known exact solutions and non-zero residual for incorrect states.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
from src.physics.equations import pendulum_residual, orbital_residual


class TestPendulumResidual:
    """Tests for the pendulum ODE residual function."""

    def test_zero_residual_at_equilibrium(self):
        """At theta=0, omega=0 (hanging straight down), the residual should be zero."""
        n = 50
        # At equilibrium: theta=0, omega=0, dtheta/dt=0, domega/dt=0
        dtheta_dt = torch.zeros(n, 1)
        domega_dt = torch.zeros(n, 1)
        theta = torch.zeros(n, 1)
        omega = torch.zeros(n, 1)

        r1, r2 = pendulum_residual(dtheta_dt, domega_dt, theta, omega)
        assert torch.allclose(r1, torch.zeros_like(r1), atol=1e-7)
        assert torch.allclose(r2, torch.zeros_like(r2), atol=1e-7)

    def test_kinematic_residual(self):
        """r1 = dtheta/dt - omega should be zero when dtheta/dt == omega."""
        n = 50
        omega = torch.randn(n, 1)
        dtheta_dt = omega.clone()  # exact kinematic relation
        domega_dt = torch.zeros(n, 1)
        theta = torch.zeros(n, 1)

        r1, _ = pendulum_residual(dtheta_dt, domega_dt, theta, omega)
        assert torch.allclose(r1, torch.zeros_like(r1), atol=1e-7)

    def test_nonzero_residual_for_wrong_derivatives(self):
        """Residual should be non-zero when derivatives don't match the ODE."""
        n = 50
        dtheta_dt = torch.ones(n, 1)
        domega_dt = torch.ones(n, 1)
        theta = torch.ones(n, 1) * 0.5
        omega = torch.zeros(n, 1)

        r1, r2 = pendulum_residual(dtheta_dt, domega_dt, theta, omega)
        assert r1.abs().sum().item() > 0
        assert r2.abs().sum().item() > 0

    def test_output_shapes(self):
        """Residual tensors should match input shapes."""
        n = 30
        dtheta_dt = torch.randn(n, 1)
        domega_dt = torch.randn(n, 1)
        theta = torch.randn(n, 1)
        omega = torch.randn(n, 1)

        r1, r2 = pendulum_residual(dtheta_dt, domega_dt, theta, omega)
        assert r1.shape == (n, 1)
        assert r2.shape == (n, 1)


class TestOrbitalResidual:
    """Tests for the orbital ODE residual function."""

    def test_zero_residual_for_consistent_state(self):
        """Residual should be zero when derivatives match the gravity ODE exactly."""
        GM = 1.0
        # A body at (1, 0) with velocity (0, 1) in a circular orbit
        n = 1
        x = torch.tensor([[1.0]])
        y = torch.tensor([[0.0]])
        vx = torch.tensor([[0.0]])
        vy = torch.tensor([[1.0]])

        # For circular orbit at r=1: acceleration = -GM/r^2 = -1 in radial direction
        dx_dt = vx.clone()
        dy_dt = vy.clone()
        dvx_dt = torch.tensor([[-GM]])  # -GM * x / r^3 = -1
        dvy_dt = torch.tensor([[0.0]])  # -GM * y / r^3 = 0

        rx, ry, rvx, rvy = orbital_residual(
            dx_dt, dy_dt, dvx_dt, dvy_dt, x, y, vx, vy, GM)

        assert torch.allclose(rx, torch.zeros_like(rx), atol=1e-6)
        assert torch.allclose(ry, torch.zeros_like(ry), atol=1e-6)
        assert torch.allclose(rvx, torch.zeros_like(rvx), atol=1e-3)
        assert torch.allclose(rvy, torch.zeros_like(rvy), atol=1e-3)

    def test_nonzero_residual_for_wrong_acceleration(self):
        """Residual should be non-zero when acceleration is wrong."""
        n = 50
        x = torch.ones(n, 1)
        y = torch.zeros(n, 1)
        vx = torch.zeros(n, 1)
        vy = torch.ones(n, 1)
        dx_dt = vx.clone()
        dy_dt = vy.clone()
        dvx_dt = torch.zeros(n, 1)  # wrong: should be -GM
        dvy_dt = torch.zeros(n, 1)

        rx, ry, rvx, rvy = orbital_residual(
            dx_dt, dy_dt, dvx_dt, dvy_dt, x, y, vx, vy, GM=1.0)
        assert rvx.abs().sum().item() > 0

    def test_output_shapes(self):
        """Residual tensors should match input shapes."""
        n = 30
        x = torch.randn(n, 1)
        y = torch.randn(n, 1)
        vx = torch.randn(n, 1)
        vy = torch.randn(n, 1)
        dx_dt = torch.randn(n, 1)
        dy_dt = torch.randn(n, 1)
        dvx_dt = torch.randn(n, 1)
        dvy_dt = torch.randn(n, 1)

        rx, ry, rvx, rvy = orbital_residual(
            dx_dt, dy_dt, dvx_dt, dvy_dt, x, y, vx, vy)
        assert rx.shape == (n, 1)
        assert ry.shape == (n, 1)
        assert rvx.shape == (n, 1)
        assert rvy.shape == (n, 1)
