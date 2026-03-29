"""
ODE/PDE residual functions, decoupled from neural network architectures.

Each function takes pre-computed derivatives and state variables,
and returns the residual tensors. This separation allows the same
physics to be reused across different network architectures.
"""

import torch


def pendulum_residual(dtheta_dt, domega_dt, theta, omega, g=9.81, L=1.0):
    """
    Pendulum ODE residuals:
        r1 = dtheta/dt - omega             (kinematic relation)
        r2 = domega/dt + (g/L)*sin(theta)  (Newton's second law)

    Args:
        dtheta_dt, domega_dt: time derivatives from autograd
        theta, omega: state variables from network output
        g: gravitational acceleration (m/s^2)
        L: pendulum length (m)

    Returns:
        (residual_1, residual_2) tensors
    """
    residual_1 = dtheta_dt - omega
    residual_2 = domega_dt + (g / L) * torch.sin(theta)
    return residual_1, residual_2


def orbital_residual(dx_dt, dy_dt, dvx_dt, dvy_dt, x, y, vx, vy, GM=1.0):
    """
    Two-body gravitational ODE residuals:
        r1: dx/dt  - vx               = 0
        r2: dy/dt  - vy               = 0
        r3: dvx/dt + GM * x / r^3     = 0
        r4: dvy/dt + GM * y / r^3     = 0

    Args:
        dx_dt, dy_dt, dvx_dt, dvy_dt: time derivatives from autograd
        x, y, vx, vy: state variables from network output
        GM: gravitational parameter

    Returns:
        (res_x, res_y, res_vx, res_vy) tensors
    """
    r = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
    r_cubed = r ** 3

    res_x = dx_dt - vx
    res_y = dy_dt - vy
    res_vx = dvx_dt + GM * x / r_cubed
    res_vy = dvy_dt + GM * y / r_cubed
    return res_x, res_y, res_vx, res_vy
