"""Loss functions for PINN training: physics, initial condition, and data losses."""

import torch


def physics_loss(model, t_collocation, **params):
    """Delegate to model's physics_loss method."""
    return model.physics_loss(t_collocation, **params)


def initial_condition_loss(model, t_zero, targets):
    """
    Generic initial condition loss at t=0.

    Args:
        model: PINN model
        t_zero: tensor of shape (1, 1) with value 0
        targets: dict mapping output indices to target values
            e.g. {0: theta_0, 1: omega_0}

    Returns:
        Scalar IC loss (sum of squared errors)
    """
    output = model(t_zero)
    loss = torch.tensor(0.0)
    for idx, target in targets.items():
        loss = loss + (output[0, idx] - target) ** 2
    return loss


def data_loss(model, t_data, y_data):
    """
    Data-fitting loss for inverse problems or supervised training.

    Args:
        model: PINN model
        t_data: time points (N, 1)
        y_data: observed data (N, output_dim)

    Returns:
        Mean squared error between predictions and observations
    """
    predictions = model(t_data)
    return torch.mean((predictions - y_data) ** 2)


def pendulum_ic_loss(model, theta_0, omega_0):
    """Pendulum-specific IC loss matching the original implementation."""
    t_zero = torch.zeros(1, 1)
    output = model(t_zero)
    theta_pred = output[:, 0]
    omega_pred = output[:, 1]
    loss_theta = (theta_pred - theta_0) ** 2
    loss_omega = (omega_pred - omega_0) ** 2
    return loss_theta.squeeze() + loss_omega.squeeze()


def orbital_ic_loss(model, x0, y0, vx0, vy0):
    """Orbital-specific IC loss matching the original implementation."""
    t_zero = torch.zeros(1, 1)
    output = model(t_zero)
    loss = ((output[0, 0] - x0) ** 2 +
            (output[0, 1] - y0) ** 2 +
            (output[0, 2] - vx0) ** 2 +
            (output[0, 3] - vy0) ** 2)
    return loss
