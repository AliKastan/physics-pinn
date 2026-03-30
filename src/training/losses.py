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


# =========================================================================
# HNN losses
# =========================================================================

def hamiltonian_loss(model, q_batch, p_batch, dqdt_true, dpdt_true):
    """
    MSE between HNN-predicted and true time derivatives.

    The HNN predicts (dq/dt, dp/dt) by differentiating its learned H(q,p)
    via Hamilton's equations.  This loss drives the network to learn the
    correct Hamiltonian up to an additive constant.

    Args:
        model: HamiltonianNN
        q_batch, p_batch: state tensors (N, 1) with requires_grad=True
        dqdt_true, dpdt_true: ground-truth derivative tensors (N, 1)

    Returns:
        Scalar MSE loss
    """
    dqdt_pred, dpdt_pred = model.time_derivative(None, q_batch, p_batch)
    return (torch.mean((dqdt_pred - dqdt_true) ** 2) +
            torch.mean((dpdt_pred - dpdt_true) ** 2))


def energy_conservation_loss(model, q_traj, p_traj):
    """
    Penalize energy drift: |H(q(t), p(t)) - H(q(0), p(0))|^2.

    Evaluated along a trajectory to encourage the learned Hamiltonian
    to be constant on physical orbits.  Useful as a regularizer on top
    of hamiltonian_loss.

    Args:
        model: HamiltonianNN
        q_traj, p_traj: trajectory tensors (T, 1), no grad required

    Returns:
        Scalar mean-squared energy deviation from H(0)
    """
    H = model(q_traj, p_traj)
    H0 = H[0].detach()
    return torch.mean((H - H0) ** 2)


# =========================================================================
# Gradient-enhanced loss
# =========================================================================

def gradient_enhanced_loss(model, t_data, y_data, dydt_data):
    """
    Gradient-enhanced data-fitting loss: match both u(t) and du/dt.

    Standard data loss only matches function values.  Adding gradient
    matching dramatically improves convergence for stiff systems where
    the derivative carries most of the information.

    Args:
        model: PINN model
        t_data: observation times (N, 1), will have requires_grad set
        y_data: observed state values (N, output_dim)
        dydt_data: observed derivatives (N, output_dim)

    Returns:
        Scalar loss = MSE(u) + MSE(du/dt)
    """
    t_data = t_data.clone().requires_grad_(True)
    predictions = model(t_data)
    loss_val = torch.mean((predictions - y_data) ** 2)

    # Compute du/dt via autograd for each output channel
    loss_grad = torch.tensor(0.0)
    for col in range(predictions.shape[1]):
        dpred_dt = torch.autograd.grad(
            predictions[:, col:col+1], t_data,
            grad_outputs=torch.ones(predictions.shape[0], 1),
            create_graph=True, retain_graph=True,
        )[0]
        loss_grad = loss_grad + torch.mean((dpred_dt - dydt_data[:, col:col+1]) ** 2)

    return loss_val + loss_grad


# =========================================================================
# PDE losses
# =========================================================================

def pde_total_loss(model, x_int, t_int, alpha,
                   n_bc, t_max, T_left, T_right, L_rod,
                   n_ic, ic_fn,
                   bc_weight=10.0, ic_weight=10.0):
    """
    Combined PDE loss:  L_pde + bc_weight * L_bc + ic_weight * L_ic.

    Convenience function that calls the model's own loss methods.
    """
    loss_pde = model.physics_loss(x_int, t_int, alpha=alpha)
    loss_bc = model.boundary_loss(n_bc, t_max, T_left, T_right, L_rod)
    loss_ic = model.ic_loss(n_ic, ic_fn, L_rod)
    return loss_pde + bc_weight * loss_bc + ic_weight * loss_ic
