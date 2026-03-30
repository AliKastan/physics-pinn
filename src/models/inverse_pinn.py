"""
Inverse PINNs — simultaneously reconstruct trajectories and infer
unknown physical parameters from noisy observational data.

InversePendulumPINN: infers g and L (or g/L) from noisy theta(t) observations.
InverseOrbitalPINN: infers GM from noisy (x, y) position observations.

The key insight: the physics loss constrains the *form* of the solution
(it must satisfy the ODE), while the data loss constrains the *specific*
trajectory.  Together they pin down both the trajectory and the unknown
parameters.

Two-phase training strategy:
    Phase 1 (warmup): data + IC loss only — the network learns a rough
    trajectory shape before physics kicks in.
    Phase 2 (main): full loss = physics + data + IC.  The physics residual
    uses trainable parameters, which the optimizer drives toward truth.
"""

import torch
import torch.nn as nn
import numpy as np


class InversePendulumPINN(nn.Module):
    """
    PINN for pendulum inverse problem: maps t -> (theta, omega) while
    simultaneously inferring unknown physical parameters g and L.

    Parameters g and L are nn.Parameters with initial guesses that the
    optimizer adjusts alongside the network weights.  The physics loss
    uses these trainable values, creating a feedback loop:

        wrong g,L -> large physics residual -> optimizer corrects g,L
    """

    def __init__(self, g_init=5.0, L_init=1.0,
                 hidden_size=64, num_hidden_layers=3):
        super().__init__()

        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 2))
        self.network = nn.Sequential(*layers)

        self.g_param = nn.Parameter(
            torch.tensor(g_init, dtype=torch.float32))
        self.L_param = nn.Parameter(
            torch.tensor(L_init, dtype=torch.float32))

    def forward(self, t):
        return self.network(t)

    @property
    def g(self):
        return self.g_param.item()

    @property
    def L(self):
        return self.L_param.item()

    @property
    def g_over_L(self):
        return self.g_param.item() / self.L_param.item()

    def clamp_parameters(self):
        """Keep physical parameters positive."""
        with torch.no_grad():
            self.g_param.clamp_(min=0.1)
            self.L_param.clamp_(min=0.1)

    def physics_loss(self, t_collocation):
        """Pendulum ODE residual using trainable g and L."""
        t_collocation.requires_grad_(True)
        output = self(t_collocation)
        theta = output[:, 0:1]
        omega = output[:, 1:2]

        dtheta_dt = torch.autograd.grad(
            theta, t_collocation,
            grad_outputs=torch.ones_like(theta),
            create_graph=True,
        )[0]
        domega_dt = torch.autograd.grad(
            omega, t_collocation,
            grad_outputs=torch.ones_like(omega),
            create_graph=True,
        )[0]

        res1 = dtheta_dt - omega
        res2 = domega_dt + (self.g_param / self.L_param) * torch.sin(theta)
        return torch.mean(res1 ** 2) + torch.mean(res2 ** 2)


class InverseOrbitalPINN(nn.Module):
    """
    PINN for orbital inverse problem: maps t -> (x, y, vx, vy) while
    simultaneously inferring the gravitational parameter GM.
    """

    def __init__(self, GM_init=0.5,
                 hidden_size=128, num_hidden_layers=4):
        super().__init__()

        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 4))
        self.network = nn.Sequential(*layers)

        self.GM_param = nn.Parameter(
            torch.tensor(GM_init, dtype=torch.float32))

    def forward(self, t):
        return self.network(t)

    @property
    def GM(self):
        return self.GM_param.item()

    def clamp_parameters(self):
        """Keep GM positive."""
        with torch.no_grad():
            self.GM_param.clamp_(min=0.01)

    def physics_loss(self, t_collocation):
        """Gravitational ODE residual using trainable GM."""
        t_collocation.requires_grad_(True)
        output = self(t_collocation)
        x = output[:, 0:1]
        y = output[:, 1:2]
        vx = output[:, 2:3]
        vy = output[:, 3:4]

        ones = torch.ones_like(x)
        dx_dt = torch.autograd.grad(
            x, t_collocation, ones, create_graph=True)[0]
        dy_dt = torch.autograd.grad(
            y, t_collocation, ones, create_graph=True)[0]
        dvx_dt = torch.autograd.grad(
            vx, t_collocation, ones, create_graph=True)[0]
        dvy_dt = torch.autograd.grad(
            vy, t_collocation, ones, create_graph=True)[0]

        r = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
        r3 = r ** 3

        res_x = dx_dt - vx
        res_y = dy_dt - vy
        res_vx = dvx_dt + self.GM_param * x / r3
        res_vy = dvy_dt + self.GM_param * y / r3

        return (torch.mean(res_x ** 2) + torch.mean(res_y ** 2) +
                torch.mean(res_vx ** 2) + torch.mean(res_vy ** 2))


def train_inverse_pendulum(model, theta_0, omega_0, t_max,
                           t_obs_t, theta_obs_t,
                           n_collocation=500, epochs=10000,
                           lr_net=1e-3, lr_param=1e-2,
                           ic_weight=20.0, data_weight=10.0,
                           warmup_epochs=1500, verbose=True):
    """
    Two-phase inverse training for the pendulum.

    Phase 1 (warmup): data + IC only — learn trajectory shape.
    Phase 2 (main):   physics + data + IC — recover parameters.

    Args:
        model: InversePendulumPINN instance
        theta_0, omega_0: initial conditions
        t_max: time domain
        t_obs_t: observation times tensor (N, 1)
        theta_obs_t: observed theta tensor (N,)
        ...training hyperparameters...

    Returns:
        g_history, L_history, loss_history: lists of per-epoch values
    """
    # Phase 1: warmup with data + IC only
    warmup_opt = torch.optim.Adam(model.network.parameters(), lr=lr_net)

    if verbose:
        print(f"Inverse Pendulum: g_init={model.g:.2f}, L_init={model.L:.2f}")
        print(f"  Warmup: {warmup_epochs} epochs | Main: {epochs} epochs")
        print("-" * 55)

    for ep in range(warmup_epochs):
        warmup_opt.zero_grad()
        out = model(t_obs_t)
        loss = data_weight * torch.mean((out[:, 0] - theta_obs_t) ** 2)
        t0 = torch.zeros(1, 1)
        out0 = model(t0)
        loss = loss + ic_weight * ((out0[0, 0] - theta_0) ** 2 +
                                   (out0[0, 1] - omega_0) ** 2)
        loss.backward()
        warmup_opt.step()

    if verbose:
        print("  Warmup done.")

    # Phase 2: full training with physics
    optimizer = torch.optim.Adam([
        {'params': model.network.parameters(), 'lr': lr_net},
        {'params': [model.g_param, model.L_param], 'lr': lr_param},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6,
    )

    loss_history = []
    g_history = []
    L_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        t_col = torch.rand(n_collocation, 1) * t_max
        loss_phys = model.physics_loss(t_col)

        out_obs = model(t_obs_t)
        loss_data = torch.mean((out_obs[:, 0] - theta_obs_t) ** 2)

        t0 = torch.zeros(1, 1)
        out0 = model(t0)
        loss_ic = (out0[0, 0] - theta_0) ** 2 + (out0[0, 1] - omega_0) ** 2

        total = loss_phys + data_weight * loss_data + ic_weight * loss_ic
        total.backward()
        optimizer.step()
        scheduler.step(total.item())
        model.clamp_parameters()

        loss_history.append(total.item())
        g_history.append(model.g)
        L_history.append(model.L)

        if verbose and (epoch + 1) % 2000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | "
                  f"Loss: {total.item():.6f} | "
                  f"g={model.g:.4f}  L={model.L:.4f}  "
                  f"g/L={model.g_over_L:.4f}")

    if verbose:
        print("-" * 55)
        print(f"  Recovered: g={model.g:.4f}, L={model.L:.4f}, "
              f"g/L={model.g_over_L:.4f}")

    return g_history, L_history, loss_history


def train_inverse_orbital(model, ics, t_max,
                          t_obs_t, x_obs_t, y_obs_t,
                          n_collocation=600, epochs=10000,
                          lr_net=1e-3, lr_param=5e-3,
                          ic_weight=50.0, data_weight=10.0,
                          warmup_epochs=2000, verbose=True):
    """
    Two-phase inverse training for orbital mechanics.

    Args:
        model: InverseOrbitalPINN instance
        ics: (x0, y0, vx0, vy0) tuple
        t_obs_t: observation times tensor (N, 1)
        x_obs_t, y_obs_t: observed position tensors (N,)

    Returns:
        GM_history, loss_history
    """
    x0, y0, vx0, vy0 = ics
    x0_t = torch.tensor(x0, dtype=torch.float32)
    y0_t = torch.tensor(y0, dtype=torch.float32)
    vx0_t = torch.tensor(vx0, dtype=torch.float32)
    vy0_t = torch.tensor(vy0, dtype=torch.float32)

    # Phase 1: warmup
    warmup_opt = torch.optim.Adam(model.network.parameters(), lr=lr_net)

    if verbose:
        print(f"Inverse Orbital: GM_init={model.GM:.2f}")
        print(f"  Warmup: {warmup_epochs} epochs | Main: {epochs} epochs")
        print("-" * 55)

    for ep in range(warmup_epochs):
        warmup_opt.zero_grad()
        out = model(t_obs_t)
        loss = data_weight * (torch.mean((out[:, 0] - x_obs_t) ** 2) +
                              torch.mean((out[:, 1] - y_obs_t) ** 2))
        t0 = torch.zeros(1, 1)
        out0 = model(t0)
        loss = loss + ic_weight * ((out0[0, 0] - x0_t) ** 2 +
                                   (out0[0, 1] - y0_t) ** 2 +
                                   (out0[0, 2] - vx0_t) ** 2 +
                                   (out0[0, 3] - vy0_t) ** 2)
        loss.backward()
        warmup_opt.step()

    if verbose:
        print("  Warmup done.")

    # Phase 2: full training
    optimizer = torch.optim.Adam([
        {'params': model.network.parameters(), 'lr': lr_net},
        {'params': [model.GM_param], 'lr': lr_param},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6,
    )

    loss_history = []
    GM_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        t_col = torch.rand(n_collocation, 1) * t_max
        loss_phys = model.physics_loss(t_col)

        out_obs = model(t_obs_t)
        loss_data = (torch.mean((out_obs[:, 0] - x_obs_t) ** 2) +
                     torch.mean((out_obs[:, 1] - y_obs_t) ** 2))

        t0 = torch.zeros(1, 1)
        out0 = model(t0)
        loss_ic = ((out0[0, 0] - x0_t) ** 2 + (out0[0, 1] - y0_t) ** 2 +
                   (out0[0, 2] - vx0_t) ** 2 + (out0[0, 3] - vy0_t) ** 2)

        total = loss_phys + data_weight * loss_data + ic_weight * loss_ic
        total.backward()
        optimizer.step()
        scheduler.step(total.item())
        model.clamp_parameters()

        loss_history.append(total.item())
        GM_history.append(model.GM)

        if verbose and (epoch + 1) % 2000 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | "
                  f"Loss: {total.item():.6f} | GM={model.GM:.4f}")

    if verbose:
        print("-" * 55)
        print(f"  Recovered: GM={model.GM:.4f}")

    return GM_history, loss_history
