"""
Residual-Based Adaptive Refinement (RAR) for collocation point sampling.

Instead of sampling collocation points uniformly, we concentrate them
where the physics residual is largest.  This is a drop-in replacement
for uniform sampling in the Trainer.

Algorithm:
    1. Evaluate physics residual on a dense grid.
    2. Build probability distribution:  P(x) ~ |residual(x)|^k
       where k controls concentration (k=1 linear, k=2 aggressive).
    3. Sample (1 - uniform_fraction) points from this distribution.
    4. Sample uniform_fraction points uniformly (prevents clustering).
    5. Concatenate and return.

The adaptive interval controls how often the distribution is refreshed —
too often is expensive, too rarely misses changes in the residual landscape.
"""

import torch
import numpy as np


def sample_rar(model, n_collocation, t_max, k=1.0,
               n_grid=1000, uniform_fraction=0.2,
               **physics_params):
    """
    Residual-based adaptive refinement for 1D collocation (ODE PINNs).

    Args:
        model: PINN with a compute_residual(t, **params) method
        n_collocation: total number of points to return
        t_max: upper bound of time domain
        k: concentration exponent (1=linear, 2=aggressive)
        n_grid: dense evaluation grid size
        uniform_fraction: fraction of uniform random points
        **physics_params: forwarded to model.compute_residual

    Returns:
        t_col: tensor (n_collocation, 1)
    """
    t_grid = torch.linspace(0, t_max, n_grid).unsqueeze(1)

    # compute_residual needs autograd enabled (it calls torch.autograd.grad),
    # but we don't need the result to be part of the training graph.
    with torch.enable_grad():
        residuals = model.compute_residual(t_grid, **physics_params)

    # Apply concentration exponent
    weights = residuals ** k
    probs = weights / (weights.sum() + 1e-16)

    n_adaptive = int(n_collocation * (1 - uniform_fraction))
    n_uniform = n_collocation - n_adaptive

    idx = torch.multinomial(probs, n_adaptive, replacement=True)
    t_adaptive = t_grid[idx]
    t_uniform = torch.rand(n_uniform, 1) * t_max

    return torch.cat([t_adaptive, t_uniform], dim=0)


def sample_rar_2d(model, n_collocation, x_max, t_max, k=1.0,
                  nx_grid=50, nt_grid=50, uniform_fraction=0.2,
                  **physics_params):
    """
    Residual-based adaptive refinement for 2D collocation (PDE PINNs).

    Evaluates the residual on a (nx_grid x nt_grid) mesh and samples
    interior points proportional to |residual|^k.

    Args:
        model: PDE PINN with compute_residual(x, t, **params) method
        n_collocation: total number of interior points
        x_max, t_max: domain bounds
        k: concentration exponent
        nx_grid, nt_grid: evaluation grid resolution
        uniform_fraction: fraction of uniform random points

    Returns:
        (x_col, t_col): tuple of tensors each (n_collocation, 1)
    """
    x_1d = torch.linspace(0, x_max, nx_grid)
    t_1d = torch.linspace(0, t_max, nt_grid)
    X, T = torch.meshgrid(x_1d, t_1d, indexing='ij')
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    with torch.enable_grad():
        residuals = model.compute_residual(x_flat, t_flat, **physics_params)

    weights = residuals ** k
    probs = weights / (weights.sum() + 1e-16)

    n_adaptive = int(n_collocation * (1 - uniform_fraction))
    n_uniform = n_collocation - n_adaptive

    idx = torch.multinomial(probs, n_adaptive, replacement=True)
    x_adaptive = x_flat[idx]
    t_adaptive = t_flat[idx]

    x_uniform = torch.rand(n_uniform, 1) * x_max
    t_uniform = torch.rand(n_uniform, 1) * t_max

    x_col = torch.cat([x_adaptive, x_uniform], dim=0)
    t_col = torch.cat([t_adaptive, t_uniform], dim=0)
    return x_col, t_col


class AdaptiveCollocationSchedule:
    """
    Manages the adaptive collocation lifecycle within a training loop.

    Caches the sampled collocation points and refreshes every
    `interval` epochs.  Tracks residual statistics for diagnostics.
    """

    def __init__(self, model, n_collocation, t_max, k=1.0,
                 interval=500, uniform_fraction=0.2,
                 n_grid=1000, physics_params=None):
        self.model = model
        self.n_collocation = n_collocation
        self.t_max = t_max
        self.k = k
        self.interval = interval
        self.uniform_fraction = uniform_fraction
        self.n_grid = n_grid
        self.physics_params = physics_params or {}

        self._cached = None
        self._residual_stats = []  # (epoch, mean_residual, max_residual)

    def sample(self, epoch):
        """Return collocation points, refreshing the cache if needed."""
        if epoch % self.interval == 0:
            self.model.eval()
            self._cached = sample_rar(
                self.model, self.n_collocation, self.t_max,
                k=self.k, n_grid=self.n_grid,
                uniform_fraction=self.uniform_fraction,
                **self.physics_params,
            )
            # Record stats
            t_grid = torch.linspace(0, self.t_max, self.n_grid).unsqueeze(1)
            with torch.enable_grad():
                res = self.model.compute_residual(t_grid, **self.physics_params)
            self._residual_stats.append(
                (epoch, res.mean().item(), res.max().item()))
            self.model.train()

        if self._cached is not None:
            return self._cached.clone()
        return torch.rand(self.n_collocation, 1) * self.t_max

    @property
    def residual_stats(self):
        """List of (epoch, mean_residual, max_residual) at each refresh."""
        return list(self._residual_stats)
