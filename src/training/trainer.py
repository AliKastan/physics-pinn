"""Generic training loop with logging and adaptive collocation support."""

import torch
import numpy as np
from .schedulers import get_scheduler


def sample_adaptive_collocation(model, n_collocation, t_max,
                                n_grid=1000, uniform_fraction=0.2,
                                **physics_params):
    """
    Sample collocation points with probability proportional to residual magnitude.

    Strategy:
        1. Evaluate residual on a dense grid.
        2. Build probability distribution: p_i = |r_i| / sum(|r_j|).
        3. Draw (1 - uniform_fraction) points from this distribution.
        4. Draw uniform_fraction points uniformly at random.
        5. Concatenate and return.
    """
    t_grid = torch.linspace(0, t_max, n_grid).unsqueeze(1)

    with torch.no_grad():
        residuals = model.compute_residual(t_grid, **physics_params)

    probs = residuals / (residuals.sum() + 1e-16)

    n_adaptive = int(n_collocation * (1 - uniform_fraction))
    n_uniform = n_collocation - n_adaptive

    idx = torch.multinomial(probs, n_adaptive, replacement=True)
    t_adaptive = t_grid[idx]
    t_uniform = torch.rand(n_uniform, 1) * t_max

    return torch.cat([t_adaptive, t_uniform], dim=0)


class Trainer:
    """
    Generic PINN trainer with support for adaptive collocation,
    LR scheduling, and logging.

    Args:
        model: BasePINN subclass
        config: dict with training hyperparameters
        physics_params: dict of physics parameters passed to physics_loss
    """

    def __init__(self, model, config, physics_params=None):
        self.model = model
        self.config = config
        self.physics_params = physics_params or {}

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("lr", 1e-3),
        )
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=config.get("scheduler", "plateau"),
            patience=config.get("patience", 500),
            factor=config.get("factor", 0.5),
            min_lr=config.get("min_lr", 1e-6),
        )

        self.epochs = config.get("epochs", 5000)
        self.n_collocation = config.get("n_collocation", 500)
        self.t_max = config.get("t_max", 10.0)
        self.ic_weight = config.get("ic_weight", 20.0)
        self.adaptive = config.get("adaptive", False)
        self.adaptive_interval = config.get("adaptive_interval", 500)
        self.uniform_fraction = config.get("uniform_fraction", 0.2)

    def train(self, ic_loss_fn, verbose=True):
        """
        Run the full training loop.

        Args:
            ic_loss_fn: callable returning scalar IC loss
            verbose: print progress every 1000 epochs

        Returns:
            loss_history: list of total loss per epoch
            collocation_snapshots: dict mapping epoch -> collocation arrays
        """
        loss_history = []
        collocation_snapshots = {}
        cached_t_col = None

        if verbose:
            mode = "adaptive" if self.adaptive else "uniform"
            print(f"Training {self.model.__class__.__name__} ({mode} sampling)")
            print(f"  Collocation points: {self.n_collocation}")
            print(f"  Epochs: {self.epochs}")
            print("-" * 50)

        for epoch in range(self.epochs):
            # Collocation point sampling
            if self.adaptive and (epoch % self.adaptive_interval == 0):
                self.model.eval()
                cached_t_col = sample_adaptive_collocation(
                    self.model, self.n_collocation, self.t_max,
                    uniform_fraction=self.uniform_fraction,
                    **self.physics_params,
                )
                self.model.train()

            if self.adaptive and cached_t_col is not None:
                t_col = cached_t_col.clone()
            else:
                t_col = torch.rand(self.n_collocation, 1) * self.t_max

            # Snapshot collocation at key epochs
            if epoch in (0, self.epochs // 2, self.epochs - 1):
                collocation_snapshots[epoch] = t_col.detach().numpy().flatten()

            # Training step
            losses = self.model.train_step(
                self.optimizer, t_col, ic_loss_fn,
                ic_weight=self.ic_weight,
                **self.physics_params,
            )

            # LR scheduling
            scheduler_type = self.config.get("scheduler", "plateau")
            if scheduler_type == "plateau":
                self.scheduler.step(losses['total'])
            else:
                self.scheduler.step()

            loss_history.append(losses['total'])

            if verbose and (epoch + 1) % 1000 == 0:
                print(f"  Epoch {epoch+1:5d}/{self.epochs} | "
                      f"Loss: {losses['total']:.6f} | "
                      f"Physics: {losses['physics']:.6f} | "
                      f"IC: {losses['ic']:.6f}")

        if verbose:
            print("-" * 50)
            print(f"Final loss: {loss_history[-1]:.6f}")

        return loss_history, collocation_snapshots
