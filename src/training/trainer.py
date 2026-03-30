"""
Generic training loop with support for advanced techniques:
  - Adaptive collocation (RAR)
  - Curriculum learning (expanding time horizon)
  - Adaptive IC weight scheduling
  - Fourier feature models (auto-detected)

All features are enabled via config dict flags.
"""

import torch
import numpy as np
from .schedulers import (
    get_scheduler, AdaptiveICWeightScheduler, CurriculumSchedule,
)
from .adaptive_collocation import (
    sample_rar, AdaptiveCollocationSchedule,
)


# Legacy function kept for backward compatibility
def sample_adaptive_collocation(model, n_collocation, t_max,
                                n_grid=1000, uniform_fraction=0.2,
                                **physics_params):
    """Sample collocation points proportional to residual (legacy wrapper)."""
    return sample_rar(model, n_collocation, t_max, k=1.0,
                      n_grid=n_grid, uniform_fraction=uniform_fraction,
                      **physics_params)


class Trainer:
    """
    Generic PINN trainer with support for advanced training techniques.

    Config flags:
        adaptive (bool):            enable RAR adaptive collocation
        adaptive_interval (int):    epochs between collocation refresh
        adaptive_k (float):         concentration exponent for RAR
        uniform_fraction (float):   fraction of uniform points in RAR

        curriculum (bool):          enable curriculum learning
        curriculum_warmup (float):  fraction of epochs for horizon expansion
        curriculum_min_frac (float): starting t_max fraction

        adaptive_ic_weight (bool):  enable dynamic IC weight adjustment
        ic_boost_factor (float):    IC weight multiplier when IC loss is high
        ic_threshold (float):       IC loss threshold to trigger boost

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
            T_0=config.get("T_0", 1000),
            T_mult=config.get("T_mult", 2),
        )

        self.epochs = config.get("epochs", 5000)
        self.n_collocation = config.get("n_collocation", 500)
        self.t_max = config.get("t_max", 10.0)
        self.ic_weight = config.get("ic_weight", 20.0)

        # Adaptive collocation (RAR)
        self.adaptive = config.get("adaptive", False)
        self.adaptive_interval = config.get("adaptive_interval", 500)
        self.uniform_fraction = config.get("uniform_fraction", 0.2)
        self.adaptive_k = config.get("adaptive_k", 1.0)

        # Curriculum learning
        self.curriculum = config.get("curriculum", False)
        self._curriculum_schedule = None
        if self.curriculum:
            self._curriculum_schedule = CurriculumSchedule(
                t_max=self.t_max,
                total_epochs=self.epochs,
                warmup_fraction=config.get("curriculum_warmup", 0.3),
                min_fraction=config.get("curriculum_min_frac", 0.1),
            )

        # Adaptive IC weight
        self.adaptive_ic = config.get("adaptive_ic_weight", False)
        self._ic_scheduler = None
        if self.adaptive_ic:
            self._ic_scheduler = AdaptiveICWeightScheduler(
                base_weight=self.ic_weight,
                boost_factor=config.get("ic_boost_factor", 3.0),
                ic_threshold=config.get("ic_threshold", 0.1),
            )

    def _get_t_max(self, epoch):
        """Return current time horizon (may grow under curriculum learning)."""
        if self._curriculum_schedule is not None:
            return self._curriculum_schedule.get_t_max(epoch)
        return self.t_max

    def _get_ic_weight(self, loss_phys, loss_ic):
        """Return current IC weight (may be boosted if IC loss is high)."""
        if self._ic_scheduler is not None:
            return self._ic_scheduler.get_weight(loss_phys, loss_ic)
        return self.ic_weight

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

        # Set up adaptive collocation manager
        rar_schedule = None
        if self.adaptive:
            rar_schedule = AdaptiveCollocationSchedule(
                self.model, self.n_collocation, self.t_max,
                k=self.adaptive_k,
                interval=self.adaptive_interval,
                uniform_fraction=self.uniform_fraction,
                physics_params=self.physics_params,
            )

        if verbose:
            features = []
            if self.adaptive:
                features.append("RAR")
            if self.curriculum:
                features.append("curriculum")
            if self.adaptive_ic:
                features.append("adaptive-IC")
            mode_str = "+".join(features) if features else "standard"
            print(f"Training {self.model.__class__.__name__} ({mode_str})")
            print(f"  Collocation points: {self.n_collocation}")
            print(f"  Epochs: {self.epochs}")
            print("-" * 50)

        for epoch in range(self.epochs):
            current_t_max = self._get_t_max(epoch)

            # Collocation point sampling
            if rar_schedule is not None:
                # Update t_max in RAR schedule for curriculum
                rar_schedule.t_max = current_t_max
                t_col = rar_schedule.sample(epoch)
            else:
                t_col = torch.rand(self.n_collocation, 1) * current_t_max

            # Snapshot collocation at key epochs
            if epoch in (0, self.epochs // 2, self.epochs - 1):
                collocation_snapshots[epoch] = t_col.detach().numpy().flatten()

            # Training step — compute losses separately for adaptive weighting
            self.optimizer.zero_grad()
            loss_phys = self.model.physics_loss(t_col, **self.physics_params)
            loss_ic = ic_loss_fn()

            ic_w = self._get_ic_weight(loss_phys.item(), loss_ic.item())
            total_loss = loss_phys + ic_w * loss_ic
            total_loss.backward()
            self.optimizer.step()

            # LR scheduling
            scheduler_type = self.config.get("scheduler", "plateau")
            if scheduler_type == "plateau":
                self.scheduler.step(total_loss.item())
            else:
                self.scheduler.step()

            loss_history.append(total_loss.item())

            if verbose and (epoch + 1) % 1000 == 0:
                t_max_str = (f" t_max={current_t_max:.2f}"
                             if self.curriculum else "")
                ic_w_str = (f" ic_w={ic_w:.1f}"
                            if self.adaptive_ic else "")
                print(f"  Epoch {epoch+1:5d}/{self.epochs} | "
                      f"Loss: {total_loss.item():.6f} | "
                      f"Phys: {loss_phys.item():.6f} | "
                      f"IC: {loss_ic.item():.6f}"
                      f"{t_max_str}{ic_w_str}")

        if verbose:
            print("-" * 50)
            print(f"Final loss: {loss_history[-1]:.6f}")

        return loss_history, collocation_snapshots
