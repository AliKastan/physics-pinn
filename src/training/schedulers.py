"""Learning rate scheduling utilities."""

import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(optimizer, scheduler_type="plateau", **kwargs):
    """
    Create a learning rate scheduler.

    Args:
        optimizer: torch optimizer
        scheduler_type: one of "plateau", "cosine", "step", "cosine_warm"
        **kwargs: scheduler-specific parameters

    Returns:
        LR scheduler instance
    """
    if scheduler_type == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=kwargs.get("patience", 500),
            factor=kwargs.get("factor", 0.5),
            min_lr=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_type == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 1000),
            eta_min=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_type == "cosine_warm":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("T_0", 1000),
            T_mult=kwargs.get("T_mult", 2),
            eta_min=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_type == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 1000),
            gamma=kwargs.get("gamma", 0.5),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class AdaptiveICWeightScheduler:
    """
    Dynamically adjusts IC weight based on current IC loss magnitude.

    If IC loss is much larger than physics loss, temporarily boost
    the IC weight so the network anchors to the correct initial
    conditions before refining the physics.

    Usage:
        scheduler = AdaptiveICWeightScheduler(base_weight=20.0)
        for epoch in range(epochs):
            ic_w = scheduler.get_weight(loss_phys, loss_ic)
    """

    def __init__(self, base_weight=20.0, boost_factor=3.0,
                 ic_threshold=0.1):
        """
        Args:
            base_weight: normal IC weight
            boost_factor: multiplier when IC loss is high
            ic_threshold: boost when ic_loss > ic_threshold
        """
        self.base_weight = base_weight
        self.boost_factor = boost_factor
        self.ic_threshold = ic_threshold

    def get_weight(self, loss_phys, loss_ic):
        """Return adjusted IC weight for this epoch."""
        if loss_ic > self.ic_threshold:
            return self.base_weight * self.boost_factor
        return self.base_weight


class CurriculumSchedule:
    """
    Curriculum learning: start training on a short time horizon and
    gradually extend it to the full domain.

    This helps the PINN learn the near-IC dynamics first, then expand
    to longer horizons where errors can compound.

    Usage:
        schedule = CurriculumSchedule(t_max=10.0, total_epochs=5000)
        for epoch in range(5000):
            t_max_current = schedule.get_t_max(epoch)
            t_col = torch.rand(N, 1) * t_max_current
    """

    def __init__(self, t_max, total_epochs, warmup_fraction=0.3,
                 min_fraction=0.1):
        """
        Args:
            t_max: final (full) time horizon
            total_epochs: total training epochs
            warmup_fraction: fraction of epochs over which to expand
            min_fraction: starting t_max as a fraction of full t_max
        """
        self.t_max = t_max
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_fraction)
        self.min_t = t_max * min_fraction

    def get_t_max(self, epoch):
        """Return the current time horizon for this epoch."""
        if epoch >= self.warmup_epochs:
            return self.t_max
        progress = epoch / self.warmup_epochs
        return self.min_t + (self.t_max - self.min_t) * progress
