"""Learning rate scheduling utilities."""

import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(optimizer, scheduler_type="plateau", **kwargs):
    """
    Create a learning rate scheduler.

    Args:
        optimizer: torch optimizer
        scheduler_type: one of "plateau", "cosine", "step"
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
    elif scheduler_type == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 1000),
            gamma=kwargs.get("gamma", 0.5),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
