"""
Transfer learning utilities for PINNs.

PINNs learn general function-approximation features in early layers and
physics-specific mappings in later layers.  Transfer learning exploits
this hierarchy: freeze early layers (which transfer well across
parameter regimes) and fine-tune only the last 1-2 layers on a new
configuration, reaching comparable accuracy in 5-10x fewer epochs.

Usage:
    # Train on source problem
    model = PINNPendulum()
    trainer.train(...)
    save_pretrained(model, "pendulum_g9.81_L1.0", config={...})

    # Fine-tune on target problem
    model_ft = load_pretrained(PINNPendulum, "pendulum_g9.81_L1.0")
    fine_tune(model_ft, freeze_layers=3)
    trainer_ft.train(...)  # converges much faster
"""

import os
import json
import copy
import torch
import torch.nn as nn
import numpy as np

_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "checkpoints",
)


def save_pretrained(model, name, config=None, checkpoint_dir=None):
    """
    Save model weights and optional config to a checkpoint directory.

    Args:
        model: trained nn.Module
        name: checkpoint name (creates a subdirectory)
        config: optional dict of training config / physics params
        checkpoint_dir: override default checkpoint location

    Returns:
        path to the saved checkpoint directory
    """
    base = checkpoint_dir or _CHECKPOINT_DIR
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(path, "model.pt"))

    meta = {
        "model_class": model.__class__.__name__,
        "config": config or {},
    }
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return path


def load_pretrained(model_class, name, checkpoint_dir=None, **model_kwargs):
    """
    Load a pre-trained model from checkpoint.

    Args:
        model_class: the nn.Module subclass to instantiate
        name: checkpoint name
        checkpoint_dir: override default location
        **model_kwargs: passed to model_class constructor

    Returns:
        (model, meta_dict) where meta_dict contains config info
    """
    base = checkpoint_dir or _CHECKPOINT_DIR
    path = os.path.join(base, name)

    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)

    model = model_class(**model_kwargs)
    state = torch.load(os.path.join(path, "model.pt"),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model, meta


def freeze_early_layers(model, n_freeze=None):
    """
    Freeze the first n_freeze layers of the model's network.

    If n_freeze is None, freeze all but the last 2 layers (1 Linear + 1 Tanh
    or just the output Linear).  This is the common transfer-learning recipe:
    early layers learn general features, late layers adapt to new physics.

    Args:
        model: nn.Module with a .network attribute (Sequential)
        n_freeze: number of nn.Module children to freeze (from the start).
                  If None, auto-detect: freeze all except last 2.

    Returns:
        (n_frozen, n_trainable) counts for logging
    """
    children = list(model.network.children())
    if n_freeze is None:
        n_freeze = max(0, len(children) - 2)

    n_frozen = 0
    n_trainable = 0
    for i, child in enumerate(children):
        if i < n_freeze:
            for p in child.parameters():
                p.requires_grad = False
                n_frozen += p.numel()
        else:
            for p in child.parameters():
                p.requires_grad = True
                n_trainable += p.numel()

    return n_frozen, n_trainable


def unfreeze_all(model):
    """Unfreeze all parameters (undo freeze_early_layers)."""
    for p in model.parameters():
        p.requires_grad = True


def fine_tune(model, new_config, ic_loss_fn, physics_params,
              epochs=500, lr=1e-3, freeze_layers=None, verbose=True):
    """
    Fine-tune a pre-trained model on a new configuration.

    Freezes early layers, trains only late layers at a lower learning rate.

    Args:
        model: pre-trained PINN (will be modified in-place)
        new_config: dict with at minimum {t_max, ic_weight, n_collocation}
        ic_loss_fn: callable returning scalar IC loss for new ICs
        physics_params: dict of new physics parameters
        epochs: fine-tuning epochs (typically 5-10x fewer than from-scratch)
        lr: learning rate (often lower than original)
        freeze_layers: number of layers to freeze (None = auto)
        verbose: print progress

    Returns:
        loss_history: per-epoch loss during fine-tuning
    """
    n_frozen, n_trainable = freeze_early_layers(model, freeze_layers)

    if verbose:
        total = n_frozen + n_trainable
        print(f"Fine-tuning {model.__class__.__name__}")
        print(f"  Frozen: {n_frozen} params ({100*n_frozen/total:.0f}%)")
        print(f"  Trainable: {n_trainable} params ({100*n_trainable/total:.0f}%)")
        print(f"  Epochs: {epochs}, lr: {lr}")
        print("-" * 50)

    # Only optimize unfrozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.5, min_lr=1e-6)

    t_max = new_config.get("t_max", 10.0)
    n_col = new_config.get("n_collocation", 500)
    ic_weight = new_config.get("ic_weight", 20.0)

    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        t_col = torch.rand(n_col, 1) * t_max
        loss_phys = model.physics_loss(t_col, **physics_params)
        loss_ic = ic_loss_fn()
        total = loss_phys + ic_weight * loss_ic
        total.backward()
        optimizer.step()
        scheduler.step(total.item())
        loss_history.append(total.item())

        if verbose and (epoch + 1) % max(epochs // 5, 1) == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | Loss: {total.item():.6f}")

    if verbose:
        print("-" * 50)
        print(f"Fine-tune final loss: {loss_history[-1]:.6f}")

    # Unfreeze all after fine-tuning so model is fully usable
    unfreeze_all(model)
    return loss_history


def compute_layer_gradients(model, t_col, ic_loss_fn, physics_params,
                            ic_weight=20.0):
    """
    Compute per-layer gradient magnitudes for a single training step.

    Useful for analysing which layers adapt most during fine-tuning.

    Returns:
        list of (layer_name, grad_magnitude) tuples
    """
    model.zero_grad()
    loss_phys = model.physics_loss(t_col, **physics_params)
    loss_ic = ic_loss_fn()
    total = loss_phys + ic_weight * loss_ic
    total.backward()

    grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads.append((name, param.grad.norm().item()))
        else:
            grads.append((name, 0.0))
    return grads


def transfer_weights_cross_physics(source_model, target_model):
    """
    Transfer compatible weights from a source model to a target model.

    Only copies layers where shapes match.  Early hidden layers tend to
    match (both are FC with same hidden_size), while input/output layers
    differ (different input/output dimensions).

    Args:
        source_model: trained nn.Module
        target_model: new nn.Module (modified in-place)

    Returns:
        (n_transferred, n_skipped) layer counts
    """
    src_state = source_model.state_dict()
    tgt_state = target_model.state_dict()

    n_transferred = 0
    n_skipped = 0

    for key in tgt_state:
        if key in src_state and src_state[key].shape == tgt_state[key].shape:
            tgt_state[key] = src_state[key].clone()
            n_transferred += 1
        else:
            n_skipped += 1

    target_model.load_state_dict(tgt_state)
    return n_transferred, n_skipped
