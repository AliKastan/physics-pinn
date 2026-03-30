"""
Transfer Learning Experiment 1 — Pendulum Parameter Transfer
=============================================================

Demonstrates that a PINN trained on one pendulum configuration can be
fine-tuned to a different configuration in ~10x fewer epochs.

Setup:
    Source: g=9.81, L=1.0, theta_0=0.5  (5000 epochs)
    Target: g=9.81, L=2.0, theta_0=0.5  (fine-tune 500 epochs vs scratch 5000)

Usage:
    python examples/transfer_learning/transfer_pendulum.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.pendulum_pinn import PINNPendulum
from src.training.trainer import Trainer
from src.training.losses import pendulum_ic_loss
from src.training.transfer import save_pretrained, load_pretrained, fine_tune
from src.utils.validation import solve_pendulum_ode
from src.utils.metrics import l2_relative_error


def train_pendulum(g, L, theta_0, epochs, seed=42, verbose=True):
    """Train a pendulum PINN from scratch and return model + losses."""
    torch.manual_seed(seed)
    model = PINNPendulum()
    trainer = Trainer(model, {
        "epochs": epochs, "n_collocation": 500, "ic_weight": 20.0,
        "t_max": 10.0, "lr": 1e-3,
    }, physics_params={"g": g, "L": L})
    ic_fn = lambda: pendulum_ic_loss(model, theta_0, 0.0)
    losses, _ = trainer.train(ic_fn, verbose=verbose)
    return model, losses


def evaluate(model, theta_0, t_max, g, L):
    """Evaluate model against ground truth, return L2 relative error."""
    t_eval = np.linspace(0, t_max, 500)
    _, theta_gt, _ = solve_pendulum_ode(theta_0, 0.0, (0, t_max), t_eval, g=g, L=L)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)).numpy()
    return l2_relative_error(out[:, 0], theta_gt)


def main():
    g = 9.81
    theta_0 = 0.5
    t_max = 10.0
    L_source = 1.0
    L_target = 2.0
    epochs_full = 3000
    epochs_ft = 300

    # ------------------------------------------------------------------
    # Step 1: Train source model
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"SOURCE: g={g}, L={L_source}, theta_0={theta_0}")
    print("=" * 60)
    model_src, losses_src = train_pendulum(g, L_source, theta_0, epochs_full)

    save_pretrained(model_src, "pendulum_source", config={
        "g": g, "L": L_source, "theta_0": theta_0,
    })
    err_src = evaluate(model_src, theta_0, t_max, g, L_source)
    print(f"Source L2 rel error: {err_src:.4e}\n")

    # ------------------------------------------------------------------
    # Step 2: Fine-tune on target
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"FINE-TUNE: L={L_source} -> L={L_target}, {epochs_ft} epochs")
    print("=" * 60)
    model_ft, _ = load_pretrained(PINNPendulum, "pendulum_source")
    ic_fn_ft = lambda: pendulum_ic_loss(model_ft, theta_0, 0.0)
    losses_ft = fine_tune(model_ft, {
        "t_max": t_max, "n_collocation": 500, "ic_weight": 20.0,
    }, ic_fn_ft, {"g": g, "L": L_target}, epochs=epochs_ft, lr=5e-4)
    err_ft = evaluate(model_ft, theta_0, t_max, g, L_target)
    print(f"Fine-tuned L2 rel error: {err_ft:.4e}\n")

    # ------------------------------------------------------------------
    # Step 3: Train from scratch on target
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"FROM SCRATCH: g={g}, L={L_target}, {epochs_full} epochs")
    print("=" * 60)
    model_scratch, losses_scratch = train_pendulum(g, L_target, theta_0, epochs_full)
    err_scratch = evaluate(model_scratch, theta_0, t_max, g, L_target)
    print(f"Scratch L2 rel error: {err_scratch:.4e}\n")

    # ------------------------------------------------------------------
    # Step 4: Transfer across theta_0 values
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TRANSFER ACROSS INITIAL CONDITIONS")
    print("=" * 60)
    theta_targets = [0.3, 0.5, 0.8, 1.0, 1.2]
    errs_ft_theta = []
    errs_scratch_theta = []
    for th in theta_targets:
        # Fine-tune
        torch.manual_seed(42)
        m_ft, _ = load_pretrained(PINNPendulum, "pendulum_source")
        ic_fn = lambda: pendulum_ic_loss(m_ft, th, 0.0)
        fine_tune(m_ft, {"t_max": t_max, "n_collocation": 500, "ic_weight": 20.0},
                  ic_fn, {"g": g, "L": L_source}, epochs=epochs_ft, lr=5e-4,
                  verbose=False)
        errs_ft_theta.append(evaluate(m_ft, th, t_max, g, L_source))

        # Scratch
        torch.manual_seed(42)
        m_sc, _ = train_pendulum(g, L_source, th, epochs_full, verbose=False)
        errs_scratch_theta.append(evaluate(m_sc, th, t_max, g, L_source))
        print(f"  theta_0={th:.1f}: FT={errs_ft_theta[-1]:.4e}, "
              f"scratch={errs_scratch_theta[-1]:.4e}")

    # ------------------------------------------------------------------
    # Plot: 2x2 figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Transfer Learning — Pendulum PINN', fontsize=14,
                 fontweight='bold')

    # (0,0) Training curves
    ax = axes[0, 0]
    ax.semilogy(losses_src, 'b-', lw=1, alpha=0.7, label=f'Source (L={L_source})')
    ax.semilogy(losses_scratch, 'r-', lw=1, alpha=0.7,
                label=f'Scratch (L={L_target})')
    # Offset fine-tune to show it starts after source
    ft_epochs = np.arange(len(losses_ft)) + len(losses_src)
    ax.semilogy(ft_epochs, losses_ft, 'g-', lw=2,
                label=f'Fine-tuned ({epochs_ft} ep)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Accuracy comparison
    ax = axes[0, 1]
    methods = ['Source\n(L=1)', f'Scratch\n(L={L_target})',
               f'Fine-tuned\n(L={L_target})']
    errors = [err_src, err_scratch, err_ft]
    colors = ['blue', 'red', 'green']
    bars = ax.bar(methods, errors, color=colors, alpha=0.7)
    ax.set_ylabel('L2 Relative Error')
    ax.set_title('Accuracy Comparison')
    for bar, e in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, e,
                f'{e:.2e}', ha='center', va='bottom', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (1,0) Transfer across theta_0
    ax = axes[1, 0]
    ax.semilogy(theta_targets, errs_scratch_theta, 'ro-', lw=2, ms=8,
                label=f'Scratch ({epochs_full} ep)')
    ax.semilogy(theta_targets, errs_ft_theta, 'gs-', lw=2, ms=8,
                label=f'Fine-tuned ({epochs_ft} ep)')
    ax.set_xlabel('Initial angle theta_0 (rad)')
    ax.set_ylabel('L2 Relative Error')
    ax.set_title('Transfer Across Initial Conditions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Epoch savings
    ax = axes[1, 1]
    savings = epochs_full / epochs_ft
    ax.bar(['From Scratch', 'Fine-Tuned'], [epochs_full, epochs_ft],
           color=['red', 'green'], alpha=0.7)
    ax.set_ylabel('Epochs')
    ax.set_title(f'Training Cost ({savings:.0f}x savings)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = 'transfer_pendulum_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
