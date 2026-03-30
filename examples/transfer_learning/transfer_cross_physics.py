"""
Transfer Learning Experiment 3 — Cross-Physics Transfer
========================================================

Train on the pendulum, then transfer weights to the orbital PINN.
Since both use the same hidden layer sizes (64 neurons), the hidden
layers' weights can be copied directly.  Only the input (1->1 same)
and output (2 vs 4) layers differ.

Analysis: early layers (general function approximation) transfer
well; late layers (physics-specific) must be retrained.

Usage:
    python examples/transfer_learning/transfer_cross_physics.py
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
from src.models.orbital_pinn import PINNOrbital
from src.training.trainer import Trainer
from src.training.losses import pendulum_ic_loss, orbital_ic_loss
from src.training.transfer import (
    save_pretrained, transfer_weights_cross_physics,
    compute_layer_gradients,
)
from src.utils.validation import solve_orbit_ode, setup_orbital_ics
from src.utils.metrics import l2_relative_error


def main():
    # ------------------------------------------------------------------
    # Step 1: Train pendulum source
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Train Pendulum (source)")
    print("=" * 60)
    torch.manual_seed(42)
    model_pend = PINNPendulum(hidden_size=64, num_hidden_layers=3)
    trainer_p = Trainer(model_pend, {
        "epochs": 3000, "n_collocation": 500, "ic_weight": 20.0,
        "t_max": 10.0, "lr": 1e-3,
    }, physics_params={"g": 9.81, "L": 1.0})
    ic_fn_p = lambda: pendulum_ic_loss(model_pend, 0.5, 0.0)
    losses_pend, _ = trainer_p.train(ic_fn_p, verbose=False)
    print(f"  Pendulum final loss: {losses_pend[-1]:.6f}")

    # ------------------------------------------------------------------
    # Step 2: Transfer hidden layers to orbital model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Transfer pendulum -> orbital (hidden layers only)")
    print("=" * 60)

    # Create an orbital model with the SAME hidden_size so layers match
    torch.manual_seed(42)
    model_transfer = PINNOrbital(hidden_size=64, num_hidden_layers=3)
    n_trans, n_skip = transfer_weights_cross_physics(model_pend, model_transfer)
    print(f"  Transferred: {n_trans} layers, Skipped: {n_skip} layers")

    # Fine-tune transferred model on orbital problem
    GM = 1.0
    x0, y0, vx0, vy0, period = setup_orbital_ics(0.3, GM)
    t_max = 0.5 * period
    x0_t = torch.tensor(x0, dtype=torch.float32)
    y0_t = torch.tensor(y0, dtype=torch.float32)
    vx0_t = torch.tensor(vx0, dtype=torch.float32)
    vy0_t = torch.tensor(vy0, dtype=torch.float32)
    epochs_ft = 2000

    ic_fn_o = lambda: orbital_ic_loss(model_transfer, x0_t, y0_t, vx0_t, vy0_t)
    optimizer_ft = torch.optim.Adam(model_transfer.parameters(), lr=1e-3)
    sched_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, patience=300, factor=0.5, min_lr=1e-6)
    losses_transfer = []
    for ep in range(epochs_ft):
        optimizer_ft.zero_grad()
        t_col = torch.rand(500, 1) * t_max
        loss_p = model_transfer.physics_loss(t_col, GM=GM)
        loss_i = ic_fn_o()
        total = loss_p + 50.0 * loss_i
        total.backward()
        optimizer_ft.step()
        sched_ft.step(total.item())
        losses_transfer.append(total.item())
    print(f"  Transfer+FT final loss: {losses_transfer[-1]:.6f}")

    # ------------------------------------------------------------------
    # Step 3: Train orbital from scratch for comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Orbital from scratch (same architecture)")
    print("=" * 60)
    torch.manual_seed(42)
    model_scratch = PINNOrbital(hidden_size=64, num_hidden_layers=3)
    trainer_sc = Trainer(model_scratch, {
        "epochs": epochs_ft, "n_collocation": 500, "ic_weight": 50.0,
        "t_max": t_max, "lr": 1e-3,
    }, physics_params={"GM": GM})
    ic_fn_sc = lambda: orbital_ic_loss(model_scratch, x0_t, y0_t, vx0_t, vy0_t)
    losses_scratch, _ = trainer_sc.train(ic_fn_sc, verbose=False)
    print(f"  Scratch final loss: {losses_scratch[-1]:.6f}")

    # ------------------------------------------------------------------
    # Step 4: Layer-wise gradient analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Layer-wise gradient analysis")
    print("=" * 60)
    t_col_grad = torch.rand(200, 1) * t_max
    grads_transfer = compute_layer_gradients(
        model_transfer, t_col_grad, ic_fn_o, {"GM": GM}, ic_weight=50.0)
    grads_scratch = compute_layer_gradients(
        model_scratch, t_col_grad, ic_fn_sc, {"GM": GM}, ic_weight=50.0)

    print("  Layer gradients (transfer vs scratch):")
    for (name_t, g_t), (name_s, g_s) in zip(grads_transfer, grads_scratch):
        print(f"    {name_t:30s}: transfer={g_t:.4f}  scratch={g_s:.4f}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    t_eval = np.linspace(0, t_max, 500)
    _, x_gt, y_gt, _, _ = solve_orbit_ode(
        x0, y0, vx0, vy0, (0, t_max), t_eval, GM=GM)

    for m in [model_transfer, model_scratch]:
        m.eval()
    with torch.no_grad():
        t_t = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)
        out_tr = model_transfer(t_t).numpy()
        out_sc = model_scratch(t_t).numpy()

    gt_stack = np.column_stack([x_gt, y_gt])
    err_tr = l2_relative_error(np.column_stack([out_tr[:, 0], out_tr[:, 1]]), gt_stack)
    err_sc = l2_relative_error(np.column_stack([out_sc[:, 0], out_sc[:, 1]]), gt_stack)

    # ------------------------------------------------------------------
    # Plot: 2x2
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-Physics Transfer: Pendulum -> Orbital',
                 fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.semilogy(losses_transfer, 'g-', lw=1.5, alpha=0.7, label='Pend->Orb transfer')
    ax.semilogy(losses_scratch, 'r-', lw=1.5, alpha=0.7, label='Scratch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    bars = ax.bar(['Cross-physics\ntransfer', 'From scratch'],
                  [err_tr, err_sc], color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('L2 Relative Error')
    ax.set_title('Final Accuracy')
    for bar, e in zip(bars, [err_tr, err_sc]):
        ax.text(bar.get_x() + bar.get_width() / 2, e,
                f'{e:.2e}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0]
    names = [n.replace('network.', '').replace('.weight', ' W').replace('.bias', ' b')
             for n, _ in grads_transfer]
    g_tr = [g for _, g in grads_transfer]
    g_sc = [g for _, g in grads_scratch]
    x_pos = np.arange(len(names))
    ax.bar(x_pos - 0.15, g_tr, 0.3, label='Transfer', color='green', alpha=0.7)
    ax.bar(x_pos + 0.15, g_sc, 0.3, label='Scratch', color='red', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Layer-wise Gradients')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    ax.text(0.5, 0.5,
            f"Cross-physics transfer analysis\n"
            f"{'='*35}\n\n"
            f"Source: Pendulum (g=9.81, L=1.0)\n"
            f"Target: Orbital (e=0.3, GM=1.0)\n\n"
            f"Transferred layers: {n_trans}\n"
            f"Skipped layers: {n_skip}\n\n"
            f"Transfer error: {err_tr:.4e}\n"
            f"Scratch error:  {err_sc:.4e}\n\n"
            f"Early layers transfer general\n"
            f"function-approximation features.\n"
            f"Late layers are physics-specific.",
            transform=ax.transAxes, fontsize=10,
            va='center', ha='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    ax.axis('off')

    plt.tight_layout()
    out_path = 'transfer_cross_physics_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
