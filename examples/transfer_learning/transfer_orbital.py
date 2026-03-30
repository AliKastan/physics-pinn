"""
Transfer Learning Experiment 2 — Orbital Eccentricity Transfer
===============================================================

Train on a circular orbit (e=0), then transfer to increasingly
elliptical orbits (e=0.3, 0.6).  Shows how transfer learning
handles increasing difficulty.

Usage:
    python examples/transfer_learning/transfer_orbital.py
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

from src.models.orbital_pinn import PINNOrbital
from src.training.trainer import Trainer
from src.training.losses import orbital_ic_loss
from src.training.transfer import save_pretrained, load_pretrained, fine_tune
from src.utils.validation import solve_orbit_ode, setup_orbital_ics
from src.utils.metrics import l2_relative_error


def main():
    GM = 1.0
    epochs_full = 3000
    epochs_ft = 300
    eccentricities = [0.0, 0.3, 0.6]

    # ------------------------------------------------------------------
    # Train source on circular orbit
    # ------------------------------------------------------------------
    e_src = 0.0
    x0, y0, vx0, vy0, period = setup_orbital_ics(e_src, GM)
    t_max = 0.5 * period
    x0_t = torch.tensor(x0, dtype=torch.float32)
    y0_t = torch.tensor(y0, dtype=torch.float32)
    vx0_t = torch.tensor(vx0, dtype=torch.float32)
    vy0_t = torch.tensor(vy0, dtype=torch.float32)

    print(f"Training source: e={e_src} (circular), {epochs_full} epochs")
    torch.manual_seed(42)
    model_src = PINNOrbital()
    trainer = Trainer(model_src, {
        "epochs": epochs_full, "n_collocation": 500, "ic_weight": 50.0,
        "t_max": t_max, "lr": 1e-3,
    }, physics_params={"GM": GM})
    ic_fn = lambda: orbital_ic_loss(model_src, x0_t, y0_t, vx0_t, vy0_t)
    losses_src, _ = trainer.train(ic_fn, verbose=False)
    save_pretrained(model_src, "orbital_circular", config={"e": e_src, "GM": GM})
    print(f"  Final loss: {losses_src[-1]:.6f}\n")

    # ------------------------------------------------------------------
    # Transfer to each eccentricity
    # ------------------------------------------------------------------
    results = {}
    for e_tgt in eccentricities:
        x0_e, y0_e, vx0_e, vy0_e, p_e = setup_orbital_ics(e_tgt, GM)
        t_max_e = 0.5 * p_e
        x0e_t = torch.tensor(x0_e, dtype=torch.float32)
        y0e_t = torch.tensor(y0_e, dtype=torch.float32)
        vx0e_t = torch.tensor(vx0_e, dtype=torch.float32)
        vy0e_t = torch.tensor(vy0_e, dtype=torch.float32)

        t_eval = np.linspace(0, t_max_e, 500)
        _, x_gt, y_gt, _, _ = solve_orbit_ode(
            x0_e, y0_e, vx0_e, vy0_e, (0, t_max_e), t_eval, GM=GM)

        # Fine-tune
        torch.manual_seed(42)
        model_ft, _ = load_pretrained(PINNOrbital, "orbital_circular")
        ic_fn_ft = lambda: orbital_ic_loss(model_ft, x0e_t, y0e_t, vx0e_t, vy0e_t)
        losses_ft = fine_tune(model_ft, {
            "t_max": t_max_e, "n_collocation": 500, "ic_weight": 50.0,
        }, ic_fn_ft, {"GM": GM}, epochs=epochs_ft, lr=5e-4, verbose=False)

        model_ft.eval()
        with torch.no_grad():
            out_ft = model_ft(torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)).numpy()
        err_ft = l2_relative_error(
            np.column_stack([out_ft[:, 0], out_ft[:, 1]]),
            np.column_stack([x_gt, y_gt]))

        # Scratch
        torch.manual_seed(42)
        model_sc = PINNOrbital()
        trainer_sc = Trainer(model_sc, {
            "epochs": epochs_full, "n_collocation": 500, "ic_weight": 50.0,
            "t_max": t_max_e, "lr": 1e-3,
        }, physics_params={"GM": GM})
        ic_fn_sc = lambda: orbital_ic_loss(model_sc, x0e_t, y0e_t, vx0e_t, vy0e_t)
        losses_sc, _ = trainer_sc.train(ic_fn_sc, verbose=False)

        model_sc.eval()
        with torch.no_grad():
            out_sc = model_sc(torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)).numpy()
        err_sc = l2_relative_error(
            np.column_stack([out_sc[:, 0], out_sc[:, 1]]),
            np.column_stack([x_gt, y_gt]))

        results[e_tgt] = {
            "err_ft": err_ft, "err_scratch": err_sc,
            "losses_ft": losses_ft, "losses_scratch": losses_sc,
        }
        print(f"  e={e_tgt}: FT={err_ft:.4e} ({epochs_ft}ep), "
              f"Scratch={err_sc:.4e} ({epochs_full}ep)")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Orbital Transfer: Circular -> Elliptical', fontsize=14,
                 fontweight='bold')

    eccs = list(results.keys())
    errs_ft = [results[e]["err_ft"] for e in eccs]
    errs_sc = [results[e]["err_scratch"] for e in eccs]

    ax = axes[0]
    ax.semilogy(eccs, errs_sc, 'ro-', ms=8, lw=2, label=f'Scratch ({epochs_full}ep)')
    ax.semilogy(eccs, errs_ft, 'gs-', ms=8, lw=2, label=f'Fine-tuned ({epochs_ft}ep)')
    ax.set_xlabel('Eccentricity')
    ax.set_ylabel('L2 Relative Error')
    ax.set_title('Accuracy vs Eccentricity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for e in eccs:
        ax.semilogy(results[e]["losses_scratch"], alpha=0.5, label=f'Scratch e={e}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('From-Scratch Training Curves')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for e in eccs:
        ax.semilogy(results[e]["losses_ft"], alpha=0.7, label=f'FT e={e}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Fine-Tuning Curves')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'transfer_orbital_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
