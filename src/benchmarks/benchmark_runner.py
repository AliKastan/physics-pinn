"""
Systematic benchmark runner for all PINN models.

Trains each problem with multiple methods and configurations, records
metrics (loss, L2 error, energy drift, training time), and saves
results to JSON.  Can also generate LaTeX-formatted tables.

Usage:
    # Quick benchmarks (for CI — fast)
    runner = BenchmarkRunner(mode="quick")
    results = runner.run_all()

    # Full benchmarks (for paper — thorough)
    runner = BenchmarkRunner(mode="full")
    results = runner.run_all()

    # Single problem
    result = runner.run_pendulum()
"""

import json
import time
import os
import sys
import numpy as np
import torch
import yaml

# Ensure imports work from any cwd
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.pendulum_pinn import PINNPendulum
from src.models.orbital_pinn import PINNOrbital
from src.models.multiscale_pinn import FourierPendulumPINN, FourierOrbitalPINN
from src.models.heat_pinn import HeatPINN, train_heat_pinn, heat_analytical
from src.models.wave_pinn import WavePINN, train_wave_pinn, wave_analytical
from src.training.trainer import Trainer
from src.training.losses import pendulum_ic_loss, orbital_ic_loss
from src.utils.validation import solve_pendulum_ode, solve_orbit_ode, setup_orbital_ics
from src.utils.metrics import (
    l2_relative_error, relative_energy_drift, compute_energy_pendulum,
    compute_energy_orbital, compute_angular_momentum,
    training_efficiency, spectral_convergence,
)


CONFIGS_PATH = os.path.join(_THIS_DIR, "benchmark_configs.yaml")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")


def _load_configs():
    with open(CONFIGS_PATH) as f:
        return yaml.safe_load(f)


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


class BenchmarkRunner:
    """
    Runs PINN benchmarks systematically and records results.

    Args:
        mode: "quick" for CI (low epochs), "full" for paper-quality
        seed: random seed for reproducibility
    """

    def __init__(self, mode="quick", seed=42):
        self.mode = mode
        self.seed = seed
        self.configs = _load_configs()
        self.results = {}

    def _cfg(self, problem):
        return self.configs[problem][self.mode]

    def run_all(self, verbose=False):
        """Run all benchmarks and return combined results dict."""
        self.results["pendulum"] = self.run_pendulum(verbose=verbose)
        self.results["orbital"] = self.run_orbital(verbose=verbose)
        self.results["heat"] = self.run_heat(verbose=verbose)
        self.results["wave"] = self.run_wave(verbose=verbose)
        self.save_results()
        return self.results

    def run_pendulum(self, verbose=False):
        """Benchmark pendulum PINN: standard vs Fourier."""
        cfg = self._cfg("pendulum")
        theta_0 = np.radians(cfg["theta_0_deg"])
        omega_0 = 0.0
        g, L, t_max = cfg["g"], cfg["L"], cfg["t_max"]
        epochs = cfg["epochs"]
        n_col = cfg["n_collocation"]

        # Ground truth
        t_eval = np.linspace(0, t_max, 500)
        _, theta_gt, omega_gt = solve_pendulum_ode(
            theta_0, omega_0, (0, t_max), t_eval, g=g, L=L)

        results = {}

        # --- Standard PINN ---
        torch.manual_seed(self.seed)
        model = PINNPendulum()
        trainer = Trainer(model, {
            "epochs": epochs, "n_collocation": n_col,
            "ic_weight": cfg["ic_weight"], "t_max": t_max, "lr": 1e-3,
        }, physics_params={"g": g, "L": L})
        ic_fn = lambda: pendulum_ic_loss(model, theta_0, omega_0)

        t0 = time.time()
        losses, _ = trainer.train(ic_fn, verbose=verbose)
        wall_time = time.time() - t0

        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)).numpy()
        E = compute_energy_pendulum(out[:, 0], out[:, 1], g, L)

        results["standard"] = {
            "final_loss": float(losses[-1]),
            "l2_rel_theta": float(l2_relative_error(out[:, 0], theta_gt)),
            "energy_drift": float(relative_energy_drift(E)),
            "spectral_err": float(spectral_convergence(out[:, 0], theta_gt)),
            "epochs_to_0.01": int(training_efficiency(losses, 0.01)),
            "wall_time_s": round(wall_time, 2),
        }

        # --- Fourier PINN ---
        torch.manual_seed(self.seed)
        model_f = FourierPendulumPINN(n_frequencies=16)
        trainer_f = Trainer(model_f, {
            "epochs": epochs, "n_collocation": n_col,
            "ic_weight": cfg["ic_weight"], "t_max": t_max, "lr": 1e-3,
        }, physics_params={"g": g, "L": L})
        ic_fn_f = lambda: pendulum_ic_loss(model_f, theta_0, omega_0)

        t0 = time.time()
        losses_f, _ = trainer_f.train(ic_fn_f, verbose=verbose)
        wall_time_f = time.time() - t0

        model_f.eval()
        with torch.no_grad():
            out_f = model_f(torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)).numpy()
        E_f = compute_energy_pendulum(out_f[:, 0], out_f[:, 1], g, L)

        results["fourier"] = {
            "final_loss": float(losses_f[-1]),
            "l2_rel_theta": float(l2_relative_error(out_f[:, 0], theta_gt)),
            "energy_drift": float(relative_energy_drift(E_f)),
            "spectral_err": float(spectral_convergence(out_f[:, 0], theta_gt)),
            "epochs_to_0.01": int(training_efficiency(losses_f, 0.01)),
            "wall_time_s": round(wall_time_f, 2),
        }

        # --- Adaptive RAR ---
        torch.manual_seed(self.seed)
        model_a = PINNPendulum()
        trainer_a = Trainer(model_a, {
            "epochs": epochs, "n_collocation": n_col,
            "ic_weight": cfg["ic_weight"], "t_max": t_max, "lr": 1e-3,
            "adaptive": True, "adaptive_interval": max(epochs // 10, 50),
            "adaptive_k": 1.5,
        }, physics_params={"g": g, "L": L})
        ic_fn_a = lambda: pendulum_ic_loss(model_a, theta_0, omega_0)

        t0 = time.time()
        losses_a, _ = trainer_a.train(ic_fn_a, verbose=verbose)
        wall_time_a = time.time() - t0

        model_a.eval()
        with torch.no_grad():
            out_a = model_a(torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)).numpy()
        E_a = compute_energy_pendulum(out_a[:, 0], out_a[:, 1], g, L)

        results["adaptive"] = {
            "final_loss": float(losses_a[-1]),
            "l2_rel_theta": float(l2_relative_error(out_a[:, 0], theta_gt)),
            "energy_drift": float(relative_energy_drift(E_a)),
            "spectral_err": float(spectral_convergence(out_a[:, 0], theta_gt)),
            "epochs_to_0.01": int(training_efficiency(losses_a, 0.01)),
            "wall_time_s": round(wall_time_a, 2),
        }

        return results

    def run_orbital(self, verbose=False):
        """Benchmark orbital PINN: standard vs Fourier vs adaptive."""
        cfg = self._cfg("orbital")
        GM = cfg["GM"]
        x0, y0, vx0, vy0, period = setup_orbital_ics(cfg["eccentricity"], GM)
        t_max = cfg["n_orbits"] * period
        epochs = cfg["epochs"]
        n_col = cfg["n_collocation"]

        t_eval = np.linspace(0, t_max, 500)
        _, x_gt, y_gt, vx_gt, vy_gt = solve_orbit_ode(
            x0, y0, vx0, vy0, (0, t_max), t_eval, GM=GM)

        x0_t = torch.tensor(x0, dtype=torch.float32)
        y0_t = torch.tensor(y0, dtype=torch.float32)
        vx0_t = torch.tensor(vx0, dtype=torch.float32)
        vy0_t = torch.tensor(vy0, dtype=torch.float32)

        results = {}

        for name, ModelClass in [("standard", PINNOrbital),
                                  ("fourier", FourierOrbitalPINN)]:
            torch.manual_seed(self.seed)
            kwargs = {"n_frequencies": 16} if name == "fourier" else {}
            model = ModelClass(**kwargs)
            trainer_cfg = {
                "epochs": epochs, "n_collocation": n_col,
                "ic_weight": cfg["ic_weight"], "t_max": t_max, "lr": 1e-3,
            }
            trainer = Trainer(model, trainer_cfg, physics_params={"GM": GM})
            ic_fn = lambda: orbital_ic_loss(model, x0_t, y0_t, vx0_t, vy0_t)

            t0 = time.time()
            losses, _ = trainer.train(ic_fn, verbose=verbose)
            wall_time = time.time() - t0

            model.eval()
            with torch.no_grad():
                out = model(torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)).numpy()
            E = compute_energy_orbital(out[:, 0], out[:, 1], out[:, 2], out[:, 3], GM)
            L_arr = compute_angular_momentum(out[:, 0], out[:, 1], out[:, 2], out[:, 3])
            pos_err = np.sqrt((out[:, 0] - x_gt) ** 2 + (out[:, 1] - y_gt) ** 2)

            results[name] = {
                "final_loss": float(losses[-1]),
                "l2_rel_pos": float(l2_relative_error(
                    np.column_stack([out[:, 0], out[:, 1]]),
                    np.column_stack([x_gt, y_gt]))),
                "max_pos_error": float(np.max(pos_err)),
                "energy_drift": float(relative_energy_drift(E)),
                "angmom_drift": float(
                    np.max(np.abs((L_arr - L_arr[0]) / (np.abs(L_arr[0]) + 1e-16)))),
                "epochs_to_0.01": int(training_efficiency(losses, 0.01)),
                "wall_time_s": round(wall_time, 2),
            }

        return results

    def run_heat(self, verbose=False):
        """Benchmark heat equation PINN."""
        cfg = self._cfg("heat")
        torch.manual_seed(self.seed)

        t0 = time.time()
        model, losses = train_heat_pinn(
            alpha=cfg["alpha"], ic_type=cfg["ic_type"],
            epochs=cfg["epochs"], n_interior=cfg["n_interior"],
            n_bc=cfg["n_bc"], n_ic=cfg["n_ic"], verbose=verbose,
        )
        wall_time = time.time() - t0

        nx, nt = 50, 50
        x_arr = np.linspace(0, 1, nx)
        t_arr = np.linspace(0, 1, nt)
        X, T = np.meshgrid(x_arr, t_arr)

        model.eval()
        with torch.no_grad():
            u_pinn = model(
                torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1),
                torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1),
            ).numpy().reshape(X.shape)
        u_exact = heat_analytical(X, T, alpha=cfg["alpha"], ic_type=cfg["ic_type"])

        return {
            "standard": {
                "final_loss": float(losses[-1]),
                "l2_rel_error": float(l2_relative_error(u_pinn.flatten(), u_exact.flatten())),
                "max_abs_error": float(np.max(np.abs(u_pinn - u_exact))),
                "epochs_to_0.01": int(training_efficiency(losses, 0.01)),
                "wall_time_s": round(wall_time, 2),
            },
        }

    def run_wave(self, verbose=False):
        """Benchmark wave equation PINN."""
        cfg = self._cfg("wave")
        torch.manual_seed(self.seed)

        t0 = time.time()
        model, losses = train_wave_pinn(
            c=cfg["c"], t_max=cfg["t_max"], ic_type=cfg["ic_type"],
            epochs=cfg["epochs"], n_interior=cfg["n_interior"],
            n_bc=cfg["n_bc"], n_ic=cfg["n_ic"], verbose=verbose,
        )
        wall_time = time.time() - t0

        nx, nt = 50, 50
        x_arr = np.linspace(0, 1, nx)
        t_arr = np.linspace(0, cfg["t_max"], nt)
        X, T = np.meshgrid(x_arr, t_arr)

        model.eval()
        with torch.no_grad():
            u_pinn = model(
                torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1),
                torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1),
            ).numpy().reshape(X.shape)
        u_exact = wave_analytical(X, T, c=cfg["c"], ic_type=cfg["ic_type"])

        return {
            "standard": {
                "final_loss": float(losses[-1]),
                "l2_rel_error": float(l2_relative_error(u_pinn.flatten(), u_exact.flatten())),
                "max_abs_error": float(np.max(np.abs(u_pinn - u_exact))),
                "epochs_to_0.01": int(training_efficiency(losses, 0.01)),
                "wall_time_s": round(wall_time, 2),
            },
        }

    def save_results(self, filename=None):
        """Save all results to JSON."""
        _ensure_results_dir()
        if filename is None:
            filename = f"benchmark_{self.mode}.json"
        path = os.path.join(RESULTS_DIR, filename)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        return path

    @staticmethod
    def load_results(filename):
        """Load results from JSON."""
        path = os.path.join(RESULTS_DIR, filename)
        with open(path) as f:
            return json.load(f)

    def generate_latex_table(self):
        """Generate a LaTeX-formatted comparison table from results."""
        if not self.results:
            return "% No results to format"

        lines = [
            r"\begin{tabular}{l l r r r r}",
            r"\hline",
            r"Problem & Method & L2 Rel Error & Energy Drift & "
            r"Epochs to 0.01 & Time (s) \\",
            r"\hline",
        ]

        for problem, methods in self.results.items():
            for method, metrics in methods.items():
                l2 = metrics.get("l2_rel_theta",
                      metrics.get("l2_rel_pos",
                      metrics.get("l2_rel_error", "—")))
                ed = metrics.get("energy_drift", "—")
                eff = metrics.get("epochs_to_0.01", "—")
                wt = metrics.get("wall_time_s", "—")

                l2_str = f"{l2:.2e}" if isinstance(l2, float) else l2
                ed_str = f"{ed:.2e}" if isinstance(ed, float) else ed

                lines.append(
                    f"  {problem} & {method} & {l2_str} & {ed_str} "
                    f"& {eff} & {wt} \\\\"
                )

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    def generate_markdown_table(self):
        """Generate a markdown comparison table from results."""
        if not self.results:
            return ""

        rows = []
        for problem, methods in self.results.items():
            for method, metrics in methods.items():
                l2 = metrics.get("l2_rel_theta",
                      metrics.get("l2_rel_pos",
                      metrics.get("l2_rel_error", None)))
                ed = metrics.get("energy_drift", None)
                eff = metrics.get("epochs_to_0.01", None)
                wt = metrics.get("wall_time_s", None)

                l2_s = f"{l2:.2e}" if l2 is not None else "—"
                ed_s = f"{ed:.2e}" if ed is not None else "—"
                eff_s = str(eff) if eff is not None else "—"
                wt_s = f"{wt:.1f}" if wt is not None else "—"
                rows.append(f"| {problem} | {method} | {l2_s} | {ed_s} | {eff_s} | {wt_s} |")

        header = "| Problem | Method | L2 Rel Error | Energy Drift | Epochs to 0.01 | Time (s) |"
        sep = "|---------|--------|-------------|-------------|---------------|---------|"
        return "\n".join([header, sep] + rows)
