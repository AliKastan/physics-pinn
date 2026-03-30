"""
Tests for the benchmarking system and advanced metrics.

Verifies:
    1. New metrics functions compute correctly
    2. BenchmarkRunner loads configs
    3. Quick benchmark runs without error for each problem
    4. Results are JSON-serializable and can be saved/loaded
    5. Table generation works
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.utils.metrics import (
    l2_relative_error, spectral_convergence, training_efficiency,
    energy_drift, angular_momentum_drift,
)
from src.benchmarks.benchmark_runner import BenchmarkRunner


# =========================================================================
# Metrics tests
# =========================================================================

class TestAdvancedMetrics:

    def test_l2_relative_error_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        assert l2_relative_error(a, a) < 1e-15

    def test_l2_relative_error_nonzero(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        err = l2_relative_error(a, b)
        assert 0 < err < 0.1

    def test_spectral_convergence_identical(self):
        sig = np.sin(np.linspace(0, 4 * np.pi, 100))
        assert spectral_convergence(sig, sig) < 1e-10

    def test_spectral_convergence_different(self):
        t = np.linspace(0, 4 * np.pi, 100)
        sig1 = np.sin(t)
        sig2 = np.sin(2 * t)
        assert spectral_convergence(sig1, sig2) > 0.1

    def test_training_efficiency_found(self):
        losses = [1.0, 0.5, 0.1, 0.05, 0.005, 0.001]
        assert training_efficiency(losses, 0.01) == 4

    def test_training_efficiency_not_found(self):
        losses = [1.0, 0.5, 0.1]
        assert training_efficiency(losses, 0.001) == 3

    def test_energy_drift(self):
        E = np.array([1.0, 1.01, 0.99, 1.02])
        d = energy_drift(E)
        assert abs(d - 0.02) < 1e-10

    def test_angular_momentum_drift(self):
        L = np.array([2.0, 2.01, 1.99, 2.05])
        d = angular_momentum_drift(L)
        assert abs(d - 0.025) < 1e-10


# =========================================================================
# BenchmarkRunner tests
# =========================================================================

class TestBenchmarkRunner:

    def test_loads_configs(self):
        runner = BenchmarkRunner(mode="quick")
        assert "pendulum" in runner.configs
        assert "orbital" in runner.configs
        assert "heat" in runner.configs
        assert "wave" in runner.configs

    def test_quick_pendulum(self):
        runner = BenchmarkRunner(mode="quick")
        results = runner.run_pendulum()
        assert "standard" in results
        assert "fourier" in results
        assert "adaptive" in results
        for method, metrics in results.items():
            assert "final_loss" in metrics
            assert "wall_time_s" in metrics
            assert metrics["final_loss"] > 0

    def test_quick_orbital(self):
        runner = BenchmarkRunner(mode="quick")
        results = runner.run_orbital()
        assert "standard" in results
        for metrics in results.values():
            assert "final_loss" in metrics
            assert "energy_drift" in metrics

    def test_quick_heat(self):
        runner = BenchmarkRunner(mode="quick")
        results = runner.run_heat()
        assert "standard" in results
        assert results["standard"]["final_loss"] > 0

    def test_quick_wave(self):
        runner = BenchmarkRunner(mode="quick")
        results = runner.run_wave()
        assert "standard" in results
        assert results["standard"]["final_loss"] > 0

    def test_save_and_load(self, tmp_path):
        runner = BenchmarkRunner(mode="quick")
        runner.results = {"test": {"method": {"loss": 0.1}}}
        # Override results dir to tmp
        import src.benchmarks.benchmark_runner as bm
        old_dir = bm.RESULTS_DIR
        bm.RESULTS_DIR = str(tmp_path)
        try:
            path = runner.save_results("test_results.json")
            loaded = BenchmarkRunner.load_results("test_results.json")
            assert loaded["test"]["method"]["loss"] == 0.1
        finally:
            bm.RESULTS_DIR = old_dir

    def test_markdown_table(self):
        runner = BenchmarkRunner(mode="quick")
        runner.results = {
            "pendulum": {
                "standard": {
                    "l2_rel_theta": 0.05,
                    "energy_drift": 0.01,
                    "epochs_to_0.01": 1000,
                    "wall_time_s": 5.0,
                },
            },
        }
        table = runner.generate_markdown_table()
        assert "pendulum" in table
        assert "standard" in table
        assert "|" in table

    def test_latex_table(self):
        runner = BenchmarkRunner(mode="quick")
        runner.results = {
            "pendulum": {
                "standard": {
                    "l2_rel_theta": 0.05,
                    "energy_drift": 0.01,
                    "epochs_to_0.01": 1000,
                    "wall_time_s": 5.0,
                },
            },
        }
        table = runner.generate_latex_table()
        assert "tabular" in table
        assert "pendulum" in table
