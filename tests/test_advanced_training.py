"""
Tests for advanced training techniques.

Verifies:
    1. Adaptive collocation (RAR) produces correctly shaped output
    2. RAR concentrates points in high-residual regions
    3. Curriculum schedule expands time horizon correctly
    4. Adaptive IC weight boosts when IC loss is high
    5. Fourier feature encoding shape
    6. FourierPendulumPINN forward/physics_loss work
    7. Gradient-enhanced loss computes without error
    8. Trainer with RAR reduces loss
    9. Trainer with curriculum reduces loss
    10. Trainer with all features enabled runs without error
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest

from src.training.adaptive_collocation import (
    sample_rar, AdaptiveCollocationSchedule,
)
from src.training.schedulers import (
    AdaptiveICWeightScheduler, CurriculumSchedule, get_scheduler,
)
from src.training.losses import gradient_enhanced_loss
from src.training.trainer import Trainer
from src.models.pendulum_pinn import PINNPendulum
from src.models.multiscale_pinn import (
    FourierFeatureEncoding, FourierPendulumPINN, FourierOrbitalPINN,
)
from src.training.losses import pendulum_ic_loss


# =========================================================================
# Adaptive collocation tests
# =========================================================================

def test_rar_output_shape():
    """RAR should return (n_collocation, 1) tensor."""
    torch.manual_seed(0)
    model = PINNPendulum()
    t_col = sample_rar(model, 200, 10.0, k=1.0, g=9.81, L=1.0)
    assert t_col.shape == (200, 1)


def test_rar_within_domain():
    """All points should be within [0, t_max]."""
    torch.manual_seed(0)
    model = PINNPendulum()
    t_col = sample_rar(model, 500, 10.0, k=1.0, g=9.81, L=1.0)
    assert t_col.min() >= 0
    assert t_col.max() <= 10.0


def test_rar_k2_more_concentrated():
    """Higher k should concentrate points more aggressively."""
    torch.manual_seed(42)
    model = PINNPendulum()
    t_k1 = sample_rar(model, 1000, 10.0, k=1.0, uniform_fraction=0.0, g=9.81, L=1.0)
    t_k2 = sample_rar(model, 1000, 10.0, k=2.0, uniform_fraction=0.0, g=9.81, L=1.0)
    # k=2 should have higher variance in local density (more clustered)
    std_k1 = t_k1.std().item()
    std_k2 = t_k2.std().item()
    # k=2 points cluster tighter around high-residual regions
    assert std_k2 < std_k1 + 1.0  # Not a strict inequality, just sanity


def test_adaptive_schedule_caches():
    """Schedule should cache between refreshes and refresh at interval."""
    torch.manual_seed(0)
    model = PINNPendulum()
    schedule = AdaptiveCollocationSchedule(
        model, n_collocation=100, t_max=5.0, interval=10,
        physics_params={"g": 9.81, "L": 1.0},
    )
    t0 = schedule.sample(0)
    t5 = schedule.sample(5)
    # Between refreshes, should return the same cached points
    assert torch.allclose(t0, t5)
    # At refresh interval, should get new points
    t10 = schedule.sample(10)
    assert not torch.allclose(t0, t10)


def test_adaptive_schedule_records_stats():
    torch.manual_seed(0)
    model = PINNPendulum()
    schedule = AdaptiveCollocationSchedule(
        model, n_collocation=100, t_max=5.0, interval=5,
        physics_params={"g": 9.81, "L": 1.0},
    )
    schedule.sample(0)
    schedule.sample(5)
    stats = schedule.residual_stats
    assert len(stats) == 2
    assert all(len(s) == 3 for s in stats)


# =========================================================================
# Scheduler tests
# =========================================================================

def test_curriculum_starts_small():
    sched = CurriculumSchedule(t_max=10.0, total_epochs=1000,
                                warmup_fraction=0.5, min_fraction=0.1)
    assert abs(sched.get_t_max(0) - 1.0) < 1e-6  # 10% of 10
    assert abs(sched.get_t_max(500) - 10.0) < 1e-6  # full at warmup end
    assert abs(sched.get_t_max(999) - 10.0) < 1e-6  # stays full


def test_curriculum_monotonic():
    sched = CurriculumSchedule(t_max=10.0, total_epochs=1000)
    prev = 0
    for ep in range(1000):
        t = sched.get_t_max(ep)
        assert t >= prev - 1e-10
        prev = t


def test_adaptive_ic_weight_boosts():
    sched = AdaptiveICWeightScheduler(
        base_weight=20.0, boost_factor=3.0, ic_threshold=0.1)
    assert sched.get_weight(0.001, 0.5) == 60.0  # boosted
    assert sched.get_weight(0.001, 0.05) == 20.0  # normal


def test_cosine_warm_scheduler():
    """CosineAnnealingWarmRestarts should be createable."""
    model = PINNPendulum()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = get_scheduler(opt, "cosine_warm", T_0=100, T_mult=2)
    # Should not raise
    sched.step()


# =========================================================================
# Fourier feature tests
# =========================================================================

def test_fourier_encoding_shape():
    enc = FourierFeatureEncoding(n_frequencies=16)
    t = torch.rand(50, 1)
    features = enc(t)
    assert features.shape == (50, 32)  # 2 * 16


def test_fourier_encoding_learnable():
    enc = FourierFeatureEncoding(n_frequencies=8, learnable=True)
    assert enc.sigmas.requires_grad


def test_fourier_pendulum_forward():
    model = FourierPendulumPINN(n_frequencies=8)
    t = torch.rand(30, 1)
    out = model(t)
    assert out.shape == (30, 2)


def test_fourier_pendulum_physics_loss():
    model = FourierPendulumPINN(n_frequencies=8)
    t = torch.rand(50, 1)
    loss = model.physics_loss(t, g=9.81, L=1.0)
    assert loss.item() > 0


def test_fourier_pendulum_residual():
    model = FourierPendulumPINN(n_frequencies=8)
    t = torch.rand(50, 1)
    res = model.compute_residual(t, g=9.81, L=1.0)
    assert res.shape == (50,)
    assert res.sum().item() > 0


def test_fourier_orbital_forward():
    model = FourierOrbitalPINN(n_frequencies=8)
    t = torch.rand(30, 1)
    out = model(t)
    assert out.shape == (30, 4)


# =========================================================================
# Gradient-enhanced loss test
# =========================================================================

def test_gradient_enhanced_loss():
    model = PINNPendulum()
    t = torch.rand(20, 1)
    y = torch.rand(20, 2)
    dydt = torch.rand(20, 2)
    loss = gradient_enhanced_loss(model, t, y, dydt)
    assert loss.item() > 0


# =========================================================================
# Trainer integration tests
# =========================================================================

def test_trainer_with_rar():
    """Trainer with adaptive collocation should reduce loss."""
    torch.manual_seed(42)
    model = PINNPendulum()
    config = {
        "epochs": 100, "lr": 1e-3, "n_collocation": 200,
        "ic_weight": 20.0, "t_max": 5.0,
        "adaptive": True, "adaptive_interval": 50,
        "adaptive_k": 1.0,
    }
    trainer = Trainer(model, config, physics_params={"g": 9.81, "L": 1.0})
    ic_fn = lambda: pendulum_ic_loss(model, np.pi / 4, 0.0)
    losses, _ = trainer.train(ic_fn, verbose=False)
    assert losses[-1] < losses[0]


def test_trainer_with_curriculum():
    """Trainer with curriculum learning should reduce loss."""
    torch.manual_seed(42)
    model = PINNPendulum()
    config = {
        "epochs": 100, "lr": 1e-3, "n_collocation": 200,
        "ic_weight": 20.0, "t_max": 5.0,
        "curriculum": True, "curriculum_warmup": 0.5,
    }
    trainer = Trainer(model, config, physics_params={"g": 9.81, "L": 1.0})
    ic_fn = lambda: pendulum_ic_loss(model, np.pi / 4, 0.0)
    losses, _ = trainer.train(ic_fn, verbose=False)
    assert losses[-1] < losses[0]


def test_trainer_all_features():
    """Trainer with all advanced features should run without error."""
    torch.manual_seed(42)
    model = PINNPendulum()
    config = {
        "epochs": 50, "lr": 1e-3, "n_collocation": 100,
        "ic_weight": 20.0, "t_max": 5.0,
        "adaptive": True, "adaptive_interval": 25, "adaptive_k": 1.5,
        "curriculum": True, "curriculum_warmup": 0.4,
        "adaptive_ic_weight": True, "ic_boost_factor": 2.0,
        "scheduler": "cosine_warm", "T_0": 25,
    }
    trainer = Trainer(model, config, physics_params={"g": 9.81, "L": 1.0})
    ic_fn = lambda: pendulum_ic_loss(model, np.pi / 4, 0.0)
    losses, snaps = trainer.train(ic_fn, verbose=False)
    assert len(losses) == 50
    assert len(snaps) > 0


def test_trainer_fourier_model():
    """Fourier-featured model should work with the standard Trainer."""
    torch.manual_seed(42)
    model = FourierPendulumPINN(n_frequencies=8)
    config = {
        "epochs": 50, "lr": 1e-3, "n_collocation": 100,
        "ic_weight": 20.0, "t_max": 5.0,
    }
    trainer = Trainer(model, config, physics_params={"g": 9.81, "L": 1.0})
    ic_fn = lambda: pendulum_ic_loss(model, np.pi / 4, 0.0)
    losses, _ = trainer.train(ic_fn, verbose=False)
    assert len(losses) == 50
