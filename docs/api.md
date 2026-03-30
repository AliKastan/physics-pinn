# API Reference

## Models (`src.models`)

### BasePINN
Abstract base class. Subclasses must implement `physics_loss()` and `compute_residual()`.

### PINNPendulum
Maps `t -> (theta, omega)`. Physics: `d^2theta/dt^2 + (g/L)*sin(theta) = 0`.

### PINNOrbital
Maps `t -> (x, y, vx, vy)`. Physics: Newtonian gravity `d^2r/dt^2 = -GM*r/|r|^3`.

### HeatPINN
Maps `(x, t) -> u`. Physics: `du/dt = alpha * d^2u/dx^2`.

### WavePINN
Maps `(x, t) -> u`. Physics: `d^2u/dt^2 = c^2 * d^2u/dx^2`.

### ThreeBodyPINN
Maps `t -> 12` state variables. Physics: three pairwise gravitational interactions.

### HamiltonianNN
Maps `(q, p) -> H` (scalar). Dynamics derived via Hamilton's equations.

### InversePendulumPINN / InverseOrbitalPINN
Standard PINNs with trainable physical parameters (`g`, `L`, `GM`).

### FourierPendulumPINN / FourierOrbitalPINN
PINNs with Fourier feature encoding on the input.

---

## Training (`src.training`)

### Trainer
Generic training loop. Config flags: `adaptive`, `curriculum`, `adaptive_ic_weight`.

### transfer.py
- `save_pretrained(model, name)` / `load_pretrained(cls, name)`
- `freeze_early_layers(model)` / `unfreeze_all(model)`
- `fine_tune(model, config, ic_fn, params)`
- `transfer_weights_cross_physics(source, target)`

### adaptive_collocation.py
- `sample_rar(model, n, t_max, k)`: P(x) ~ |residual(x)|^k
- `AdaptiveCollocationSchedule`: managed lifecycle with caching

### losses.py
- `physics_loss`, `pendulum_ic_loss`, `orbital_ic_loss`
- `gradient_enhanced_loss`: matches values + derivatives
- `hamiltonian_loss`, `energy_conservation_loss`

### schedulers.py
- `get_scheduler(optimizer, type)`: plateau, cosine, cosine_warm, step
- `CurriculumSchedule`: expanding time horizon
- `AdaptiveICWeightScheduler`: dynamic IC weight

---

## Utils (`src.utils`)

### metrics.py
- `l2_relative_error`, `spectral_convergence`, `training_efficiency`
- `compute_energy_pendulum`, `compute_energy_orbital`, `compute_angular_momentum`
- `relative_energy_drift`, `angular_momentum_drift`

### validation.py
- `solve_pendulum_ode`, `solve_orbit_ode`: scipy RK45/DOP853 solvers
- `setup_orbital_ics`: vis-viva equation for orbital ICs

### data_generation.py
- `generate_noisy_pendulum_data`, `generate_noisy_orbital_data`

---

## Benchmarks (`src.benchmarks`)

### BenchmarkRunner
- `run_all()`, `run_pendulum()`, `run_orbital()`, `run_heat()`, `run_wave()`
- `save_results()` / `load_results()`
- `generate_markdown_table()` / `generate_latex_table()`
