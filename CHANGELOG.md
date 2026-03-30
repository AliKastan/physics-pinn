# Changelog

All notable changes to the physics-pinn project.

## [1.0.0] - 2024

### Core Framework
- Modular Python package structure under `src/`
- Abstract `BasePINN` class with `forward()`, `physics_loss()`, `compute_residual()`, `train_step()`
- Generic `Trainer` class with configurable features via YAML/dict config
- Decoupled physics residual functions in `src/physics/equations.py`
- YAML-based hyperparameter configs in `configs/`

### Physical Systems
- **Simple Pendulum** (`PINNPendulum`): nonlinear ODE, energy conservation validation
- **Two-Body Orbital Mechanics** (`PINNOrbital`): Keplerian orbits, angular momentum tracking
- **1D Heat Equation** (`HeatPINN`): parabolic PDE, Fourier series analytical validation
- **1D Wave Equation** (`WavePINN`): hyperbolic PDE, mode decomposition, energy conservation
- **Three-Body Problem** (`ThreeBodyPINN`): chaotic N-body dynamics, 4 preset configurations (figure-eight, Lagrange triangle, Pythagorean, Sun-Earth-Moon)

### Specialised Architectures
- **Hamiltonian Neural Network** (`HamiltonianNN`): learns H(q,p), derives dynamics via Hamilton's equations, ~200x better energy conservation
- **Inverse PINNs** (`InversePendulumPINN`, `InverseOrbitalPINN`): trainable physical parameters, two-phase warmup training, parameter recovery from noisy data
- **Fourier Feature PINNs** (`FourierPendulumPINN`, `FourierOrbitalPINN`): overcome spectral bias via sinusoidal input encoding

### Advanced Training
- **Adaptive Collocation (RAR)**: residual-based point concentration with tunable exponent k
- **Curriculum Learning**: gradually expanding time horizon
- **Adaptive IC Weight**: dynamic loss weight adjustment based on IC loss magnitude
- **Cosine Annealing with Warm Restarts**: cyclic learning rate schedule
- **Gradient-Enhanced Loss**: matches both function values and derivatives

### Transfer Learning
- Save/load pretrained model checkpoints with metadata
- Layer freezing with auto-detection of transfer boundary
- Fine-tuning pipeline with ~10x training speedup
- Cross-physics weight transfer for matching hidden layers
- Layer-wise gradient analysis for understanding what transfers

### Evaluation & Benchmarking
- `BenchmarkRunner` with quick (CI) and full (paper) modes
- Metrics: L2 relative error, spectral convergence, energy drift, angular momentum drift, training efficiency
- JSON result persistence with LaTeX and Markdown table generation
- Noisy synthetic data generators for inverse problem benchmarking

### Interactive Web App (Streamlit)
- 9 interactive pages: Pendulum, Orbital, Inverse Problem, Heat Equation, Wave Equation, Three-Body Problem, Transfer Learning, Benchmarks
- Real-time training with progress bars
- Plotly interactive visualisations
- HNN comparison mode with energy conservation analysis
- Mode decomposition and energy tracking for wave equation

### Testing & CI
- 131 automated tests across 11 test files
- GitHub Actions CI: pytest + benchmark regression on Python 3.10-3.12
- Tests cover: shapes, residuals, training convergence, conservation laws, analytical validation, parameter recovery, transfer learning, benchmarking

### Documentation
- Comprehensive README with Mermaid architecture diagram
- CONTRIBUTING.md with guidelines for adding new systems
- CHANGELOG.md (this file)
- BibTeX citation block
