# Physics-Informed Neural Networks: Learning Physics from Differential Equations

A PyTorch implementation of Physics-Informed Neural Networks (PINNs) applied to classical mechanics problems — simple pendulum motion and two-body orbital mechanics. Includes an interactive Streamlit web interface for real-time parameter exploration.

---

## What Are PINNs?

Physics-Informed Neural Networks embed known physical laws directly into the training process of a neural network. Rather than learning from labeled simulation data, the network is trained to satisfy the governing differential equations at randomly sampled points in the domain. Automatic differentiation provides exact derivatives of the network output, which are substituted into the equations of motion to form a "physics residual" loss. The result is a continuous, differentiable surrogate model that respects the underlying physics by construction.

This approach was formalized by [Raissi, Perdikaris & Karniadakis (2019)](https://doi.org/10.1016/j.jcp.2018.10.045) and has since been applied to fluid dynamics, heat transfer, quantum mechanics, and beyond.

---

## Physics Background

### 1. Simple Pendulum

A rigid pendulum of length *L* under gravitational acceleration *g* satisfies the nonlinear ODE:

```
d²θ/dt² + (g/L) sin(θ) = 0
```

Decomposed into a first-order system with angular velocity ω = dθ/dt:

```
dθ/dt = ω
dω/dt = -(g/L) sin(θ)
```

The total mechanical energy is conserved:

```
E = ½ m L² ω² - m g L cos(θ) = const.
```

Unlike the small-angle approximation (sin θ ≈ θ), this formulation preserves the amplitude-dependent period of the real pendulum.

### 2. Two-Body Orbital Mechanics

A body orbiting a central mass under Newtonian gravity obeys:

```
d²x/dt² = -GM x / r³
d²y/dt² = -GM y / r³

where r = sqrt(x² + y²)
```

Expanded into four first-order ODEs with velocities vx = dx/dt, vy = dy/dt:

```
dx/dt  = vx              dy/dt  = vy
dvx/dt = -GM x / r³      dvy/dt = -GM y / r³
```

Two conserved quantities serve as validation metrics:

```
Energy:             E = ½(vx² + vy²) - GM/r           (constant for Kepler orbits)
Angular momentum:   L = x·vy - y·vx                    (Kepler's second law)
```

Initial conditions are derived from the vis-viva equation at perihelion for a given eccentricity *e* and semi-major axis *a*.

---

## How It Works

### Network Architecture

| Component | Pendulum | Orbital |
|-----------|----------|---------|
| Input | t (1 neuron) | t (1 neuron) |
| Hidden layers | 3 × 64 neurons | 4 × 128 neurons |
| Activation | tanh | tanh |
| Output | θ, ω (2 neurons) | x, y, vx, vy (4 neurons) |

**Why tanh?** The physics loss requires computing first- and second-order derivatives through the network via backpropagation. Tanh is infinitely differentiable; ReLU has discontinuous derivatives that degrade PINN convergence.

### Loss Function

The total loss is a weighted sum of two terms:

```
L_total = L_physics + λ · L_initial_conditions
```

**Physics loss** — At each training step, *N* collocation points are sampled uniformly from the time domain. The network's output and its autograd-computed derivatives are substituted into the ODE. The mean squared residual measures how well the current network satisfies the equation:

```
L_physics = (1/N) Σ [ |dθ/dt - ω|² + |dω/dt + (g/L)sin(θ)|² ]
```

**Initial condition loss** — Evaluated at t = 0 to pin the trajectory:

```
L_IC = |θ_pred(0) - θ₀|² + |ω_pred(0) - ω₀|²
```

The IC weight λ is set higher than 1 (20 for pendulum, 50 for orbital) because the initial conditions uniquely determine the solution — a small IC error propagates across the entire trajectory.

### Training Details

- **Optimizer**: Adam with ReduceLROnPlateau scheduler
- **Collocation resampling**: New random points each epoch prevent overfitting to a fixed grid
- **Numerical stability**: Small epsilon (1e-8) added inside sqrt for the orbital 1/r³ term to prevent NaN gradients during early training

### Validation

Solutions are compared against scipy's `solve_ivp` (Runge-Kutta 4/5) with tight tolerances (rtol=1e-10 to 1e-12). Key metrics:

- Absolute prediction error vs. classical solver
- Relative energy drift |ΔE/E₀| (should remain near machine precision for the classical solver; measures how well the PINN implicitly conserves energy)
- Angular momentum drift (orbital case) — tests Kepler's second law

---

## Results & Visualizations

### Pendulum

| Angular Displacement | Phase Portrait |
|:---:|:---:|
| ![Pendulum displacement](pendulum_pinn_results.png) | *(generated at runtime)* |

The PINN tracks the classical solution closely over multiple oscillation periods. The phase portrait (ω vs. θ) forms a closed curve, indicating energy conservation.

### Orbital Mechanics

| Orbital Trajectory | Energy Conservation |
|:---:|:---:|
| ![Orbital results](orbital_pinn_results.png) | *(generated at runtime)* |

The PINN reproduces the elliptical orbit and captures the velocity increase at perihelion. Energy and angular momentum plots reveal how faithfully the network has internalized the conservation laws without being explicitly trained on them.

### Interactive Web App

| Streamlit Interface |
|:---:|
| *(screenshot of the running app)* |

The Streamlit app provides interactive Plotly charts with hover data, zoom, and pan. Training progress is shown in real time.

> **Note**: Run the standalone scripts first (`pinn_pendulum.py`, `pinn_orbital.py`) to generate the `.png` result files shown above.

---

## How to Run Locally

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone <repository-url>
cd physics-pinn
pip install -r requirements.txt
```

### Run the standalone simulations

```bash
# Pendulum PINN (trains ~5000 epochs, generates plots)
python pinn_pendulum.py

# Orbital mechanics PINN (trains ~8000 epochs, generates plots)
python pinn_orbital.py
```

### Run the interactive web app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Use the sidebar to adjust physical parameters and training settings, then click **Train & Compare**.

---

## Project Structure

```
physics-pinn/
├── pinn_pendulum.py      # Pendulum PINN: model, training, visualization
├── pinn_orbital.py       # Orbital PINN: model, training, energy analysis
├── app.py                # Streamlit web interface (self-contained)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Future Work

- **Hamiltonian Neural Networks** — Replace the standard MLP with a Hamiltonian-preserving architecture (HNN) that conserves energy by construction, rather than hoping the network learns it implicitly.
- **Three-body problem** — Extend the orbital simulation to three gravitationally interacting bodies, where no closed-form solution exists and PINNs could offer advantages over traditional integrators for long-horizon predictions.
- **Inverse problems** — Given noisy observational data of a pendulum or orbit, use the PINN to simultaneously infer unknown parameters (gravity, mass, pendulum length) while reconstructing the trajectory.
- **Transfer learning** — Train a PINN on one set of physical parameters, then fine-tune on a different configuration to study whether the network can generalize across parameter space.
- **Adaptive collocation** — Concentrate collocation points in regions of high residual (e.g., near perihelion in the orbital case) rather than sampling uniformly, improving accuracy where the dynamics are stiffest.
- **PDE extensions** — Apply the same framework to partial differential equations: heat equation, wave equation, or Navier-Stokes for fluid flow.

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707.
2. Greydanus, S., Dzamba, M., & Cranmer, M. (2019). *Hamiltonian Neural Networks.* NeurIPS 2019.
3. Cranmer, M. et al. (2020). *Lagrangian Neural Networks.* ICLR 2020 Workshop on Integration of Deep Neural Models and Differential Equations.

---

## License

MIT
