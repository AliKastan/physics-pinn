# Physics-Informed Neural Networks: Learning Physics from Differential Equations

A PyTorch implementation of Physics-Informed Neural Networks (PINNs) and Hamiltonian Neural Networks (HNNs) applied to classical mechanics. Includes forward simulation, inverse parameter inference, energy-conserving architectures, and an interactive Streamlit web interface.

---

## What Are PINNs?

Physics-Informed Neural Networks embed known physical laws directly into the training process of a neural network. Rather than learning from labeled simulation data, the network is trained to satisfy the governing differential equations at randomly sampled points in the domain. Automatic differentiation provides exact derivatives of the network output, which are substituted into the equations of motion to form a "physics residual" loss. The result is a continuous, differentiable surrogate model that respects the underlying physics by construction.

This approach was formalized by [Raissi, Perdikaris & Karniadakis (2019)](https://doi.org/10.1016/j.jcp.2018.10.045) and has since been applied to fluid dynamics, heat transfer, quantum mechanics, and beyond.

---

## Physics Background

### 1. Simple Pendulum

A rigid pendulum of length *L* under gravitational acceleration *g* satisfies the nonlinear ODE:

```
d^2 theta/dt^2 + (g/L) sin(theta) = 0
```

Decomposed into a first-order system with angular velocity omega = d theta/dt:

```
d theta/dt = omega
d omega/dt = -(g/L) sin(theta)
```

The total mechanical energy is conserved:

```
E = 1/2 m L^2 omega^2 - m g L cos(theta) = const.
```

Unlike the small-angle approximation (sin theta ~ theta), this formulation preserves the amplitude-dependent period of the real pendulum.

### 2. Two-Body Orbital Mechanics

A body orbiting a central mass under Newtonian gravity obeys:

```
d^2 x/dt^2 = -GM x / r^3
d^2 y/dt^2 = -GM y / r^3

where r = sqrt(x^2 + y^2)
```

Two conserved quantities serve as validation metrics:

```
Energy:             E = 1/2(vx^2 + vy^2) - GM/r       (constant for Kepler orbits)
Angular momentum:   L = x*vy - y*vx                    (Kepler's second law)
```

---

## Project Components

### Forward PINN (Pendulum & Orbital)

The standard PINN maps time to state variables — `t -> (theta, omega)` for the pendulum, `t -> (x, y, vx, vy)` for orbits. The network learns to satisfy the ODE everywhere in the time domain using only the equation and initial conditions. No simulation data is needed.

### Inverse Problem (`pinn_inverse.py`)

Given noisy observational data of a pendulum trajectory, the inverse PINN simultaneously infers the unknown physical parameters **g** (gravity) and **L** (pendulum length) while reconstructing the smooth trajectory. This works by treating g and L as trainable `torch.Parameter`s alongside the network weights.

The optimizer uses **two parameter groups** with different learning rates:
- **Network weights** (lr=1e-3): many parameters, standard NN training
- **Physical parameters g, L** (lr=1e-2): only 2 parameters, but they control the entire dynamics

The loss function combines three terms:
1. **Physics loss**: ODE residual at collocation points (using the current g and L estimates)
2. **Data loss**: MSE between predictions and noisy observations
3. **Initial condition loss**: pins the trajectory start

Starting from deliberately wrong initial guesses (g=5.0 instead of 9.81, L=2.0 instead of 1.0), the method converges to the true values within a few percent error.

### Hamiltonian Neural Network (`hnn_pendulum.py`)

Instead of learning the trajectory directly, the HNN learns the **Hamiltonian** H(q, p) — the total energy function. Equations of motion are derived via automatic differentiation:

```
dq/dt =  dH/dp     (Hamilton's first equation)
dp/dt = -dH/dq     (Hamilton's second equation)
```

The key advantage: **energy conservation is exact by construction**. The time derivative of H along any trajectory is:

```
dH/dt = (dH/dq)(dH/dp) + (dH/dp)(-dH/dq) = 0
```

This holds regardless of the network's accuracy — it's a structural property of Hamiltonian mechanics, not something the network needs to learn. The standard PINN, by contrast, only conserves energy as well as its training loss allows, and can drift over long time horizons.

### Comparison Notebook (`comparison.ipynb`)

A Jupyter notebook with head-to-head comparisons:
- **PINN vs HNN energy conservation**: the HNN maintains near-constant energy while the PINN drifts
- **PINN vs scipy accuracy**: error analysis over short (10s) and long (30s) time horizons
- **Inverse problem convergence**: watching g and L converge from wrong initial guesses to their true values

---

## Results

### Pendulum PINN

![Pendulum PINN Results](pendulum_pinn_results.png)

The PINN tracks the classical RK45 solution closely over multiple oscillation periods. The phase portrait (omega vs theta) forms a closed curve, indicating energy conservation. Error grows slowly at later times.

### Orbital Mechanics PINN

![Orbital PINN Results](orbital_pinn_results.png)

The PINN reproduces the elliptical orbit and captures the velocity increase at perihelion. Energy and angular momentum plots reveal how faithfully the network has internalized the conservation laws without being explicitly trained on them.

### Inverse Problem

![Inverse Problem Results](inverse_results.png)

Starting from wrong guesses (g=5.0, L=2.0), the inverse PINN converges to the true parameters (g=9.81, L=1.0) while simultaneously denoising the trajectory. The top-left plot shows the reconstructed trajectory threading through noisy observations, and the convergence plots show how g and L evolve during training.

### Hamiltonian Neural Network

![HNN Results](hnn_results.png)

The energy conservation comparison is the key result: the HNN (green) maintains near-constant energy over the full simulation, while the standard PINN (red) shows measurable drift. The learned Hamiltonian landscape (bottom-center) closely matches the true analytical Hamiltonian, confirming that the network has internalized the physics.

---

## Connection to Robot Learning

This work directly relates to a growing research direction in robotics: **learning dynamics from physical laws rather than copying fixed trajectories**.

Traditional robot learning (imitation learning, behavioral cloning) trains a policy to mimic demonstrated movements. This works for specific tasks but fails when conditions change — a robot arm trained to move a 1kg object can't automatically adapt to 2kg without retraining on new demonstrations.

Physics-informed approaches offer a fundamentally different paradigm:

- **Inverse PINNs for system identification**: A robot can observe its own arm swinging and infer physical parameters (link masses, joint friction, motor constants) from sensor data — the same way our inverse PINN infers g and L from noisy observations. This replaces manual calibration with automated, data-driven identification.

- **HNNs for energy-aware control**: A robot that learns its own Hamiltonian can predict how energy flows through its joints. This enables energy-efficient motion planning and guarantees that the learned model respects conservation laws — critical for contact-rich tasks where energy violations cause instability.

- **Generalization through physics**: A PINN-based dynamics model trained on one set of conditions (payload mass, joint configuration) can generalize to new conditions by re-solving with updated parameters, rather than collecting new demonstration data. The physics provides the inductive bias that raw neural networks lack.

- **Sim-to-real transfer**: Physics-informed models bridge the gap between simulation and reality. Rather than training entirely in simulation and hoping it transfers, PINNs can be fine-tuned on sparse real-world observations while maintaining consistency with known physics — reducing the amount of real robot data needed.

This is an active area of research with applications in legged locomotion, manipulation, and soft robotics, where accurate physics models are essential but hard to obtain analytically.

---

## How to Run Locally

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/AliKastan/physics-pinn.git
cd physics-pinn
pip install -r requirements.txt
```

### Run the simulations

```bash
# Pendulum PINN (trains ~5000 epochs, generates pendulum_pinn_results.png)
python pinn_pendulum.py

# Orbital mechanics PINN (trains ~8000 epochs, generates orbital_pinn_results.png)
python pinn_orbital.py

# Inverse problem (trains ~8000 epochs, generates inverse_results.png)
python pinn_inverse.py

# Hamiltonian Neural Network comparison (generates hnn_results.png)
python hnn_pendulum.py
```

### Run the comparison notebook

```bash
jupyter notebook comparison.ipynb
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
├── pinn_pendulum.py      # Forward PINN for pendulum dynamics
├── pinn_orbital.py       # Forward PINN for two-body orbital mechanics
├── pinn_inverse.py       # Inverse problem: infer g and L from noisy data
├── hnn_pendulum.py       # Hamiltonian Neural Network with PINN comparison
├── comparison.ipynb      # Jupyter notebook comparing all approaches
├── app.py                # Streamlit interactive web interface
├── requirements.txt      # Python dependencies
└── README.md
```

---

## How It Works

### Network Architecture

| Component | Pendulum PINN | Orbital PINN | HNN | Inverse PINN |
|-----------|--------------|-------------|-----|-------------|
| Input | t (1) | t (1) | q, p (2) | t (1) |
| Hidden layers | 3 x 64 | 4 x 128 | 3 x 64 | 3 x 64 |
| Activation | tanh | tanh | tanh | tanh |
| Output | theta, omega (2) | x, y, vx, vy (4) | H (1) | theta, omega (2) |
| Extra params | -- | -- | -- | g, L (trainable) |

**Why tanh?** The physics loss requires computing derivatives through the network via backpropagation. Tanh is infinitely differentiable; ReLU has discontinuous derivatives that degrade PINN convergence.

### Loss Functions

**Forward PINN**: `L = L_physics + lambda * L_IC`

**Inverse PINN**: `L = L_physics + alpha * L_data + lambda * L_IC` (physics uses trainable g, L)

**HNN**: `L = MSE(predicted derivatives, true derivatives)` (derivatives from Hamilton's equations)

### Validation

Solutions are compared against scipy's `solve_ivp` (Runge-Kutta 4/5) with tight tolerances (rtol=1e-10 to 1e-12). Key metrics:
- Absolute prediction error vs classical solver
- Relative energy drift |dE/E_0|
- Angular momentum drift (orbital case)
- Parameter recovery accuracy (inverse case)

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707.
2. Greydanus, S., Dzamba, M., & Cranmer, M. (2019). *Hamiltonian Neural Networks.* NeurIPS 2019.
3. Cranmer, M. et al. (2020). *Lagrangian Neural Networks.* ICLR 2020 Workshop on Integration of Deep Neural Models and Differential Equations.

---

## License

MIT
