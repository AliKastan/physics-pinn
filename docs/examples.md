# Running the Examples

All examples are standalone scripts that can be run from the project root.

## Prerequisites

```bash
pip install -e ".[dev]"
```

## Example Scripts

### HNN vs PINN Comparison
```bash
python examples/hnn_vs_pinn_pendulum.py
```
Trains both a standard PINN and a Hamiltonian Neural Network on the pendulum, then compares energy conservation. Generates `hnn_vs_pinn_results.png`.

### Inverse Problem Demo
```bash
python examples/inverse_problem_demo.py
```
Infers pendulum parameters `g` and `L` from noisy observations. Generates `inverse_problem_results.png`.

### Wave vs Heat Comparison
```bash
python examples/wave_vs_heat_comparison.py
```
Shows diffusion (heat) vs propagation (wave) with the same framework. Generates `wave_vs_heat_results.png`.

### Three-Body Chaos
```bash
python examples/threebody_chaos.py
```
Demonstrates sensitivity to initial conditions with epsilon=1e-6 perturbation. Estimates Lyapunov exponent. Generates `threebody_chaos_results.png`.

### Adaptive vs Uniform Collocation
```bash
python examples/adaptive_vs_uniform.py
```
Compares RAR adaptive collocation against uniform sampling for orbital mechanics. Generates `adaptive_vs_uniform_orbital.png`.

### Transfer Learning Experiments
```bash
python examples/transfer_learning/transfer_pendulum.py
python examples/transfer_learning/transfer_orbital.py
python examples/transfer_learning/transfer_cross_physics.py
```
Three experiments demonstrating transfer across pendulum lengths, orbital eccentricities, and different physical systems.

## Interactive Web App

```bash
streamlit run src/app.py
```
Opens a browser with pages for all 9 systems. Each page has interactive controls, real-time training, and comparison plots.
