# Mathematical Background

## Physics-Informed Neural Networks (PINNs)

### Core Idea

A PINN solves a differential equation by training a neural network `N(t; theta)` to minimise the equation's residual. For an ODE `F(t, u, du/dt, d^2u/dt^2) = 0` with initial condition `u(0) = u_0`:

```
L_total = L_physics + lambda_ic * L_ic
        = (1/N) sum_i |F(t_i, N(t_i), N'(t_i), N''(t_i))|^2
        + lambda_ic * |N(0) - u_0|^2
```

where `t_i` are **collocation points** sampled randomly in the domain.

Derivatives `N'(t)` and `N''(t)` are computed exactly via PyTorch's automatic differentiation -- no finite differences needed.

### Why Tanh Activation?

The physics loss requires derivatives through the network. `tanh` is `C^infinity` (infinitely differentiable), while ReLU has a discontinuous second derivative that degrades PINN training. All models in this framework use tanh.

---

## Hamiltonian Neural Networks (HNNs)

Instead of learning `u(t)` directly, an HNN learns the scalar Hamiltonian `H(q, p)` and derives dynamics via Hamilton's equations:

```
dq/dt =  dH/dp
dp/dt = -dH/dq
```

**Energy conservation is structural:** `dH/dt = (dH/dq)(dq/dt) + (dH/dp)(dp/dt) = (dH/dq)(dH/dp) + (dH/dp)(-dH/dq) = 0`. This holds exactly regardless of network accuracy.

The trade-off: HNNs need training data (state-derivative pairs), while standard PINNs need only the equation.

---

## PDE Extension

For PDEs like the heat equation `du/dt = alpha * d^2u/dx^2` on `[0, L] x [0, T]`:

- **Input:** `(x, t)` (2D instead of 1D)
- **Interior loss:** PDE residual at random interior points
- **Boundary loss:** Dirichlet/Neumann conditions at `x = 0, L`
- **Initial condition loss:** `u(x, 0) = f(x)` at random `x`

The key challenge is computing **second-order spatial derivatives** via two passes of autograd:

```python
du_dx = autograd.grad(u, x, create_graph=True)[0]
d2u_dx2 = autograd.grad(du_dx, x, create_graph=True)[0]
```

---

## Inverse Problems

In the inverse setting, physical parameters (e.g., gravity `g`, pendulum length `L`) are promoted to `nn.Parameter` and optimised alongside network weights. The data loss constrains the trajectory to match observations; the physics loss constrains the form of the solution. Together they identify both the trajectory and the parameters.

---

## Transfer Learning

PINNs learn a hierarchy of features: early layers capture general function approximation (smooth basis functions), while late layers encode physics-specific mappings. Transfer learning freezes early layers and fine-tunes only the last 1-2 layers on a new configuration, reusing the learned feature representations.
