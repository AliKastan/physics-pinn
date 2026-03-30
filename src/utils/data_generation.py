"""
Synthetic noisy data generation for inverse problem benchmarking.

Each generator solves the true ODE with known parameters, then adds
Gaussian noise to simulate real-world sensor data.
"""

import numpy as np
from scipy.integrate import solve_ivp


def generate_noisy_pendulum_data(theta_0, omega_0, t_max,
                                 g=9.81, L=1.0,
                                 n_obs=50, noise_std=0.05,
                                 n_dense=1000):
    """
    Generate noisy pendulum observations.

    Solves the pendulum ODE with true (g, L), then samples sparse noisy
    measurements of theta(t).

    Args:
        theta_0, omega_0: true initial conditions
        t_max: time horizon
        g, L: true physical parameters
        n_obs: number of noisy observation points
        noise_std: standard deviation of Gaussian noise on theta
        n_dense: number of dense ground-truth points

    Returns:
        dict with keys:
            t_obs, theta_obs: noisy observation arrays (n_obs,)
            t_dense, theta_true, omega_true: clean ground truth (n_dense,)
            g_true, L_true: the true parameters used
    """
    g_over_L = g / L

    # Dense ground truth
    t_dense = np.linspace(0, t_max, n_dense)
    sol = solve_ivp(
        lambda t, y: [y[1], -g_over_L * np.sin(y[0])],
        (0, t_max), [theta_0, omega_0],
        t_eval=t_dense, method='RK45', rtol=1e-10, atol=1e-12,
    )
    theta_true = sol.y[0]
    omega_true = sol.y[1]

    # Sparse noisy observations (avoid t=0, which is handled by IC loss)
    t_obs = np.sort(np.random.uniform(0.1, t_max, n_obs))
    sol_obs = solve_ivp(
        lambda t, y: [y[1], -g_over_L * np.sin(y[0])],
        (0, t_max), [theta_0, omega_0],
        t_eval=t_obs, method='RK45', rtol=1e-10, atol=1e-12,
    )
    theta_obs = sol_obs.y[0] + np.random.normal(0, noise_std, n_obs)

    return {
        't_obs': t_obs,
        'theta_obs': theta_obs,
        't_dense': t_dense,
        'theta_true': theta_true,
        'omega_true': omega_true,
        'g_true': g,
        'L_true': L,
    }


def generate_noisy_orbital_data(eccentricity=0.5, GM=1.0,
                                n_obs=80, noise_std=0.05,
                                n_dense=1500):
    """
    Generate noisy orbital position observations.

    Solves the two-body problem with true GM, then samples sparse noisy
    (x, y) measurements.

    Args:
        eccentricity: orbital eccentricity (0 < e < 1)
        GM: true gravitational parameter
        n_obs: number of noisy observation points
        noise_std: standard deviation of Gaussian noise on positions
        n_dense: dense ground-truth points

    Returns:
        dict with keys:
            t_obs, x_obs, y_obs: noisy observations (n_obs,)
            t_dense, x_true, y_true, vx_true, vy_true: ground truth
            ics: (x0, y0, vx0, vy0)
            t_max: simulation time (one period)
            GM_true: the true parameter
    """
    a = 1.0
    e = eccentricity
    x0 = a * (1.0 - e)
    y0 = 0.0
    vx0 = 0.0
    vy0 = np.sqrt(GM / a * (1.0 + e) / (1.0 - e))
    period = 2.0 * np.pi * np.sqrt(a ** 3 / GM)
    t_max = period

    def rhs(t, s):
        x, y, vx, vy = s
        r3 = (x ** 2 + y ** 2) ** 1.5
        return [vx, vy, -GM * x / r3, -GM * y / r3]

    # Dense ground truth
    t_dense = np.linspace(0, t_max, n_dense)
    sol = solve_ivp(
        rhs, (0, t_max), [x0, y0, vx0, vy0],
        t_eval=t_dense, method='RK45', rtol=1e-12, atol=1e-14,
    )
    x_true, y_true = sol.y[0], sol.y[1]
    vx_true, vy_true = sol.y[2], sol.y[3]

    # Sparse noisy position observations
    t_obs = np.sort(
        np.random.uniform(0.05 * t_max, 0.95 * t_max, n_obs))
    sol_obs = solve_ivp(
        rhs, (0, t_max), [x0, y0, vx0, vy0],
        t_eval=t_obs, method='RK45', rtol=1e-12, atol=1e-14,
    )
    x_obs = sol_obs.y[0] + np.random.normal(0, noise_std, n_obs)
    y_obs = sol_obs.y[1] + np.random.normal(0, noise_std, n_obs)

    return {
        't_obs': t_obs,
        'x_obs': x_obs,
        'y_obs': y_obs,
        't_dense': t_dense,
        'x_true': x_true,
        'y_true': y_true,
        'vx_true': vx_true,
        'vy_true': vy_true,
        'ics': (x0, y0, vx0, vy0),
        't_max': t_max,
        'GM_true': GM,
    }
