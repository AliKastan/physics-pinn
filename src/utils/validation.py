"""Ground-truth solvers using scipy solve_ivp for comparison."""

import numpy as np
from scipy.integrate import solve_ivp


def solve_pendulum_ode(theta_0, omega_0, t_span, t_eval, g=9.81, L=1.0):
    """
    Solve the pendulum ODE using scipy's RK45.

    System: dy/dt = [omega, -(g/L)*sin(theta)]

    Returns:
        t, theta, omega arrays
    """
    def pendulum_rhs(t, y):
        theta, omega = y
        return [omega, -(g / L) * np.sin(theta)]

    sol = solve_ivp(
        pendulum_rhs, t_span, [theta_0, omega_0],
        t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12,
    )
    return sol.t, sol.y[0], sol.y[1]


def solve_orbit_ode(x0, y0, vx0, vy0, t_span, t_eval, GM=1.0):
    """
    Solve the two-body problem using scipy's RK45.

    State vector: [x, y, vx, vy]

    Returns:
        t, x, y, vx, vy arrays
    """
    def gravity_rhs(t, state):
        x, y, vx, vy = state
        r = np.sqrt(x ** 2 + y ** 2)
        r3 = r ** 3
        ax = -GM * x / r3
        ay = -GM * y / r3
        return [vx, vy, ax, ay]

    sol = solve_ivp(
        gravity_rhs, t_span, [x0, y0, vx0, vy0],
        t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-14,
    )
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def setup_orbital_ics(eccentricity=0.5, GM=1.0):
    """
    Set up initial conditions for an elliptical orbit starting at perihelion.

    Returns:
        x0, y0, vx0, vy0, period
    """
    a = 1.0
    e = eccentricity
    x0 = a * (1.0 - e)
    y0 = 0.0
    vx0 = 0.0
    vy0 = np.sqrt(GM / a * (1.0 + e) / (1.0 - e))
    period = 2.0 * np.pi * np.sqrt(a ** 3 / GM)
    return x0, y0, vx0, vy0, period
