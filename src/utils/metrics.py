"""Energy drift, L2 error, and angular momentum metrics."""

import numpy as np


def compute_energy_pendulum(theta, omega, g=9.81, L=1.0, m=1.0):
    """
    Pendulum mechanical energy: E = 0.5*m*L^2*omega^2 - m*g*L*cos(theta)

    Args:
        theta: angular displacement array
        omega: angular velocity array
        g, L, m: physical parameters

    Returns:
        E: total energy array
    """
    kinetic = 0.5 * m * L ** 2 * omega ** 2
    potential = -m * g * L * np.cos(theta)
    return kinetic + potential


def compute_energy_orbital(x, y, vx, vy, GM=1.0, m=1.0):
    """
    Orbital mechanical energy: E = 0.5*m*(vx^2 + vy^2) - GM*m/r

    Args:
        x, y, vx, vy: position and velocity arrays
        GM: gravitational parameter
        m: orbiting body mass

    Returns:
        E: total energy array
    """
    r = np.sqrt(x ** 2 + y ** 2)
    kinetic = 0.5 * m * (vx ** 2 + vy ** 2)
    potential = -GM * m / r
    return kinetic + potential


def compute_angular_momentum(x, y, vx, vy, m=1.0):
    """
    Angular momentum: L = m*(x*vy - y*vx)

    Conservation of L is Kepler's second law.

    Returns:
        L: angular momentum array
    """
    return m * (x * vy - y * vx)


def relative_energy_drift(E):
    """Compute max |dE/E0| — relative energy drift from initial value."""
    E0 = E[0]
    return np.max(np.abs((E - E0) / (np.abs(E0) + 1e-16)))


def l2_error(predicted, reference):
    """Compute L2 error between predicted and reference trajectories."""
    return np.sqrt(np.mean((predicted - reference) ** 2))


def compute_hamiltonian_pendulum(q, p, g=9.81, L=1.0, m=1.0):
    """
    True pendulum Hamiltonian: H = p^2/(2mL^2) - mgL*cos(q).

    Args:
        q: angle array (theta)
        p: angular momentum array (m*L^2*omega)
        g, L, m: physical parameters

    Returns:
        H: Hamiltonian (total energy) array
    """
    return p ** 2 / (2 * m * L ** 2) - m * g * L * np.cos(q)
