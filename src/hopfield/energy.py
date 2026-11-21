import numpy as np


def calculate_energy(state, weights):
    """E = -1/2 * s^T W s  (W symmetric, zero diagonal)."""
    state = np.asarray(state)
    return float(-0.5 * state @ weights @ state)
