import numpy as np


def hebbian_learning(patterns):
    """Hebbian weights W = (1/N) * sum_mu xi^mu xi^mu^T, zero diagonal.

    patterns: array (num_patterns, num_neurons) with values in {-1, +1}.
    """
    patterns = np.asarray(patterns)
    if patterns.ndim != 2:
        raise ValueError("Patterns must be a 2D array.")
    if not np.all(np.isin(patterns, [-1, 1])):
        raise ValueError("Patterns must contain only -1 and +1.")

    num_neurons = patterns.shape[1]
    weights = patterns.T @ patterns / num_neurons
    np.fill_diagonal(weights, 0)
    return weights
