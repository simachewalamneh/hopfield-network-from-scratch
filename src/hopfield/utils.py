import numpy as np


def random_patterns(num_patterns, num_neurons, rng):
    return rng.choice([-1, 1], size=(num_patterns, num_neurons))


def corrupt(pattern, num_flips, rng):
    noisy = np.array(pattern, copy=True)
    idx = rng.choice(len(noisy), size=num_flips, replace=False)
    noisy[idx] *= -1
    return noisy


def hamming(a, b):
    return int(np.sum(np.asarray(a) != np.asarray(b)))
