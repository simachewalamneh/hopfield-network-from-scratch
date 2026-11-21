import numpy as np
from hopfield import HopfieldNetwork, calculate_energy


def test_energy_formula():
    w = np.array([[0, 1], [1, 0]], dtype=float)
    assert calculate_energy(np.array([1, 1]), w) == -1.0
    assert calculate_energy(np.array([1, -1]), w) == 1.0


def test_energy_never_increases():
    rng = np.random.default_rng(2)
    net = HopfieldNetwork()
    net.train(rng.choice([-1, 1], size=(3, 50)))
    out = net.recall(rng.choice([-1, 1], size=50), seed=0)
    assert np.all(np.diff(out["energy_history"]) <= 1e-9)
