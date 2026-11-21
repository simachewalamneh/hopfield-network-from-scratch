import numpy as np
import pytest
from hopfield import HopfieldNetwork


def test_recall_before_train_raises():
    with pytest.raises(RuntimeError):
        HopfieldNetwork().recall(np.ones(5))


def test_single_pattern_is_stable():
    p = np.random.default_rng(1).choice([-1, 1], size=(1, 30))
    net = HopfieldNetwork()
    net.train(p)
    out = net.recall(p[0], seed=0)
    assert np.array_equal(out["state"], p[0])
    assert out["converged"]


def test_noisy_recall():
    rng = np.random.default_rng(3)
    p = rng.choice([-1, 1], size=(3, 100))
    net = HopfieldNetwork()
    net.train(p)
    noisy = p[0].copy()
    noisy[:8] *= -1
    out = net.recall(noisy, seed=0)
    assert np.array_equal(out["state"], p[0])


def test_negated_pattern_also_stable():
    rng = np.random.default_rng(4)
    p = rng.choice([-1, 1], size=(2, 60))
    net = HopfieldNetwork()
    net.train(p)
    out = net.recall(-p[0], seed=0)
    assert np.array_equal(out["state"], -p[0])
