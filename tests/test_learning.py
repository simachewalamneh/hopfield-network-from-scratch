import numpy as np
import pytest
from hopfield import hebbian_learning


def test_symmetric_zero_diagonal():
    p = np.random.default_rng(0).choice([-1, 1], size=(3, 20))
    w = hebbian_learning(p)
    assert np.allclose(w, w.T)
    assert np.all(np.diag(w) == 0)


def test_rejects_non_bipolar():
    with pytest.raises(ValueError):
        hebbian_learning(np.array([[0, 1, 1]]))


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        hebbian_learning(np.array([1, -1, 1]))
