import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from hopfield import HopfieldNetwork
from hopfield.utils import corrupt

LETTERS = {
    "A": ["..#..", ".#.#.", "#...#", "#####", "#...#", "#...#", "#...#"],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
}


def to_vec(rows):
    return np.array([1 if c == "#" else -1 for r in rows for c in r])


def show(vec, title):
    print(title)
    for r in vec.reshape(7, 5):
        print("".join("#" if v == 1 else "." for v in r))
    print()


rng = np.random.default_rng(1)
patterns = np.array([to_vec(v) for v in LETTERS.values()])
net = HopfieldNetwork()
net.train(patterns)

noisy = corrupt(patterns[0], 5, rng)
out = net.recall(noisy, seed=0)

show(patterns[0], "Original (A)")
show(noisy, "Noisy (5 flips)")
show(out["state"], "Recalled")
print("Exact recovery:", np.array_equal(out["state"], patterns[0]))
