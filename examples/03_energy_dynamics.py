import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from hopfield import HopfieldNetwork, calculate_energy
from hopfield.utils import random_patterns

rng = np.random.default_rng(0)
patterns = random_patterns(4, 100, rng)
net = HopfieldNetwork()
net.train(patterns)

print("Energy of stored patterns:",
      np.round([calculate_energy(p, net.weights) for p in patterns], 3))

start = rng.choice([-1, 1], size=100)
out = net.recall(start, seed=0)
print("Energy per sweep from random start:", np.round(out["energy_history"], 3))
print("Monotonically non-increasing:", bool(np.all(np.diff(out["energy_history"]) <= 1e-9)))
print("Converged:", out["converged"], "in", out["sweeps"], "sweeps")
