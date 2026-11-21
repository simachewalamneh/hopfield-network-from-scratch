import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from hopfield import HopfieldNetwork
from hopfield.utils import random_patterns, corrupt, hamming

rng = np.random.default_rng(0)
patterns = random_patterns(3, 100, rng)

net = HopfieldNetwork()
net.train(patterns)

noisy = corrupt(patterns[0], 10, rng)
result = net.recall(noisy, seed=42)

print("Bit errors before:", hamming(noisy, patterns[0]))
print("Bit errors after: ", hamming(result["state"], patterns[0]))
print("Exact recovery:   ", np.array_equal(result["state"], patterns[0]))
print("Converged:        ", result["converged"], "in", result["sweeps"], "sweeps")
print("Energy history:   ", np.round(result["energy_history"], 3))
