import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from hopfield import HopfieldNetwork
from hopfield.utils import random_patterns, corrupt, hamming

N, P, TRIALS = 100, 5, 30
rng = np.random.default_rng(0)

print(f"N={N}, P={P}")
print(f"{'noise':>6} {'bit accuracy':>13} {'exact recovery':>15}")
for frac in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    flips = int(frac * N)
    acc, exact = [], []
    for _ in range(TRIALS):
        patterns = random_patterns(P, N, rng)
        net = HopfieldNetwork()
        net.train(patterns)
        for p in patterns:
            noisy = corrupt(p, flips, rng)
            out = net.recall(noisy, seed=int(rng.integers(1_000_000)))["state"]
            e = hamming(out, p)
            acc.append(1 - e / N)
            exact.append(e == 0)
    print(f"{frac:>6.2f} {np.mean(acc):>13.3f} {np.mean(exact):>15.3f}")
