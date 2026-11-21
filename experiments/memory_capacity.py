import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from hopfield import HopfieldNetwork
from hopfield.utils import random_patterns, hamming

N, TRIALS = 100, 20
rng = np.random.default_rng(0)

print(f"N={N}")
print(f"{'P':>4} {'P/N':>6} {'exact recall':>13} {'mean bit err':>13}")
for P in [1, 2, 5, 8, 10, 12, 15, 20, 30]:
    exact, errs = [], []
    for _ in range(TRIALS):
        patterns = random_patterns(P, N, rng)
        net = HopfieldNetwork()
        net.train(patterns)
        for p in patterns:
            out = net.recall(p, seed=int(rng.integers(1_000_000)))["state"]
            e = hamming(out, p)
            errs.append(e)
            exact.append(e == 0)
    print(f"{P:>4} {P/N:>6.2f} {np.mean(exact):>13.2f} {np.mean(errs):>13.2f}")
