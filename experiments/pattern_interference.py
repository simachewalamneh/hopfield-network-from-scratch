import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from hopfield import HopfieldNetwork
from hopfield.utils import random_patterns, corrupt, hamming

# Paper: N=100, 7 random memories + an 8th at Hamming distance 30/20/10 from one.
N, TRIALS = 100, 30
rng = np.random.default_rng(0)

print(f"{'dist':>5} {'both stable':>12} {'dist between recalled':>22}")
for dist in [50, 30, 20, 10]:
    both, final_d = [], []
    for _ in range(TRIALS):
        base = random_patterns(7, N, rng)
        eighth = corrupt(base[0], dist, rng)
        patterns = np.vstack([base, eighth])
        net = HopfieldNetwork()
        net.train(patterns)
        a = net.recall(patterns[0], seed=0)["state"]
        b = net.recall(patterns[7], seed=0)["state"]
        both.append(hamming(a, patterns[0]) == 0 and hamming(b, patterns[7]) == 0)
        final_d.append(hamming(a, b))
    print(f"{dist:>5} {np.mean(both):>12.2f} {np.mean(final_d):>22.1f}")
