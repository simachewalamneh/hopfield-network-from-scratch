# hopfield-network-from-scratch

Implementing and exploring the original Hopfield network (1982) using NumPy: Hebbian learning, asynchronous dynamics, energy, associative memory, capacity experiments.

Reference: J. J. Hopfield, "Neural networks and physical systems with emergent collective computational abilities", PNAS 79, 2554-2558 (1982).

## Model
- States s_i in {-1, +1} (the paper's 0/1 form is equivalent via V = (s+1)/2).
- Learning: W = (1/N) sum_mu xi^mu (xi^mu)^T, zero diagonal.
- Dynamics: s_i <- sign(sum_j W_ij s_j), asynchronous, random order.
- Energy: E = -1/2 s^T W s; never increases under asynchronous updates.

## Install and run
```
pip install -r requirements.txt
python examples/01_basic_memory.py
python examples/02_pattern_reconstruction.py
python examples/03_energy_dynamics.py
python experiments/memory_capacity.py
python experiments/noise_robustness.py
python experiments/pattern_interference.py
pytest
```

## Expected (paper, N=100)
- P=5: memories almost always stable.
- P=15 (~0.15N): about half recovered with fewer than 5 errors.
- Memories at Hamming distance ~10 tend to fuse.

## Limitations
Capacity ~0.15N, spurious attractors, correlated patterns interfere.
