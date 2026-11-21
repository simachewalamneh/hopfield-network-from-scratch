# hopfield-network-from-scratch

Implementing and exploring the original Hopfield network (1982) using NumPy: Hebbian learning, asynchronous dynamics, energy, associative memory, and capacity experiments.

## Paper

J. J. Hopfield, "Neural networks and physical systems with emergent collective computational abilities", *PNAS* 79(8), 2554-2558 (1982).
https://www.pnas.org/doi/10.1073/pnas.79.8.2554

## Model

- States s_i in {-1, +1} (the paper's 0/1 form is equivalent via V = (s+1)/2).
- Learning: W = (1/N) sum_mu xi^mu (xi^mu)^T, zero diagonal.
- Dynamics: s_i <- sign(sum_j W_ij s_j), asynchronous, random order.
- Energy: E = -1/2 s^T W s, which never increases under asynchronous updates.

## Installation

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
python examples/01_basic_memory.py
python examples/02_pattern_reconstruction.py
python examples/03_energy_dynamics.py
python experiments/memory_capacity.py
python experiments/noise_robustness.py
python experiments/pattern_interference.py
pytest
```

## Expected results (paper, N=100)

- P=5: memories are almost always stable.
- P=15 (~0.15N): about half are recovered with fewer than 5 errors.
- Memories at Hamming distance ~10 tend to fuse.

## Limitations

Capacity is about 0.15N, spurious attractors exist, and correlated patterns interfere.