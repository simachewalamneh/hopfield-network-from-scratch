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

## Results

All experiments use N = 100 neurons, random bipolar patterns, and fixed seeds.

### Memory capacity (`experiments/memory_capacity.py`)

| P | P/N | Exact recall | Mean bit errors |
|---|-----|--------------|-----------------|
| 1 | 0.01 | 1.00 | 0.00 |
| 5 | 0.05 | 1.00 | 0.00 |
| 8 | 0.08 | 1.00 | 0.00 |
| 10 | 0.10 | 0.97 | 0.08 |
| 12 | 0.12 | 0.90 | 0.38 |
| 15 | 0.15 | 0.72 | 1.79 |
| 20 | 0.20 | 0.42 | 6.06 |
| 30 | 0.30 | 0.04 | 21.74 |

Recall is perfect up to P = 8 and degrades sharply beyond P/N ~ 0.15, consistent with the paper's capacity estimate of about 0.15N.

### Noise robustness (`experiments/noise_robustness.py`, P = 5)

| Noise | Bit accuracy | Exact recovery |
|-------|--------------|----------------|
| 0.00 | 1.000 | 1.000 |
| 0.10 | 1.000 | 1.000 |
| 0.20 | 0.998 | 0.987 |
| 0.30 | 0.986 | 0.927 |
| 0.40 | 0.843 | 0.500 |
| 0.50 | 0.506 | 0.000 |

Up to 30% corruption is almost always corrected. At 50% the input is uncorrelated with the memory and accuracy falls to chance. Bit accuracy and exact recovery differ: at 40% noise, 84% of bits are right but only half of patterns are recovered exactly.

### Pattern interference (`experiments/pattern_interference.py`)

Seven random memories plus an eighth at a given Hamming distance from one of them.

| Distance | Both stable | Distance between recalled states |
|----------|-------------|----------------------------------|
| 50 (random) | 0.97 | 50.0 |
| 30 | 0.73 | 28.9 |
| 20 | 0.13 | 12.6 |
| 10 | 0.03 | 2.2 |

Similar memories displace each other and, at distance 10, fuse into one attractor, as the paper reports.

### Energy dynamics (`examples/03_energy_dynamics.py`)

Energy never increased during asynchronous updates, and the network converged in 2 sweeps from a random start. Stored patterns sit at energy near -49.

### Letter patterns (`examples/02_pattern_reconstruction.py`)

Recall of a noisy "A" (5 flips) stored with "H" and "L" failed. These letters overlap heavily, so the memories are correlated and the network converged to a spurious mixed state. This is the correlated-pattern limitation described in the paper.

## Limitations

Capacity is about 0.15N, spurious attractors exist, and correlated patterns interfere.