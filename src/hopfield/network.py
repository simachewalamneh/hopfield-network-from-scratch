import numpy as np

from .learning import hebbian_learning
from .energy import calculate_energy


class HopfieldNetwork:
    def __init__(self):
        self.weights = None
        self.num_neurons = None

    def train(self, patterns):
        patterns = np.asarray(patterns)
        self.num_neurons = patterns.shape[1]
        self.weights = hebbian_learning(patterns)

    def update_neuron(self, state, i):
        """Asynchronous update of neuron i (threshold 0). Returns True if changed."""
        net_input = self.weights[i] @ state
        if net_input > 0:
            new = 1
        elif net_input < 0:
            new = -1
        else:
            new = state[i]  # tie: keep current state
        changed = new != state[i]
        state[i] = new
        return bool(changed)

    def recall(self, initial_state, max_sweeps=100, seed=None, synchronous=False):
        """Run dynamics until stable (asynchronous, random order, by default).

        Returns dict: state, energy_history (initial + after each sweep),
        converged, sweeps.
        """
        if self.weights is None:
            raise RuntimeError("Call train() first.")
        rng = np.random.default_rng(seed)
        state = np.array(initial_state, dtype=int)
        energy_history = [calculate_energy(state, self.weights)]
        converged = False
        sweeps = 0

        for sweeps in range(1, max_sweeps + 1):
            if synchronous:
                net = self.weights @ state
                new = np.where(net > 0, 1, np.where(net < 0, -1, state))
                changed = not np.array_equal(new, state)
                state = new
            else:
                changed = False
                for i in rng.permutation(self.num_neurons):
                    changed |= self.update_neuron(state, i)
            energy_history.append(calculate_energy(state, self.weights))
            if not changed:
                converged = True
                break

        return {
            "state": state,
            "energy_history": energy_history,
            "converged": converged,
            "sweeps": sweeps,
        }
