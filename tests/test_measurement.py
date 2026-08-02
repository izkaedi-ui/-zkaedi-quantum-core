# -*- coding: utf-8 -*-
"""
Tests for Quantum Core measurement, statevector collapse, and shot sampling.
"""

from __future__ import annotations

import unittest
import numpy as np

from quantum_core.measurement import (
    marginal_probabilities,
    measure_qubit,
    probabilities,
    sample_counts,
)
from quantum_core.simulator import execute_gate_sequence


class TestQuantumMeasurement(unittest.TestCase):

    def test_probabilities_sum_to_one(self) -> None:
        state = execute_gate_sequence([["h", 0], ["rx", 1, 0.42]], n_qubits=2)
        probs = probabilities(state)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=10)

    def test_bell_state_correlations(self) -> None:
        """Requirement: 10,000 shots on Bell state |Phi+>: 00 ~ 50%, 11 ~ 50%, 01 ~ 0%, 10 ~ 0%."""
        state = execute_gate_sequence([["h", 0], ["cx", 0, 1]], n_qubits=2)
        rng = np.random.default_rng(1337)

        counts = sample_counts(state, shots=10000, n_qubits=2, rng=rng)

        # 01 and 10 must not occur
        self.assertEqual(counts.get("01", 0), 0)
        self.assertEqual(counts.get("10", 0), 0)

        # 00 and 11 each approx 50% (5000 +/- 300)
        c00 = counts.get("00", 0)
        c11 = counts.get("11", 0)

        self.assertGreater(c00, 4700)
        self.assertLess(c00, 5300)
        self.assertGreater(c11, 4700)
        self.assertLess(c11, 5300)
        self.assertEqual(c00 + c11, 10000)

    def test_single_qubit_measurement_and_collapse(self) -> None:
        state = execute_gate_sequence([["h", 0]], n_qubits=1)
        rng = np.random.default_rng(42)

        outcome, collapsed = measure_qubit(state, qubit=0, n_qubits=1, rng=rng)

        self.assertIn(outcome, (0, 1))
        collapsed_probs = probabilities(collapsed)

        if outcome == 0:
            self.assertAlmostEqual(collapsed_probs[0], 1.0, places=10)
            self.assertAlmostEqual(collapsed_probs[1], 0.0, places=10)
        else:
            self.assertAlmostEqual(collapsed_probs[0], 0.0, places=10)
            self.assertAlmostEqual(collapsed_probs[1], 1.0, places=10)

    def test_marginal_probabilities(self) -> None:
        state = execute_gate_sequence([["h", 0], ["x", 1]], n_qubits=2)
        marginals = marginal_probabilities(state, qubits=[1], n_qubits=2)

        # Qubit 1 is flipped to 1, so marginal for qubit 1 being '1' should be 1.0
        self.assertAlmostEqual(marginals.get("1", 0.0), 1.0, places=10)
        self.assertAlmostEqual(marginals.get("0", 0.0), 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
