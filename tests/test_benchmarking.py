# -*- coding: utf-8 -*-
"""
Tests for Quantum Core Randomized Benchmarking (RB) & Clifford sequence analytics.
"""

from __future__ import annotations

import unittest
import numpy as np

from quantum_core.benchmarking import (
    append_inversion_gate,
    generate_random_clifford_sequence,
    run_randomized_benchmarking_trial,
)
from quantum_core.error_models import inject_stochastic_pauli_noise
from quantum_core.simulator import execute_gate_sequence


class TestRandomizedBenchmarking(unittest.TestCase):

    def test_generate_random_clifford_sequence(self) -> None:
        seq = generate_random_clifford_sequence(length=10, target_qubit=0, n_qubits=1)
        self.assertGreater(len(seq), 0)

    def test_inversion_closure_noiseless(self) -> None:
        rng = np.random.default_rng(42)
        raw_seq = generate_random_clifford_sequence(length=15, target_qubit=0, n_qubits=1, rng=rng)
        closed_seq = append_inversion_gate(raw_seq, target_qubit=0, n_qubits=1)

        final_state = execute_gate_sequence(closed_seq, n_qubits=1)

        # Final state must be ground state |0>
        self.assertAlmostEqual(abs(final_state[0]), 1.0, places=8)
        self.assertAlmostEqual(abs(final_state[1]), 0.0, places=8)

    def test_run_randomized_benchmarking_trial(self) -> None:
        rng = np.random.default_rng(1337)
        prob_noiseless = run_randomized_benchmarking_trial(
            length=20, target_qubit=0, n_qubits=1, rng=rng
        )
        self.assertAlmostEqual(prob_noiseless, 1.0, places=8)

        # Noisy trial
        def noise_inj(circuit):
            return inject_stochastic_pauli_noise(circuit, p_error=0.05, n_qubits=1, rng=rng)

        prob_noisy = run_randomized_benchmarking_trial(
            length=20, noise_injector=noise_inj, target_qubit=0, n_qubits=1, rng=rng
        )
        self.assertLess(prob_noisy, 1.0)

    def test_fit_rb_decay(self) -> None:
        from quantum_core.benchmarking import fit_rb_decay

        lengths = [5, 10, 15, 20, 30]
        p_true = 0.97
        A_true, B_true = 0.95, 0.03
        probs = [A_true * (p_true ** m) + B_true for m in lengths]

        fit = fit_rb_decay(lengths, probs, n_qubits=1)
        self.assertTrue(fit["success"])
        self.assertAlmostEqual(float(fit["p"]), p_true, places=2)
        self.assertAlmostEqual(float(fit["r"]), (1.0 - p_true) * 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
