# -*- coding: utf-8 -*-
"""
Tests for Quantum Error Models & Coherent vs Stochastic Noise Evaluation.
"""

from __future__ import annotations

import unittest
import numpy as np

from quantum_core.error_models import (
    evaluate_palindromic_coherent_suppression,
    inject_coherent_rotation_noise,
    inject_stochastic_pauli_noise,
    state_fidelity,
)
from quantum_core.simulator import execute_gate_sequence, zero_state


class TestQuantumErrorModels(unittest.TestCase):

    def test_state_fidelity(self) -> None:
        state = execute_gate_sequence([["h", 0]], n_qubits=1)
        self.assertAlmostEqual(state_fidelity(state, state), 1.0, places=10)

        orthogonal = zero_state(1)
        orthogonal[0] = 0.0
        orthogonal[1] = 1.0
        self.assertAlmostEqual(state_fidelity(zero_state(1), orthogonal), 0.0, places=10)

    def test_inject_coherent_rotation_noise(self) -> None:
        circuit = [["rx", 0, 1.0]]
        noisy = inject_coherent_rotation_noise(circuit, over_rotation_scale=0.10, n_qubits=1)
        self.assertEqual(noisy[0].parameters[0], 1.10)

    def test_palindromic_coherent_suppression(self) -> None:
        # Palindromic circuit: Rx(theta) followed by Rx(-theta)
        palindromic = [["rx", 0, 1.5], ["rx", 0, -1.5]]

        # Non-palindromic: two independent rotations Rx(theta), Ry(phi)
        non_palindromic = [["rx", 0, 1.5], ["ry", 0, 1.5]]

        eval_res = evaluate_palindromic_coherent_suppression(
            palindromic, non_palindromic, over_rotation_scale=0.05, n_qubits=1
        )

        # Palindromic circuit cancels over-rotation to first order -> fidelity ~ 1.0
        self.assertGreater(eval_res["palindromic_fidelity"], 0.99)
        self.assertGreater(eval_res["palindromic_fidelity"], eval_res["non_palindromic_fidelity"])


if __name__ == "__main__":
    unittest.main()
