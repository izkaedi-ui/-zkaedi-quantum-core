# -*- coding: utf-8 -*-
"""
Tests for Quantum Core statevector simulator and differential matrix equivalence.
"""

from __future__ import annotations

import unittest
import numpy as np

from quantum_core.gates import HADAMARD, rx_matrix, rz_matrix
from quantum_core.simulator import (
    cnot_matrix,
    embed_single_qubit_gate,
    execute_gate_sequence,
    zero_state,
)
from quantum_core.types import COMPLEX_DTYPE, DEFAULT_ATOL


class TestQuantumSimulator(unittest.TestCase):

    def test_bell_state_generation(self) -> None:
        state = execute_gate_sequence([["h", 0], ["cx", 0, 1]], n_qubits=2)
        expected = np.array(
            [1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)],
            dtype=COMPLEX_DTYPE,
        )
        self.assertTrue(np.allclose(state, expected, atol=DEFAULT_ATOL))

    def test_statevector_and_full_matrix_paths_agree(self) -> None:
        gates = [
            ["h", 0],
            ["rx", 1, 0.73],
            ["cx", 0, 1],
            ["rz", 0, -0.19],
        ]
        fast_state = execute_gate_sequence(gates, n_qubits=2)

        reference = zero_state(2)
        for operation in [
            ("h", 0),
            ("rx", 1, 0.73),
            ("cx", 0, 1),
            ("rz", 0, -0.19),
        ]:
            if operation[0] == "h":
                operator = embed_single_qubit_gate(HADAMARD, 0, 2)
            elif operation[0] == "rx":
                operator = embed_single_qubit_gate(rx_matrix(operation[2]), 1, 2)
            elif operation[0] == "rz":
                operator = embed_single_qubit_gate(rz_matrix(operation[2]), 0, 2)
            elif operation[0] == "cx":
                operator = cnot_matrix(0, 1, 2)
            else:
                self.fail(f"unexpected test gate: {operation[0]}")
            reference = operator @ reference

        self.assertTrue(np.allclose(fast_state, reference, atol=DEFAULT_ATOL))


if __name__ == "__main__":
    unittest.main()
