# -*- coding: utf-8 -*-
"""
Tests for Quantum Core Matrix Product State (MPS) Tensor Network Simulator Backend.
"""

from __future__ import annotations

import unittest
import numpy as np

from quantum_core.gates import HADAMARD, PAULI_X, cnot_matrix
from quantum_core.mps import (
    apply_single_qubit_gate_mps,
    apply_two_qubit_gate_mps,
    mps_norm,
    mps_to_statevector,
    mps_zero_state,
)
from quantum_core.simulator import execute_gate_sequence, zero_state


class TestMPSSimulator(unittest.TestCase):

    def test_mps_zero_state(self) -> None:
        mps = mps_zero_state(n_qubits=4, max_bond_dim=16)
        self.assertEqual(mps.n_qubits, 4)
        self.assertAlmostEqual(mps_norm(mps), 1.0, places=10)

        vec = mps_to_statevector(mps)
        self.assertEqual(len(vec), 16)
        self.assertAlmostEqual(abs(vec[0]), 1.0, places=10)

    def test_mps_single_qubit_gate(self) -> None:
        mps = mps_zero_state(n_qubits=2)
        apply_single_qubit_gate_mps(mps, HADAMARD, target=0)

        vec = mps_to_statevector(mps)
        # Expected: 1/sqrt(2) (|00> + |10>) -> index 0 and index 2
        self.assertAlmostEqual(abs(vec[0]), 1.0 / np.sqrt(2.0), places=8)
        self.assertAlmostEqual(abs(vec[2]), 1.0 / np.sqrt(2.0), places=8)

    def test_mps_bell_state_fidelity_against_exact_simulator(self) -> None:
        # Create Bell state |Phi+> = 1/sqrt(2) (|00> + |11>) via MPS
        mps = mps_zero_state(n_qubits=2, max_bond_dim=4)
        apply_single_qubit_gate_mps(mps, HADAMARD, target=0)
        apply_two_qubit_gate_mps(mps, cnot_matrix(), q0=0, q1=1)

        mps_vec = mps_to_statevector(mps)

        # Exact statevector
        exact_vec = execute_gate_sequence([["h", 0], ["cx", 0, 1]], n_qubits=2)

        fidelity = float(abs(np.vdot(exact_vec, mps_vec)) ** 2)
        self.assertAlmostEqual(fidelity, 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
