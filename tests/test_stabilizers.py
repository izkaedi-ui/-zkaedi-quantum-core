# -*- coding: utf-8 -*-
"""
Tests for Stabilizer Code & QEC Syndrome Extraction Engine.
"""

from __future__ import annotations

import unittest
import numpy as np

from quantum_core.gates import PAULI_X, PAULI_Z
from quantum_core.simulator import execute_gate_sequence, zero_state
from quantum_core.stabilizers import (
    PauliOperator,
    extract_syndrome,
    steane_code_7_1_3,
    surface_code_distance_3,
)


class TestStabilizerEngine(unittest.TestCase):

    def test_pauli_operator_matrix_construction(self) -> None:
        op_x = PauliOperator("X")
        self.assertTrue(np.allclose(op_x.to_matrix(), PAULI_X))

        op_z = PauliOperator("Z")
        self.assertTrue(np.allclose(op_z.to_matrix(), PAULI_Z))

    def test_steane_code_zero_syndrome(self) -> None:
        code = steane_code_7_1_3()
        self.assertEqual(code.n_qubits, 7)
        self.assertEqual(code.n_logical, 1)
        self.assertEqual(code.distance, 3)

        # Zero state of 7 qubits initialized to |0000000>
        # Under Z-stabilizers, |0000000> is +1 eigenstate (syndrome bit 0)
        state = zero_state(7)

        # Z-stabilizers are index 3, 4, 5
        for stab in code.stabilizers[3:]:
            mat = stab.to_matrix()
            exp = float(np.real(np.vdot(state, mat @ state)))
            self.assertAlmostEqual(exp, 1.0, places=5)

    def test_surface_code_distance_3_structure(self) -> None:
        code = surface_code_distance_3()
        self.assertEqual(code.n_qubits, 9)
        self.assertEqual(code.n_logical, 1)
        self.assertEqual(code.distance, 3)
        self.assertEqual(len(code.stabilizers), 8)

    def test_surface_code_distance_5_structure(self) -> None:
        from quantum_core.stabilizers import surface_code_distance_5
        code = surface_code_distance_5()
        self.assertEqual(code.n_qubits, 25)
        self.assertEqual(code.n_logical, 1)
        self.assertEqual(code.distance, 5)
        self.assertEqual(len(code.stabilizers), 24)

        # Logical X and Z must have weight equal to distance (5)
        lx_str = code.logical_x[0].pauli_string
        lz_str = code.logical_z[0].pauli_string
        self.assertEqual(lx_str.count("X"), 5)
        self.assertEqual(lz_str.count("Z"), 5)


if __name__ == "__main__":
    unittest.main()
