# -*- coding: utf-8 -*-
"""
Tests for Quantum Core gate primitives, matrix properties, and linear algebra checks.
"""

from __future__ import annotations

import unittest
import numpy as np

from quantum_core.gates import (
    HADAMARD,
    IDENTITY_2,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PHASE_S,
    PHASE_T,
    compose_gates,
    global_phase_equivalent,
    h_matrix,
    i_matrix,
    is_hermitian,
    is_unitary,
    rx_matrix,
    ry_matrix,
    rz_matrix,
    s_matrix,
    t_matrix,
    x_matrix,
    y_matrix,
    z_matrix,
)


class TestQuantumGates(unittest.TestCase):

    def test_fixed_gates_are_unitary(self) -> None:
        for name, gate in {
            "I": i_matrix(),
            "H": h_matrix(),
            "X": x_matrix(),
            "Y": y_matrix(),
            "Z": z_matrix(),
            "S": s_matrix(),
            "T": t_matrix(),
        }.items():
            with self.subTest(gate=name):
                self.assertTrue(is_unitary(gate))

    def test_pauli_gates_are_hermitian(self) -> None:
        for name, gate in {
            "X": x_matrix(),
            "Y": y_matrix(),
            "Z": z_matrix(),
        }.items():
            with self.subTest(gate=name):
                self.assertTrue(is_hermitian(gate))

    def test_rotations_are_unitary(self) -> None:
        for theta in (0.0, np.pi / 7.0, np.pi, 2.0 * np.pi):
            for name, rotation in {
                "Rx": rx_matrix,
                "Ry": ry_matrix,
                "Rz": rz_matrix,
            }.items():
                with self.subTest(gate=name, theta=theta):
                    self.assertTrue(is_unitary(rotation(theta)))

    def test_zero_rotations_equal_identity(self) -> None:
        for rotation in (rx_matrix, ry_matrix, rz_matrix):
            with self.subTest(rotation=rotation.__name__):
                self.assertTrue(np.allclose(rotation(0.0), IDENTITY_2))

    def test_pi_rotations_match_paulis_up_to_global_phase(self) -> None:
        self.assertTrue(global_phase_equivalent(rx_matrix(np.pi), PAULI_X))
        self.assertTrue(global_phase_equivalent(ry_matrix(np.pi), PAULI_Y))
        self.assertTrue(global_phase_equivalent(rz_matrix(np.pi), PAULI_Z))

    def test_hadamard_is_self_inverse(self) -> None:
        self.assertTrue(np.allclose(HADAMARD @ HADAMARD, IDENTITY_2))

    def test_cached_rotations_are_read_only(self) -> None:
        gate = rx_matrix(0.5)
        with self.assertRaises(ValueError):
            gate[0, 0] = 123.0  # type: ignore[index]

    def test_compose_gates(self) -> None:
        h = h_matrix()
        z = z_matrix()
        composed = compose_gates(h, z, h)  # H Z H = X
        self.assertTrue(global_phase_equivalent(composed, x_matrix()))


if __name__ == "__main__":
    unittest.main()
