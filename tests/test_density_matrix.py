# -*- coding: utf-8 -*-
"""
Tests for Quantum Core Density-Matrix & Quantum Channel Backend.
"""

from __future__ import annotations

import unittest
import numpy as np

from quantum_core.density_matrix import (
    amplitude_damping_kraus,
    apply_kraus_channel,
    apply_unitary_density_matrix,
    bit_flip_kraus,
    density_matrix_from_statevector,
    density_matrix_purity,
    density_matrix_zero,
    depolarizing_kraus,
    phase_flip_kraus,
)
from quantum_core.gates import HADAMARD, PAULI_X
from quantum_core.simulator import zero_state


class TestDensityMatrixBackend(unittest.TestCase):

    def test_density_matrix_zero_and_purity(self) -> None:
        rho = density_matrix_zero(2)
        self.assertEqual(rho.shape, (4, 4))
        self.assertAlmostEqual(density_matrix_purity(rho), 1.0, places=10)

    def test_density_matrix_from_statevector(self) -> None:
        psi = zero_state(1)
        rho = density_matrix_from_statevector(psi)
        self.assertAlmostEqual(density_matrix_purity(rho), 1.0, places=10)
        self.assertAlmostEqual(float(np.real(np.trace(rho))), 1.0, places=10)

    def test_apply_unitary_density_matrix(self) -> None:
        rho = density_matrix_zero(1)
        rho_x = apply_unitary_density_matrix(rho, PAULI_X, qubits=[0], n_qubits=1)
        # X|0><0|X = |1><1| -> entry [1,1] must be 1.0
        self.assertAlmostEqual(abs(rho_x[1, 1]), 1.0, places=10)

    def test_bit_flip_kraus_channel(self) -> None:
        rho = density_matrix_zero(1)
        kraus = bit_flip_kraus(p=0.1)
        rho_noisy = apply_kraus_channel(rho, kraus, target_qubit=0, n_qubits=1)

        # Trace preserving check: Tr(rho) == 1.0
        self.assertAlmostEqual(float(np.real(np.trace(rho_noisy))), 1.0, places=10)

        # Purity check: mixed state purity < 1.0
        self.assertLess(density_matrix_purity(rho_noisy), 1.0)

    def test_depolarizing_and_amplitude_damping(self) -> None:
        rho = density_matrix_zero(1)

        # Depolarizing
        depol_kraus = depolarizing_kraus(p=0.2)
        rho_depol = apply_kraus_channel(rho, depol_kraus, target_qubit=0, n_qubits=1)
        self.assertAlmostEqual(float(np.real(np.trace(rho_depol))), 1.0, places=10)

        # Amplitude damping
        ad_kraus = amplitude_damping_kraus(gamma=0.3)
        rho_ad = apply_kraus_channel(rho, ad_kraus, target_qubit=0, n_qubits=1)
        self.assertAlmostEqual(float(np.real(np.trace(rho_ad))), 1.0, places=10)


    def test_apply_multi_qubit_unitary_density_matrix(self) -> None:
        from quantum_core.gates import cnot_matrix
        rho = density_matrix_zero(2)
        # Apply X on qubit 0 -> |10><10| (index 2)
        rho_x0 = apply_unitary_density_matrix(rho, PAULI_X, qubits=[0], n_qubits=2)
        self.assertAlmostEqual(abs(rho_x0[2, 2]), 1.0, places=10)

        # Apply CNOT(0, 1) -> |11><11| (index 3)
        rho_cnot = apply_unitary_density_matrix(rho_x0, cnot_matrix(), qubits=[0, 1], n_qubits=2)
        self.assertAlmostEqual(abs(rho_cnot[3, 3]), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
