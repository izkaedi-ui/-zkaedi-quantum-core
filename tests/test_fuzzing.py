# -*- coding: utf-8 -*-
"""
Quantum Core — Property-Based & Fuzz Testing Suite.
Validates mathematical invariants, statevector normalization, gate inverses,
and parameter bounds using randomized property testing.
"""

from __future__ import annotations

import math
import unittest
import numpy as np

from quantum_core.gates import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    cnot_matrix,
    h_matrix,
    is_hermitian,
    is_unitary,
    rx_matrix,
    ry_matrix,
    rz_matrix,
    swap_matrix,
)
from quantum_core.optimization import optimize_circuit
from quantum_core.simulator import execute_gate_sequence, zero_state
from quantum_core.types import CircuitValidationError
from quantum_core.validation import normalize_gate


class TestPropertyBasedFuzzing(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(2026)

    def test_single_qubit_gate_unitarity_fuzz(self) -> None:
        """Property: All generated single-qubit rotations Rx, Ry, Rz are unitary for any real theta."""
        for _ in range(100):
            theta = float(self.rng.uniform(-10.0 * math.pi, 10.0 * math.pi))
            self.assertTrue(is_unitary(rx_matrix(theta)))
            self.assertTrue(is_unitary(ry_matrix(theta)))
            self.assertTrue(is_unitary(rz_matrix(theta)))

    def test_gate_inverse_cancellation_fuzz(self) -> None:
        """Property: Applying U then U^dagger (or U U for self-inverses) restores initial statevector."""
        for _ in range(50):
            n_qubits = int(self.rng.integers(1, 4))
            state = zero_state(n_qubits)
            target = int(self.rng.integers(0, n_qubits))

            # Apply H H
            c1 = [["h", target], ["h", target]]
            res1 = execute_gate_sequence(c1, n_qubits, initial_state=state)
            fidelity1 = float(abs(np.vdot(state, res1)) ** 2)
            self.assertAlmostEqual(fidelity1, 1.0, places=10)

            # Apply X X
            c2 = [["x", target], ["x", target]]
            res2 = execute_gate_sequence(c2, n_qubits, initial_state=state)
            fidelity2 = float(abs(np.vdot(state, res2)) ** 2)
            self.assertAlmostEqual(fidelity2, 1.0, places=10)

    def test_rotation_addition_identity_fuzz(self) -> None:
        """Property: Rz(a) followed by Rz(-a) is identity."""
        for _ in range(50):
            theta = float(self.rng.uniform(-2.0 * math.pi, 2.0 * math.pi))
            circuit = [["rz", 0, theta], ["rz", 0, -theta]]
            opt = optimize_circuit(circuit)
            self.assertEqual(len(opt), 0)

    def test_statevector_norm_conservation_fuzz(self) -> None:
        """Property: Any valid random circuit preserves statevector norm equal to 1.0."""
        single_gates = ["h", "x", "y", "z", "s", "t"]
        for _ in range(50):
            n_qubits = int(self.rng.integers(1, 5))
            state = zero_state(n_qubits)
            n_ops = int(self.rng.integers(1, 20))

            circuit = []
            for _ in range(n_ops):
                gtype = self.rng.choice(["single", "rotation", "cx"])
                if gtype == "single":
                    gname = str(self.rng.choice(single_gates))
                    q = int(self.rng.integers(0, n_qubits))
                    circuit.append([gname, q])
                elif gtype == "rotation":
                    gname = str(self.rng.choice(["rx", "ry", "rz"]))
                    q = int(self.rng.integers(0, n_qubits))
                    angle = float(self.rng.uniform(-math.pi, math.pi))
                    circuit.append([gname, q, angle])
                elif gtype == "cx" and n_qubits >= 2:
                    ctrl, tgt = self.rng.choice(n_qubits, size=2, replace=False)
                    circuit.append(["cx", int(ctrl), int(tgt)])

            res = execute_gate_sequence(circuit, n_qubits, initial_state=state)
            norm = float(np.linalg.norm(res))
            self.assertAlmostEqual(norm, 1.0, places=10)

    def test_malformed_gate_specifications_rejection_fuzz(self) -> None:
        """Property: Fail-closed validation rejects malformed gates, out-of-bounds qubits, and non-finite params."""
        for _ in range(50):
            # Duplicate control and target
            with self.assertRaises(CircuitValidationError):
                normalize_gate(["cx", 0, 0], n_qubits=2)

            # Out of bounds qubit
            with self.assertRaises(CircuitValidationError):
                normalize_gate(["h", 5], n_qubits=3)

            # Non-finite parameter
            with self.assertRaises(CircuitValidationError):
                normalize_gate(["rz", 0, float("nan")], n_qubits=1)


if __name__ == "__main__":
    unittest.main()
