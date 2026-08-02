# -*- coding: utf-8 -*-
"""
Tests for Quantum Core circuit validation, normalization, and parameter binding.
"""

from __future__ import annotations

import unittest

from quantum_core.circuit import bind_parameters
from quantum_core.types import CircuitValidationError
from quantum_core.validation import normalize_gate, normalize_gates


class TestQuantumCircuits(unittest.TestCase):

    def test_normalize_gate_mapping(self) -> None:
        op = normalize_gate({"gate": "hadamard", "targets": [0]}, n_qubits=2)
        self.assertEqual(op.name, "h")
        self.assertEqual(op.qubits, (0,))
        self.assertEqual(op.parameters, ())

    def test_normalize_gate_sequence(self) -> None:
        op = normalize_gate(["cnot", 0, 1], n_qubits=2)
        self.assertEqual(op.name, "cx")
        self.assertEqual(op.qubits, (0, 1))

    def test_unknown_gate_rejected(self) -> None:
        with self.assertRaises(CircuitValidationError):
            normalize_gates([["mystery", 0]], n_qubits=1)

    def test_duplicate_qubits_rejected(self) -> None:
        with self.assertRaises(CircuitValidationError):
            normalize_gates([["cx", 0, 0]], n_qubits=2)

    def test_out_of_range_qubit_rejected(self) -> None:
        with self.assertRaises(CircuitValidationError):
            normalize_gates([["h", 5]], n_qubits=2)

    def test_bind_parameters(self) -> None:
        circuit = [["ry", 0, "theta_0"], ["rz", 0, 0.5]]
        bound = bind_parameters(circuit, {"theta_0": 0.75}, n_qubits=1)

        self.assertEqual(bound[0].parameters, (0.75,))
        self.assertEqual(bound[1].parameters, (0.5,))

    def test_unbound_symbolic_parameter_raises(self) -> None:
        circuit = [["rx", 0, "unbound_param"]]
        with self.assertRaises(CircuitValidationError):
            bind_parameters(circuit, {"other_param": 1.0}, n_qubits=1)


if __name__ == "__main__":
    unittest.main()
