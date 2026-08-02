# -*- coding: utf-8 -*-
"""
Tests for Quantum Core Circuit Optimization Compiler Pass Layer.
"""

from __future__ import annotations

import math
import unittest

from quantum_core.optimization import (
    circuit_depth,
    gate_count,
    optimize_circuit,
    two_qubit_gate_count,
)


class TestCircuitOptimization(unittest.TestCase):

    def test_gate_count_and_depth(self) -> None:
        circuit = [
            ["h", 0],
            ["cx", 0, 1],
            ["rz", 1, 0.5],
        ]
        self.assertEqual(gate_count(circuit), 3)
        self.assertEqual(two_qubit_gate_count(circuit), 1)
        self.assertEqual(circuit_depth(circuit), 3)

    def test_self_inverse_cancellation(self) -> None:
        circuit = [
            ["h", 0],
            ["h", 0],
            ["x", 1],
            ["x", 1],
            ["cx", 0, 1],
            ["cx", 0, 1],
        ]
        opt = optimize_circuit(circuit)
        self.assertEqual(len(opt), 0)

    def test_rotation_merging(self) -> None:
        circuit = [
            ["rz", 0, 0.5],
            ["rz", 0, 0.3],
        ]
        opt = optimize_circuit(circuit)
        self.assertEqual(len(opt), 1)
        self.assertEqual(opt[0][0], "rz")
        self.assertEqual(opt[0][1], 0)
        self.assertAlmostEqual(float(opt[0][2]), 0.8, places=6)

    def test_rotation_cancellation_to_zero(self) -> None:
        circuit = [
            ["rz", 0, 1.0],
            ["rz", 0, -1.0],
        ]
        opt = optimize_circuit(circuit)
        self.assertEqual(len(opt), 0)


if __name__ == "__main__":
    unittest.main()
