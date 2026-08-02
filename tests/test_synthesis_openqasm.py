# -*- coding: utf-8 -*-
"""
Tests for Quantum Core Clifford+T Synthesis Engine & OpenQASM 3.0 Exporter.
"""

from __future__ import annotations

import unittest
from quantum_core.openqasm import from_openqasm3, to_openqasm3
from quantum_core.synthesis import decompose_rz_to_clifford_t, synthesize_circuit_clifford_t


class TestSynthesisAndOpenQASM(unittest.TestCase):

    def test_decompose_rz_to_clifford_t(self) -> None:
        t_seq = decompose_rz_to_clifford_t(0.8, precision=1e-3)
        self.assertGreater(len(t_seq), 0)
        self.assertIn("t", t_seq)

    def test_synthesize_circuit_clifford_t(self) -> None:
        circuit = [["h", 0], ["rz", 0, 0.8], ["cx", 0, 1]]
        syn_circuit, t_count = synthesize_circuit_clifford_t(circuit, precision=1e-3)
        self.assertGreater(t_count, 0)
        self.assertGreater(len(syn_circuit), len(circuit))

    def test_to_and_from_openqasm3(self) -> None:
        circuit = [["h", 0], ["cx", 0, 1], ["rz", 0, 0.8]]
        qasm_str = to_openqasm3(circuit, n_qubits=2)

        self.assertIn("OPENQASM 3.0;", qasm_str)
        self.assertIn("qubit[2] q;", qasm_str)
        self.assertIn("h q[0];", qasm_str)
        self.assertIn("cx q[0], q[1];", qasm_str)
        self.assertIn("rz(0.8) q[0];", qasm_str)

        parsed_circuit, n_q = from_openqasm3(qasm_str)
        self.assertEqual(n_q, 2)
        self.assertEqual(len(parsed_circuit), 3)
        self.assertEqual(parsed_circuit[0], ["h", 0])
        self.assertEqual(parsed_circuit[1], ["cx", 0, 1])


if __name__ == "__main__":
    unittest.main()
