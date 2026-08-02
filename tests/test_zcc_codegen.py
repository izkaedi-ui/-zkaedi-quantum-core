# -*- coding: utf-8 -*-
"""
Tests for Quantum Core ZCC Compiler C Code Generator (zcc_codegen.py).
"""

from __future__ import annotations

import unittest
from quantum_core.zcc_codegen import generate_zcc_c_code


class TestZCCCodegen(unittest.TestCase):

    def test_generate_zcc_c_code(self) -> None:
        circuit = [["h", 0], ["cx", 0, 1]]
        c_code = generate_zcc_c_code(circuit, n_qubits=2)

        self.assertIn("#include <stdio.h>", c_code)
        self.assertIn("Complex statevector[4];", c_code)
        self.assertIn("apply_h(0, 2);", c_code)
        self.assertIn("apply_cx(0, 1, 2);", c_code)
        self.assertIn("ZCC QUANTUM ENGINE", c_code)


if __name__ == "__main__":
    unittest.main()
