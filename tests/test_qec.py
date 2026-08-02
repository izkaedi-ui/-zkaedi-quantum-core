# -*- coding: utf-8 -*-
"""
Tests for Quantum Error Correction (QEC) 3-qubit bit-flip recovery.
"""

from __future__ import annotations

import unittest
import numpy as np

from examples.qec_demo import run_3qubit_bit_flip_qec
from quantum_core.measurement import probabilities
from quantum_core.simulator import execute_gate_sequence


class TestQuantumErrorCorrection(unittest.TestCase):

    def test_qec_repetition_code_no_error(self) -> None:
        result = run_3qubit_bit_flip_qec(error_qubit=None)
        state = result["final_state"]
        self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=10)

    def test_qec_repetition_code_corrects_qubit_1_error(self) -> None:
        clean = run_3qubit_bit_flip_qec(error_qubit=None)["final_state"]
        corrupted = run_3qubit_bit_flip_qec(error_qubit=1)["final_state"]

        # Probabilities on logical data qubit 0 must match
        p_clean = probabilities(clean)
        p_corr = probabilities(corrupted)

        self.assertTrue(np.allclose(p_clean, p_corr, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
