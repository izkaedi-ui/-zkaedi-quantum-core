# -*- coding: utf-8 -*-
"""
Tests for Quantum Core Magic State Preparation, Teleportation & Distillation Engine.
"""

from __future__ import annotations

import unittest
import numpy as np

from quantum_core.gates import PHASE_T
from quantum_core.magic_states import (
    distill_magic_t_state_toy,
    inject_t_gate_via_teleportation,
    prepare_magic_t_state,
)
from quantum_core.simulator import apply_single_qubit_gate, zero_state


class TestMagicStateEngine(unittest.TestCase):

    def test_prepare_magic_t_state(self) -> None:
        state = prepare_magic_t_state()
        self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=10)

        # Expected: T|+> = 1/sqrt(2) (|0> + exp(i pi/4)|1>)
        self.assertAlmostEqual(abs(state[0]), 1.0 / np.sqrt(2.0), places=8)
        self.assertAlmostEqual(abs(state[1]), 1.0 / np.sqrt(2.0), places=8)

    def test_inject_t_gate_via_teleportation(self) -> None:
        """Verifies that injecting a T-gate via teleportation matches direct T-gate application."""
        rng = np.random.default_rng(42)

        # Initial state: |+>
        data_initial = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=complex)

        # Direct application
        expected = data_initial.copy()
        apply_single_qubit_gate(expected, PHASE_T, target=0, n_qubits=1)

        # Teleported application
        teleported = inject_t_gate_via_teleportation(data_initial, rng=rng)

        # Modulo global phase overlap fidelity must be 1.0
        overlap = abs(np.vdot(expected, teleported)) ** 2
        self.assertAlmostEqual(overlap, 1.0, places=8)

    def test_distill_magic_t_state_toy(self) -> None:
        rng = np.random.default_rng(1337)
        noisy_1 = prepare_magic_t_state(angle_error=0.01)
        noisy_2 = prepare_magic_t_state(angle_error=0.01)
        noisy_3 = prepare_magic_t_state(angle_error=0.01)

        success, distilled = distill_magic_t_state_toy(
            [noisy_1, noisy_2, noisy_3], rng=rng
        )

        self.assertIsInstance(success, bool)
        self.assertAlmostEqual(float(np.linalg.norm(distilled)), 1.0, places=10)

    def test_bravyi_haah_14_to_2_distillation(self) -> None:
        from quantum_core.magic_states import (
            bravyi_haah_14_to_2_distillation,
            bravyi_haah_syndrome_extraction_circuit,
        )

        rng = np.random.default_rng(123)
        success, outs = bravyi_haah_14_to_2_distillation(angle_error=0.0, rng=rng)
        self.assertIsInstance(success, bool)
        self.assertEqual(len(outs), 2)
        for s in outs:
            self.assertAlmostEqual(float(np.linalg.norm(s)), 1.0, places=8)

        circuit = bravyi_haah_syndrome_extraction_circuit()
        self.assertGreater(len(circuit), 0)


if __name__ == "__main__":
    unittest.main()
