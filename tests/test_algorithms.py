# -*- coding: utf-8 -*-
"""
Tests auditing all 13 discovered algorithms and the cross-algorithm synergy matrix.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np

from quantum_core.registry import (
    REGISTRY_PATH,
    import_module_from_path,
    load_json_object,
    require_finite_metric,
    require_mapping,
    resolve_project_path,
)
from quantum_core.simulator import execute_gate_sequence


class TestRegisteredAlgorithms(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_json_object(REGISTRY_PATH)
        cls.algorithms = require_mapping(
            cls.registry.get("algorithms", {}),
            field="algorithms",
        )

    def test_01_qalgo_search_2(self) -> None:
        meta = self.algorithms["QAlgo-Search-2"]
        self.assertEqual(meta["domain"], "quantum_search")
        self.assertEqual(meta["metrics"]["fidelity"], 1.0)
        gates = meta.get("circuit", [])
        if gates:
            state = execute_gate_sequence(gates, n_qubits=4)
            self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=10)

    def test_02_qalgo_optimization_1(self) -> None:
        meta = self.algorithms["QAlgo-Optimization-1"]
        self.assertEqual(meta["domain"], "quantum_optimization")
        artifact = load_json_object(resolve_project_path(meta["file_location"]))
        gate_seq = artifact["quantum_circuit"]["gate_sequence"]
        state = execute_gate_sequence(gate_seq, n_qubits=4)
        self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=10)

    def test_03_qalgo_cryptography_4(self) -> None:
        meta = self.algorithms["QAlgo-Cryptography-4"]
        self.assertEqual(meta["domain"], "quantum_cryptography")

    def test_04_qalgo_simulation_5(self) -> None:
        meta = self.algorithms["QAlgo-Simulation-5"]
        self.assertEqual(meta["domain"], "quantum_simulation")

    def test_05_qalgo_ml_3(self) -> None:
        meta = self.algorithms["QAlgo-Ml-3"]
        self.assertEqual(meta["domain"], "quantum_ml")

    def test_06_qalgo_error_s2_1(self) -> None:
        meta = self.algorithms["QAlgo-Error-S2-1"]
        self.assertEqual(meta["domain"], "quantum_error_correction")

    def test_07_qalgo_communication_s2_2(self) -> None:
        meta = self.algorithms["QAlgo-Communication-S2-2"]
        self.assertEqual(meta["domain"], "quantum_communication")

    def test_08_qalgo_chemistry_s2_3(self) -> None:
        meta = self.algorithms["QAlgo-Chemistry-S2-3"]
        self.assertEqual(meta["domain"], "quantum_chemistry")

    def test_09_qalgo_optimization_s2_4(self) -> None:
        meta = self.algorithms["QAlgo-Optimization-S2-4"]
        self.assertEqual(meta["domain"], "quantum_optimization")

    def test_10_qalgo_search_s2_5(self) -> None:
        meta = self.algorithms["QAlgo-Search-S2-5"]
        self.assertEqual(meta["domain"], "quantum_search")

    def test_11_zk_anyon_512(self) -> None:
        meta = self.algorithms["ZK-ANYON-512"]
        artifact = load_json_object(resolve_project_path(meta["file_location"]))
        self.assertIn("Fibonacci", str(artifact["anyon_braiding"]["topological_nature"]))

    def test_12_alien_math_primitive(self) -> None:
        meta = self.algorithms["ALIEN-MATH-PRIMITIVE"]
        source = resolve_project_path(meta["file_location"]).read_text(encoding="utf-8")
        self.assertIn("assembly", source)

    def test_13_zcc_quantum_engine(self) -> None:
        meta = self.algorithms["ZCC-QUANTUM-ENGINE"]
        source = resolve_project_path(meta["file_location"]).read_text(encoding="utf-8")
        self.assertIn("#include", source)

    def test_14_zkaedi_ideakz(self) -> None:
        meta = self.algorithms["ZKAEDI-IDEAKZ"]
        self.assertEqual(meta["domain"], "quantum_error_correction")
        self.assertEqual(meta["metrics"]["fidelity"], 1.0)
        self.assertEqual(meta["metrics"]["quantum_advantage"], 1337.0)
        gates = meta.get("circuit", [])
        state = execute_gate_sequence(gates, n_qubits=12)
        self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=10)

    def test_15_synergy_fusion_matrix(self) -> None:
        synergy_path = REGISTRY_PATH.parent.parent / "tools" / "fuse_synergy_algorithms.py"
        self.assertTrue(synergy_path.is_file())
        synergy_module = import_module_from_path("synergy_mod", synergy_path)

        synergy_map = getattr(synergy_module, "SYNERGY_MAP")
        compute_phi = getattr(synergy_module, "compute_phi")

        for algo_id in self.algorithms:
            self.assertIn(algo_id, synergy_map)
            relation = synergy_map[algo_id]
            partner = relation["partner"]
            self.assertIn(partner, self.algorithms)
            self.assertNotEqual(partner, algo_id)
            phi = float(compute_phi(relation["coherence"], relation["diversity"], relation["actionability"], relation["kappa"]))
            self.assertTrue(phi >= 0.85)


if __name__ == "__main__":
    unittest.main()
