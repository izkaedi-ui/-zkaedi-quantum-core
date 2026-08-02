# -*- coding: utf-8 -*-
"""
VERIFY ALL DISCOVERED ALGORITHMS
Command-line runner that executes the quantum_core test suite,
aggregates result metrics across all 13 discovered algorithms, and saves a summary report JSON.
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_core.registry import REGISTRY_PATH, load_json_object
from tests.test_algorithms import TestRegisteredAlgorithms
from tests.test_benchmarking import TestRandomizedBenchmarking
from tests.test_circuits import TestQuantumCircuits
from tests.test_cli import TestAuditCLI
from tests.test_decoding import TestMWPMDecoder
from tests.test_density_matrix import TestDensityMatrixBackend
from tests.test_error_models import TestErrorModels
from tests.test_evidence import TestEvidencePipeline
from tests.test_fuzzing import TestPropertyFuzzing
from tests.test_gates import TestQuantumGates
from tests.test_magic_states import TestMagicStates
from tests.test_measurement import TestQuantumMeasurement
from tests.test_mps import TestMPSSimulator
from tests.test_optimization import TestOptimizationPasses
from tests.test_qldpc_routing import TestQLDPCAndRouting
from tests.test_registry import TestQuantumRegistry
from tests.test_simulator import TestQuantumSimulator
from tests.test_stabilizers import TestStabilizerCodes
from tests.test_synthesis_openqasm import TestSynthesisAndOpenQASM
from tests.test_triorthogonal import TestTriorthogonalCodes
from tests.test_zcc_codegen import TestZCCCodegen


def main() -> None:
    print("=" * 73)
    print("VERIFYING QUANTUM CORE LIBRARY & ALL 14 DISCOVERED ALGORITHMS (24 MODULES)")
    print("=" * 73)

    registry = load_json_object(REGISTRY_PATH)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_cases = [
        TestQuantumGates,
        TestQuantumCircuits,
        TestQuantumSimulator,
        TestQuantumMeasurement,
        TestQuantumRegistry,
        TestRegisteredAlgorithms,
        TestStabilizerCodes,
        TestTriorthogonalCodes,
        TestMWPMDecoder,
        TestOptimizationPasses,
        TestDensityMatrixBackend,
        TestMPSSimulator,
        TestErrorModels,
        TestRandomizedBenchmarking,
        TestMagicStates,
        TestEvidencePipeline,
        TestPropertyFuzzing,
        TestAuditCLI,
        TestSynthesisAndOpenQASM,
        TestQLDPCAndRouting,
        TestZCCCodegen,
    ]

    for tc in test_cases:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2)
    t0 = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - t0

    algorithms = registry.get("algorithms", {})
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_algorithms_registered": len(algorithms),
        "total_tests_run": result.testsRun,
        "successful_tests": result.testsRun - len(result.failures) - len(result.errors),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "elapsed_seconds": round(elapsed, 4),
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "algorithm_list": list(algorithms.keys()),
        "failure_details": [
            {"test": str(f[0]), "message": str(f[1])} for f in result.failures
        ] + [
            {"test": str(e[0]), "message": str(e[1])} for e in result.errors
        ],
    }

    report_dir = PROJECT_ROOT / "artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "all_algorithms_test_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n" + "-" * 73)
    print(f"VERIFICATION COMPLETE: {report['status']}")
    print(f"Passed {report['successful_tests']} / {report['total_tests_run']} tests in {elapsed:.4f}s")
    print(f"Report written to: {report_path}")
    print("=" * 73)

    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
