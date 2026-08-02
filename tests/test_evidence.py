# -*- coding: utf-8 -*-
"""
Tests for Quantum Core Reproducible Evidence & Benchmark Metric Pipeline.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from quantum_core.evidence import (
    MetricEvidence,
    compute_sha256_hash,
    generate_reproducible_evidence,
)


class TestEvidencePipeline(unittest.TestCase):

    def test_compute_sha256_hash_string_and_dict(self) -> None:
        h1 = compute_sha256_hash("quantum_core_evidence")
        self.assertEqual(len(h1), 64)

        data = {"metric": "fidelity", "value": 0.999}
        h2 = compute_sha256_hash(data)
        self.assertEqual(len(h2), 64)

    def test_generate_reproducible_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            def dummy_benchmark(rng):
                return float(rng.normal(loc=0.995, scale=0.001))

            evidence = generate_reproducible_evidence(
                metric_name="fidelity_bench",
                run_func=dummy_benchmark,
                baseline=1.0,
                n_runs=5,
                seed=42,
                output_dir=tmpdir,
            )

            self.assertIsInstance(evidence, MetricEvidence)
            self.assertEqual(evidence.runs, 5)
            self.assertEqual(evidence.seed, 42)
            self.assertEqual(len(evidence.sha256_hash), 64)
            self.assertTrue(os.path.exists(evidence.evidence_path))

            # Verify saved JSON payload integrity
            with open(evidence.evidence_path, "r", encoding="utf-8") as f:
                saved_json = json.load(f)
            self.assertEqual(saved_json["sha256_hash"], evidence.sha256_hash)


if __name__ == "__main__":
    unittest.main()
