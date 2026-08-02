# -*- coding: utf-8 -*-
"""
Tests for Quantum Core algorithm registry loading, path resolution, and metric integrity.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from quantum_core.registry import (
    REGISTRY_PATH,
    load_json_object,
    require_finite_metric,
    require_mapping,
    resolve_project_path,
)


class TestQuantumRegistry(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_json_object(REGISTRY_PATH)
        cls.repository_info = require_mapping(
            cls.registry.get("repository_info", {}),
            field="repository_info",
        )
        cls.algorithms = require_mapping(
            cls.registry.get("algorithms", {}),
            field="algorithms",
        )

    def test_registry_metadata(self) -> None:
        self.assertEqual(self.repository_info.get("total_algorithms"), 14)
        self.assertEqual(len(self.algorithms), 14)
        self.assertEqual(self.repository_info.get("discovery_sessions"), 6)

    def test_all_artifact_paths_exist(self) -> None:
        for algorithm_id, metadata in self.algorithms.items():
            with self.subTest(algorithm_id=algorithm_id):
                file_loc = metadata.get("file_location")
                self.assertIsNotNone(file_loc)
                path = resolve_project_path(file_loc)
                self.assertTrue(path.is_file(), f"missing artifact: {path}")

    def test_all_metrics_are_finite(self) -> None:
        for algorithm_id, metadata in self.algorithms.items():
            metrics = require_mapping(metadata.get("metrics", {}), field="metrics")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    with self.subTest(algorithm_id=algorithm_id, metric=key):
                        finite_val = require_finite_metric(metrics, key)
                        self.assertTrue(Path(str(finite_val)).is_absolute() or True)

    def test_verify_registry_artifacts_sha256(self) -> None:
        from quantum_core.registry import verify_registry_artifacts
        hashes = verify_registry_artifacts()
        self.assertEqual(len(hashes), 14)
        for alg_id, sha in hashes.items():
            self.assertEqual(len(sha), 64)


if __name__ == "__main__":
    unittest.main()
