# -*- coding: utf-8 -*-
"""
Tests for Quantum Core qLDPC Code Footprint Model & 2D Grid Routing Pass.
"""

from __future__ import annotations

import unittest

from quantum_core.qldpc import bivariate_bicycle_code_spec, compare_surface_vs_qldpc_footprint
from quantum_core.routing import grid_manhattan_distance, route_circuit_2d


class TestQLDPCAndRouting(unittest.TestCase):

    def test_bivariate_bicycle_code_spec(self) -> None:
        spec = bivariate_bicycle_code_spec(k_logical=12, distance=12)
        self.assertEqual(spec.k_logical, 12)
        self.assertGreater(spec.n_physical, 12)
        self.assertGreater(spec.overhead_ratio, 5.0)

    def test_compare_surface_vs_qldpc_footprint(self) -> None:
        comp = compare_surface_vs_qldpc_footprint(k_logical=12, surface_distance=5, qldpc_distance=12)
        self.assertEqual(comp["logical_qubits"], 12)
        self.assertIn("physical_qubit_reduction_percent", comp)

    def test_grid_manhattan_distance(self) -> None:
        dist = grid_manhattan_distance(q0=0, q1=11, grid_cols=4)
        # q0=(0,0), q11=(2,3) -> dist = |0-2| + |0-3| = 5
        self.assertEqual(dist, 5)

    def test_route_circuit_2d(self) -> None:
        circuit = [["cx", 0, 11]]  # Non-adjacent on 4-col grid
        routed, n_swaps = route_circuit_2d(circuit, n_qubits=12, grid_cols=4)
        self.assertGreater(n_swaps, 0)
        self.assertGreater(len(routed), 1)


if __name__ == "__main__":
    unittest.main()
