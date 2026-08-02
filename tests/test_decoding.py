# -*- coding: utf-8 -*-
"""
Tests for Quantum Core Minimum-Weight Perfect Matching (MWPM) Decoder.
"""

from __future__ import annotations

import unittest

from quantum_core.decoding import (
    build_matching_graph,
    decode_mwpm,
    minimum_weight_perfect_matching,
)
from quantum_core.stabilizers import surface_code_distance_3


class TestMWPMDecoder(unittest.TestCase):

    def test_mwpm_empty_syndrome(self) -> None:
        code = surface_code_distance_3()
        syndrome = (0,) * len(code.stabilizers)
        corr = decode_mwpm(syndrome, code)
        self.assertEqual(len(corr), 0)

    def test_build_matching_graph_single_pair(self) -> None:
        code = surface_code_distance_3()
        syndrome = (1, 1, 0, 0, 0, 0, 0, 0)
        defects, edges = build_matching_graph(syndrome, code)
        self.assertEqual(defects, [0, 1])
        self.assertIn((0, 1), edges)

    def test_minimum_weight_perfect_matching_two_defects(self) -> None:
        defects = [0, 1]
        edges = {(0, 1): 2.0, (1, 0): 2.0}
        matching = minimum_weight_perfect_matching(defects, edges)
        self.assertEqual(matching, [(0, 1)])


if __name__ == "__main__":
    unittest.main()
