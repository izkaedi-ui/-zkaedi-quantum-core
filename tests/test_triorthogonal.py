# -*- coding: utf-8 -*-
"""
Tests for Quantum Core Triorthogonal Matrices & Bravyi-Haah Magic State Distillation.
"""

from __future__ import annotations

import unittest
import numpy as np

from quantum_core.triorthogonal import (
    bravyi_haah_14_2_matrix,
    bravyi_kitaev_15_1_matrix,
    is_triorthogonal,
)


class TestTriorthogonalMatrices(unittest.TestCase):

    def test_triorthogonal_check_identity(self) -> None:
        # Simple triorthogonal matrix
        matrix = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
        ], dtype=int)
        self.assertTrue(is_triorthogonal(matrix))

    def test_bravyi_kitaev_15_1_matrix_is_triorthogonal(self) -> None:
        mat = bravyi_kitaev_15_1_matrix()
        self.assertEqual(mat.shape, (4, 15))
        self.assertTrue(is_triorthogonal(mat))

    def test_bravyi_haah_14_2_matrix_is_triorthogonal(self) -> None:
        mat = bravyi_haah_14_2_matrix()
        self.assertEqual(mat.shape, (4, 14))
        self.assertTrue(is_triorthogonal(mat))


if __name__ == "__main__":
    unittest.main()
