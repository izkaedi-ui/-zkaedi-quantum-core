# -*- coding: utf-8 -*-
"""
Quantum Core — Triorthogonal Binary Matrices & Bravyi-Haah Magic State Distillation.
Supports triorthogonality verification, Reed-Muller 15-to-1 matrix, and Bravyi-Haah 14-to-2 matrix.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def is_triorthogonal(matrix: NDArray[np.int_]) -> bool:
    """Verifies whether a binary matrix G (m x n) is triorthogonal.

    Triorthogonality conditions:
      1. Every pair of rows (a, b) has even dot product mod 2:
         sum_j G_{a,j} G_{b,j} = 0 (mod 2)
      2. Every triple of rows (a, b, c) has even entrywise product sum mod 2:
         sum_j G_{a,j} G_{b,j} G_{c,j} = 0 (mod 2)
    """
    g = np.asarray(matrix, dtype=int) % 2
    if g.ndim != 2:
        return False

    m, n = g.shape
    if m < 1 or n < 1:
        return False

    # Check row pair overlap (mod 2)
    for a in range(m):
        for b in range(a + 1, m):
            pair_sum = int(np.sum(g[a] & g[b])) % 2
            if pair_sum != 0:
                return False

    # Check row triple overlap (mod 2)
    for a in range(m):
        for b in range(a + 1, m):
            for c in range(b + 1, m):
                triple_sum = int(np.sum(g[a] & g[b] & g[c])) % 2
                if triple_sum != 0:
                    return False

    return True


def bravyi_kitaev_15_1_matrix() -> NDArray[np.int_]:
    """Constructs the 4 x 15 binary triorthogonal matrix for 15-to-1 Reed-Muller T-state distillation."""
    # 4 rows, 15 columns (non-zero binary 4-tuples)
    cols: list[list[int]] = []
    for val in range(1, 16):
        cols.append([(val >> 3) & 1, (val >> 2) & 1, (val >> 1) & 1, val & 1])

    mat = np.array(cols, dtype=int).T
    return mat


def bravyi_haah_14_2_matrix() -> NDArray[np.int_]:
    """Constructs the 4 x 14 binary triorthogonal matrix for Bravyi-Haah 14-to-2 T-state distillation."""
    # Representative 4 x 14 triorthogonal matrix (2 logical qubits, distance 2)
    raw = [
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    ]
    return np.array(raw, dtype=int)
