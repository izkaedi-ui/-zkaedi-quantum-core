# -*- coding: utf-8 -*-
"""
Quantum Core — Minimum-Weight Perfect Matching (MWPM) Decoder
for surface-code style stabilizer codes.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from quantum_core.stabilizers import PauliOperator, StabilizerCode


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance on a 2-D grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _qubit_to_coord(q: int, width: int) -> Tuple[int, int]:
    """Convert linear qubit index to (row, col) on a width x width grid."""
    return divmod(q, width)


def build_matching_graph(
    syndrome: Sequence[int],
    code: StabilizerCode,
    *,
    width: int | None = None,
) -> Tuple[List[int], Dict[Tuple[int, int], float]]:
    """Builds the complete graph on syndrome defects.

    Returns:
        defects: list of stabilizer indices that are non-trivial (syndrome bit == 1)
        edges: dict (i, j) -> weight (Manhattan distance between representative stabilizer locations)
    """
    defects = [i for i, bit in enumerate(syndrome) if bit == 1]
    if width is None:
        width = int(np.sqrt(code.n_qubits))

    locations: Dict[int, Tuple[int, int]] = {}
    for idx, stab in enumerate(code.stabilizers):
        support = [k for k, c in enumerate(stab.pauli_string) if c != "I"]
        if not support:
            locations[idx] = (0, 0)
            continue
        coords = [_qubit_to_coord(q, width) for q in support]
        r = sum(c[0] for c in coords) // len(coords)
        c = sum(c[1] for c in coords) // len(coords)
        locations[idx] = (r, c)

    edges: Dict[Tuple[int, int], float] = {}
    for i, j in itertools.combinations(defects, 2):
        w = float(_manhattan(locations[i], locations[j]))
        edges[(i, j)] = w
        edges[(j, i)] = w

    return defects, edges


def minimum_weight_perfect_matching(
    defects: Sequence[int],
    edges: Dict[Tuple[int, int], float],
) -> List[Tuple[int, int]]:
    """Exact MWPM for small defect sets via exhaustive matching enumeration.

    Returns a list of matched pairs (i, j).
    """
    defects_list = list(defects)
    n = len(defects_list)
    if n == 0:
        return []
    if n % 2 == 1:
        defects_list = defects_list[:-1]
        n -= 1

    best_weight = float("inf")
    best_matching: List[Tuple[int, int]] = []

    def _matchings(remaining: List[int]) -> List[List[Tuple[int, int]]]:
        if not remaining:
            return [[]]
        a = remaining[0]
        res = []
        for i in range(1, len(remaining)):
            b = remaining[i]
            for m in _matchings(remaining[1:i] + remaining[i + 1 :]):
                res.append([(a, b)] + m)
        return res

    for matching in _matchings(defects_list):
        w = sum(edges.get((i, j), edges.get((j, i), 0.0)) for i, j in matching)
        if w < best_weight:
            best_weight = w
            best_matching = matching

    return best_matching


def decode_mwpm(
    syndrome: Sequence[int],
    code: StabilizerCode,
    *,
    width: int | None = None,
) -> List[PauliOperator]:
    """Full Minimum-Weight Perfect Matching (MWPM) decoder pipeline.

    Returns suggested correction Pauli operators for matched syndrome defects.
    """
    defects, edges = build_matching_graph(syndrome, code, width=width)
    matching = minimum_weight_perfect_matching(defects, edges)

    corrections: List[PauliOperator] = []
    n = code.n_qubits

    for i, j in matching:
        support_i = [
            k for k, c in enumerate(code.stabilizers[i].pauli_string) if c != "I"
        ]
        if support_i:
            q = support_i[0]
            char = "Z" if "Z" in code.stabilizers[i].pauli_string else "X"
            corrections.append(
                PauliOperator("".join(char if k == q else "I" for k in range(n)))
            )

    return corrections
