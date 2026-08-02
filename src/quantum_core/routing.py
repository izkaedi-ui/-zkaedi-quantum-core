# -*- coding: utf-8 -*-
"""
Quantum Core — 2D Grid Qubit Routing & SWAP Overhead Pass.
Calculates nearest-neighbor routing overhead and SWAP insertion depth expansion for hardware architectures.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
from quantum_core.types import GateOperation


def grid_manhattan_distance(q0: int, q1: int, grid_cols: int = 4) -> int:
    """Computes Manhattan distance between two qubit indices on a 2D grid of width grid_cols."""
    r0, c0 = q0 // grid_cols, q0 % grid_cols
    r1, c1 = q1 // grid_cols, q1 % grid_cols
    return abs(r0 - r1) + abs(c0 - c1)


def route_circuit_2d(
    circuit: Sequence[Any], n_qubits: int, grid_cols: int = 4
) -> tuple[list[list[Any]], int]:
    """Inserts routing SWAPs for non-adjacent two-qubit gates on a 2D grid layout.

    Returns:
        (routed_circuit_sequence, total_swaps_inserted)
    """
    routed: list[list[Any]] = []
    total_swaps = 0

    for gate in circuit:
        if isinstance(gate, GateOperation):
            name = gate.name.lower()
            qubits = list(gate.qubits)
        elif isinstance(gate, (list, tuple)):
            name = str(gate[0]).lower()
            qubits = [int(q) for q in gate[1:] if isinstance(q, int)]
        else:
            continue

        if name in ("cx", "cz", "swap") and len(qubits) >= 2:
            q0, q1 = qubits[0], qubits[1]
            dist = grid_manhattan_distance(q0, q1, grid_cols=grid_cols)

            if dist > 1:
                # Insert routing SWAP operations along path
                swaps_needed = dist - 1
                for s in range(swaps_needed):
                    routed.append(["swap", q0, q0 + 1 if q0 < q1 else q0 - 1])
                    total_swaps += 1

                routed.append(list(gate) if isinstance(gate, (list, tuple)) else [name, q0, q1])

                for s in range(swaps_needed):
                    routed.append(["swap", q0, q0 + 1 if q0 < q1 else q0 - 1])
                    total_swaps += 1
            else:
                routed.append(list(gate) if isinstance(gate, (list, tuple)) else [name, q0, q1])
        else:
            routed.append(list(gate) if isinstance(gate, (list, tuple)) else [name] + qubits)

    return routed, total_swaps
