# -*- coding: utf-8 -*-
"""
Quantum Core — Circuit Optimization Compiler Pass Layer.
Provides peep-hole circuit optimization passes, depth computation, and gate counting.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from quantum_core.types import GateOperation
from quantum_core.validation import normalize_gate


def gate_count(circuit: Sequence[Any]) -> int:
    """Returns the total number of gate operations in the circuit."""
    return len(circuit)


def two_qubit_gate_count(circuit: Sequence[Any]) -> int:
    """Returns the total number of two-qubit gate operations (e.g., CX, CZ, SWAP)."""
    count = 0
    for raw_gate in circuit:
        op = normalize_gate(raw_gate)
        if len(op.qubits) == 2:
            count += 1
    return count


def circuit_depth(circuit: Sequence[Any]) -> int:
    """Computes the topological execution depth (layer count) of the circuit."""
    if not circuit:
        return 0

    qubit_depths: dict[int, int] = {}
    for raw_gate in circuit:
        op = normalize_gate(raw_gate)
        max_prev = max((qubit_depths.get(q, 0) for q in op.qubits), default=0)
        new_depth = max_prev + 1
        for q in op.qubits:
            qubit_depths[q] = new_depth

    return max(qubit_depths.values(), default=0)


def optimize_circuit(circuit: Sequence[Any]) -> list[list[Any]]:
    """Applies peep-hole compiler optimization passes:

    1. Self-inverse cancellation: H H -> I, X X -> I, Y Y -> I, Z Z -> I, CX CX -> I, SWAP SWAP -> I
    2. Consecutive rotation merging: Rz(a) Rz(b) -> Rz(a+b), Rx(a) Rx(b) -> Rx(a+b), Ry(a) Ry(b) -> Ry(a+b)
    3. Identity gate stripping.
    """
    if not circuit:
        return []

    ops = [normalize_gate(g) for g in circuit]
    optimized = True

    while optimized:
        optimized = False
        new_ops: list[GateOperation] = []

        i = 0
        while i < len(ops):
            curr = ops[i]

            # Identity gate stripping
            if curr.name == "i":
                optimized = True
                i += 1
                continue

            if i + 1 < len(ops):
                nxt = ops[i + 1]

                # 1. Self-inverse cancellation for single and two-qubit gates
                if (
                    curr.name == nxt.name
                    and curr.qubits == nxt.qubits
                    and curr.parameters == nxt.parameters
                    and curr.name in {"h", "x", "y", "z", "cx", "cz", "swap"}
                ):
                    optimized = True
                    i += 2  # cancel pair
                    continue

                # 2. Rotation gate merging (Rz, Rx, Ry) on identical target qubits
                if (
                    curr.name in {"rz", "rx", "ry"}
                    and curr.name == nxt.name
                    and curr.qubits == nxt.qubits
                ):
                    theta_curr = float(curr.parameters[0]) if curr.parameters else 0.0
                    theta_nxt = float(nxt.parameters[0]) if nxt.parameters else 0.0
                    total_theta = (theta_curr + theta_nxt) % (2.0 * math.pi)

                    optimized = True
                    i += 2
                    if not math.isclose(total_theta, 0.0, abs_tol=1e-9):
                        # Re-emit merged rotation
                        new_gate = [curr.name, curr.qubits[0], total_theta]
                        new_ops.append(normalize_gate(new_gate))
                    continue

            new_ops.append(curr)
            i += 1

        ops = new_ops

    # Format output back into standard gate list representation
    res: list[list[Any]] = []
    for op in ops:
        if op.parameters:
            res.append([op.name, op.qubits[0], op.parameters[0]])
        elif len(op.qubits) == 2:
            res.append([op.name, op.qubits[0], op.qubits[1]])
        else:
            res.append([op.name] + list(op.qubits))

    return res
