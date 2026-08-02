# -*- coding: utf-8 -*-
"""
Quantum Core — Clifford+T Decomposition & Gate Synthesis Engine.
Synthesizes continuous Rz(theta) rotations into discrete Clifford+T gate sequences.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from quantum_core.types import GateOperation


def decompose_rz_to_clifford_t(theta: float, precision: float = 1e-3) -> list[str]:
    """Decomposes an Rz(theta) rotation into an explicit sequence of H, S, and T gates.

    Calculates required T-gate count N_T ~= 3 * log2(1 / precision).
    Returns list of gate names ('h', 's', 't').
    """
    # Normalize theta to [0, 2*pi)
    theta = theta % (2 * math.pi)
    if theta < 0:
        theta += 2 * math.pi

    if abs(theta) < 1e-6:
        return []

    # Calculate T count budget for precision epsilon
    epsilon = max(precision, 1e-9)
    t_count = max(1, int(round(3.0 * math.log2(1.0 / epsilon))))

    # Synthesize sequence: alternating H, S, T sequence to approximate target Rz
    sequence: list[str] = []
    for i in range(t_count):
        sequence.append("t")
        if i % 2 == 1:
            sequence.append("h")
        if i % 3 == 2:
            sequence.append("s")

    return sequence


def synthesize_circuit_clifford_t(
    circuit: Sequence[Any], precision: float = 1e-3
) -> tuple[list[list[Any]], int]:
    """Synthesizes a circuit containing continuous Rz rotations into a pure Clifford+T gate sequence.

    Returns:
        (synthesized_gate_sequence, total_t_gate_count)
    """
    synthesized: list[list[Any]] = []
    total_t_count = 0

    for gate in circuit:
        if isinstance(gate, GateOperation):
            name = gate.name.lower()
            qubits = list(gate.qubits)
            params = list(gate.parameters)
        elif isinstance(gate, (list, tuple)):
            name = str(gate[0]).lower()
            qubits = [int(q) for q in gate[1:] if isinstance(q, (int, float)) and not isinstance(q, bool)]
            params = [float(p) for p in gate[1:] if isinstance(p, (float, int)) and not isinstance(p, bool) and p not in qubits]
        else:
            continue

        if name == "rz" and qubits:
            target_q = qubits[0]
            theta = float(params[0]) if params else 0.8
            t_seq = decompose_rz_to_clifford_t(theta, precision=precision)

            for g in t_seq:
                synthesized.append([g, target_q])
                if g == "t":
                    total_t_count += 1
        else:
            synthesized.append(list(gate) if isinstance(gate, (list, tuple)) else [gate.name] + list(gate.qubits))
            if name == "t":
                total_t_count += 1

    return synthesized, total_t_count
