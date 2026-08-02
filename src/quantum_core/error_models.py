# -*- coding: utf-8 -*-
"""
Quantum Core — Quantum Error Injection & Coherent vs Stochastic Noise Evaluation.
Supports coherent control over-rotations, stochastic Pauli noise channel insertions,
and quantum state overlap fidelity metrics.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from quantum_core.circuit import bind_parameters
from quantum_core.simulator import execute_gate_sequence
from quantum_core.types import GateOperation, StateVector
from quantum_core.validation import normalize_gates


def inject_coherent_rotation_noise(
    circuit: Iterable[Any],
    over_rotation_scale: float = 0.03,  # e.g., +3% over-rotation
    *,
    n_qubits: int,
) -> list[GateOperation]:
    """Injects systematic coherent over-rotation noise into all Rx, Ry, Rz rotation gates.

    Converts angle theta -> theta * (1 + over_rotation_scale).
    """
    normalized = normalize_gates(circuit, n_qubits=n_qubits)
    noisy_gates: list[GateOperation] = []

    for op in normalized:
        if op.name in {"rx", "ry", "rz"} and op.parameters:
            noisy_params = tuple(
                float(p) * (1.0 + over_rotation_scale) if isinstance(p, (int, float)) else p
                for p in op.parameters
            )
            noisy_gates.append(
                GateOperation(name=op.name, qubits=op.qubits, parameters=noisy_params)
            )
        else:
            noisy_gates.append(op)

    return noisy_gates


def inject_stochastic_pauli_noise(
    circuit: Iterable[Any],
    p_error: float = 0.01,
    *,
    n_qubits: int,
    rng: np.random.Generator | None = None,
) -> list[GateOperation]:
    """Injects stochastic single-qubit Pauli X or Z errors after each gate with probability `p_error`."""
    if rng is None:
        rng = np.random.default_rng()

    normalized = normalize_gates(circuit, n_qubits=n_qubits)
    noisy_gates: list[GateOperation] = []

    for op in normalized:
        noisy_gates.append(op)

        # After each gate, check for stochastic error insertion on affected qubits
        for target_qubit in op.qubits:
            if rng.random() < p_error:
                error_type = "x" if rng.random() < 0.5 else "z"
                noisy_gates.append(GateOperation(name=error_type, qubits=(target_qubit,)))

    return noisy_gates


def state_fidelity(state_a: StateVector, state_b: StateVector) -> float:
    """Calculates quantum state overlap fidelity F = |<state_a | state_b>|^2."""
    overlap = complex(np.vdot(state_a, state_b))
    return float(np.abs(overlap) ** 2)


def evaluate_palindromic_coherent_suppression(
    palindromic_circuit: Iterable[Any],
    non_palindromic_circuit: Iterable[Any],
    over_rotation_scale: float = 0.05,
    *,
    n_qubits: int,
) -> dict[str, float]:
    """Evaluates logical fidelity under coherent over-rotation noise comparing palindromic vs non-palindromic circuits."""
    ideal_pal = execute_gate_sequence(palindromic_circuit, n_qubits=n_qubits)
    noisy_pal_gates = inject_coherent_rotation_noise(
        palindromic_circuit, over_rotation_scale, n_qubits=n_qubits
    )
    noisy_pal = execute_gate_sequence(noisy_pal_gates, n_qubits=n_qubits)
    fid_pal = state_fidelity(ideal_pal, noisy_pal)

    ideal_non = execute_gate_sequence(non_palindromic_circuit, n_qubits=n_qubits)
    noisy_non_gates = inject_coherent_rotation_noise(
        non_palindromic_circuit, over_rotation_scale, n_qubits=n_qubits
    )
    noisy_non = execute_gate_sequence(noisy_non_gates, n_qubits=n_qubits)
    fid_non = state_fidelity(ideal_non, noisy_non)

    return {
        "over_rotation_scale": over_rotation_scale,
        "palindromic_fidelity": fid_pal,
        "non_palindromic_fidelity": fid_non,
        "fidelity_gain": fid_pal - fid_non,
    }
