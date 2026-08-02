# -*- coding: utf-8 -*-
"""
Quantum Core — Randomized Benchmarking (RB) & Clifford Sequence Analytics.
Supports standard single-qubit and multi-qubit Clifford RB sequence generation,
inversion closure, survival probability estimation, and decay fitting.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

from quantum_core.gates import (
    HADAMARD,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PHASE_S,
    global_phase_equivalent,
    is_unitary,
)
from quantum_core.measurement import probabilities
from quantum_core.simulator import execute_gate_sequence
from quantum_core.types import COMPLEX_DTYPE, ComplexMatrix, GateOperation


# 1-Qubit Clifford Group generator gate definitions
_SINGLE_QUBIT_CLIFFORD_GATES: tuple[tuple[str, ...], ...] = (
    ("i",),
    ("h",),
    ("x",),
    ("y",),
    ("z",),
    ("s",),
    ("h", "s"),
    ("s", "h"),
    ("h", "s", "h"),
    ("s", "s"),
    ("h", "x"),
    ("h", "y"),
    ("h", "z"),
    ("s", "x"),
    ("s", "y"),
    ("s", "z"),
    ("h", "s", "x"),
    ("h", "s", "y"),
    ("h", "s", "z"),
    ("s", "h", "x"),
    ("s", "h", "y"),
    ("s", "h", "z"),
    ("h", "s", "h", "s"),
    ("s", "s", "h"),
)


def generate_random_clifford_sequence(
    length: int,
    *,
    target_qubit: int = 0,
    n_qubits: int = 1,
    rng: np.random.Generator | None = None,
) -> list[GateOperation]:
    """Generates a random sequence of `length` single-qubit Clifford gates."""
    if length < 0:
        raise ValueError("sequence length must be non-negative")
    if rng is None:
        rng = np.random.default_rng()

    sequence: list[GateOperation] = []
    num_cliffords = len(_SINGLE_QUBIT_CLIFFORD_GATES)

    for _ in range(length):
        choice_idx = int(rng.integers(0, num_cliffords))
        gate_names = _SINGLE_QUBIT_CLIFFORD_GATES[choice_idx]
        for name in gate_names:
            sequence.append(GateOperation(name=name, qubits=(target_qubit,)))

    return sequence


def append_inversion_gate(
    sequence: Sequence[GateOperation],
    *,
    target_qubit: int = 0,
    n_qubits: int = 1,
) -> list[GateOperation]:
    """Computes the composed unitary of `sequence` and appends an inverting gate sequence so the ideal final state is |0>."""
    final_state = execute_gate_sequence(sequence, n_qubits=n_qubits)

    # Find matching Clifford inverse that maps final_state back to zero_state |0>
    full_sequence = list(sequence)

    for inverse_gates in _SINGLE_QUBIT_CLIFFORD_GATES:
        test_seq = list(sequence) + [
            GateOperation(name=name, qubits=(target_qubit,))
            for name in inverse_gates
        ]
        test_state = execute_gate_sequence(test_seq, n_qubits=n_qubits)
        if np.isclose(abs(test_state[0]), 1.0, atol=1e-8):
            for name in inverse_gates:
                full_sequence.append(GateOperation(name=name, qubits=(target_qubit,)))
            return full_sequence

    raise RuntimeError("failed to find valid Clifford inverting gate")


def run_randomized_benchmarking_trial(
    length: int,
    *,
    noise_injector: Callable[[list[GateOperation]], list[GateOperation]] | None = None,
    target_qubit: int = 0,
    n_qubits: int = 1,
    rng: np.random.Generator | None = None,
) -> float:
    """Runs a single Randomized Benchmarking trial of given sequence `length`.

    Returns survival probability P(|0>) of measuring the initial ground state.
    """
    raw_seq = generate_random_clifford_sequence(
        length, target_qubit=target_qubit, n_qubits=n_qubits, rng=rng
    )
    closed_seq = append_inversion_gate(
        raw_seq, target_qubit=target_qubit, n_qubits=n_qubits
    )

    if noise_injector is not None:
        closed_seq = noise_injector(closed_seq)

    final_state = execute_gate_sequence(closed_seq, n_qubits=n_qubits)
    probs = probabilities(final_state)
    return float(probs[0])


def fit_rb_decay(
    lengths: Sequence[int],
    survival_probs: Sequence[float],
    *,
    n_qubits: int = 1,
) -> dict[str, float | bool]:
    """Fits standard RB exponential decay model P(m) = A * p^m + B.

    Returns dictionary containing:
        - p: decay parameter
        - A: amplitude (SPAM)
        - B: offset
        - r: average error rate per Clifford gate r = (d-1)/d * (1 - p)
        - r_std: fit residual error estimate
        - success: boolean fit status
    """
    lengths_arr = np.asarray(lengths, dtype=float)
    probs_arr = np.asarray(survival_probs, dtype=float)

    if lengths_arr.size < 3:
        raise ValueError("need at least 3 data points for a reliable fit")

    B_guess = float(np.min(probs_arr))
    shifted = np.clip(probs_arr - B_guess, 1e-12, None)

    try:
        log_shifted = np.log(shifted)
        X = np.column_stack([np.ones_like(lengths_arr), lengths_arr])
        coeffs, _, _, _ = np.linalg.lstsq(X, log_shifted, rcond=None)
        log_A, log_p = coeffs
        A = float(np.exp(log_A))
        p = float(np.exp(log_p))
    except Exception:
        return {
            "p": 0.0,
            "A": 0.0,
            "B": 0.0,
            "r": 1.0,
            "r_std": 1.0,
            "success": False,
        }

    B = float(np.clip(B_guess, 0.0, 0.5))
    d = 1 << n_qubits
    r = ((d - 1) / d) * (1.0 - p)

    res = probs_arr - (A * (p ** lengths_arr) + B)
    rmse = float(np.sqrt(np.mean(res ** 2))) if res.size else 1.0
    r_std = abs(r) * (rmse / (float(np.mean(probs_arr)) + 1e-12))

    return {
        "p": p,
        "A": A,
        "B": B,
        "r": float(r),
        "r_std": float(r_std),
        "success": True,
    }

