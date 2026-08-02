# -*- coding: utf-8 -*-
"""
Quantum Core — Quantum state measurement, statevector collapse, and shot sampling.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from quantum_core.types import StateVector


def probabilities(state: StateVector) -> NDArray[np.float64]:
    """Computes exact statevector measurement probabilities |c_i|^2.

    Validates that total probability sums to 1.0 within numerical tolerance.
    """
    candidate = np.asarray(state, dtype=np.complex128)
    probs = np.abs(candidate) ** 2
    total = float(np.sum(probs))

    if not math.isfinite(total) or not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            f"statevector probabilities must sum to 1.0, observed {total:.16g}"
        )

    return probs.astype(np.float64)


def measure_qubit(
    state: StateVector,
    qubit: int,
    *,
    n_qubits: int,
    rng: np.random.Generator | None = None,
) -> tuple[int, StateVector]:
    """Measures a single target qubit, collapses the statevector, and re-normalizes.

    Returns (outcome_bit, collapsed_statevector).
    """
    if rng is None:
        rng = np.random.default_rng()

    probs = probabilities(state)
    mask = 1 << (n_qubits - 1 - qubit)

    prob_one = float(sum(p for idx, p in enumerate(probs) if idx & mask))
    prob_zero = 1.0 - prob_one

    outcome = 1 if rng.random() < prob_one else 0
    selected_prob = prob_one if outcome == 1 else prob_zero

    if selected_prob <= 0.0:
        raise ValueError("attempted to measure zero-probability subspace")

    collapsed = state.copy()
    for idx in range(collapsed.size):
        bit = 1 if (idx & mask) else 0
        if bit != outcome:
            collapsed[idx] = 0.0

    collapsed /= np.sqrt(selected_prob)
    return outcome, collapsed


def sample_counts(
    state: StateVector,
    shots: int = 1000,
    *,
    n_qubits: int,
    rng: np.random.Generator | None = None,
) -> dict[str, int]:
    """Samples measurement outcomes for a full statevector over `shots` repetitions.

    Returns big-endian bitstring frequency mapping (e.g. `{"00": 5012, "11": 4988}`).
    """
    if shots < 1:
        raise ValueError("shots must be at least 1")

    if rng is None:
        rng = np.random.default_rng()

    probs = probabilities(state)
    indices = rng.choice(len(probs), size=shots, p=probs)

    fmt = f"0{n_qubits}b"
    counts: dict[str, int] = {}

    for idx in indices:
        bitstring = format(int(idx), fmt)
        counts[bitstring] = counts.get(bitstring, 0) + 1

    return counts


def marginal_probabilities(
    state: StateVector,
    qubits: Sequence[int],
    *,
    n_qubits: int,
) -> dict[str, float]:
    """Computes marginal probability distribution over a subset of target qubits."""
    probs = probabilities(state)
    fmt = f"0{len(qubits)}b"
    marginals: dict[str, float] = {}

    for state_idx, prob in enumerate(probs):
        sub_bits = "".join(
            "1" if (state_idx & (1 << (n_qubits - 1 - q))) else "0"
            for q in qubits
        )
        marginals[sub_bits] = marginals.get(sub_bits, 0.0) + float(prob)

    return marginals
