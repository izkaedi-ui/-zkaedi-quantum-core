# -*- coding: utf-8 -*-
"""
Quantum Core — Density-Matrix & Quantum Channel Noise Backend.
Supports open quantum system simulation via density matrices, Kraus operators,
and common physical noise channels (bit-flip, phase-flip, depolarizing, amplitude damping).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from quantum_core.gates import PAULI_X, PAULI_Y, PAULI_Z, i_matrix
from quantum_core.types import COMPLEX_DTYPE, ComplexMatrix, StateVector


def density_matrix_zero(n_qubits: int) -> ComplexMatrix:
    """Initializes the density matrix for |0...0><0...0| state of n_qubits (2^N x 2^N)."""
    dim = 1 << n_qubits
    rho = np.zeros((dim, dim), dtype=COMPLEX_DTYPE)
    rho[0, 0] = 1.0 + 0.0j
    return rho


def density_matrix_from_statevector(state: StateVector) -> ComplexMatrix:
    """Constructs density matrix rho = |psi><psi| from a statevector."""
    s = np.asarray(state, dtype=COMPLEX_DTYPE).reshape(-1, 1)
    return np.matmul(s, s.conj().T)


def density_matrix_purity(rho: ComplexMatrix) -> float:
    """Computes the purity Tr(rho^2) of density matrix rho.

    Purity == 1.0 for pure states, < 1.0 for mixed states.
    """
    rho_sq = np.matmul(rho, rho)
    return float(np.real(np.trace(rho_sq)))


def embed_operator(
    matrix: ComplexMatrix,
    qubits: Sequence[int],
    n_qubits: int,
) -> ComplexMatrix:
    """Embeds an M-qubit operator matrix (2^M x 2^M) into an N-qubit Hilbert space (2^N x 2^N)."""
    m = len(qubits)
    dim = 1 << n_qubits
    full_mat = np.zeros((dim, dim), dtype=COMPLEX_DTYPE)

    mask_other = (dim - 1)
    for q in qubits:
        mask_other &= ~(1 << (n_qubits - 1 - q))

    for i in range(dim):
        other_bits = i & mask_other
        i_sub = 0
        for k, q in enumerate(qubits):
            bit = (i >> (n_qubits - 1 - q)) & 1
            i_sub = (i_sub << 1) | bit

        for j_sub in range(1 << m):
            j = other_bits
            for k, q in enumerate(qubits):
                bit = (j_sub >> (m - 1 - k)) & 1
                j |= (bit << (n_qubits - 1 - q))

            full_mat[i, j] = matrix[i_sub, j_sub]

    return full_mat


def apply_unitary_density_matrix(
    rho: ComplexMatrix,
    unitary: ComplexMatrix,
    qubits: Sequence[int],
    n_qubits: int,
) -> ComplexMatrix:
    """Applies a multi-qubit unitary transformation U to density matrix rho: rho' = U rho U^dagger."""
    full_u = embed_operator(unitary, qubits, n_qubits)
    rho_prime = np.matmul(full_u, np.matmul(rho, full_u.conj().T))
    return rho_prime


def apply_kraus_channel(
    rho: ComplexMatrix,
    kraus_ops: Sequence[ComplexMatrix],
    target_qubit: int,
    n_qubits: int,
) -> ComplexMatrix:
    """Applies a quantum channel defined by Kraus operators {E_k}: E(rho) = sum_k E_k rho E_k^\dagger."""
    new_rho = np.zeros_like(rho, dtype=COMPLEX_DTYPE)

    for ek in kraus_ops:
        full_ek = np.array([[1.0]], dtype=COMPLEX_DTYPE)
        for i in range(n_qubits):
            mat = ek if i == target_qubit else i_matrix()
            full_ek = np.kron(full_ek, mat)

        new_rho += np.matmul(full_ek, np.matmul(rho, full_ek.conj().T))

    return new_rho


def bit_flip_kraus(p: float) -> list[ComplexMatrix]:
    """Kraus operators for single-qubit bit-flip channel with probability p."""
    e0 = np.sqrt(1.0 - p) * i_matrix()
    e1 = np.sqrt(p) * PAULI_X
    return [e0, e1]


def phase_flip_kraus(p: float) -> list[ComplexMatrix]:
    """Kraus operators for single-qubit phase-flip channel with probability p."""
    e0 = np.sqrt(1.0 - p) * i_matrix()
    e1 = np.sqrt(p) * PAULI_Z
    return [e0, e1]


def depolarizing_kraus(p: float) -> list[ComplexMatrix]:
    """Kraus operators for single-qubit depolarizing channel with error probability p."""
    e0 = np.sqrt(1.0 - p) * i_matrix()
    e1 = np.sqrt(p / 3.0) * PAULI_X
    e2 = np.sqrt(p / 3.0) * PAULI_Y
    e3 = np.sqrt(p / 3.0) * PAULI_Z
    return [e0, e1, e2, e3]


def amplitude_damping_kraus(gamma: float) -> list[ComplexMatrix]:
    """Kraus operators for single-qubit amplitude damping channel with parameter gamma."""
    e0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=COMPLEX_DTYPE)
    e1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=COMPLEX_DTYPE)
    return [e0, e1]
