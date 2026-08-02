# -*- coding: utf-8 -*-
"""
Quantum Core — Matrix Product State (MPS) Tensor Network Simulator Backend.
Enables simulation of low-entanglement 1D quantum circuits with bounded bond dimension chi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantum_core.types import COMPLEX_DTYPE, ComplexMatrix, StateVector


@dataclass
class MPSState:
    """Represents an N-qubit Matrix Product State (MPS) with bounded bond dimension chi.

    Tensors A[i] have shape (d_left, 2, d_right) for qubit i in 0..N-1.
    """

    tensors: list[np.ndarray]  # List of 3D complex arrays
    max_bond_dim: int = 64

    @property
    def n_qubits(self) -> int:
        return len(self.tensors)


def mps_zero_state(n_qubits: int, max_bond_dim: int = 64) -> MPSState:
    """Initializes an N-qubit MPS in the |0...0> ground state."""
    tensors: list[np.ndarray] = []
    for i in range(n_qubits):
        # Shape: (left_bond, physical=2, right_bond)
        t = np.zeros((1, 2, 1), dtype=COMPLEX_DTYPE)
        t[0, 0, 0] = 1.0 + 0.0j
        tensors.append(t)

    return MPSState(tensors=tensors, max_bond_dim=max_bond_dim)


def apply_single_qubit_gate_mps(
    mps: MPSState,
    gate_matrix: ComplexMatrix,
    target: int,
) -> None:
    """Applies a 1-qubit gate U (2x2) in-place to tensor A[target]."""
    # A[target] shape: (dl, 2, dr)
    # U @ A[target] along physical dimension
    t = mps.tensors[target]
    # np.einsum('ij,ljk->lik', gate_matrix, t)
    res = np.tensordot(gate_matrix, t, axes=([1], [1]))
    # res shape: (2_physical, dl, dr) -> transpose to (dl, 2_physical, dr)
    mps.tensors[target] = np.transpose(res, (1, 0, 2))


def apply_two_qubit_gate_mps(
    mps: MPSState,
    gate_matrix: ComplexMatrix,
    q0: int,
    q1: int,
) -> None:
    """Applies a 2-qubit gate U (4x4) to adjacent qubits (q0, q0+1) and truncates bond dimension to max_bond_dim via SVD."""
    if abs(q0 - q1) != 1:
        raise ValueError(f"MPS gate application currently requires adjacent qubits: ({q0}, {q1})")

    left_q = min(q0, q1)
    right_q = max(q0, q1)

    t_left = mps.tensors[left_q]    # (d_l, 2, d_m)
    t_right = mps.tensors[right_q]  # (d_m, 2, d_r)

    dl = t_left.shape[0]
    dm = t_left.shape[2]
    dr = t_right.shape[2]

    # Contract t_left and t_right -> theta: (dl, 2_left, 2_right, dr)
    theta = np.tensordot(t_left, t_right, axes=([2], [0]))  # (dl, 2, 2, dr)

    # Reshape gate_matrix 4x4 to (2, 2, 2, 2) [out1, out2, in1, in2]
    u_4x4 = gate_matrix.reshape(2, 2, 2, 2)

    # Apply 2-qubit gate onto theta
    # theta_prime: (dl, 2_left, 2_right, dr)
    theta_prime = np.tensordot(u_4x4, theta, axes=([2, 3], [1, 2]))
    # theta_prime shape: (2_left, 2_right, dl, dr) -> transpose to (dl, 2_left, 2_right, dr)
    theta_prime = np.transpose(theta_prime, (2, 0, 1, 3))

    # Reshape for SVD: (dl * 2_left, 2_right * dr)
    matrix_for_svd = theta_prime.reshape(dl * 2, 2 * dr)

    # Perform SVD decomposition
    u, s, vh = np.linalg.svd(matrix_for_svd, full_matrices=False)

    # Truncate to max_bond_dim
    chi = min(len(s), mps.max_bond_dim)
    u_trunc = u[:, :chi]
    s_trunc = np.diag(s[:chi])
    vh_trunc = vh[:chi, :]

    # Update left tensor: (dl, 2, chi)
    mps.tensors[left_q] = u_trunc.reshape(dl, 2, chi)

    # Update right tensor: (chi, 2, dr) = s_trunc @ vh_trunc -> (chi, 2 * dr) -> (chi, 2, dr)
    sv_trunc = np.matmul(s_trunc, vh_trunc)
    mps.tensors[right_q] = sv_trunc.reshape(chi, 2, dr)


def mps_to_statevector(mps: MPSState) -> StateVector:
    """Contracts all MPS tensors into a dense 2^N statevector."""
    n = mps.n_qubits
    if n == 0:
        return np.array([1.0], dtype=COMPLEX_DTYPE)

    curr = mps.tensors[0]  # (1, 2, d_r)
    for i in range(1, n):
        nxt = mps.tensors[i]  # (d_l, 2, d_r)
        # Contract right bond of curr with left bond of nxt
        curr = np.tensordot(curr, nxt, axes=([-1], [0]))

    # Squeeze boundary bonds (shape 1, 2, 2, ..., 1) -> flatten to 2^N
    return curr.flatten()


def mps_norm(mps: MPSState) -> float:
    """Computes the norm |||psi>||_2 of the MPS statevector."""
    vec = mps_to_statevector(mps)
    return float(np.linalg.norm(vec))
