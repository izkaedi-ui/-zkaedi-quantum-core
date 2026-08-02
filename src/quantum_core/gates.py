# -*- coding: utf-8 -*-
"""
Quantum Core — Gate primitive matrices and linear algebra checks.
"""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from quantum_core.types import (
    COMPLEX_DTYPE,
    DEFAULT_ATOL,
    ComplexMatrix,
    GateAxis,
)

IDENTITY_2: Final[ComplexMatrix] = np.eye(2, dtype=COMPLEX_DTYPE)
PAULI_X: Final[ComplexMatrix] = np.array(
    [[0.0, 1.0], [1.0, 0.0]], dtype=COMPLEX_DTYPE
)
PAULI_Y: Final[ComplexMatrix] = np.array(
    [[0.0, -1.0j], [1.0j, 0.0]], dtype=COMPLEX_DTYPE
)
PAULI_Z: Final[ComplexMatrix] = np.array(
    [[1.0, 0.0], [0.0, -1.0]], dtype=COMPLEX_DTYPE
)
HADAMARD: Final[ComplexMatrix] = np.array(
    [[1.0, 1.0], [1.0, -1.0]], dtype=COMPLEX_DTYPE
) / np.sqrt(2.0)
PHASE_S: Final[ComplexMatrix] = np.array(
    [[1.0, 0.0], [0.0, 1.0j]], dtype=COMPLEX_DTYPE
)
PHASE_T: Final[ComplexMatrix] = np.array(
    [[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]],
    dtype=COMPLEX_DTYPE,
)

for _constant_gate in (
    IDENTITY_2,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    HADAMARD,
    PHASE_S,
    PHASE_T,
):
    _constant_gate.setflags(write=False)


def _validate_real_scalar(value: Real, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


def _readonly_copy(matrix: ComplexMatrix) -> ComplexMatrix:
    result = np.array(matrix, dtype=COMPLEX_DTYPE, copy=True)
    result.setflags(write=False)
    return result


def is_unitary(
    matrix: NDArray[np.complexfloating[Any, Any]],
    *,
    atol: float = DEFAULT_ATOL,
) -> bool:
    candidate = np.asarray(matrix, dtype=COMPLEX_DTYPE)

    if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
        return False
    if not np.all(np.isfinite(candidate)):
        return False

    identity = np.eye(candidate.shape[0], dtype=COMPLEX_DTYPE)
    return bool(
        np.allclose(
            candidate.conj().T @ candidate,
            identity,
            atol=atol,
            rtol=0.0,
        )
    )


def is_hermitian(
    matrix: NDArray[np.complexfloating[Any, Any]],
    *,
    atol: float = DEFAULT_ATOL,
) -> bool:
    candidate = np.asarray(matrix, dtype=COMPLEX_DTYPE)

    return bool(
        candidate.ndim == 2
        and candidate.shape[0] == candidate.shape[1]
        and np.all(np.isfinite(candidate))
        and np.allclose(candidate, candidate.conj().T, atol=atol, rtol=0.0)
    )


def global_phase_equivalent(
    left: NDArray[np.complexfloating[Any, Any]],
    right: NDArray[np.complexfloating[Any, Any]],
    *,
    atol: float = DEFAULT_ATOL,
) -> bool:
    lhs = np.asarray(left, dtype=COMPLEX_DTYPE)
    rhs = np.asarray(right, dtype=COMPLEX_DTYPE)

    if lhs.shape != rhs.shape or lhs.ndim != 2:
        return False

    overlap = np.vdot(rhs.ravel(), lhs.ravel())
    if np.isclose(abs(overlap), 0.0, atol=atol):
        return bool(np.allclose(lhs, rhs, atol=atol, rtol=0.0))

    phase = overlap / abs(overlap)
    return bool(np.allclose(lhs, phase * rhs, atol=atol, rtol=0.0))


def rotation_matrix(axis: GateAxis, theta: Real) -> ComplexMatrix:
    angle = _validate_real_scalar(theta, name="theta")
    half = angle / 2.0
    cosine = np.cos(half)
    sine = np.sin(half)

    if axis == "x":
        return np.array(
            [[cosine, -1.0j * sine], [-1.0j * sine, cosine]],
            dtype=COMPLEX_DTYPE,
        )

    if axis == "y":
        return np.array(
            [[cosine, -sine], [sine, cosine]],
            dtype=COMPLEX_DTYPE,
        )

    if axis == "z":
        return np.array(
            [
                [np.exp(-1.0j * half), 0.0],
                [0.0, np.exp(1.0j * half)],
            ],
            dtype=COMPLEX_DTYPE,
        )

    raise ValueError(f"unsupported rotation axis: {axis!r}")


@lru_cache(maxsize=1024)
def rx_matrix(theta: float) -> ComplexMatrix:
    return _readonly_copy(rotation_matrix("x", theta))


@lru_cache(maxsize=1024)
def ry_matrix(theta: float) -> ComplexMatrix:
    return _readonly_copy(rotation_matrix("y", theta))


@lru_cache(maxsize=1024)
def rz_matrix(theta: float) -> ComplexMatrix:
    return _readonly_copy(rotation_matrix("z", theta))


def i_matrix() -> ComplexMatrix:
    return IDENTITY_2


def h_matrix() -> ComplexMatrix:
    return HADAMARD


def x_matrix() -> ComplexMatrix:
    return PAULI_X


def y_matrix() -> ComplexMatrix:
    return PAULI_Y


def z_matrix() -> ComplexMatrix:
    return PAULI_Z


def s_matrix() -> ComplexMatrix:
    return PHASE_S


def t_matrix() -> ComplexMatrix:
    return PHASE_T


def compose_gates(
    *gates: NDArray[np.complexfloating[Any, Any]],
) -> ComplexMatrix:
    """Compose gates in application order. ``compose_gates(a, b, c)`` returns ``c @ b @ a``."""
    result = IDENTITY_2.copy()

    for index, gate in enumerate(gates):
        candidate = np.asarray(gate, dtype=COMPLEX_DTYPE)
        if candidate.shape != (2, 2):
            raise ValueError(
                f"gate {index} must have shape (2, 2), got {candidate.shape}"
            )
        if not is_unitary(candidate):
            raise ValueError(f"gate {index} is not unitary")
        result = candidate @ result

    return result
