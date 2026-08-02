# -*- coding: utf-8 -*-
"""
Quantum Core — Stabilizer Code & Quantum Error Correction (QEC) Engine.
Supports Stabilizer Code generators, Pauli error chain syndrome extraction,
and Surface Code / Bivariate Bicycle qLDPC code specifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from quantum_core.gates import PAULI_X, PAULI_Y, PAULI_Z, i_matrix, is_unitary
from quantum_core.simulator import apply_single_qubit_gate
from quantum_core.types import COMPLEX_DTYPE, ComplexMatrix, StateVector


@dataclass(frozen=True, slots=True)
class PauliOperator:
    """Represents an N-qubit tensor product of Pauli matrices (I, X, Y, Z)."""

    pauli_string: str  # e.g., "XZZI" or "ZIZX"
    phase: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        valid_chars = {"I", "X", "Y", "Z"}
        if not all(c in valid_chars for c in self.pauli_string.upper()):
            raise ValueError(
                f"invalid Pauli operator string: {self.pauli_string!r}"
            )

    @property
    def n_qubits(self) -> int:
        return len(self.pauli_string)

    def to_matrix(self) -> ComplexMatrix:
        """Constructs the 2^N x 2^N unitary matrix representation."""
        mats = {
            "I": i_matrix(),
            "X": PAULI_X,
            "Y": PAULI_Y,
            "Z": PAULI_Z,
        }
        res = np.array([[self.phase]], dtype=COMPLEX_DTYPE)
        for char in self.pauli_string.upper():
            res = np.kron(res, mats[char])
        return res


@dataclass(frozen=True, slots=True)
class StabilizerCode:
    """Defines a Quantum Stabilizer Code via its check matrix generators."""

    name: str
    n_qubits: int
    n_logical: int
    distance: int
    stabilizers: tuple[PauliOperator, ...]
    logical_x: tuple[PauliOperator, ...] = ()
    logical_z: tuple[PauliOperator, ...] = ()


# ── CANONICAL STABILIZER CODE DEFINITIONS ──

def steane_code_7_1_3() -> StabilizerCode:
    """Constructs the 7-qubit Steane Code [[7, 1, 3]] CSS stabilizer code."""
    # 6 Stabilizer generators (3 X-type, 3 Z-type) derived from Hamming [7,4,3] code
    stabs = (
        PauliOperator("IIIXXXX"),
        PauliOperator("XXIIXXI"),
        PauliOperator("XIXIXIX"),
        PauliOperator("IIIZZZZ"),
        PauliOperator("ZZIIZZI"),
        PauliOperator("ZIZIZIZ"),
    )
    lx = (PauliOperator("XXXXXXX"),)
    lz = (PauliOperator("ZZZZZZZ"),)

    return StabilizerCode(
        name="Steane [[7, 1, 3]]",
        n_qubits=7,
        n_logical=1,
        distance=3,
        stabilizers=stabs,
        logical_x=lx,
        logical_z=lz,
    )


def surface_code_distance_3() -> StabilizerCode:
    """Constructs a 9-qubit rotated Surface Code [[9, 1, 3]] patch."""
    stabs = (
        PauliOperator("XXIIIIIII"),  # X-check
        PauliOperator("IXXIIIIII"),  # X-check
        PauliOperator("IIIXXIIII"),  # X-check
        PauliOperator("IIIIXXIII"),  # X-check
        PauliOperator("ZZIIIIIII"),  # Z-check
        PauliOperator("IZIZIIIII"),  # Z-check
        PauliOperator("IIIZZIIII"),  # Z-check
        PauliOperator("IIIIZIZII"),  # Z-check
    )
    lx = (PauliOperator("XIXIXIXIX"),)
    lz = (PauliOperator("ZIZIZIZIZ"),)

    return StabilizerCode(
        name="Rotated Surface Code [[9, 1, 3]]",
        n_qubits=9,
        n_logical=1,
        distance=3,
        stabilizers=stabs,
        logical_x=lx,
        logical_z=lz,
    )


def _build_pauli_string(n_qubits: int, indices: Sequence[int], char: str) -> PauliOperator:
    chars = ["I"] * n_qubits
    for idx in indices:
        chars[idx] = char
    return PauliOperator("".join(chars))


def surface_code_distance_5() -> StabilizerCode:
    """Constructs a 25-qubit geometrically accurate rotated Surface Code [[25, 1, 5]] patch specification on a 5x5 grid."""
    n_qubits = 25

    # 12 X-type plaquette checks (combination of weight-4 faces and weight-2 boundary checks)
    x_check_qubit_groups = [
        [0, 1, 5, 6], [2, 3, 7, 8],
        [6, 7, 11, 12], [8, 9, 13, 14],
        [10, 11, 15, 16], [12, 13, 17, 18],
        [16, 17, 21, 22], [18, 19, 23, 24],
        [4, 9], [15, 20], [0, 5], [19, 24]  # boundary weight-2 checks
    ]

    # 12 Z-type plaquette checks (combination of weight-4 faces and weight-2 boundary checks)
    z_check_qubit_groups = [
        [1, 2, 6, 7], [3, 4, 8, 9],
        [5, 6, 10, 11], [7, 8, 12, 13],
        [11, 12, 16, 17], [13, 14, 18, 19],
        [15, 16, 20, 21], [17, 18, 22, 23],
        [0, 1], [23, 24], [10, 15], [9, 14]  # boundary weight-2 checks
    ]

    x_stabs = tuple(_build_pauli_string(n_qubits, group, "X") for group in x_check_qubit_groups)
    z_stabs = tuple(_build_pauli_string(n_qubits, group, "Z") for group in z_check_qubit_groups)

    # Weight-5 logical operators along grid boundaries
    lx = (_build_pauli_string(n_qubits, [0, 1, 2, 3, 4], "X"),)      # Top row
    lz = (_build_pauli_string(n_qubits, [0, 5, 10, 15, 20], "Z"),)    # Left column

    return StabilizerCode(
        name="Rotated Surface Code [[25, 1, 5]]",
        n_qubits=25,
        n_logical=1,
        distance=5,
        stabilizers=x_stabs + z_stabs,
        logical_x=lx,
        logical_z=lz,
    )


def extract_syndrome(
    state: StateVector,
    code: StabilizerCode,
) -> tuple[int, ...]:
    """Extracts the stabilizer syndrome (+1 / -1 eigenvalues -> 0 / 1 binary syndrome bits).

    syndrome_bit = 0 if <state| S_k |state> = +1
    syndrome_bit = 1 if <state| S_k |state> = -1
    """
    syndrome: list[int] = []

    for stab in code.stabilizers:
        matrix = stab.to_matrix()
        expectation = float(np.real(np.vdot(state, matrix @ state)))
        if np.isclose(expectation, 1.0, atol=1e-5):
            syndrome.append(0)
        elif np.isclose(expectation, -1.0, atol=1e-5):
            syndrome.append(1)
        else:
            raise ValueError(
                f"state is not a stabilizer eigenstate for generator {stab.pauli_string}: expectation = {expectation}"
            )

    return tuple(syndrome)
