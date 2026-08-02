# -*- coding: utf-8 -*-
"""
Quantum Core — Fast statevector simulator.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Iterable

import numpy as np

from quantum_core.gates import (
    HADAMARD,
    IDENTITY_2,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PHASE_S,
    PHASE_T,
    is_unitary,
    rx_matrix,
    ry_matrix,
    rz_matrix,
)
from quantum_core.types import (
    COMPLEX_DTYPE,
    MAX_SIMULATED_QUBITS,
    CircuitValidationError,
    ComplexMatrix,
    GateOperation,
    StateVector,
)
from quantum_core.validation import normalize_gate, normalize_gates


def _validate_qubit_count(n_qubits: int) -> int:
    if isinstance(n_qubits, (bool, np.bool_)) or not isinstance(
        n_qubits, Integral
    ):
        raise TypeError("n_qubits must be an integer")

    result = int(n_qubits)
    if not 1 <= result <= MAX_SIMULATED_QUBITS:
        raise ValueError(
            f"n_qubits must be between 1 and {MAX_SIMULATED_QUBITS}, got {result}"
        )

    return result


def zero_state(n_qubits: int) -> StateVector:
    count = _validate_qubit_count(n_qubits)
    state = np.zeros(1 << count, dtype=COMPLEX_DTYPE)
    state[0] = 1.0
    return state


def _validate_state(state: StateVector, n_qubits: int) -> StateVector:
    candidate = np.asarray(state, dtype=COMPLEX_DTYPE)

    expected = 1 << n_qubits
    if candidate.ndim != 1 or candidate.size != expected:
        raise ValueError(
            f"state must be one-dimensional with {expected} amplitudes"
        )
    if not np.all(np.isfinite(candidate)):
        raise ValueError("state contains NaN or infinite amplitudes")

    return candidate


def _bit_mask(qubit: int, n_qubits: int) -> int:
    return 1 << (n_qubits - 1 - qubit)


def apply_single_qubit_gate(
    state: StateVector,
    gate: ComplexMatrix,
    *,
    target: int,
    n_qubits: int,
) -> None:
    candidate = np.asarray(gate, dtype=COMPLEX_DTYPE)
    if candidate.shape != (2, 2):
        raise ValueError(f"single-qubit gate must have shape (2, 2), got {candidate.shape}")
    if not is_unitary(candidate):
        raise ValueError("single-qubit gate must be unitary")

    mask = _bit_mask(target, n_qubits)

    for base in range(state.size):
        if base & mask:
            continue

        paired = base | mask
        amp_zero = state[base]
        amp_one = state[paired]

        state[base] = candidate[0, 0] * amp_zero + candidate[0, 1] * amp_one
        state[paired] = candidate[1, 0] * amp_zero + candidate[1, 1] * amp_one


def apply_controlled_x(
    state: StateVector,
    *,
    control: int,
    target: int,
    n_qubits: int,
) -> None:
    control_mask = _bit_mask(control, n_qubits)
    target_mask = _bit_mask(target, n_qubits)

    for index in range(state.size):
        if index & control_mask and not index & target_mask:
            paired = index | target_mask
            state[index], state[paired] = state[paired], state[index]


def apply_controlled_z(
    state: StateVector,
    *,
    control: int,
    target: int,
    n_qubits: int,
) -> None:
    control_mask = _bit_mask(control, n_qubits)
    target_mask = _bit_mask(target, n_qubits)

    for index in range(state.size):
        if index & control_mask and index & target_mask:
            state[index] *= -1.0


def apply_swap(
    state: StateVector,
    *,
    q1: int,
    q2: int,
    n_qubits: int,
) -> None:
    mask_1 = _bit_mask(q1, n_qubits)
    mask_2 = _bit_mask(q2, n_qubits)

    for index in range(state.size):
        bit_1 = bool(index & mask_1)
        bit_2 = bool(index & mask_2)

        if not bit_1 and bit_2:
            paired = index ^ (mask_1 | mask_2)
            state[index], state[paired] = state[paired], state[index]


def apply_toffoli(
    state: StateVector,
    *,
    control_1: int,
    control_2: int,
    target: int,
    n_qubits: int,
) -> None:
    control_1_mask = _bit_mask(control_1, n_qubits)
    control_2_mask = _bit_mask(control_2, n_qubits)
    target_mask = _bit_mask(target, n_qubits)

    for index in range(state.size):
        controls_set = (
            index & control_1_mask
            and index & control_2_mask
        )
        if controls_set and not index & target_mask:
            paired = index | target_mask
            state[index], state[paired] = state[paired], state[index]


def apply_operation(
    state: StateVector,
    operation: GateOperation,
    *,
    n_qubits: int,
) -> None:
    name = operation.name
    qubits = operation.qubits
    parameters = operation.parameters

    if name == "i":
        return

    if any(isinstance(p, str) for p in parameters):
        raise CircuitValidationError(
            f"cannot simulate gate with unresolved symbolic parameter: {name} {parameters}"
        )

    float_params = [float(p) for p in parameters]

    if name in {"h", "x", "y", "z", "s", "t", "rx", "ry", "rz"}:
        fixed_gates: dict[str, ComplexMatrix] = {
            "h": HADAMARD,
            "x": PAULI_X,
            "y": PAULI_Y,
            "z": PAULI_Z,
            "s": PHASE_S,
            "t": PHASE_T,
        }

        if name in fixed_gates:
            gate = fixed_gates[name]
        elif name == "rx":
            gate = rx_matrix(float_params[0])
        elif name == "ry":
            gate = ry_matrix(float_params[0])
        else:
            gate = rz_matrix(float_params[0])

        apply_single_qubit_gate(
            state,
            gate,
            target=qubits[0],
            n_qubits=n_qubits,
        )
        return

    if name == "cx":
        apply_controlled_x(
            state,
            control=qubits[0],
            target=qubits[1],
            n_qubits=n_qubits,
        )
        return

    if name == "cz":
        apply_controlled_z(
            state,
            control=qubits[0],
            target=qubits[1],
            n_qubits=n_qubits,
        )
        return

    if name == "swap":
        apply_swap(
            state,
            q1=qubits[0],
            q2=qubits[1],
            n_qubits=n_qubits,
        )
        return

    if name == "ccx":
        apply_toffoli(
            state,
            control_1=qubits[0],
            control_2=qubits[1],
            target=qubits[2],
            n_qubits=n_qubits,
        )
        return

    raise CircuitValidationError(f"unhandled normalized gate: {name!r}")


def execute_gate_sequence(
    gates: Iterable[Any],
    n_qubits: int,
    *,
    initial_state: StateVector | None = None,
    norm_tolerance: float = 1e-10,
) -> StateVector:
    count = _validate_qubit_count(n_qubits)
    operations = normalize_gates(gates, n_qubits=count)

    if initial_state is None:
        state = zero_state(count)
    else:
        state = _validate_state(initial_state, count).copy()
        initial_norm = float(np.linalg.norm(state))
        if not np.isclose(initial_norm, 1.0, atol=norm_tolerance, rtol=0.0):
            raise ValueError(
                f"initial state must be normalized, observed norm {initial_norm:.16g}"
            )

    for index, operation in enumerate(operations):
        apply_operation(state, operation, n_qubits=count)

        if not np.all(np.isfinite(state)):
            raise FloatingPointError(
                f"gate {index} ({operation.name}) produced non-finite amplitudes"
            )

    final_norm = float(np.linalg.norm(state))
    if not np.isclose(final_norm, 1.0, atol=norm_tolerance, rtol=0.0):
        raise AssertionError(
            f"circuit failed norm preservation: observed {final_norm:.16g}"
        )

    return state


# ---------------------------------------------------------------------------
# Full matrix constructors for differential verification
# ---------------------------------------------------------------------------

def embed_single_qubit_gate(
    gate: ComplexMatrix,
    target: int,
    n_qubits: int,
) -> ComplexMatrix:
    count = _validate_qubit_count(n_qubits)
    candidate = np.asarray(gate, dtype=COMPLEX_DTYPE)

    if candidate.shape != (2, 2) or not is_unitary(candidate):
        raise ValueError("gate must be a 2x2 unitary matrix")

    result = np.array([[1.0]], dtype=COMPLEX_DTYPE)
    for qubit in range(count):
        operand = candidate if qubit == target else IDENTITY_2
        result = np.kron(result, operand)

    return result


def _operator_from_basis_mapping(
    n_qubits: int,
    mapping: Any,
) -> ComplexMatrix:
    count = _validate_qubit_count(n_qubits)
    dimension = 1 << count
    matrix = np.zeros((dimension, dimension), dtype=COMPLEX_DTYPE)

    for source in range(dimension):
        destination, phase = mapping(source)
        matrix[destination, source] = phase

    return matrix


def cnot_matrix(control: int, target: int, n_qubits: int) -> ComplexMatrix:
    count = _validate_qubit_count(n_qubits)
    operation = normalize_gate(["cx", control, target], n_qubits=count)
    control_mask = _bit_mask(operation.qubits[0], count)
    target_mask = _bit_mask(operation.qubits[1], count)

    return _operator_from_basis_mapping(
        count,
        lambda index: (
            index ^ target_mask if index & control_mask else index,
            1.0,
        ),
    )


def cz_matrix(control: int, target: int, n_qubits: int) -> ComplexMatrix:
    count = _validate_qubit_count(n_qubits)
    operation = normalize_gate(["cz", control, target], n_qubits=count)
    control_mask = _bit_mask(operation.qubits[0], count)
    target_mask = _bit_mask(operation.qubits[1], count)

    return _operator_from_basis_mapping(
        count,
        lambda index: (
            index,
            -1.0 if index & control_mask and index & target_mask else 1.0,
        ),
    )


def swap_matrix(q1: int, q2: int, n_qubits: int) -> ComplexMatrix:
    count = _validate_qubit_count(n_qubits)
    operation = normalize_gate(["swap", q1, q2], n_qubits=count)
    mask_1 = _bit_mask(operation.qubits[0], count)
    mask_2 = _bit_mask(operation.qubits[1], count)

    def mapping(index: int) -> tuple[int, complex]:
        bit_1 = bool(index & mask_1)
        bit_2 = bool(index & mask_2)
        destination = index ^ (mask_1 | mask_2) if bit_1 != bit_2 else index
        return destination, 1.0

    return _operator_from_basis_mapping(count, mapping)


def ccx_matrix(
    control_1: int,
    control_2: int,
    target: int,
    n_qubits: int,
) -> ComplexMatrix:
    count = _validate_qubit_count(n_qubits)
    operation = normalize_gate(
        ["ccx", control_1, control_2, target],
        n_qubits=count,
    )
    control_1_mask = _bit_mask(operation.qubits[0], count)
    control_2_mask = _bit_mask(operation.qubits[1], count)
    target_mask = _bit_mask(operation.qubits[2], count)

    return _operator_from_basis_mapping(
        count,
        lambda index: (
            index ^ target_mask
            if index & control_1_mask and index & control_2_mask
            else index,
            1.0,
        ),
    )
