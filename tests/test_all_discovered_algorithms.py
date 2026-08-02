# -*- coding: utf-8 -*-
"""
Discovered Quantum Algorithms — comprehensive verification suite.

This module validates the algorithm registry, referenced artifacts, circuit
definitions, statevector normalization, gate semantics, and cross-algorithm
synergy mappings.

Design goals
------------
* Fail closed on malformed or unknown gates.
* Avoid constructing full 2**n by 2**n matrices during circuit execution.
* Validate qubit indices, gate arity, parameters, and numerical finiteness.
* Preserve a simple unittest entry point with deterministic diagnostics.
* Keep fixed gate matrices immutable and cached rotations safe to reuse.

Notes
-----
The suite verifies repository consistency and simulator behavior. Numeric claims
stored in the registry (for example "quantum_advantage") are treated as declared
metadata unless independently reproduced by a test.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import unittest
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Iterable, Literal, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REGISTRY_PATH: Final[Path] = PROJECT_ROOT / "catalog" / "algorithm_registry.json"


# ---------------------------------------------------------------------------
# Types and numeric constants
# ---------------------------------------------------------------------------

ComplexMatrix: TypeAlias = NDArray[np.complex128]
StateVector: TypeAlias = NDArray[np.complex128]
GateAxis: TypeAlias = Literal["x", "y", "z"]

COMPLEX_DTYPE: Final = np.complex128
DEFAULT_ATOL: Final[float] = 1e-12
MAX_SIMULATED_QUBITS: Final[int] = 20

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


# ---------------------------------------------------------------------------
# Exceptions and normalized gate representation
# ---------------------------------------------------------------------------

class CircuitValidationError(ValueError):
    """Raised when a circuit or gate specification is malformed."""


class RegistryValidationError(ValueError):
    """Raised when the algorithm registry is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class GateOperation:
    """Normalized gate operation."""

    name: str
    qubits: tuple[int, ...]
    parameters: tuple[float, ...] = ()


_GATE_ALIASES: Final[dict[str, str]] = {
    "cnot": "cx",
    "toffoli": "ccx",
    "hadamard": "h",
    "pauli-x": "x",
    "pauli-y": "y",
    "pauli-z": "z",
    "identity": "i",
    "id": "i",
}

_GATE_ARITY: Final[dict[str, int]] = {
    "i": 1,
    "h": 1,
    "x": 1,
    "y": 1,
    "z": 1,
    "s": 1,
    "t": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "cx": 2,
    "cz": 2,
    "swap": 2,
    "ccx": 3,
}

_PARAMETER_COUNTS: Final[dict[str, int]] = {
    "rx": 1,
    "ry": 1,
    "rz": 1,
}


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

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
    """Compose gates in application order.

    ``compose_gates(a, b, c)`` returns ``c @ b @ a``.
    """
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


# ---------------------------------------------------------------------------
# Circuit parsing and validation
# ---------------------------------------------------------------------------

def _normalize_gate_name(value: Any) -> str:
    name = str(value).strip().lower()
    name = _GATE_ALIASES.get(name, name)

    if name not in _GATE_ARITY:
        raise CircuitValidationError(f"unsupported gate: {value!r}")

    return name


def _normalize_qubit(value: Any, *, n_qubits: int, position: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise CircuitValidationError(
            f"qubit at position {position} must be an integer"
        )

    qubit = int(value)
    if not 0 <= qubit < n_qubits:
        raise CircuitValidationError(
            f"qubit index {qubit} is outside [0, {n_qubits - 1}]"
        )

    return qubit


def _normalize_parameter(value: Any, *, position: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CircuitValidationError(
            f"parameter at position {position} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise CircuitValidationError(
            f"parameter at position {position} must be finite"
        )

    return result


def normalize_gate(
    raw_gate: Any,
    *,
    n_qubits: int,
) -> GateOperation:
    if isinstance(raw_gate, Mapping):
        name = _normalize_gate_name(raw_gate.get("gate", raw_gate.get("name", "")))
        raw_qubits = raw_gate.get(
            "qubits",
            raw_gate.get("targets", raw_gate.get("wires", ())),
        )
        raw_parameters = raw_gate.get(
            "parameters",
            raw_gate.get("params", raw_gate.get("angles", ())),
        )
    elif (
        isinstance(raw_gate, Sequence)
        and not isinstance(raw_gate, (str, bytes, bytearray))
        and raw_gate
    ):
        name = _normalize_gate_name(raw_gate[0])
        arity = _GATE_ARITY[name]
        raw_qubits = raw_gate[1 : 1 + arity]
        raw_parameters = raw_gate[1 + arity :]
    else:
        raise CircuitValidationError(
            f"gate must be a non-empty mapping or sequence, got {type(raw_gate).__name__}"
        )

    if isinstance(raw_qubits, Integral) and not isinstance(
        raw_qubits, (bool, np.bool_)
    ):
        raw_qubits = [raw_qubits]

    if raw_qubits is None:
        raw_qubits = ()
    if raw_parameters is None:
        raw_parameters = ()

    if not isinstance(raw_qubits, Sequence) or isinstance(
        raw_qubits, (str, bytes, bytearray)
    ):
        raise CircuitValidationError(f"qubits for gate {name!r} must be a sequence")

    if not isinstance(raw_parameters, Sequence) or isinstance(
        raw_parameters, (str, bytes, bytearray)
    ):
        raw_parameters = [raw_parameters]

    qubits = tuple(
        _normalize_qubit(value, n_qubits=n_qubits, position=index)
        for index, value in enumerate(raw_qubits)
    )
    parameters = tuple(
        _normalize_parameter(value, position=index)
        for index, value in enumerate(raw_parameters)
    )

    expected_arity = _GATE_ARITY[name]
    if len(qubits) != expected_arity:
        raise CircuitValidationError(
            f"gate {name!r} requires {expected_arity} qubit(s), got {len(qubits)}"
        )

    if len(set(qubits)) != len(qubits):
        raise CircuitValidationError(
            f"gate {name!r} cannot use the same qubit more than once: {qubits}"
        )

    expected_parameters = _PARAMETER_COUNTS.get(name, 0)
    if len(parameters) != expected_parameters:
        raise CircuitValidationError(
            f"gate {name!r} requires {expected_parameters} parameter(s), "
            f"got {len(parameters)}"
        )

    return GateOperation(name=name, qubits=qubits, parameters=parameters)


def normalize_gates(
    raw_gates: Iterable[Any],
    *,
    n_qubits: int,
) -> list[GateOperation]:
    if isinstance(raw_gates, (str, bytes, bytearray, Mapping)):
        raise CircuitValidationError("circuit must be an iterable of gate definitions")

    return [
        normalize_gate(raw_gate, n_qubits=n_qubits)
        for raw_gate in raw_gates
    ]


# ---------------------------------------------------------------------------
# Efficient statevector simulator
# ---------------------------------------------------------------------------

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
    # Big-endian qubit convention, matching the original implementation.
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
            gate = rx_matrix(parameters[0])
        elif name == "ry":
            gate = ry_matrix(parameters[0])
        else:
            gate = rz_matrix(parameters[0])

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
# Compatibility matrix constructors
# ---------------------------------------------------------------------------

def embed_single_qubit_gate(
    gate: ComplexMatrix,
    target: int,
    n_qubits: int,
) -> ComplexMatrix:
    """Construct a full embedded operator.

    Retained for diagnostics and compatibility. Circuit execution uses the
    in-place statevector path above.
    """
    count = _validate_qubit_count(n_qubits)
    normalized_target = _normalize_qubit(target, n_qubits=count, position=0)
    candidate = np.asarray(gate, dtype=COMPLEX_DTYPE)

    if candidate.shape != (2, 2) or not is_unitary(candidate):
        raise ValueError("gate must be a 2x2 unitary matrix")

    result = np.array([[1.0]], dtype=COMPLEX_DTYPE)
    for qubit in range(count):
        operand = candidate if qubit == normalized_target else IDENTITY_2
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


# ---------------------------------------------------------------------------
# Registry and artifact helpers
# ---------------------------------------------------------------------------

def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RegistryValidationError(
            f"invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(document, dict):
        raise RegistryValidationError(
            f"top-level JSON document must be an object: {path}"
        )

    return document


def resolve_project_path(relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise RegistryValidationError("file_location must be a non-empty string")

    root = PROJECT_ROOT.resolve()
    candidate = (root / relative_path).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RegistryValidationError(
            f"file_location escapes project root: {relative_path!r}"
        ) from exc

    return candidate


def require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryValidationError(f"{field} must be an object")
    return value


def require_finite_metric(
    metrics: Mapping[str, Any],
    key: str,
) -> float:
    if key not in metrics:
        raise RegistryValidationError(f"missing metric: {key}")

    try:
        result = float(metrics[key])
    except (TypeError, ValueError) as exc:
        raise RegistryValidationError(f"metric {key!r} must be numeric") from exc

    if not math.isfinite(result):
        raise RegistryValidationError(f"metric {key!r} must be finite")

    return result


def import_module_from_path(module_name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to create module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQuantumGatePrimitives(unittest.TestCase):
    """Tests for the simulator and quantum gate primitives."""

    def test_fixed_gates_are_unitary(self) -> None:
        for name, gate in {
            "I": i_matrix(),
            "H": h_matrix(),
            "X": x_matrix(),
            "Y": y_matrix(),
            "Z": z_matrix(),
            "S": s_matrix(),
            "T": t_matrix(),
        }.items():
            with self.subTest(gate=name):
                self.assertTrue(is_unitary(gate))

    def test_pauli_gates_are_hermitian(self) -> None:
        for name, gate in {
            "X": x_matrix(),
            "Y": y_matrix(),
            "Z": z_matrix(),
        }.items():
            with self.subTest(gate=name):
                self.assertTrue(is_hermitian(gate))

    def test_rotations_are_unitary(self) -> None:
        for theta in (0.0, np.pi / 7.0, np.pi, 2.0 * np.pi):
            for name, rotation in {
                "Rx": rx_matrix,
                "Ry": ry_matrix,
                "Rz": rz_matrix,
            }.items():
                with self.subTest(gate=name, theta=theta):
                    self.assertTrue(is_unitary(rotation(theta)))

    def test_zero_rotations_equal_identity(self) -> None:
        for rotation in (rx_matrix, ry_matrix, rz_matrix):
            with self.subTest(rotation=rotation.__name__):
                self.assertTrue(np.allclose(rotation(0.0), IDENTITY_2))

    def test_pi_rotations_match_paulis_up_to_global_phase(self) -> None:
        self.assertTrue(global_phase_equivalent(rx_matrix(np.pi), PAULI_X))
        self.assertTrue(global_phase_equivalent(ry_matrix(np.pi), PAULI_Y))
        self.assertTrue(global_phase_equivalent(rz_matrix(np.pi), PAULI_Z))

    def test_hadamard_is_self_inverse(self) -> None:
        self.assertTrue(np.allclose(HADAMARD @ HADAMARD, IDENTITY_2))

    def test_cached_rotations_are_read_only(self) -> None:
        gate = rx_matrix(0.5)
        with self.assertRaises(ValueError):
            gate[0, 0] = 123.0

    def test_unknown_gate_is_rejected(self) -> None:
        with self.assertRaises(CircuitValidationError):
            execute_gate_sequence([["mystery", 0]], n_qubits=1)

    def test_duplicate_control_and_target_are_rejected(self) -> None:
        with self.assertRaises(CircuitValidationError):
            execute_gate_sequence([["cx", 0, 0]], n_qubits=1)

    def test_bell_state(self) -> None:
        state = execute_gate_sequence(
            [["h", 0], ["cx", 0, 1]],
            n_qubits=2,
        )
        expected = np.array(
            [1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)],
            dtype=COMPLEX_DTYPE,
        )
        self.assertTrue(np.allclose(state, expected, atol=DEFAULT_ATOL))

    def test_statevector_and_full_matrix_paths_agree(self) -> None:
        gates = [
            ["h", 0],
            ["rx", 1, 0.73],
            ["cx", 0, 1],
            ["rz", 0, -0.19],
        ]
        fast_state = execute_gate_sequence(gates, n_qubits=2)

        reference = zero_state(2)
        for operation in normalize_gates(gates, n_qubits=2):
            if operation.name == "h":
                operator = embed_single_qubit_gate(HADAMARD, 0, 2)
            elif operation.name == "rx":
                operator = embed_single_qubit_gate(
                    rx_matrix(operation.parameters[0]),
                    operation.qubits[0],
                    2,
                )
            elif operation.name == "rz":
                operator = embed_single_qubit_gate(
                    rz_matrix(operation.parameters[0]),
                    operation.qubits[0],
                    2,
                )
            elif operation.name == "cx":
                operator = cnot_matrix(
                    operation.qubits[0],
                    operation.qubits[1],
                    2,
                )
            else:
                self.fail(f"unexpected test gate: {operation.name}")
            reference = operator @ reference

        self.assertTrue(np.allclose(fast_state, reference, atol=DEFAULT_ATOL))


class TestAllDiscoveredAlgorithms(unittest.TestCase):
    """Repository-level audit suite for all registered algorithms."""

    EXPECTED_ALGORITHM_COUNT: Final[int] = 13

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_json_object(REGISTRY_PATH)
        cls.repository_info = require_mapping(
            cls.registry.get("repository_info", {}),
            field="repository_info",
        )
        cls.algorithms = require_mapping(
            cls.registry.get("algorithms", {}),
            field="algorithms",
        )

    def get_algorithm(self, algorithm_id: str) -> Mapping[str, Any]:
        self.assertIn(algorithm_id, self.algorithms)
        metadata = self.algorithms[algorithm_id]
        self.assertIsInstance(metadata, Mapping)
        return metadata

    def get_metrics(self, metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        metrics = metadata.get("metrics")
        self.assertIsInstance(metrics, Mapping)
        return metrics

    def get_artifact_path(self, metadata: Mapping[str, Any]) -> Path:
        self.assertIn("file_location", metadata)
        path = resolve_project_path(metadata["file_location"])
        self.assertTrue(path.is_file(), f"missing artifact: {path}")
        return path

    def assert_metric_almost_equal(
        self,
        metadata: Mapping[str, Any],
        key: str,
        expected: float,
        *,
        places: int,
    ) -> None:
        metrics = self.get_metrics(metadata)
        actual = require_finite_metric(metrics, key)
        self.assertAlmostEqual(actual, expected, places=places)

    def test_00_registry_structure_and_count(self) -> None:
        declared_count = self.repository_info.get("total_algorithms")
        self.assertEqual(declared_count, self.EXPECTED_ALGORITHM_COUNT)
        self.assertEqual(len(self.algorithms), self.EXPECTED_ALGORITHM_COUNT)
        self.assertEqual(len(self.algorithms), declared_count)
        self.assertEqual(self.repository_info.get("discovery_sessions"), 5)

        for algorithm_id, metadata in self.algorithms.items():
            with self.subTest(algorithm_id=algorithm_id):
                self.assertIsInstance(algorithm_id, str)
                self.assertTrue(algorithm_id.strip())
                self.assertIsInstance(metadata, Mapping)
                self.assertIn("domain", metadata)
                self.assertIn("metrics", metadata)
                self.get_artifact_path(metadata)

    def test_01_all_declared_numeric_metrics_are_finite(self) -> None:
        for algorithm_id, metadata in self.algorithms.items():
            metrics = self.get_metrics(metadata)
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    with self.subTest(algorithm_id=algorithm_id, metric=key):
                        self.assertTrue(
                            math.isfinite(float(value)),
                            f"{algorithm_id}.{key} is non-finite",
                        )

    def test_02_qalgo_search_2(self) -> None:
        algorithm_id = "QAlgo-Search-2"
        metadata = self.get_algorithm(algorithm_id)

        self.assertEqual(metadata["domain"], "quantum_search")
        self.assert_metric_almost_equal(metadata, "fidelity", 1.0, places=12)
        self.assert_metric_almost_equal(
            metadata,
            "quantum_advantage",
            4.0,
            places=12,
        )

        artifact = load_json_object(self.get_artifact_path(metadata))
        algorithm_info = require_mapping(
            artifact.get("algorithm_info", {}),
            field=f"{algorithm_id}.algorithm_info",
        )
        self.assertEqual(algorithm_info.get("id"), algorithm_id)

        gates = metadata.get("circuit", [])
        if gates:
            state = execute_gate_sequence(gates, n_qubits=4)
            self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=12)

    def test_03_qalgo_optimization_1(self) -> None:
        algorithm_id = "QAlgo-Optimization-1"
        metadata = self.get_algorithm(algorithm_id)

        self.assertEqual(metadata["domain"], "quantum_optimization")
        self.assert_metric_almost_equal(metadata, "fidelity", 0.9778, places=4)
        self.assert_metric_almost_equal(
            metadata,
            "quantum_advantage",
            3.91,
            places=2,
        )

        artifact = load_json_object(self.get_artifact_path(metadata))
        circuit = require_mapping(
            artifact.get("quantum_circuit", {}),
            field=f"{algorithm_id}.quantum_circuit",
        )
        gate_sequence = circuit.get("gate_sequence")
        self.assertIsInstance(gate_sequence, list)
        self.assertGreater(len(gate_sequence), 0)

        state = execute_gate_sequence(gate_sequence, n_qubits=4)
        self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=12)

    def test_04_qalgo_cryptography_4(self) -> None:
        metadata = self.get_algorithm("QAlgo-Cryptography-4")
        self.assertEqual(metadata["domain"], "quantum_cryptography")
        self.assert_metric_almost_equal(metadata, "fidelity", 0.9668, places=4)
        self.get_artifact_path(metadata)

    def test_05_qalgo_simulation_5(self) -> None:
        metadata = self.get_algorithm("QAlgo-Simulation-5")
        self.assertEqual(metadata["domain"], "quantum_simulation")
        self.assert_metric_almost_equal(metadata, "fidelity", 0.9573, places=4)
        self.get_artifact_path(metadata)

    def test_06_qalgo_ml_3(self) -> None:
        metadata = self.get_algorithm("QAlgo-Ml-3")
        self.assertEqual(metadata["domain"], "quantum_ml")
        self.assert_metric_almost_equal(metadata, "fidelity", 0.4688, places=4)
        self.get_artifact_path(metadata)

    def test_07_qalgo_error_s2_1(self) -> None:
        metadata = self.get_algorithm("QAlgo-Error-S2-1")
        self.assertEqual(metadata["domain"], "quantum_error_correction")
        self.assert_metric_almost_equal(metadata, "fidelity", 0.9707, places=4)
        metrics = self.get_metrics(metadata)
        self.assertEqual(metrics.get("speedup_class"), "super-exponential")

    def test_08_qalgo_communication_s2_2(self) -> None:
        metadata = self.get_algorithm("QAlgo-Communication-S2-2")
        self.assertEqual(metadata["domain"], "quantum_communication")
        self.assert_metric_almost_equal(metadata, "fidelity", 1.0, places=12)
        self.assert_metric_almost_equal(
            metadata,
            "quantum_advantage",
            6.67,
            places=2,
        )

    def test_09_qalgo_chemistry_s2_3(self) -> None:
        metadata = self.get_algorithm("QAlgo-Chemistry-S2-3")
        self.assertEqual(metadata["domain"], "quantum_chemistry")
        self.assert_metric_almost_equal(metadata, "fidelity", 0.9844, places=4)

    def test_10_qalgo_optimization_s2_4(self) -> None:
        metadata = self.get_algorithm("QAlgo-Optimization-S2-4")
        self.assertEqual(metadata["domain"], "quantum_optimization")
        self.assert_metric_almost_equal(metadata, "fidelity", 1.0, places=12)
        self.assert_metric_almost_equal(
            metadata,
            "quantum_advantage",
            6.67,
            places=2,
        )

    def test_11_qalgo_search_s2_5(self) -> None:
        metadata = self.get_algorithm("QAlgo-Search-S2-5")
        self.assertEqual(metadata["domain"], "quantum_search")
        self.assert_metric_almost_equal(metadata, "fidelity", 1.0, places=12)

    def test_12_zk_anyon_512(self) -> None:
        algorithm_id = "ZK-ANYON-512"
        metadata = self.get_algorithm(algorithm_id)

        self.assertEqual(metadata["domain"], "quantum_cryptography")
        metrics = self.get_metrics(metadata)
        self.assertEqual(require_finite_metric(metrics, "qubit_allocation"), 512.0)
        self.assertEqual(require_finite_metric(metrics, "quantum_advantage"), 512.0)

        artifact = load_json_object(self.get_artifact_path(metadata))
        braiding = require_mapping(
            artifact.get("anyon_braiding", {}),
            field=f"{algorithm_id}.anyon_braiding",
        )
        self.assertIn("Fibonacci", str(braiding.get("topological_nature", "")))

    def test_13_alien_math_primitive(self) -> None:
        metadata = self.get_algorithm("ALIEN-MATH-PRIMITIVE")
        metrics = self.get_metrics(metadata)
        self.assertEqual(require_finite_metric(metrics, "quantum_advantage"), 1024.0)

        artifact_path = self.get_artifact_path(metadata)
        source = artifact_path.read_text(encoding="utf-8")

        self.assertIn("assembly", source)
        self.assertIn("0x00", source)

    def test_14_zcc_quantum_engine(self) -> None:
        metadata = self.get_algorithm("ZCC-QUANTUM-ENGINE")
        metrics = self.get_metrics(metadata)
        self.assertEqual(require_finite_metric(metrics, "quantum_advantage"), 9999.0)

        artifact_path = self.get_artifact_path(metadata)
        source = artifact_path.read_text(encoding="utf-8")

        self.assertIn("#include", source)

    def test_15_synergy_fusion_matrix(self) -> None:
        synergy_path = PROJECT_ROOT / "tools" / "fuse_synergy_algorithms.py"
        self.assertTrue(synergy_path.is_file(), f"missing file: {synergy_path}")

        synergy_module = import_module_from_path(
            "fuse_synergy_algorithms_test_target",
            synergy_path,
        )

        synergy_map = getattr(synergy_module, "SYNERGY_MAP", None)
        compute_phi = getattr(synergy_module, "compute_phi", None)

        self.assertIsInstance(synergy_map, Mapping)
        self.assertTrue(callable(compute_phi))

        for algorithm_id in self.algorithms:
            with self.subTest(algorithm_id=algorithm_id):
                self.assertIn(algorithm_id, synergy_map)

                relation = synergy_map[algorithm_id]
                self.assertIsInstance(relation, Mapping)

                partner = relation.get("partner")
                self.assertIn(partner, self.algorithms)
                self.assertNotEqual(
                    partner,
                    algorithm_id,
                    "an algorithm should not be its own synergy partner",
                )

                required = (
                    "coherence",
                    "diversity",
                    "actionability",
                    "kappa",
                )
                numeric_values: list[float] = []

                for key in required:
                    self.assertIn(key, relation)
                    numeric_values.append(
                        _validate_real_scalar(relation[key], name=key)
                    )

                phi = float(compute_phi(*numeric_values))
                self.assertTrue(math.isfinite(phi))
                self.assertGreaterEqual(
                    phi,
                    0.85,
                    f"low synergy score for {algorithm_id}: {phi}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
