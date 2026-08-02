# -*- coding: utf-8 -*-
"""
Quantum Core — Circuit normalization, gate alias resolution, and validation.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np

from quantum_core.types import CircuitValidationError, GateOperation

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


def _normalize_parameter(value: Any, *, position: int) -> float | str:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise CircuitValidationError("symbolic parameter name cannot be empty")
        return cleaned

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise CircuitValidationError(
            f"parameter at position {position} must be numeric or a string symbol"
        )

    result = float(value)
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
