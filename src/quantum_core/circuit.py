# -*- coding: utf-8 -*-
"""
Quantum Core — Circuit objects and symbolic parameter binding.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from quantum_core.types import CircuitValidationError, GateOperation
from quantum_core.validation import normalize_gate, normalize_gates


def bind_parameters(
    circuit: Iterable[Any],
    parameter_map: Mapping[str, float],
    *,
    n_qubits: int,
) -> list[GateOperation]:
    """Binds symbolic parameters in a circuit to numeric values.

    Example:
    >>> bound = bind_parameters([["ry", 0, "theta_0"]], {"theta_0": 0.75}, n_qubits=1)
    >>> bound[0].parameters
    (0.75,)
    """
    normalized = normalize_gates(circuit, n_qubits=n_qubits)
    bound_gates: list[GateOperation] = []

    for op in normalized:
        resolved_params: list[float] = []
        for param in op.parameters:
            if isinstance(param, str):
                if param not in parameter_map:
                    raise CircuitValidationError(
                        f"unresolved symbolic parameter: {param!r}"
                    )
                val = float(parameter_map[param])
                resolved_params.append(val)
            else:
                resolved_params.append(float(param))

        bound_gates.append(
            GateOperation(
                name=op.name,
                qubits=op.qubits,
                parameters=tuple(resolved_params),
            )
        )

    return bound_gates
