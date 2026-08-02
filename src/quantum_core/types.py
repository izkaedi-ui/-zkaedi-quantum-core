# -*- coding: utf-8 -*-
"""
Quantum Core — Type definitions, dataclasses, and domain exception classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias
import numpy as np
from numpy.typing import NDArray

ComplexMatrix: TypeAlias = NDArray[np.complex128]
StateVector: TypeAlias = NDArray[np.complex128]
GateAxis: TypeAlias = Literal["x", "y", "z"]

COMPLEX_DTYPE: Final = np.complex128
DEFAULT_ATOL: Final[float] = 1e-12
MAX_SIMULATED_QUBITS: Final[int] = 20


class CircuitValidationError(ValueError):
    """Raised when a circuit or gate specification is malformed."""


class RegistryValidationError(ValueError):
    """Raised when the algorithm registry is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class GateOperation:
    """Normalized gate operation with optional symbolic or resolved numeric parameters."""

    name: str
    qubits: tuple[int, ...]
    parameters: tuple[float | str, ...] = ()
