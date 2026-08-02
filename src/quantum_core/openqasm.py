# -*- coding: utf-8 -*-
"""
Quantum Core — OpenQASM 3.0 Exporter & Importer Module.
Enables interoperability with Qiskit, AWS Braket, and hardware vendor backends.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
from quantum_core.types import GateOperation


def to_openqasm3(circuit: Sequence[Any], n_qubits: int) -> str:
    """Exports a quantum_core gate sequence to standard OpenQASM 3.0 string format."""
    lines = [
        'OPENQASM 3.0;',
        'include "stdgates.inc";',
        f'qubit[{n_qubits}] q;',
        f'bit[{n_qubits}] c;',
        ''
    ]

    for gate in circuit:
        if isinstance(gate, GateOperation):
            name = gate.name.lower()
            qubits = list(gate.qubits)
            params = list(gate.parameters)
        elif isinstance(gate, (list, tuple)):
            name = str(gate[0]).lower()
            # Parse targets and params
            qubits = []
            params = []
            for item in gate[1:]:
                if isinstance(item, int):
                    qubits.append(item)
                elif isinstance(item, float):
                    params.append(item)
        else:
            continue

        if name in ("h", "x", "y", "z", "s", "t"):
            if qubits:
                lines.append(f"{name} q[{qubits[0]}];")
        elif name == "cx" and len(qubits) >= 2:
            lines.append(f"cx q[{qubits[0]}], q[{qubits[1]}];")
        elif name == "cz" and len(qubits) >= 2:
            lines.append(f"cz q[{qubits[0]}], q[{qubits[1]}];")
        elif name == "swap" and len(qubits) >= 2:
            lines.append(f"swap q[{qubits[0]}], q[{qubits[1]}];")
        elif name in ("rx", "ry", "rz") and qubits:
            theta = params[0] if params else 0.0
            lines.append(f"{name}({theta}) q[{qubits[0]}];")

    return "\n".join(lines)


def from_openqasm3(qasm_str: str) -> tuple[list[list[Any]], int]:
    """Parses OpenQASM 3.0 code string into quantum_core gate sequence and qubit count."""
    circuit: list[list[Any]] = []
    n_qubits = 1

    # Extract qubit count
    q_match = re.search(r'qubit\[(\d+)\]', qasm_str)
    if q_match:
        n_qubits = int(q_match.group(1))

    # Parse gate lines
    for line in qasm_str.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("OPENQASM") or line.startswith("include") or line.startswith("qubit") or line.startswith("bit"):
            continue

        # Single qubit gate e.g. h q[0];
        m_single = re.match(r'^(h|x|y|z|s|t)\s+q\[(\d+)\];', line, re.IGNORECASE)
        if m_single:
            g_name = m_single.group(1).lower()
            q_idx = int(m_single.group(2))
            circuit.append([g_name, q_idx])
            continue

        # 2-qubit gate e.g. cx q[0], q[1];
        m_two = re.match(r'^(cx|cz|swap)\s+q\[(\d+)\]\s*,\s*q\[(\d+)\];', line, re.IGNORECASE)
        if m_two:
            g_name = m_two.group(1).lower()
            q0 = int(m_two.group(2))
            q1 = int(m_two.group(3))
            circuit.append([g_name, q0, q1])
            continue

        # Rotation gate e.g. rz(0.8) q[0];
        m_rot = re.match(r'^(rx|ry|rz)\(([^)]+)\)\s+q\[(\d+)\];', line, re.IGNORECASE)
        if m_rot:
            g_name = m_rot.group(1).lower()
            theta = float(m_rot.group(2))
            q_idx = int(m_rot.group(3))
            circuit.append([g_name, q_idx, theta])
            continue

    return circuit, n_qubits
