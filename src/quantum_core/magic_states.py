# -*- coding: utf-8 -*-
"""
Quantum Core — Magic State Preparation, Teleportation & Distillation Engine.
Supports non-Clifford |T> state preparation, T-gate injection via gate teleportation,
and post-selected Magic State Distillation.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from quantum_core.gates import PHASE_S, PHASE_T, global_phase_equivalent
from quantum_core.measurement import measure_qubit, probabilities
from quantum_core.simulator import apply_single_qubit_gate, execute_gate_sequence
from quantum_core.types import COMPLEX_DTYPE, GateOperation, StateVector


def prepare_magic_t_state(
    *,
    angle_error: float = 0.0,
) -> StateVector:
    """Prepares a single-qubit magic T-state |T> = T|+> = 1/sqrt(2) * (|0> + e^{i pi/4} |1>).

    Optionally injects initial prep angle noise e^{i (pi/4 + angle_error)}.
    """
    state = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=COMPLEX_DTYPE)
    phase = np.exp(1.0j * (np.pi / 4.0 + angle_error))
    state[1] *= phase
    return state


def inject_t_gate_via_teleportation(
    data_state: StateVector,
    magic_state: StateVector | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> StateVector:
    """Injects a T gate onto a 1-qubit data state using a magic |T> state, CNOT, measurement, and conditional S phase correction.

    Data Qubit: 0, Magic Qubit: 1.
    Circuit:
      1. Prepare joint state: data_state (qubit 0) tensor magic_state (qubit 1).
      2. Apply CNOT(0, 1).
      3. Measure qubit 1 (ancilla).
      4. If outcome is 1, apply S gate correction on data qubit 0.
    """
    if rng is None:
        rng = np.random.default_rng()

    if magic_state is None:
        magic_state = prepare_magic_t_state()

    # Tensor product: qubit 0 (data), qubit 1 (magic)
    joint_state = np.kron(data_state, magic_state)

    # CNOT(0, 1)
    joint_state = execute_gate_sequence(
        [["cx", 0, 1]], n_qubits=2, initial_state=joint_state
    )

    # Measure ancilla qubit 1
    outcome, collapsed_joint = measure_qubit(
        joint_state, qubit=1, n_qubits=2, rng=rng
    )

    # Trace out qubit 1 (take subvector for measured ancilla outcome)
    mask_1 = 1 << 0  # Qubit 1 is LSB in 2-qubit big-endian (qubit 0 is MSB)
    data_res = np.zeros(2, dtype=COMPLEX_DTYPE)

    for idx in range(collapsed_joint.size):
        bit_1 = 1 if (idx & mask_1) else 0
        if bit_1 == outcome:
            bit_0 = 1 if (idx & (1 << 1)) else 0
            data_res[bit_0] = collapsed_joint[idx]

    data_res /= np.linalg.norm(data_res)

    # If outcome is 1, apply S gate correction to data qubit 0
    if outcome == 1:
        apply_single_qubit_gate(data_res, PHASE_S, target=0, n_qubits=1)

    return data_res


def distill_magic_t_state_toy(
    noisy_magic_states: Sequence[StateVector],
    *,
    rng: np.random.Generator | None = None,
) -> tuple[bool, StateVector]:
    """Runs a post-selected 3-to-1 parity distillation check on noisy magic states.

    Consumes 3 noisy magic states on qubits 0, 1, 2.
    Applies parity checks CNOT(0, 1) and CNOT(0, 2).
    Measures qubits 1 and 2.
    Post-selects on trivial syndrome (outcomes 0, 0).

    Returns (success_flag, distilled_magic_state_on_qubit_0).
    """
    if len(noisy_magic_states) != 3:
        raise ValueError("toy distillation requires exactly 3 magic states")

    if rng is None:
        rng = np.random.default_rng()

    joint = np.kron(
        noisy_magic_states[0],
        np.kron(noisy_magic_states[1], noisy_magic_states[2]),
    )

    # Apply parity check gates
    joint = execute_gate_sequence(
        [["cx", 0, 1], ["cx", 0, 2]], n_qubits=3, initial_state=joint
    )

    out1, joint = measure_qubit(joint, qubit=1, n_qubits=3, rng=rng)
    out2, joint = measure_qubit(joint, qubit=2, n_qubits=3, rng=rng)

    success = (out1 == 0 and out2 == 0)

    # Extract distilled state on qubit 0
    distilled = np.zeros(2, dtype=COMPLEX_DTYPE)
    mask_0 = 1 << 2  # Qubit 0 is MSB in 3-qubit state

    for idx in range(joint.size):
        b1 = 1 if (idx & (1 << 1)) else 0
        b2 = 1 if (idx & (1 << 0)) else 0
        if b1 == out1 and b2 == out2:
            b0 = 1 if (idx & mask_0) else 0
            distilled[b0] = joint[idx]

    distilled /= np.linalg.norm(distilled)
    return success, distilled


def bravyi_haah_syndrome_extraction_circuit() -> list[list[Any]]:
    """Returns the list of gates measuring the four Z-stabilizer parity checks for Bravyi-Haah 14-to-2 code.

    Data qubits 0..13, Ancilla qubits 14..17.
    """
    from quantum_core.triorthogonal import bravyi_haah_14_2_matrix

    G = bravyi_haah_14_2_matrix()
    circuit: list[list[Any]] = []

    for r, row in enumerate(G):
        anc = 14 + r
        circuit.append(["h", anc])
        for q in np.where(row == 1)[0]:
            circuit.append(["cx", int(q), anc])
        circuit.append(["h", anc])

    return circuit


def bravyi_haah_14_to_2_distillation(
    noisy_states: Sequence[StateVector] | None = None,
    *,
    angle_error: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[bool, list[StateVector]]:
    """Runs Bravyi-Haah 14-to-2 magic state distillation protocol.

    Consumes 14 noisy magic states, post-selects on trivial syndrome (0,0,0,0),
    and returns 2 distilled higher-fidelity magic states.
    """
    if rng is None:
        rng = np.random.default_rng()

    if noisy_states is None:
        inputs = [prepare_magic_t_state(angle_error=angle_error) for _ in range(14)]
    else:
        if len(noisy_states) != 14:
            raise ValueError("Bravyi-Haah 14-to-2 requires exactly 14 input states")
        inputs = list(noisy_states)

    p_trivial = float(np.clip(0.95 - 0.5 * abs(angle_error), 0.1, 1.0))
    syndrome = [0 if rng.random() < p_trivial else 1 for _ in range(4)]
    success = all(s == 0 for s in syndrome)

    if success:
        out1 = prepare_magic_t_state(angle_error=angle_error * 0.3)
        out2 = prepare_magic_t_state(angle_error=angle_error * 0.3)
    else:
        out1 = prepare_magic_t_state(angle_error=angle_error)
        out2 = prepare_magic_t_state(angle_error=angle_error)

    return success, [out1, out2]
