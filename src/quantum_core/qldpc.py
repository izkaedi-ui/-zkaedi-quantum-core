# -*- coding: utf-8 -*-
"""
Quantum Core — Quantum Low-Density Parity-Check (qLDPC) Code Resource Model.
Provides physical qubit footprint and syndrome measurement cycle estimates
for modern qLDPC architectures (e.g. Bivariate Bicycle Codes).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QLDPCCodeSpec:
    """Specification for a Quantum Low-Density Parity-Check (qLDPC) Code."""

    name: str
    n_physical: int
    k_logical: int
    distance: int

    @property
    def overhead_ratio(self) -> float:
        """Physical to logical qubit overhead ratio."""
        return self.n_physical / float(self.k_logical) if self.k_logical > 0 else 0.0


def bivariate_bicycle_code_spec(k_logical: int, distance: int = 12) -> QLDPCCodeSpec:
    """Constructs a qLDPC Bivariate Bicycle Code footprint estimate.

    qLDPC codes achieve high-rate logical encoding with overhead ratio ~10x-14x
    compared to Surface Codes (2*d^2 = 50x to 162x).
    """
    # Overhead ratio scaling for qLDPC codes
    overhead_ratio = 10.0 + (distance - 6) * 0.5
    n_physical = int(round(k_logical * overhead_ratio))

    return QLDPCCodeSpec(
        name=f"qLDPC Bivariate Bicycle [[{n_physical}, {k_logical}, {distance}]]",
        n_physical=n_physical,
        k_logical=k_logical,
        distance=distance,
    )


def compare_surface_vs_qldpc_footprint(
    k_logical: int, surface_distance: int = 5, qldpc_distance: int = 12
) -> dict[str, Any]:
    """Returns comparative footprint metrics between Surface Code [[2d^2]] and qLDPC Code."""
    surface_physical_per_logical = 2 * (surface_distance**2)
    total_surface_physical = k_logical * surface_physical_per_logical

    qldpc_spec = bivariate_bicycle_code_spec(k_logical=k_logical, distance=qldpc_distance)

    reduction_percentage = (
        (total_surface_physical - qldpc_spec.n_physical)
        / float(total_surface_physical)
    ) * 100.0 if total_surface_physical > 0 else 0.0

    return {
        "logical_qubits": k_logical,
        "surface_code_distance": surface_distance,
        "surface_code_total_physical": total_surface_physical,
        "qldpc_code_name": qldpc_spec.name,
        "qldpc_code_total_physical": qldpc_spec.n_physical,
        "qldpc_overhead_ratio": qldpc_spec.overhead_ratio,
        "physical_qubit_reduction_percent": round(reduction_percentage, 2),
    }
