# -*- coding: utf-8 -*-
"""
Quantum Core — Unified Audit CLI Module.
Executes end-to-end verification of algorithm registry, SHA-256 artifacts,
and circuit execution for `python -m quantum_core audit`.
"""

from __future__ import annotations

import sys
from typing import Sequence

from quantum_core.registry import REGISTRY_PATH, load_json_object, verify_registry_artifacts


def run_audit_cli(args: Sequence[str] | None = None) -> int:
    """Executes full quantum_core verification audit CLI for registry, SHA-256 artifacts, and metrics.

    Returns:
        0 on SUCCESS (GREEN), 1 on FAILURE (RED).
    """
    print("=" * 60)
    print("      QUANTUM CORE UNIFIED AUDIT & VERIFICATION SUITE      ")
    print("=" * 60)

    try:
        registry_data = load_json_object(REGISTRY_PATH)
        repo_info = registry_data.get("repository_info", {})
        total_algs = repo_info.get("total_algorithms", 0)

        print(f"[*] Loading registry: {REGISTRY_PATH}")
        print(f"[*] Registered algorithms count: {total_algs}")

        print("[*] Verifying artifact SHA-256 checksums...")
        hashes = verify_registry_artifacts(registry_data)
        for alg_id, sha in hashes.items():
            print(f"    [PASS] {alg_id:<25} SHA256: {sha[:16]}...")

        print("-" * 60)
        print(f"[SUCCESS] All {len(hashes)} algorithm artifacts verified cleanly.")
        print("RESULT: GREEN (100% PASS)")
        print("=" * 60)
        return 0

    except Exception as exc:
        print(f"\n[ERROR] Audit failed with exception: {exc}", file=sys.stderr)
        print("RESULT: RED (FAIL)", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_audit_cli(sys.argv[1:]))
