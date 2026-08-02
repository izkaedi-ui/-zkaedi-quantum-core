# -*- coding: utf-8 -*-
"""
Quantum Core — Main Module Entry Point for CLI (`python -m quantum_core`).
"""

from __future__ import annotations

import sys
from quantum_core.cli import run_audit_cli

if __name__ == "__main__":
    sys.exit(run_audit_cli(sys.argv[1:]))
