# -*- coding: utf-8 -*-
"""
Tests for Quantum Core Unified Audit CLI Entry Point.
"""

from __future__ import annotations

import unittest
from quantum_core.cli import run_audit_cli


class TestAuditCLI(unittest.TestCase):

    def test_run_audit_cli_returns_zero(self) -> None:
        exit_code = run_audit_cli()
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
