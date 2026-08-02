# -*- coding: utf-8 -*-
"""
Quantum Core — Reproducible Evidence & Benchmark Metric Verification Pipeline.
Replaces static declared registry metrics with reproducible empirical benchmark evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class MetricEvidence:
    """Represents a reproducible empirical benchmark metric result payload."""

    metric_name: str
    value: float
    method: str
    baseline: float
    dataset: str
    runs: int
    seed: int
    sha256_hash: str
    evidence_path: str


def compute_sha256_hash(data: bytes | str | Dict[str, Any] | Path) -> str:
    """Computes SHA-256 checksum string for a string, bytes, dictionary payload, or file path."""
    hasher = hashlib.sha256()

    if isinstance(data, Path) or (isinstance(data, str) and os.path.exists(data)):
        path = Path(data)
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    if isinstance(data, dict):
        encoded = json.dumps(data, sort_keys=True).encode("utf-8")
        hasher.update(encoded)
        return hasher.hexdigest()

    if isinstance(data, str):
        hasher.update(data.encode("utf-8"))
        return hasher.hexdigest()

    if isinstance(data, bytes):
        hasher.update(data)
        return hasher.hexdigest()

    raise TypeError(f"unsupported data type for SHA-256 calculation: {type(data)}")


def generate_reproducible_evidence(
    metric_name: str,
    run_func: Callable[[np.random.Generator], float],
    *,
    baseline: float = 1.0,
    n_runs: int = 10,
    seed: int = 1337,
    dataset: str = "synthetic_fuzz_corpus",
    output_dir: str | Path = "docs/evidence",
) -> MetricEvidence:
    """Runs a deterministic benchmark pipeline over n_runs, computes mean value & SHA-256, and writes evidence JSON artifact.

    Returns:
        MetricEvidence object with recorded benchmark results.
    """
    rng = np.random.default_rng(seed)
    results: list[float] = []

    for _ in range(n_runs):
        val = run_func(rng)
        results.append(float(val))

    mean_val = float(np.mean(results))
    raw_payload = {
        "metric_name": metric_name,
        "value": mean_val,
        "baseline": baseline,
        "runs": n_runs,
        "seed": seed,
        "dataset": dataset,
        "raw_results": results,
    }

    content_hash = compute_sha256_hash(raw_payload)
    out_path = Path(output_dir) / f"evidence_{metric_name}_{seed}.json"
    os.makedirs(out_path.parent, exist_ok=True)

    evidence_dict = {
        **raw_payload,
        "method": "reproducible_monte_carlo_trial",
        "sha256_hash": content_hash,
        "evidence_path": str(out_path),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence_dict, f, indent=2)

    return MetricEvidence(
        metric_name=metric_name,
        value=mean_val,
        method="reproducible_monte_carlo_trial",
        baseline=baseline,
        dataset=dataset,
        runs=n_runs,
        seed=seed,
        sha256_hash=content_hash,
        evidence_path=str(out_path),
    )
