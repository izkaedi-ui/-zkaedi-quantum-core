# -*- coding: utf-8 -*-
"""
Quantum Core — Registry and artifact resolution helpers.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from quantum_core.types import RegistryValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = PROJECT_ROOT / "catalog" / "algorithm_registry.json"


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RegistryValidationError(
            f"invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(document, dict):
        raise RegistryValidationError(
            f"top-level JSON document must be an object: {path}"
        )

    return document


def resolve_project_path(relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise RegistryValidationError("file_location must be a non-empty string")

    root = PROJECT_ROOT.resolve()
    candidate = (root / relative_path).resolve()

    if not candidate.is_file():
        catalog_candidate = (root / "catalog" / relative_path).resolve()
        if catalog_candidate.is_file():
            candidate = catalog_candidate

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RegistryValidationError(
            f"file_location escapes project root: {relative_path!r}"
        ) from exc

    return candidate


def require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryValidationError(f"{field} must be an object")
    return value


def require_finite_metric(
    metrics: Mapping[str, Any],
    key: str,
) -> float:
    if key not in metrics:
        raise RegistryValidationError(f"missing metric: {key}")

    try:
        result = float(metrics[key])
    except (TypeError, ValueError) as exc:
        raise RegistryValidationError(f"metric {key!r} must be numeric") from exc

    if not math.isfinite(result):
        raise RegistryValidationError(f"metric {key!r} must be finite")

    return result


def import_module_from_path(module_name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to create module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_registry_artifacts(
    registry_data: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Verifies that all entries in algorithm_registry.json have valid files and computes SHA-256 hashes.

    Supports both dictionary format ("algorithms": { ... }) and list format ("discovered_algorithms": [ ... ]).
    Returns mapping of algorithm ID -> SHA-256 hash string of referenced artifact file.
    """
    from quantum_core.evidence import compute_sha256_hash

    if registry_data is None:
        registry_data = load_json_object(REGISTRY_PATH)

    raw_algs = registry_data.get("algorithms")
    if raw_algs is None:
        raw_algs = registry_data.get("discovered_algorithms")

    algorithms: list[dict[str, Any]] = []

    if isinstance(raw_algs, dict):
        for k, v in raw_algs.items():
            if isinstance(v, dict):
                algorithms.append({"id": k, **v})
    elif isinstance(raw_algs, list):
        for item in raw_algs:
            if isinstance(item, dict):
                algorithms.append(item)

    if len(algorithms) == 0:
        raise RegistryValidationError(
            "registry contains no valid 'algorithms' mapping or 'discovered_algorithms' list"
        )

    artifact_hashes: dict[str, str] = {}

    for entry in algorithms:
        alg_id = str(entry.get("id", ""))
        if not alg_id:
            raise RegistryValidationError("algorithm entry missing 'id'")

        loc = entry.get("file_location")
        if not loc:
            raise RegistryValidationError(f"algorithm {alg_id} missing 'file_location'")

        path = resolve_project_path(loc)
        if not path.is_file():
            raise RegistryValidationError(f"artifact file missing for {alg_id}: {path}")

        sha = compute_sha256_hash(path)
        artifact_hashes[alg_id] = sha

    return artifact_hashes
