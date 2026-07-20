from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.refactor_baseline import binding_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Canonical live binding manifest. Historical binding counts remain frozen in
# their phase audit artifacts; the live universe is derived from this manifest.
BASELINE_PATH = REPOSITORY_ROOT / "ci" / "native-binding-manifest.json"
PHASE10_AUDIT_PATH = (
    REPOSITORY_ROOT / "docs" / "dev" / "audit" / "phase10-legacy-dead-binding.json"
)
PHASE13_MIGRATION_DELTA_PATH = (
    REPOSITORY_ROOT / "docs" / "dev" / "audit" / "phase13-migration-delta.json"
)


def _semantic_projection(
    manifest: dict[str, Any], *, removed_parameters: dict[str, set[str]] | None = None
) -> list[dict[str, Any]]:
    """Discard source locations while retaining the complete pybind contract."""

    removed_parameters = removed_parameters or {}

    return sorted(
        (
            {
                "name": symbol["name"],
                "target": symbol["target"],
                "parameters": [
                    {
                        "name": parameter["name"],
                        "type": parameter["type"],
                        "default": parameter["default"],
                    }
                    for parameter in symbol["parameters"]
                    if parameter["name"]
                    not in removed_parameters.get(symbol["name"], set())
                ],
                "pybind_arguments": symbol["pybind_arguments"],
                "return_type": symbol["return_type"],
                "return_arity": symbol["return_arity"],
                "doc": symbol["doc"],
            }
            for symbol in manifest["symbols"]
        ),
        key=lambda symbol: symbol["name"],
    )


def test_native_binding_semantics_match_the_phase_zero_baseline() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    current = binding_manifest(REPOSITORY_ROOT)
    audit = json.loads(PHASE10_AUDIT_PATH.read_text(encoding="utf-8"))
    migration = json.loads(PHASE13_MIGRATION_DELTA_PATH.read_text(encoding="utf-8"))
    retirement = audit["dummy_parameter_projection"]
    parameter_name = retirement["parameter_name"]
    binding_names = set(retirement["bindings"])
    renames = migration["renames_applied"]
    deletions = {
        entry.get("binding", entry.get("symbol"))
        for entry in migration["deletions_applied"]
        if entry.get("binding", entry.get("symbol")) is not None
    }
    rename_projection = {entry["from"]: entry["to"] for entry in renames}
    projected_binding_names = {
        rename_projection.get(name, name) for name in binding_names
    }

    baseline_names = [symbol["name"] for symbol in baseline["symbols"]]
    current_names = [symbol["name"] for symbol in current["symbols"]]
    baseline_name_set = set(baseline_names)
    current_name_set = set(current_names)
    rename_sources = set(rename_projection)
    rename_targets = set(rename_projection.values())

    expected_binding_count = len(baseline_names)
    assert len(current_names) == expected_binding_count
    assert len(baseline_name_set) == expected_binding_count
    assert len(current_name_set) == expected_binding_count
    assert baseline["duplicate_symbols"] == []
    assert current["duplicate_symbols"] == []
    assert migration["current_phase"] == 12
    assert migration["current_subphase"] == "profiling-driven performance closure"
    assert len(renames) == len(rename_sources) == len(rename_targets) == 21
    assert rename_sources.isdisjoint(rename_targets)
    assert baseline_name_set & (rename_sources | rename_targets) == rename_targets
    assert current_name_set & (rename_sources | rename_targets) == rename_targets
    assert len(binding_names) == len(projected_binding_names) == 22
    assert rename_sources & binding_names == binding_names - projected_binding_names
    historical_projection_universe = binding_names | projected_binding_names
    live_projected_binding_names = projected_binding_names - deletions
    assert len(deletions) == 9
    assert (
        baseline_name_set & historical_projection_universe
        == live_projected_binding_names
    )
    assert (
        current_name_set & historical_projection_universe
        == live_projected_binding_names
    )
    baseline_dummy_bindings = {
        symbol["name"]
        for symbol in baseline["symbols"]
        if any(
            parameter["name"] == parameter_name for parameter in symbol["parameters"]
        )
    }
    current_dummy_bindings = {
        symbol["name"]
        for symbol in current["symbols"]
        if any(
            parameter["name"] == parameter_name for parameter in symbol["parameters"]
        )
    }
    assert retirement["status"] == "removed"
    assert baseline_dummy_bindings == set()
    assert current_dummy_bindings == set()
    assert _semantic_projection(current) == _semantic_projection(baseline)
