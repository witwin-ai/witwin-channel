from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.refactor_baseline import binding_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Canonical live binding manifest. The phase-0 copy under
# docs/dev/baselines/0892d855.../static/bindings.json is an immutable
# historical artifact and must never be rewritten; symbol additions are
# re-frozen HERE, with the delta explained in the introducing change
# (ADR-010 added the 5 scattering/rough-reflection kernels: 174 -> 179;
# ADR-014 added the 4 scattering JVP/VJP companions: 179 -> 183;
# ADR-015 added the 4 scattering table-eval / table-build JVP/VJP
# companions: 183 -> 187).
BASELINE_PATH = REPOSITORY_ROOT / "ci" / "native-binding-manifest.json"
EXPECTED_BINDING_COUNT = 187
PHASE10_AUDIT_PATH = (
    REPOSITORY_ROOT / "docs" / "dev" / "audit" / "phase10-legacy-dead-binding.json"
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
    retirement = audit["dummy_parameter_projection"]
    parameter_name = retirement["parameter_name"]
    binding_names = set(retirement["bindings"])
    removed_parameters = {name: {parameter_name} for name in binding_names}

    baseline_names = [symbol["name"] for symbol in baseline["symbols"]]
    current_names = [symbol["name"] for symbol in current["symbols"]]

    assert len(baseline_names) == EXPECTED_BINDING_COUNT
    assert len(current_names) == EXPECTED_BINDING_COUNT
    assert len(set(baseline_names)) == EXPECTED_BINDING_COUNT
    assert len(set(current_names)) == EXPECTED_BINDING_COUNT
    assert baseline["duplicate_symbols"] == []
    assert current["duplicate_symbols"] == []
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
    assert baseline_dummy_bindings == binding_names
    assert current_dummy_bindings == set()
    for symbol in baseline["symbols"]:
        if symbol["name"] in binding_names:
            parameter_names = [parameter["name"] for parameter in symbol["parameters"]]
            assert parameter_names.count(parameter_name) == 1
            assert parameter_names[-1] == parameter_name
    assert _semantic_projection(
        current, removed_parameters=removed_parameters
    ) == _semantic_projection(baseline, removed_parameters=removed_parameters)
