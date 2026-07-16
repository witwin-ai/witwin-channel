from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.refactor_baseline import binding_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "dev"
    / "baselines"
    / "0892d855b27ee851521a181f5158b0bf41091eda"
    / "static"
    / "bindings.json"
)
EXPECTED_BINDING_COUNT = 174


def _semantic_projection(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Discard source locations while retaining the complete pybind contract."""

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

    baseline_names = [symbol["name"] for symbol in baseline["symbols"]]
    current_names = [symbol["name"] for symbol in current["symbols"]]

    assert len(baseline_names) == EXPECTED_BINDING_COUNT
    assert len(current_names) == EXPECTED_BINDING_COUNT
    assert len(set(baseline_names)) == EXPECTED_BINDING_COUNT
    assert len(set(current_names)) == EXPECTED_BINDING_COUNT
    assert baseline["duplicate_symbols"] == []
    assert current["duplicate_symbols"] == []
    assert _semantic_projection(current) == _semantic_projection(baseline)
