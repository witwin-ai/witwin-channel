from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_shared_rf_helper_decision_covers_frozen_closure_once() -> None:
    baseline = _json("docs/dev/audit/phase13-shared-rf-helper-ledger.json")
    decision = _json(
        "docs/dev/audit/phase13-shared-rf-helper-ownership-decision.json"
    )

    baseline_names = [record["name"] for record in baseline["helpers"]]  # type: ignore[index]
    decided_names = [
        name
        for group in decision["groups"]  # type: ignore[index]
        for name in group["helpers"]
    ]

    assert len(baseline_names) == len(decided_names) == 129
    assert len(set(decided_names)) == 129
    assert set(decided_names) == set(baseline_names)
    assert decision["totals"] == {
        "helper_count": 129,
        "adr_024_rayd": 112,
        "channel_boundary": 10,
        "adr_026_rayd": 7,
        "pending": 0,
    }
    assert decision["baseline"]["sha256"] == (
        "975b75db559aa4facd46659f62dbd74053b0fdb60a9b3e9309ce520d3f1563f4"
    )


def test_scattering_table_reservation_is_realized_by_phase10a() -> None:
    graph = _json("docs/dev/audit/phase13-shared-rf-dependency-graph.json")
    node = next(
        record
        for record in graph["nodes"]  # type: ignore[index]
        if record["id"].endswith("scattering_table.cuh")
    )
    assert node["current_source_owner"] == "RayD"
    assert node["target_source_owner"] == "RayD"
    assert graph["phase10a_activation"]["status"] == "active"  # type: ignore[index]
    assert graph["phase10a_activation"][  # type: ignore[index]
        "scattering_table_helpers_activated"
    ] == 7


def test_architecture_guides_remain_identical_and_list_adr_024() -> None:
    agents = (ROOT / "AGENTS.md").read_bytes()
    claude = (ROOT / "CLAUDE.md").read_bytes()
    assert agents == claude
    assert b"adr-024-shared-rf-transmission-ownership.md" in agents
