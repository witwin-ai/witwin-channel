from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_shared_rf_helper_decision_covers_frozen_closure_once() -> None:
    baseline_path = ROOT / "docs/dev/audit/phase13-shared-rf-helper-ledger.json"
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
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == decision[
        "baseline"
    ]["sha256"]


def test_scattering_table_is_reserved_for_adr_026() -> None:
    graph = _json("docs/dev/audit/phase13-shared-rf-dependency-graph.json")
    node = next(
        record
        for record in graph["nodes"]  # type: ignore[index]
        if record["id"].endswith("scattering_table.cuh")
    )
    assert node["target_source_owner"] == "RayD after ADR-026"


def test_architecture_guides_remain_identical_and_list_adr_024() -> None:
    agents = (ROOT / "AGENTS.md").read_bytes()
    claude = (ROOT / "CLAUDE.md").read_bytes()
    assert agents == claude
    assert b"adr-024-shared-rf-transmission-ownership.md" in agents
