from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/dev/audit"
RAYD_COMMIT = "4cb400acbfcc2da7fda4110d1298d311816905f1"
INTEGRATION_V2_SHA256 = (
    "c8e162c55a0e5abe789e4f1b19cd6ab00ee4ef59d70244cfc55d58166aeb646b"
)
LAYER_STACK_SYMBOLS = {
    "em_layer_stack_eval",
    "em_layer_stack_backward",
    "em_layer_stack_jvp",
}
REMOVED_CHANNEL_SOURCES = {
    "native/channel_native/em/complex.cuh",
    "native/channel_native/em/medium.cuh",
    "native/channel_native/em/fresnel.cuh",
    "native/channel_native/em/layer_stack.cuh",
    "native/channel_native/field_transport.cuh",
    "native/channel_native/field_transport_ad.cuh",
    "native/channel_native/kernels/em_debug.cu",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase6a_evidence_is_preserved_after_later_pin() -> None:
    inventory = _json(AUDIT / "phase13-current-native-owner-inventory.json")
    migration = _json(AUDIT / "phase13-migration-delta.json")

    inventory_phase6a = inventory["phase6a_shared_rf_and_layer_stack"]  # type: ignore[index]
    migration_phase6a = migration["phase6a_current"]  # type: ignore[index]
    assert (
        inventory_phase6a["rayd_commit"]
        == migration_phase6a["rayd_commit"]
        == RAYD_COMMIT
    )
    assert (
        inventory_phase6a["integration_header_sha256"]
        == migration_phase6a["integration_header_sha256"]
        == INTEGRATION_V2_SHA256
    )
    assert migration_phase6a["owner_counts"] == {
        "RayD": 20,
        "layered": 2,
        "Channel Native": 180,
    }
    assert len(inventory_phase6a["binding_manifest_sha256"]) == 64
    assert (
        inventory_phase6a["binding_manifest_sha256"]
        == migration_phase6a["binding_manifest_sha256"]
    )

    owners = {
        record["symbol"]: record["numerical_owner"]
        for record in inventory["symbols"]  # type: ignore[index]
    }
    assert {symbol for symbol, owner in owners.items() if owner == "RayD"} >= LAYER_STACK_SYMBOLS
    assert all(owners[symbol] == "RayD" for symbol in LAYER_STACK_SYMBOLS)


def test_phase6a_helper_partition_is_112_10_7() -> None:
    decision = _json(AUDIT / "phase13-shared-rf-helper-ownership-decision.json")
    ledger = _json(AUDIT / "phase13-shared-rf-helper-ledger.json")
    groups = decision["groups"]  # type: ignore[index]

    active_rayd = sum(
        len(group["helpers"])
        for group in groups
        if group["current_owner"] == "RayD"
    )
    channel_boundary = sum(
        len(group["helpers"])
        for group in groups
        if group["accepted_target_owner"] == "Channel Native"
    )
    channel_pending_adr026 = sum(
        len(group["helpers"])
        for group in groups
        if group["accepted_target_owner"] == "RayD after ADR-026"
    )

    assert (active_rayd, channel_boundary, channel_pending_adr026) == (112, 10, 7)
    assert ledger["phase6a_activation"]["ownership_projection"] == {  # type: ignore[index]
        "RayD ADR-024 active": 112,
        "Channel boundary retained": 10,
        "Channel pending ADR-026": 7,
    }


def test_phase6a_dependency_graph_has_no_deleted_channel_rf_owner() -> None:
    graph = _json(AUDIT / "phase13-shared-rf-dependency-graph.json")
    node_ids = {node["id"] for node in graph["nodes"]}  # type: ignore[index]
    edges = graph["edges"]  # type: ignore[index]

    assert not (REMOVED_CHANNEL_SOURCES & node_ids)
    assert all(
        not (
            edge["from"].startswith("RayD:")
            and edge["to"].startswith("native/channel_native/")
        )
        for edge in edges
    )
    assert all(not (ROOT / source).exists() for source in REMOVED_CHANNEL_SOURCES)


def test_phase6a_transmission_activation_history_is_preserved() -> None:
    contracts = _json(AUDIT / "phase13-transmission-contracts.json")
    by_symbol = {
        record["symbol"]: record
        for record in contracts["contracts"]  # type: ignore[index]
    }

    assert all(
        by_symbol[symbol]["current_numerical_owner"] == "RayD"
        and by_symbol[symbol]["activation_phase"] == "6A"
        for symbol in LAYER_STACK_SYMBOLS
    )
    pending = set(by_symbol) - LAYER_STACK_SYMBOLS
    assert len(pending) == 3
    assert (
        contracts["phase6a_activation"]["pending_phase6b_contract_count"]  # type: ignore[index]
        == 3
    )
    assert all(
        by_symbol[symbol]["current_numerical_owner"] == "RayD"
        and by_symbol[symbol]["activation_phase"] == "6B"
        for symbol in pending
    )
