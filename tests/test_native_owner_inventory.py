from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.refactor_baseline import cpp_body_hashes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    REPOSITORY_ROOT / "docs/dev/audit/phase9-native-owner-inventory.json"
)
CURRENT_OWNER_INVENTORY_PATH = (
    REPOSITORY_ROOT
    / "docs/dev/audit/phase13-current-native-owner-inventory.json"
)
MIGRATION_DELTA_PATH = (
    REPOSITORY_ROOT / "docs/dev/audit/phase13-migration-delta.json"
)
EXPECTED_OWNER_IDS = {
    "path.compaction",
    "path.topology",
    "path.core",
    "field_transport.free_space",
    "field_transport.reflection_sequence",
    "field_transport.rough_scale",
    "field_transport.transmission_sequence",
    "field_wedge.diffraction",
    "field_wedge.coupled_rd",
    "field_wedge.project_complex3",
    "field_wedge.coupled_prepare",
    "bdpt.mis",
    "bdpt.endpoint_connection",
    "bdpt.diffraction_connection",
    "bdpt.connection_storage",
    "legacy_slab.primal",
    "legacy_slab.dual",
    # ADR-010 native scattering / rough-reflection kernel owners.
    "scattering.ensemble_eval",
    "scattering.patch_integral",
}
HASH_FIELDS = (
    "name",
    "token_count",
    "signature_sha256",
    "body_sha256",
)


def _load_inventory() -> dict[str, object]:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def _hash_tuple(entry: dict[str, object]) -> tuple[object, ...]:
    return tuple(entry[field] for field in HASH_FIELDS)


def test_native_owner_inventory_digest_is_frozen() -> None:
    inventory = _load_inventory()
    expected = inventory.pop("manifest_sha256")
    payload = json.dumps(
        inventory,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert hashlib.sha256(payload).hexdigest() == expected


def test_native_owner_inventory_covers_the_planned_abi_fusion_owners() -> None:
    inventory = _load_inventory()
    owners = inventory["owners"]

    assert {owner["id"] for owner in owners} == EXPECTED_OWNER_IDS
    assert len(owners) == len(EXPECTED_OWNER_IDS)
    for owner in owners:
        assert owner["abi_owner"]
        assert owner["semantic_surface"]
        assert owner["tape_lifetime"]
        assert owner["fusion_boundary"]
        assert owner["non_split_reason"]
        assert owner["expiry_condition"]
        assert owner["lockstep_tests"]
        assert owner["cpp_body_hash_multiset"]


def test_path_compaction_abi_owner_is_complete_and_frozen() -> None:
    owners = {owner["id"]: owner for owner in _load_inventory()["owners"]}
    expected = {
        "cn_path_concat_vec3_cuda",
        "cn_path_los_visibility_inputs_cuda",
        "cn_path_filter_los_cuda",
        "cn_path_filter_block_cuda",
        "cn_path_diffraction_block_cuda",
        "cn_path_finalize_blocks_cuda",
    }
    frozen_names = {
        entry["name"]
        for owner in owners.values()
        for entry in owner["cpp_body_hash_multiset"]
    }

    assert set(owners["path.compaction"]["abi_owner"]) == expected
    assert expected <= frozen_names
    assert {
        "cn_deterministic_los_topology_block",
        "cn_deterministic_reflection_order1_compact",
        "cn_deterministic_reflection_sequence_compact",
        "cn_deterministic_diffraction_order1_compact",
    } <= {
        entry["name"]
        for entry in owners["path.topology"]["cpp_body_hash_multiset"]
    }


def test_bdpt_abi_owners_are_complete_and_frozen() -> None:
    owners = {owner["id"]: owner for owner in _load_inventory()["owners"]}
    expected_by_owner = {
        "bdpt.mis": {"cn_bdpt_mis_weights_cuda"},
        "bdpt.endpoint_connection": {
            "cn_bdpt_endpoint_connection_samples_cuda",
            "cn_bdpt_endpoint_connection_visibility_inputs_cuda",
        },
        "bdpt.diffraction_connection": {
            "cn_bdpt_diffraction_connection_samples_from_tape_cuda",
            "cn_bdpt_diffraction_point_connection_samples_cuda",
        },
        "bdpt.connection_storage": {
            "cn_bdpt_accumulate_connection_samples_cuda",
            "cn_bdpt_filter_connection_samples_cuda",
            "cn_bdpt_count_valid_connection_samples_cuda",
            "cn_bdpt_compact_connection_samples_cuda",
            "cn_bdpt_concat_connection_samples_cuda",
            "cn_bdpt_connection_variance_cuda",
        },
    }

    for owner_id, expected in expected_by_owner.items():
        owner = owners[owner_id]
        frozen_names = {
            entry["name"] for entry in owner["cpp_body_hash_multiset"]
        }
        assert set(owner["abi_owner"]) == expected
        assert expected <= frozen_names


def test_frozen_native_body_hash_multisets_still_exist_after_function_moves() -> None:
    inventory = _load_inventory()
    current_inventory = json.loads(
        CURRENT_OWNER_INVENTORY_PATH.read_text(encoding="utf-8")
    )
    migration = json.loads(MIGRATION_DELTA_PATH.read_text(encoding="utf-8"))
    expected = Counter(
        _hash_tuple(entry)
        for owner in inventory["owners"]
        for entry in owner["cpp_body_hash_multiset"]
    )
    current_hashes = cpp_body_hashes(REPOSITORY_ROOT)
    actual = Counter(_hash_tuple(entry) for entry in current_hashes)
    transfers = migration["phase3_current"][
        "approved_phase9_body_hash_transfer_multiset"
    ]
    phase4 = migration["phase4_current"]
    phase6a = migration["phase6a_current"]
    phase6b = migration["phase6b_current"]
    phase8a = migration["phase8a_current"]
    phase10a = migration["phase10a_current"]
    phase10b = migration["phase10b_current"]
    deleted_bindings = set(phase4["deleted_bindings"])
    approved_deletions = Counter(
        _hash_tuple(entry)
        for entry in (
            phase4["approved_phase9_body_hash_deletions"]
            + phase6a["approved_phase9_body_hash_deletions"]
                + phase6b["approved_phase9_body_hash_deletions"]
                + phase8a["approved_phase9_body_hash_deletions"]
                + phase10a["approved_phase9_body_hash_deletions"]
                + phase10b["approved_phase9_body_hash_deletions"]
        )
    )
    live_transfers = [
        transfer
        for transfer in transfers
        if transfer["binding_symbol"] not in deleted_bindings
    ]
    approved_before = Counter(
        _hash_tuple(transfer["before"]) for transfer in live_transfers
    )
    approved_after = Counter(
        _hash_tuple(transfer["after"]) for transfer in live_transfers
    )
    transferred_names = {
        transfer["before"]["name"] for transfer in live_transfers
    }
    actual_transferred = Counter(
        _hash_tuple(entry)
        for entry in current_hashes
        if entry["name"] in transferred_names
    )
    phase9_owners = {
        owner["id"]: Counter(
            _hash_tuple(entry) for entry in owner["cpp_body_hash_multiset"]
        )
        for owner in inventory["owners"]
    }
    current_owners = {
        entry["symbol"]: entry["numerical_owner"]
        for entry in current_inventory["symbols"]
    }

    assert migration["current_phase"] == 11
    assert migration["current_subphase"] == "11A"
    assert current_inventory["current_phase"] == 10
    assert current_inventory["current_subphase"] == "10B"
    assert len(live_transfers) == len(transferred_names)
    assert expected - actual == approved_before + approved_deletions
    assert actual_transferred == approved_after
    for transfer in transfers:
        assert set(transfer) == {
            "owner_id",
            "binding_symbol",
            "owner_before",
            "owner_after",
            "transfer_kind",
            "before",
            "after",
        }
        assert set(transfer["before"]) == set(HASH_FIELDS)
        assert set(transfer["after"]) == set(HASH_FIELDS)
        assert transfer["before"]["name"] == transfer["after"]["name"]
        assert transfer["owner_before"] == "Channel Native"
        if transfer["binding_symbol"] in deleted_bindings:
            assert _hash_tuple(transfer["before"]) in approved_deletions
        else:
            assert current_owners[transfer["binding_symbol"]] == transfer["owner_after"]
        assert phase9_owners[transfer["owner_id"]][
            _hash_tuple(transfer["before"])
        ] == 1
    for entry in (
        entry
        for owner in inventory["owners"]
        for entry in owner["cpp_body_hash_multiset"]
    ):
        assert set(entry) == set(HASH_FIELDS)
        assert isinstance(entry["token_count"], int)
        assert len(entry["signature_sha256"]) == 64
        assert len(entry["body_sha256"]) == 64


def test_source_launch_and_sync_snapshot_is_complete_and_specific() -> None:
    inventory = _load_inventory()
    evidence = {entry["path"]: entry for entry in inventory["source_evidence"]}

    assert {
        path: (
            entry["line_count"],
            len(entry["kernel_launch_sites"]),
            len(entry["explicit_sync_sites"]),
        )
        for path, entry in evidence.items()
    } == {
        "native/channel_native/kernels/path_trace.cu": (4270, 51, 11),
        "native/channel_native/kernels/field_transport_ad.cu": (2496, 9, 0),
        "native/channel_native/kernels/field_wedge_ad.cu": (2473, 9, 0),
        # ADR-019 coherent (2) + ADR-022 coherent/power AD (4) launch sites
        # extended the frozen bdpt_connect.cu ledger from 17 to 23.
        "native/channel_native/kernels/bdpt_connect.cu": (2356, 23, 2),
    }
    for entry in evidence.values():
        assert all(site["kernel"].endswith("_kernel") for site in entry["kernel_launch_sites"])
        assert all(site["call"].endswith("Synchronize") for site in entry["explicit_sync_sites"])


def test_legacy_slab_primal_and_dual_lockstep_sets_are_explicit() -> None:
    owners = {owner["id"]: owner for owner in _load_inventory()["owners"]}
    primal = {
        entry["name"] for entry in owners["legacy_slab.primal"]["cpp_body_hash_multiset"]
    }
    dual = {
        entry["name"] for entry in owners["legacy_slab.dual"]["cpp_body_hash_multiset"]
    }

    assert primal == {
        "legacy_add",
        "legacy_sub",
        "legacy_mul",
        "legacy_scale",
        "legacy_div",
        "legacy_div_floor",
        "legacy_sqrt",
        "legacy_interface_sqrt",
        "legacy_interface_fresnel",
        "legacy_exp_neg_2i",
        "legacy_sionna_slab_fresnel",
    }
    assert dual == {
        "dlc_const",
        "dlc_make",
        "dlc_add",
        "dlc_sub",
        "dlc_mul",
        "dlc_scale_dual",
        "dlc_div",
        "dlc_sqrt",
        "dlc_exp_neg_2i",
        "legacy_sionna_slab_fresnel_dual",
    }
