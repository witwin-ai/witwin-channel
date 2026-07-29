# Copyright Xingyu Chen.
# Tests native owner inventory.

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
    current_hashes = cpp_body_hashes(
        REPOSITORY_ROOT, adr033_predecessor_identity=True
    )
    actual = Counter(_hash_tuple(entry) for entry in current_hashes)
    compact_count_helper_transformations = {
        "cn_path_filter_los_cuda",
        "cn_deterministic_los_topology_block",
        "cn_deterministic_reflection_order1_compact",
        "cn_deterministic_reflection_sequence_compact",
        "cn_deterministic_diffraction_order1_compact",
        "cn_path_filter_block_cuda",
        "cn_path_diffraction_block_cuda",
    }
    compact_count_helper_before = Counter(
        _hash_tuple(entry)
        for owner in inventory["owners"]
        for entry in owner["cpp_body_hash_multiset"]
        if entry["name"] in compact_count_helper_transformations
    )
    compact_count_helper_after_entries = [
        entry
        for entry in current_hashes
        if entry["name"] in compact_count_helper_transformations
    ]
    compact_count_helper_after = Counter(
        _hash_tuple(entry) for entry in compact_count_helper_after_entries
    )
    transfers = migration["phase3_current"][
        "approved_phase9_body_hash_transfer_multiset"
    ]
    phase4 = migration["phase4_current"]
    phase6a = migration["phase6a_current"]
    phase6b = migration["phase6b_current"]
    phase8a = migration["phase8a_current"]
    phase10a = migration["phase10a_current"]
    phase10b = migration["phase10b_current"]
    phase11b = migration["phase11b_current"]
    shared_math = migration["shared_math_current"]
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
                + shared_math["approved_phase9_body_hash_deletions"]
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
    phase11b_transformations = phase11b[
        "approved_phase9_body_hash_transformations"
    ]
    phase11b_before = Counter(
        _hash_tuple(transformation["before"])
        for transformation in phase11b_transformations
    )
    phase11b_after = Counter(
        _hash_tuple(transformation["after"])
        for transformation in phase11b_transformations
    )
    phase11b_names = {
        transformation["before"]["name"]
        for transformation in phase11b_transformations
    }
    shared_math_transformations = shared_math[
        "approved_phase9_body_hash_transformations"
    ]
    shared_math_before = Counter(
        _hash_tuple(transformation["before"])
        for transformation in shared_math_transformations
    )
    shared_math_after = Counter(
        _hash_tuple(transformation["after"])
        for transformation in shared_math_transformations
    )
    shared_math_names = {
        transformation["before"]["name"]
        for transformation in shared_math_transformations
    }
    actual_shared_math = Counter(
        _hash_tuple(entry)
        for entry in current_hashes
        if entry["name"] in shared_math_names
    )
    # ADR-043 gave the two Channel-owned field transports an optional
    # arrival-direction cotangent input and a direction tangent output. Eight
    # frozen bodies moved and nothing else did, so they are recorded here by
    # before/after hash rather than silently absorbed.
    adr043_transformations = migration["adr043_current"][
        "approved_phase9_body_hash_transformations"
    ]
    adr043_names = {
        transformation["before"]["name"]
        for transformation in adr043_transformations
    }
    adr043_before = Counter(
        _hash_tuple(transformation["before"])
        for transformation in adr043_transformations
    )
    adr043_after = Counter(
        _hash_tuple(transformation["after"])
        for transformation in adr043_transformations
    )
    actual_adr043 = Counter(
        _hash_tuple(entry)
        for entry in current_hashes
        if entry["name"] in adr043_names
    )
    actual_phase11b = Counter(
        _hash_tuple(entry)
        for entry in current_hashes
        if entry["name"] in phase11b_names
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

    # This is an immutable Phase-9 body-hash test.  Later phases advance the
    # live cursor, so validate the recorded transformations rather than pinning
    # the repository-wide current phase to the historical cut.
    assert migration["phase11b_current"]["status"] == (
        "historical Phase 11B implementation snapshot; frozen duplication "
        "acceptance met and later release closure completed"
    )
    assert current_inventory["phase10b_scattering_chains"]["rayd_commit"] == (
        "768b96e42a95f70c32d55f98a72000085317e288"
    )
    assert len(live_transfers) == len(transferred_names)
    assert expected - actual == (
        approved_before
        + approved_deletions
        + phase11b_before
        + compact_count_helper_before
        + adr043_before
        + shared_math_before
    )
    assert actual_shared_math == shared_math_after
    assert shared_math["owner"] == "native/channel/kernels/math.cuh"
    assert len(shared_math_transformations) == len(shared_math_names) == 1
    for transformation in shared_math_transformations:
        assert set(transformation) == {
            "owner_before",
            "owner_after",
            "transformation_kind",
            "before",
            "after",
        }
        assert set(transformation["before"]) == set(HASH_FIELDS)
        assert set(transformation["after"]) == set(HASH_FIELDS)
        assert transformation["before"]["name"] == transformation["after"]["name"]
        assert (
            transformation["before"]["signature_sha256"]
            == transformation["after"]["signature_sha256"]
        )
        assert transformation["owner_before"] == transformation["owner_after"] == (
            "Channel Native"
        )
    assert actual_adr043 == adr043_after
    assert len(adr043_transformations) == len(adr043_names) == 8
    for transformation in adr043_transformations:
        assert transformation["owner_before"] == "Channel Native"
        assert transformation["owner_after"] == "Channel Native"
        assert set(transformation["before"]) == set(HASH_FIELDS)
        assert set(transformation["after"]) == set(HASH_FIELDS)
        assert transformation["before"]["name"] == transformation["after"]["name"]
        # Every one of them really did change body, and only the four that
        # gained a parameter changed signature.
        assert (
            transformation["before"]["body_sha256"]
            != transformation["after"]["body_sha256"]
        )
        assert transformation["signature_changed"] == (
            transformation["before"]["signature_sha256"]
            != transformation["after"]["signature_sha256"]
        )
        assert phase9_owners[transformation["owner_id"]][
            _hash_tuple(transformation["before"])
        ] == 1
    assert (
        sum(
            transformation["signature_changed"]
            for transformation in adr043_transformations
        )
        == 6
    )
    assert {
        entry["name"] for entry in compact_count_helper_after_entries
    } == compact_count_helper_transformations
    assert len(compact_count_helper_after) == len(
        compact_count_helper_transformations
    )
    frozen_compact_signatures = {
        entry["name"]: entry["signature_sha256"]
        for owner in inventory["owners"]
        for entry in owner["cpp_body_hash_multiset"]
        if entry["name"] in compact_count_helper_transformations
    }
    assert {
        entry["name"]: entry["signature_sha256"]
        for entry in compact_count_helper_after_entries
    } == frozen_compact_signatures
    assert actual_transferred == approved_after
    assert actual_phase11b == phase11b_after
    assert len(phase11b_transformations) == len(phase11b_names) == 2
    for transformation in phase11b_transformations:
        assert set(transformation) == {
            "owner_id",
            "binding_symbol",
            "owner_before",
            "owner_after",
            "transformation_kind",
            "before",
            "after",
        }
        assert set(transformation["before"]) == set(HASH_FIELDS)
        assert set(transformation["after"]) == set(HASH_FIELDS)
        assert transformation["before"]["name"] == transformation["after"]["name"]
        assert (
            transformation["before"]["signature_sha256"]
            == transformation["after"]["signature_sha256"]
        )
        assert transformation["owner_before"] == transformation["owner_after"] == "Channel Native"
        assert phase9_owners[transformation["owner_id"]][
            _hash_tuple(transformation["before"])
        ] == 1
        assert current_owners[transformation["binding_symbol"]] == "Channel Native"
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