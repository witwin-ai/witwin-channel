from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from tools.refactor_baseline import cpp_body_hashes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = REPOSITORY_ROOT / "native/channel_native/kernels"
INVENTORY_PATH = (
    REPOSITORY_ROOT / "docs/dev/audit/phase9-native-owner-inventory.json"
)
MIGRATION_DELTA_PATH = (
    REPOSITORY_ROOT / "docs/dev/audit/phase13-migration-delta.json"
)
RAYD_ROOT = Path(
    os.environ.get("RAYD_SOURCE_DIR", REPOSITORY_ROOT.parent.parent / "RayDi")
)
RAYD_WEDGE_SOURCE = (
    RAYD_ROOT / "backends/torch/src/torch_ext/rf/diffraction_wedge.cu"
)
FIELDS_BINDING = REPOSITORY_ROOT / "native/channel_native/binding/fields.cpp"

TRANSLATION_UNITS = {
    "coupled": KERNEL_ROOT / "field_wedge_ad_coupled.cu",
    "project": KERNEL_ROOT / "field_wedge_ad_project.cu",
    "prepare": KERNEL_ROOT / "field_wedge_ad_prepare.cu",
}
ABI_BY_OWNER = {
    "coupled": {
        "cn_field_coupled_rd_backward",
        "cn_field_coupled_rd_jvp",
    },
    "project": {
        "cn_field_project_complex3_backward",
        "cn_field_project_complex3_jvp",
    },
    "prepare": {
        "cn_coupled_rd_prepare_backward",
        "cn_coupled_rd_prepare_jvp",
    },
}
PURE_WEDGE_ABI = {
    "cn_field_diffraction_wedge",
    "cn_field_diffraction_wedge_backward",
    "cn_field_diffraction_wedge_jvp",
}
COMMON_HELPERS = {
    "load3f",
    "to_c10",
    "from_c10",
    "launch_blocks",
    "zero_scalar",
    "optional_tensor_arg",
    "opt_ptr",
    "opt_mut_ptr",
}


def _function_names_by_path() -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for entry in cpp_body_hashes(REPOSITORY_ROOT):
        names.setdefault(entry["path"], set()).add(entry["name"])
    return names


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*\([^;]*?\)\s*\{{", source, re.DOTALL)
    assert match is not None, f"missing function body: {name}"
    start = source.index("{", match.start())
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function body: {name}")


def _dict_keys(body: str) -> set[str]:
    return set(re.findall(r'out\["([^"]+)"\]', body))


def test_field_wedge_abi_has_one_semantic_translation_unit_owner() -> None:
    names = _function_names_by_path()
    all_abi = set().union(*ABI_BY_OWNER.values())

    for owner, path in TRANSLATION_UNITS.items():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        assert ABI_BY_OWNER[owner] <= names[relative]
        assert not (all_abi - ABI_BY_OWNER[owner]) & names[relative]

    binding = FIELDS_BINDING.relative_to(REPOSITORY_ROOT).as_posix()
    assert PURE_WEDGE_ABI <= names[binding]


def test_field_wedge_split_preserves_launch_and_sync_multisets() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    migration_delta = json.loads(MIGRATION_DELTA_PATH.read_text(encoding="utf-8"))
    source_evidence = next(
        entry
        for entry in inventory["source_evidence"]
        if entry["path"] == "native/channel_native/kernels/field_wedge_ad.cu"
    )
    approved_additions = migration_delta["phase3_current"][
        "approved_post_phase9_field_wedge_launch_additions"
    ]
    source_by_owner = {
        owner: path.read_text(encoding="utf-8-sig")
        for owner, path in TRANSLATION_UNITS.items()
    }
    sources = "\n".join(source_by_owner.values())
    actual_launches = Counter(
        re.findall(
            r"\b([A-Za-z_]\w*_kernel)(?:\s*<[^;{}]*?>)?\s*<<<",
            sources,
        )
    )
    expected_launches = Counter(
        site["kernel"]
        for site in source_evidence["kernel_launch_sites"]
        if not site["kernel"].startswith("diffraction_wedge_")
    )
    expected_launches.update(
        {
            addition["kernel"]: addition["launch_count"]
            for addition in approved_additions
        }
    )

    assert actual_launches == expected_launches
    assert sources.count("cudaStreamSynchronize(") == 0
    expected_launch_counts_by_owner = {
        "coupled": 2,
        "project": 2,
        "prepare": 2,
    }
    for addition in approved_additions:
        expected_launch_counts_by_owner[addition["split_owner"]] += addition[
            "launch_count"
        ]
    assert {
        owner: source.count("<<<") for owner, source in source_by_owner.items()
    } == expected_launch_counts_by_owner


def test_rayd_pure_wedge_preserves_launch_and_fast_math_boundary() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    source_evidence = next(
        entry
        for entry in inventory["source_evidence"]
        if entry["path"] == "native/channel_native/kernels/field_wedge_ad.cu"
    )
    source = RAYD_WEDGE_SOURCE.read_text(encoding="utf-8-sig")
    actual = Counter(
        re.findall(
            r"\b([A-Za-z_]\w*_kernel)(?:\s*<[^;{}]*?>)?\s*<<<",
            source,
        )
    )
    expected = Counter(
        site["kernel"]
        for site in source_evidence["kernel_launch_sites"]
        if site["kernel"].startswith("diffraction_wedge_")
    )
    assert actual == expected
    assert sum(actual.values()) == 3
    assert source.count("cudaStreamSynchronize(") == 0

    cmake = (RAYD_ROOT / "backends/torch/CMakeLists.txt").read_text(
        encoding="utf-8-sig"
    )
    relative = RAYD_WEDGE_SOURCE.relative_to(
        RAYD_ROOT / "backends/torch"
    ).as_posix()
    assert cmake.count(relative) == 2
    source_property = re.compile(
        rf"set_source_files_properties\(\s*{re.escape(relative)}\s+"
        r"PROPERTIES\s+COMPILE_OPTIONS\s+"
        r'"\$<\$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>"\)',
        re.DOTALL,
    )
    assert source_property.search(cmake)


def test_pure_wedge_typed_adapter_preserves_channel_schemas() -> None:
    source = FIELDS_BINDING.read_text(encoding="utf-8-sig")
    assert source.count("#include <rayd/torch/integration.h>") == 1
    for entry in (
        "field_diffraction_wedge",
        "field_diffraction_wedge_backward",
        "field_diffraction_wedge_jvp",
    ):
        assert source.count(f"rayd::torch::{entry}(") == 1
    assert "<<<" not in source

    assert _dict_keys(_function_body(source, "diffraction_wedge_result_dict")) == {
        "field_vector",
        "direction",
    }
    assert _dict_keys(
        _function_body(source, "cn_field_diffraction_wedge_backward")
    ) == {
        "grad_source",
        "grad_target",
        "grad_face0_eps_r",
        "grad_face0_sigma_e",
        "grad_face0_gain",
        "grad_face1_eps_r",
        "grad_face1_sigma_e",
        "grad_face1_gain",
        "grad_frequency",
        "grad_vertex_v0",
        "grad_vertex_v1",
        "grad_vertex_opp0",
        "grad_vertex_opp1",
    }
    assert _dict_keys(
        _function_body(source, "diffraction_wedge_jvp_result_dict")
    ) == {"tangent_field_vector", "tangent_direction"}

    backward = _function_body(source, "cn_field_diffraction_wedge_backward")
    for flag in (
        "need_grad_material",
        "need_grad_frequency",
        "need_grad_geometry",
        "need_grad_vertices",
    ):
        assert f"request.{flag} = {flag};" in backward
    request = _function_body(source, "diffraction_wedge_request")
    for tensor in (
        "valid",
        "source",
        "target",
        "edge_position",
        "edge_direction",
        "edge_t_min",
        "edge_t_max",
        "edge_n0",
        "edge_n1",
        "exterior_angle",
        "face0_valid",
        "face0_eps_r",
        "face0_sigma_e",
        "face0_mu_r",
        "face0_gain",
        "face1_valid",
        "face1_eps_r",
        "face1_sigma_e",
        "face1_mu_r",
        "face1_gain",
        "tx_power",
    ):
        assert f"request.{tensor} = std::move({tensor});" in request
    for optional in (
        "vertex_v0",
        "vertex_v1",
        "vertex_opp0",
        "vertex_opp1",
        "edge_boundary",
    ):
        assert f"request.{optional} = optional_tensor({optional});" in request
    assert "request.frequency_hz = frequency_hz;" in request
    assert (
        "request.isb_boundary_taper_width = isb_boundary_taper_width;" in request
    )

    for cotangent in ("grad_field_vector", "grad_direction"):
        assert (
            f"request.{cotangent} = optional_tensor({cotangent});" in backward
        )

    jvp = _function_body(source, "cn_field_diffraction_wedge_jvp")
    for tangent in (
        "tangent_source",
        "tangent_target",
        "tangent_face0_eps_r",
        "tangent_face0_sigma_e",
        "tangent_face0_gain",
        "tangent_face1_eps_r",
        "tangent_face1_sigma_e",
        "tangent_face1_gain",
        "tangent_vertex_v0",
        "tangent_vertex_v1",
        "tangent_vertex_opp0",
        "tangent_vertex_opp1",
    ):
        assert f"request.{tangent} = optional_tensor({tangent});" in jvp
    assert "request.tangent_frequency = tangent_frequency;" in jvp


def test_field_wedge_common_and_owner_local_plumbing_are_isolated() -> None:
    names = _function_names_by_path()
    common = "native/channel_native/kernels/field_wedge_ad_common.cuh"

    assert COMMON_HELPERS == names[common]
    for owner, path in TRANSLATION_UNITS.items():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        assert not COMMON_HELPERS & names[relative]
        assert source.count('#include "field_wedge_ad_common.cuh"') == 1
        assert "#define WEDGE_" not in source
        if owner != "coupled":
            assert "#define COUPLED_" not in source
            assert "check_coupled_primal_rows" not in source


def test_field_wedge_split_is_registered_once_and_below_budget() -> None:
    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    policy = inventory["translation_unit_policy"]
    common = "native/channel_native/kernels/field_wedge_ad_common.cuh"

    assert "native/channel_native/kernels/field_wedge_ad.cu" not in cmake
    assert "native/channel_native/kernels/field_wedge_ad_diffraction.cu" not in cmake
    assert "CHANNEL_NATIVE_FAST_MATH_WEDGE_TU" not in cmake
    assert "--use_fast_math" not in cmake
    assert not (KERNEL_ROOT / "field_wedge_ad_diffraction.cu").exists()
    assert "native/channel_native/kernels/field_wedge_ad.cu" not in policy[
        "planned_owner_debt"
    ]
    assert common not in cmake
    for path in TRANSLATION_UNITS.values():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        assert cmake.count(relative) == 1
        assert len(path.read_text(encoding="utf-8-sig").splitlines()) < policy[
            "recommended_limit_lines"
        ]
