# Copyright Xingyu Chen.
# Tests field transport native boundaries.

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from tools.refactor_baseline import cpp_body_hashes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = REPOSITORY_ROOT / "native/channel/kernels"
INVENTORY_PATH = REPOSITORY_ROOT / "docs/dev/audit/phase9-native-owner-inventory.json"
RAYD_ROOT = Path(os.environ.get("RAYD_SOURCE_DIR", REPOSITORY_ROOT.parent / "RayD"))
RAYD_FIELD_TRANSPORT_AD = RAYD_ROOT / "src/field_transport_ad.cuh"
RAYD_TRANSMISSION_SOURCE = RAYD_ROOT / "src/transmission.cu"
FIELDS_BINDING = REPOSITORY_ROOT / "native/channel/binding/fields.cpp"

TRANSLATION_UNITS = {
    "merged": KERNEL_ROOT / "fields.cu",
}
ABI_BY_OWNER = {
    "merged": {
        "channel_field_free_space_fwd64",
        "channel_field_free_space_backward",
        "channel_field_free_space_jvp",
        "channel_field_reflection_sequence_backward",
        "channel_field_reflection_sequence_jvp",
    },
}
TRANSMISSION_ABI = {
    "channel_field_transmission_sequence",
    "channel_field_transmission_sequence_backward",
    "channel_field_transmission_sequence_jvp",
}
REMOVED_TRANSMISSION_TU = KERNEL_ROOT / "field_transport_transmission.cu"
COMMON_HELPERS = {
    "load3",
    "load3f",
    "load_sequence3f",
    "load_dual3f",
    "load_dual_sequence3f",
    "complex_of",
    "to_c10",
    "launch_blocks",
    "zero_filled",
    "optional_grad",
    "grad_ptr",
}
RAYD_NUMERICAL_HELPERS = {
    "fold_output_cotangents",
    "write_output_tangents",
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


def test_field_transport_abi_has_one_semantic_translation_unit_owner() -> None:
    names = _function_names_by_path()
    all_abi = set().union(*ABI_BY_OWNER.values())

    for owner, path in TRANSLATION_UNITS.items():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        assert ABI_BY_OWNER[owner] <= names[relative]
        assert not (all_abi - ABI_BY_OWNER[owner]) & names[relative]

    binding = FIELDS_BINDING.relative_to(REPOSITORY_ROOT).as_posix()
    assert TRANSMISSION_ABI <= names[binding]


def test_rayd_transmission_preserves_migrated_launch_budget() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    source_evidence = next(
        entry
        for entry in inventory["source_evidence"]
        if entry["path"] == "native/channel_native/kernels/field_transport_ad.cu"
    )
    ad_source = RAYD_TRANSMISSION_SOURCE.read_text(encoding="utf-8-sig")
    actual = Counter(
        kernel
        for kernel in re.findall(
            r"\b([A-Za-z_]\w*_kernel)(?:\s*<[^;{}]*?>)?\s*<<<",
            ad_source,
        )
        if kernel.startswith("transmission_sequence_") and kernel != "transmission_sequence_kernel"
    )
    expected = Counter(
        site["kernel"]
        for site in source_evidence["kernel_launch_sites"]
        if site["kernel"].startswith("transmission_sequence_")
    )
    assert actual == expected
    assert sum(actual.values()) == 2
    assert ad_source.count("cudaStreamSynchronize(") == 0

    primal_source = RAYD_TRANSMISSION_SOURCE.read_text(encoding="utf-8-sig")
    assert len(re.findall(r"\btransmission_sequence_kernel\s*<<<", primal_source)) == 1
    assert primal_source.count("cudaStreamSynchronize(") == 0


def test_transmission_sequence_typed_adapter_preserves_channel_schemas() -> None:
    source = FIELDS_BINDING.read_text(encoding="utf-8-sig")
    assert source.count("#include <rayd/integration.h>") == 1
    for entry in (
        "field_transmission_sequence",
        "field_transmission_sequence_backward",
        "field_transmission_sequence_jvp",
    ):
        assert source.count(f"rayd::torch::{entry}(") == 1
    assert "<<<" not in source
    request = _function_body(source, "transmission_sequence_request")
    assert request.index("std::move(path_valid)") < request.index("std::move(source)")
    for entry in TRANSMISSION_ABI:
        signature = re.search(rf"\b{entry}\s*\((?P<args>[^;]*?)\)\s*\{{", source, re.DOTALL)
        assert signature is not None
        assert signature.group("args").lstrip().startswith("torch::Tensor path_valid,")

    assert _dict_keys(_function_body(source, "transmission_sequence_result_dict")) == {
        "field_vector",
        "coefficient",
        "path_field",
        "path_gain",
        "path_length_m",
        "delay_s",
        "direction",
    }
    assert _dict_keys(_function_body(source, "channel_field_transmission_sequence_backward")) == {
        "grad_layer_thickness_m",
        "grad_layer_eps_r",
        "grad_layer_sigma_e",
        "grad_frequency",
        "grad_source",
        "grad_target",
        "grad_interaction_positions",
        "grad_interaction_normals",
    }
    assert _dict_keys(_function_body(source, "transmission_sequence_jvp_result_dict")) == {
        "field_vector",
        "coefficient",
        "path_field",
        "path_gain",
        "path_length_m",
        "delay_s",
    }
    backward = _function_body(source, "channel_field_transmission_sequence_backward")
    assert 'out["grad_interaction_positions"] = pybind11::none();' in backward


def test_field_transport_common_helpers_have_one_source() -> None:
    names = _function_names_by_path()
    common = "native/channel/kernels/field_ad.cuh"
    merged = KERNEL_ROOT / "fields.cu"

    assert COMMON_HELPERS == names[common]
    assert merged.read_text(encoding="utf-8-sig").count('#include "field_ad.cuh"') == 2


def test_output_chain_ad_helpers_are_defined_only_in_locked_rayd_header() -> None:
    common_source = (KERNEL_ROOT / "field_ad.cuh").read_text(encoding="utf-8-sig")
    rayd_source = RAYD_FIELD_TRANSPORT_AD.read_text(encoding="utf-8-sig")

    for helper in RAYD_NUMERICAL_HELPERS:
        definition = re.compile(rf"__device__\s+__forceinline__[^;{{}}]*\b{helper}\s*\(")
        assert len(definition.findall(rayd_source)) == 1
        assert not definition.search(common_source)
        assert common_source.count(f"using ad::{helper};") == 1


def test_field_transport_consolidation_is_registered_once() -> None:
    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    common = "native/channel/kernels/field_ad.cuh"
    merged = "native/channel/kernels/fields.cu"

    assert cmake.count(merged) == 1
    assert "native/channel/kernels/field_transport_free_space.cu" not in cmake
    assert "native/channel/kernels/field_transport_reflection.cu" not in cmake
    assert "native/channel/kernels/field_transport_transmission.cu" not in cmake
    assert not REMOVED_TRANSMISSION_TU.exists()
    assert common not in cmake

    rayd_cmake = (RAYD_ROOT / "torch/CMakeLists.txt").read_text(encoding="utf-8-sig")
    relative = RAYD_TRANSMISSION_SOURCE.relative_to(RAYD_ROOT).as_posix()
    assert rayd_cmake.count(f"${{RAYD_ROOT_DIR}}/{relative}") == 1
