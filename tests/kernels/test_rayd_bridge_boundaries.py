import re
from pathlib import Path


RAYD_BRIDGE_SOURCES = (
    "scene.cpp",
    "geometry.cpp",
    "reflection.cpp",
    "diffraction.cpp",
)

WRAPPERS_BY_SOURCE = {
    "scene.cpp": {
        "channel_rayd_scene_create",
        "channel_rayd_scene_edge_records",
    },
    "geometry.cpp": {
        "channel_rayd_intersect_forward",
        "channel_rayd_visibility_forward",
        "channel_rayd_intersect_backward",
        "channel_rayd_intersect_jvp",
        "channel_rayd_segment_penetration_forward",
        "channel_rayd_segment_penetration_forward_tape",
        "channel_rayd_segment_penetration_backward",
        "channel_rayd_segment_penetration_jvp",
        "channel_coupled_rd_geometry_forward",
        "channel_coupled_dd_geometry_forward",
    },
    "reflection.cpp": {
        "channel_rayd_trace_reflections_forward",
        "channel_rayd_reflection_epc_paths_forward",
        "channel_rayd_trace_reflections_forward_tape",
        "channel_rayd_trace_reflections_backward",
        "channel_rayd_trace_reflections_jvp",
        "channel_rayd_reflection_epc_paths_backward",
        "channel_rayd_reflection_epc_paths_jvp",
        "channel_rayd_scene_face_normals_backward",
        "channel_rayd_scene_face_normals_jvp",
    },
    "diffraction.cpp": {
        "channel_diffraction_tx_visible_state_plan",
        "channel_rayd_diffraction_paths_order1_forward",
        "channel_rayd_diffraction_sample_tape_forward",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _channel_wrapper_definitions(source: str) -> set[str]:
    signature = re.compile(
        r"(?m)^(?:pybind11::(?:tuple|dict)|(?:at|torch)::Tensor|"
        r"std::shared_ptr<RayDSceneResource>)\s+"
        r"(channel_[A-Za-z0-9_]+)\s*\("
    )
    definitions: set[str] = set()
    for match in signature.finditer(source):
        depth = 1
        cursor = match.end()
        while depth and cursor < len(source):
            depth += (source[cursor] == "(") - (source[cursor] == ")")
            cursor += 1
        if depth == 0 and source[cursor:].lstrip().startswith("{"):
            definitions.add(match.group(1))
    return definitions


def test_rayd_direct_integration_has_only_the_modular_sources():
    bridge_root = _repo_root() / "native" / "channel" / "rayd"

    assert not (_repo_root() / "native" / "channel" / "rayd_bridge.cpp").exists()
    assert not (bridge_root / "bridge.h").exists()
    assert not (bridge_root / "common.cpp").exists()
    assert (bridge_root / "resource.h").is_file()
    assert tuple(path.name for path in sorted(bridge_root.glob("*.cpp"))) == tuple(
        sorted(RAYD_BRIDGE_SOURCES)
    )


def test_rayd_wrapper_definitions_are_unique_and_owned_by_responsibility():
    bridge_root = _repo_root() / "native" / "channel" / "rayd"
    expected_wrappers = set().union(*WRAPPERS_BY_SOURCE.values())
    owners: dict[str, list[str]] = {}

    for source_name in RAYD_BRIDGE_SOURCES:
        definitions = _channel_wrapper_definitions((bridge_root / source_name).read_text())
        assert definitions == WRAPPERS_BY_SOURCE.get(source_name, set())
        for definition in definitions:
            owners.setdefault(definition, []).append(source_name)

    assert len(expected_wrappers) == 24
    assert set(owners) == expected_wrappers
    assert all(len(source_names) == 1 for source_names in owners.values())


def test_cmake_builds_every_rayd_source_without_legacy_exception_boundary():
    cmake = (_repo_root() / "CMakeLists.txt").read_text()
    source_paths = tuple(
        f"native/channel/rayd/{source_name}"
        for source_name in RAYD_BRIDGE_SOURCES
    )

    extension_sources = re.search(
        r"Python_add_library\(\s*_channel\s+MODULE\s+WITH_SOABI(.*?)\n\)",
        cmake,
        re.DOTALL,
    )
    assert extension_sources is not None
    for source_path in source_paths:
        assert extension_sources.group(1).count(source_path) == 1

    exception_source_groups = re.findall(
        r"set_source_files_properties\((.*?)"
        r"PROPERTIES\s+COMPILE_OPTIONS\s+\"/EHc-\"\s*\)",
        cmake,
        re.DOTALL,
    )
    assert exception_source_groups == []


def test_diffraction_visibility_plan_calls_the_typed_rayd_axial_operation() -> None:
    source = (_repo_root() / "native/channel/rayd/diffraction.cpp").read_text(
        encoding="utf-8-sig"
    )

    assert source.count("channel_diffraction_tx_visible_state_plan(") == 1
    assert source.count("rayd::torch::AxialEdgeVisibilityRequest request{") == 1
    assert source.count("rayd::torch::axial_edge_visibility_forward(") == 1
    assert "kDiffractionStateCapacity = 4'194'304" in source
    body = source.split("channel_diffraction_tx_visible_state_plan(", 1)[1].split(
        "channel_rayd_diffraction_paths_order1_forward(", 1
    )[0]
    assert "state_src" not in body
    assert "cudaStreamSynchronize" not in body
    assert ".cpu()" not in body


def test_segment_penetration_bridge_preserves_the_typed_api6_contract() -> None:
    root = _repo_root()
    resource = (root / "native/channel/rayd/resource.h").read_text(
        encoding="utf-8-sig"
    )
    geometry = (root / "native/channel/rayd/geometry.cpp").read_text(
        encoding="utf-8-sig"
    )
    binding = (root / "native/channel/binding/rayd.cpp").read_text(
        encoding="utf-8-sig"
    )

    assert "rayd::torch::kIntegrationApiVersion == 6u" in resource
    assert 'std::string_view{"rayd.torch.integration"}' in resource
    assert "switch (policy)" in geometry
    assert geometry.count("case 0:") == 1
    assert geometry.count("case 1:") == 1
    assert "(bit & (bit - 1u)) == 0u" in geometry

    typed_entries = (
        "segment_penetration_forward",
        "segment_penetration_forward_tape",
        "segment_penetration_backward",
        "segment_penetration_jvp",
    )
    for entry in typed_entries:
        assert geometry.count(f"rayd::torch::{entry}(") == 1
        assert binding.count(f'"rayd_{entry}"') == 1

    result_pack = geometry.split("pack_segment_penetration_result(", 1)[1].split(
        "pack_segment_penetration_tape(", 1
    )[0]
    result_fields = (
        "out.valid",
        "out.num_hits",
        "out.reached_target",
        "out.overflow",
        "out.distance",
        "out.direction",
        "out.t",
        "out.position",
        "out.normal",
        "out.geometric_normal",
        "out.global_primitive_id",
    )
    assert [result_pack.index(field) for field in result_fields] == sorted(
        result_pack.index(field) for field in result_fields
    )

    tape_pack = geometry.split("pack_segment_penetration_tape(", 1)[1].split(
        "}  // namespace", 1
    )[0]
    tape_fields = tuple(f"out.result.{field[4:]}" for field in result_fields) + (
        "out.tape_primitive_id",
        "out.tape_barycentric",
        "out.tape_restart_epsilon",
        "out.tape_restart_branch",
        "out.tape_restart_tie_mask",
        "out.tape_direction_denominator_branch",
    )
    assert [tape_pack.index(field) for field in tape_fields] == sorted(
        tape_pack.index(field) for field in tape_fields
    )

    for gradient in ("vertices", "origins", "targets"):
        assert f"tensor_or_none(out.grad_{gradient})" in geometry
    assert "capacity_failure_state" in geometry
    assert "input_active_any" in geometry
