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
        "cn_rayd_scene_create",
        "cn_rayd_scene_edge_records",
    },
    "geometry.cpp": {
        "cn_rayd_intersect_forward",
        "cn_rayd_visibility_forward",
        "cn_rayd_intersect_backward",
        "cn_rayd_intersect_jvp",
        "cn_coupled_rd_geometry_forward",
        "cn_coupled_dd_geometry_forward",
    },
    "reflection.cpp": {
        "cn_rayd_trace_reflections_forward",
        "cn_rayd_reflection_epc_paths_forward",
        "cn_rayd_trace_reflections_forward_tape",
        "cn_rayd_trace_reflections_backward",
        "cn_rayd_trace_reflections_jvp",
        "cn_rayd_reflection_epc_paths_backward",
        "cn_rayd_reflection_epc_paths_jvp",
        "cn_rayd_scene_face_normals_backward",
        "cn_rayd_scene_face_normals_jvp",
    },
    "diffraction.cpp": {
        "cn_diffraction_tx_visible_state_plan",
        "cn_rayd_diffraction_paths_order1_forward",
        "cn_rayd_diffraction_sample_tape_forward",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cn_wrapper_definitions(source: str) -> set[str]:
    signature = re.compile(
        r"(?m)^(?:pybind11::(?:tuple|dict)|(?:at|torch)::Tensor|"
        r"std::shared_ptr<RayDSceneResource>)\s+"
        r"(cn_[A-Za-z0-9_]+)\s*\("
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
    bridge_root = _repo_root() / "native" / "channel_native" / "rayd"

    assert not (_repo_root() / "native" / "channel_native" / "rayd_bridge.cpp").exists()
    assert not (bridge_root / "bridge.h").exists()
    assert not (bridge_root / "common.cpp").exists()
    assert (bridge_root / "resource.h").is_file()
    assert tuple(path.name for path in sorted(bridge_root.glob("*.cpp"))) == tuple(
        sorted(RAYD_BRIDGE_SOURCES)
    )


def test_rayd_wrapper_definitions_are_unique_and_owned_by_responsibility():
    bridge_root = _repo_root() / "native" / "channel_native" / "rayd"
    expected_wrappers = set().union(*WRAPPERS_BY_SOURCE.values())
    owners: dict[str, list[str]] = {}

    for source_name in RAYD_BRIDGE_SOURCES:
        definitions = _cn_wrapper_definitions((bridge_root / source_name).read_text())
        assert definitions == WRAPPERS_BY_SOURCE.get(source_name, set())
        for definition in definitions:
            owners.setdefault(definition, []).append(source_name)

    assert len(expected_wrappers) == 20
    assert set(owners) == expected_wrappers
    assert all(len(source_names) == 1 for source_names in owners.values())


def test_cmake_builds_every_rayd_source_without_legacy_exception_boundary():
    cmake = (_repo_root() / "CMakeLists.txt").read_text()
    source_paths = tuple(
        f"native/channel_native/rayd/{source_name}"
        for source_name in RAYD_BRIDGE_SOURCES
    )

    extension_sources = re.search(
        r"Python_add_library\(\s*_channel_native\s+MODULE\s+WITH_SOABI(.*?)\n\)",
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
    source = (_repo_root() / "native/channel_native/rayd/diffraction.cpp").read_text(
        encoding="utf-8-sig"
    )

    assert source.count("cn_diffraction_tx_visible_state_plan(") == 1
    assert source.count("rayd::torch::AxialEdgeVisibilityRequest request{") == 1
    assert source.count("rayd::torch::axial_edge_visibility_forward(") == 1
    assert "kDiffractionStateCapacity = 4'194'304" in source
    body = source.split("cn_diffraction_tx_visible_state_plan(", 1)[1].split(
        "cn_rayd_diffraction_paths_order1_forward(", 1
    )[0]
    assert "state_src" not in body
    assert "cudaStreamSynchronize" not in body
    assert ".cpu()" not in body
