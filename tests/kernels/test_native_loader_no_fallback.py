import inspect
from pathlib import Path

from witwin.channel_native.core.kernels import extension, raydn_backend
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.core.runtime import raydn as raydn_runtime
from witwin.channel_native.montecarlo.basic import solver as mc_basic_solver


def test_channel_native_extension_loader_has_no_artifact_fallback():
    source = inspect.getsource(extension.native_extension)

    assert "artifacts" not in source
    assert "sys.path" not in source
    assert "ModuleNotFoundError" not in source
    assert "return None" not in source


def test_raydn_extension_loader_has_no_artifact_fallback():
    source = inspect.getsource(raydn_backend.native_extension)

    assert "artifacts" not in source
    assert "rglob" not in source
    assert "spec_from_file_location" not in source
    assert "sys.path" not in source
    assert "ModuleNotFoundError" not in source
    assert "import_module" not in source
    assert "return None" in source


def test_bdpt_visibility_uses_channel_native_bridge():
    source = inspect.getsource(ops.bdpt_visibility_forward)

    assert "_required_raydn_op" not in source
    assert "torch.ops" not in source
    assert "_required_native_op" in source
    assert "bdpt_visibility_forward" in source


def test_bdpt_intersection_uses_channel_native_bridge():
    source = inspect.getsource(ops.bdpt_intersect_forward)

    assert "_required_raydn_op" not in source
    assert "torch.ops" not in source
    assert "_required_native_op" in source
    assert "bdpt_intersect_forward" in source
    assert "return 0" in inspect.getsource(ops._raydn_module_handle)


def test_bdpt_reflection_accumulation_uses_channel_native_bridge():
    source = inspect.getsource(ops.bdpt_reflection_accumulation_forward)

    assert "_required_raydn_op" not in source
    assert "torch.ops" not in source
    assert "_required_native_op" in source
    assert "bdpt_reflection_accumulation_forward" in source


def test_bdpt_diffraction_uses_channel_native_bridge():
    for fn in (
        ops.bdpt_diffraction_discover_edges,
        ops.bdpt_diffraction_discover_edges_counted,
        ops.bdpt_diffraction_accumulation_forward,
    ):
        source = inspect.getsource(fn)
        assert "_required_raydn_op" not in source
        assert "torch.ops" not in source
        assert "_required_native_op" in source


def test_raydn_path_exports_use_channel_native_bridge():
    for fn in (
        ops.raydn_reflection_epc_paths_forward,
        ops.raydn_diffraction_paths_order1_forward,
    ):
        source = inspect.getsource(fn)
        assert "_required_raydn_op" not in source
        assert "torch.ops" not in source
        assert "_required_native_op" in source


def test_simplified_coherent_diffraction_grid_api_is_not_public():
    assert not hasattr(ops, "deterministic_diffraction_coherent_accumulation_forward")
    assert not hasattr(
        extension.native_extension(),
        "deterministic_diffraction_coherent_accumulation_forward",
    )


def test_raydn_scene_builder_uses_channel_native_scene_bridge():
    source = inspect.getsource(raydn_runtime.build_scene_from_structures)
    edge_source = inspect.getsource(raydn_runtime.RayDNScene.edge_records)

    assert "torch.classes" not in source
    assert "raydn_backend" not in source
    assert "raydn_scene_create" in source
    assert "raydn_scene_edge_records" in edge_source


def test_production_sources_have_no_raydn_dispatch_or_loader_fallbacks():
    repo = Path(__file__).resolve().parents[2]
    roots = (
        repo / "src" / "witwin" / "channel_native",
        repo / "native" / "channel_native",
    )
    forbidden = (
        "torch.ops.raydn",
        "_required_raydn_op",
        "allow_python_fallback",
        "python_fallback",
        "LoadLibrary",
        "dlopen",
        "sys.path",
        "spec_from_file_location",
        "rglob",
    )

    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".cpp", ".cu", ".h", ".hpp"}:
                continue
            source = path.read_text()
            for token in forbidden:
                if token in source:
                    offenders.append(f"{path.relative_to(repo)}: {token}")
    assert offenders == []


def test_production_sources_have_no_legacy_fallback_state_terms():
    repo = Path(__file__).resolve().parents[2]
    roots = (
        repo / "src" / "witwin" / "channel_native",
        repo / "native" / "channel_native",
    )
    forbidden = (
        "fallback",
        "unsupported",
        "fusion_debt",
        "require_reflection",
        "require_diffraction",
        "torch_cuda",
    )

    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".cpp", ".cu", ".h", ".hpp"}:
                continue
            source = path.read_text()
            for token in forbidden:
                if token in source:
                    offenders.append(f"{path.relative_to(repo)}: {token}")
    assert offenders == []


def test_native_kernels_do_not_use_aten_compute_or_cpu_tensor_readback():
    repo = Path(__file__).resolve().parents[2]
    root = repo / "native" / "channel_native" / "kernels"
    forbidden = ("at::zeros", ".cpu()")

    offenders: list[str] = []
    for path in root.rglob("*"):
        if path.suffix not in {".cu", ".cpp"}:
            continue
        source = path.read_text()
        for token in forbidden:
            if token in source:
                offenders.append(f"{path.relative_to(repo)}: {token}")
    assert offenders == []


def test_mc_bdpt_hot_paths_do_not_make_python_layout_copies():
    repo = Path(__file__).resolve().parents[2]
    roots = (
        repo / "src" / "witwin" / "channel_native" / "montecarlo" / "basic",
        repo / "src" / "witwin" / "channel_native" / "montecarlo" / "bdpt",
    )
    extra_files = (
        repo / "src" / "witwin" / "channel_native" / "core" / "kernels" / "ops.py",
    )
    forbidden = (".contiguous(", ".reshape(")

    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*.py"):
            source = path.read_text()
            for token in forbidden:
                if token in source:
                    offenders.append(f"{path.relative_to(repo)}: {token}")
    for path in extra_files:
        source = path.read_text()
        for token in forbidden:
            if token in source:
                offenders.append(f"{path.relative_to(repo)}: {token}")
    assert offenders == []


def test_path_solver_uses_the_typed_enumerated_engine():
    repo = Path(__file__).resolve().parents[2]
    source = (
        repo / "src" / "witwin" / "channel_native" / "path" / "solver.py"
    ).read_text()

    assert "propagation.enumerated.engine import" in source
    assert "evaluate_enumerated_paths" in source
    assert "core.path_topology import export_topology" not in source
    assert "reflection_paths_order1" not in source
    assert "diffraction_paths_order1" not in source


def test_mc_bdpt_hot_paths_do_not_make_python_empty_wedge_sentinels():
    repo = Path(__file__).resolve().parents[2]
    targets = (
        repo
        / "src"
        / "witwin"
        / "channel_native"
        / "montecarlo"
        / "basic"
        / "backend.py",
        repo
        / "src"
        / "witwin"
        / "channel_native"
        / "montecarlo"
        / "basic"
        / "raydn_components.py",
        repo
        / "src"
        / "witwin"
        / "channel_native"
        / "montecarlo"
        / "bdpt"
        / "subpaths.py",
    )

    offenders: list[str] = []
    for path in targets:
        source = path.read_text()
        for token in ("_empty_wedge_events", "torch.empty((0,"):
            if token in source:
                offenders.append(f"{path.relative_to(repo)}: {token}")
    assert offenders == []


def test_mc_basic_solver_uses_native_scene_and_store_material_paths():
    solve_source = inspect.getsource(mc_basic_solver.solve)
    module_source = inspect.getsource(mc_basic_solver)

    # Plan 07 AD-3: materials come from the compiled store in BOTH
    # ad_mode="none" and the AD modes (one source, same values); the old
    # host-float flattening cannot carry a gradient and is gone.
    assert "scene.raydn_scene()" in solve_source
    assert "_host_material_tensors" not in module_source
    assert "bdpt_face_material_tensors_from_host" not in module_source
    assert "face_material_field_bundle" in module_source


def test_bdpt_solver_does_not_use_derived_variance_or_component_map_path_export():
    from witwin.channel_native.montecarlo.bdpt import solver as bdpt_solver

    source = inspect.getsource(bdpt_solver)

    assert "bdpt_variance_estimate" not in source
    assert "bdpt_export_component_paths" not in source
    assert "bdpt_export_paths(" not in source
    assert "reflection_component_maps_with_wedges" not in source
    assert "diffraction_paths_order1" not in source
    assert "bdpt_sample_path_block" not in source


def test_bdpt_package_does_not_reintroduce_python_los_visibility_helpers():
    repo = Path(__file__).resolve().parents[2]
    root = repo / "src" / "witwin" / "channel_native" / "montecarlo" / "bdpt"
    forbidden = (
        "direct_los_path_gain",
        "visible_los_path_gain",
        "direct_los_component_map",
        "bdpt_los_visibility_inputs",
        "bdpt_apply_los_visibility",
    )

    offenders: list[str] = []
    for path in root.rglob("*.py"):
        source = path.read_text()
        for token in forbidden:
            if path == root / "kernels" / "maps.py" and token in {
                "bdpt_los_visibility_inputs",
                "bdpt_apply_los_visibility",
            }:
                continue
            if token in source:
                offenders.append(f"{path.relative_to(repo)}: {token}")
    assert offenders == []


def test_rayd_bridge_is_source_linked_without_dso_lookup():
    repo = Path(__file__).resolve().parents[2]
    bridge_source = (
        repo / "native" / "channel_native" / "raydn_bridge.cpp"
    ).read_text()
    ops_source = inspect.getsource(ops._raydn_module_handle)

    assert "LoadLibrary" not in bridge_source
    assert "dlopen" not in bridge_source
    assert "GetProcAddress" not in bridge_source
    assert "dlsym" not in bridge_source
    assert "__file__" not in ops_source
    assert "return 0" in ops_source
    assert "rayd_torch_native_scene_create" in bridge_source
