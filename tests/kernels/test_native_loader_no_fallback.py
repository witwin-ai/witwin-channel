import inspect
from pathlib import Path

import pytest

from witwin.channel.core.kernels import extension
from witwin.channel.montecarlo.basic.kernels import sampling as mc_sampling
from witwin.channel.propagation.geometry.kernels import bridge as ops
from witwin.channel.runtime import symbols
from witwin.channel.scene.kernels import rayd_scene


def test_channel_extension_loader_has_no_artifact_fallback():
    source = inspect.getsource(extension.native_extension)

    assert "artifacts" not in source
    assert "sys.path" not in source
    assert "ModuleNotFoundError" not in source
    assert "return None" not in source


def test_rayd_uses_the_validated_fail_loud_native_loader():
    source = inspect.getsource(symbols.native_extension)

    assert extension.native_extension is symbols.native_extension
    assert "_load_native_extension" in source
    assert "return None" not in source
    assert symbols.native_extension() is not None


def test_rayd_visibility_uses_channel_bridge():
    source = inspect.getsource(ops.rayd_visibility_forward)

    assert "_required_rayd_op" not in source
    assert "torch.ops" not in source
    assert "_required_native_op" in source
    assert "rayd_visibility_forward" in source


def test_diffraction_visibility_plan_uses_channel_bridge():
    source = inspect.getsource(ops.diffraction_tx_visible_state_plan)

    assert "_required_rayd_op" not in source
    assert "torch.ops" not in source
    assert "_required_native_op" in source
    assert "diffraction_tx_visible_state_plan" in source


def test_rayd_intersection_uses_channel_bridge():
    source = inspect.getsource(ops.rayd_intersect_forward)

    assert "_required_rayd_op" not in source
    assert "torch.ops" not in source
    assert "_required_native_op" in source
    assert "rayd_intersect_forward" in source
    assert "_rayd_resource" not in ops.__dict__


def test_rayd_diffraction_sample_tape_uses_channel_bridge():
    for fn in (ops.rayd_diffraction_sample_tape_forward,):
        source = inspect.getsource(fn)
        assert "_required_rayd_op" not in source
        assert "torch.ops" not in source
        assert "_required_native_op" in source


def test_mc_diffraction_discovery_requires_native_symbols(monkeypatch):
    for fn in (
        mc_sampling.mc_diffraction_discover_edges,
        mc_sampling.mc_diffraction_discover_edges_counted,
    ):
        source = inspect.getsource(fn)
        assert "required_symbol" in source
        assert "optional_symbol" not in source
        assert "torch.ops" not in source

    monkeypatch.setattr(
        mc_sampling,
        "_validate_mc_diffraction_discovery_args",
        lambda *args, **kwargs: None,
    )

    def missing(name: str):
        raise symbols.NativeSymbolError(
            f"_channel.{name} CUDA kernel is required"
        )

    monkeypatch.setattr(mc_sampling, "required_symbol", missing)
    with pytest.raises(symbols.NativeSymbolError, match="mc_diffraction_discover_edges"):
        mc_sampling.mc_diffraction_discover_edges()
    with pytest.raises(
        symbols.NativeSymbolError,
        match="mc_diffraction_discover_edges_counted",
    ):
        mc_sampling.mc_diffraction_discover_edges_counted()


def test_rayd_path_exports_use_channel_bridge():
    for fn in (
        ops.rayd_reflection_epc_paths_forward,
        ops.rayd_diffraction_paths_order1_forward,
    ):
        source = inspect.getsource(fn)
        assert "_required_rayd_op" not in source
        assert "torch.ops" not in source
        assert "_required_native_op" in source


def test_simplified_coherent_diffraction_grid_api_is_not_public():
    assert not hasattr(ops, "deterministic_diffraction_coherent_accumulation_forward")
    assert not hasattr(
        extension.native_extension(),
        "deterministic_diffraction_coherent_accumulation_forward",
    )


def test_rayd_scene_builder_uses_channel_scene_bridge():
    source = inspect.getsource(rayd_scene.build_scene_from_structures)
    edge_source = inspect.getsource(rayd_scene.RayDSceneResource.edge_records)

    assert "torch.classes" not in source
    assert "rayd_backend" not in source
    assert "rayd_scene_create" in source
    assert "rayd_scene_edge_records" in edge_source


def test_production_sources_have_no_rayd_dispatch_or_loader_fallbacks():
    repo = Path(__file__).resolve().parents[2]
    roots = (
        repo / "src" / "witwin" / "channel",
        repo / "native" / "channel",
    )
    forbidden = (
        "torch.ops.rayd",
        "_required_rayd_op",
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
        repo / "src" / "witwin" / "channel",
        repo / "native" / "channel",
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
    root = repo / "native" / "channel" / "kernels"
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
        repo / "src" / "witwin" / "channel" / "montecarlo" / "basic",
        repo / "src" / "witwin" / "channel" / "montecarlo" / "bdpt",
    )
    forbidden = (".contiguous(", ".reshape(")

    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*.py"):
            source = path.read_text()
            for token in forbidden:
                if token in source:
                    offenders.append(f"{path.relative_to(repo)}: {token}")
    assert offenders == []


def test_path_solver_uses_the_typed_enumerated_engine():
    repo = Path(__file__).resolve().parents[2]
    source = (
        repo / "src" / "witwin" / "channel" / "path" / "solver.py"
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
        / "channel"
        / "montecarlo"
        / "basic"
        / "backend.py",
        repo
        / "src"
        / "witwin"
        / "channel"
        / "montecarlo"
        / "basic"
        / "rayd_components.py",
        repo
        / "src"
        / "witwin"
        / "channel"
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
    from witwin.channel.montecarlo.basic import pipeline as mc_basic_pipeline

    solve_source = inspect.getsource(mc_basic_pipeline.solve_pipeline)
    module_source = inspect.getsource(mc_basic_pipeline)

    # Plan 07 AD-3: materials come from the compiled store in BOTH
    # ad_mode="none" and the AD modes (one source, same values); the old
    # host-float flattening cannot carry a gradient and is gone.
    assert "scene.rayd_scene()" in solve_source
    assert "_host_material_tensors" not in module_source
    assert "bdpt_face_material_tensors_from_host" not in module_source
    assert "face_material_field_bundle" in module_source


def test_bdpt_pipeline_does_not_use_derived_variance_or_component_map_path_export():
    from witwin.channel.montecarlo.bdpt import pipeline as bdpt_pipeline

    source = inspect.getsource(bdpt_pipeline)

    assert "bdpt_variance_estimate" not in source
    assert "bdpt_export_component_paths" not in source
    assert "bdpt_export_paths(" not in source
    assert "reflection_component_maps_with_wedges" not in source
    assert "diffraction_paths_order1" not in source
    assert "bdpt_sample_path_block" not in source


def test_bdpt_package_does_not_reintroduce_python_los_visibility_helpers():
    repo = Path(__file__).resolve().parents[2]
    root = repo / "src" / "witwin" / "channel" / "montecarlo" / "bdpt"
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
    bridge_root = repo / "native" / "channel" / "rayd"
    bridge_paths = [bridge_root / "resource.h", *sorted(bridge_root.glob("*.cpp"))]
    bridge_source = "\n".join(path.read_text() for path in bridge_paths)

    assert "LoadLibrary" not in bridge_source
    assert "dlopen" not in bridge_source
    assert "GetProcAddress" not in bridge_source
    assert "dlsym" not in bridge_source
    assert "module_handle" not in bridge_source
    assert "rayd::torch::create_scene" in bridge_source
    assert "rayd_torch_native_" not in bridge_source
    assert not (bridge_root / "bridge.h").exists()
