from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from witwin.core import scene as core_scene
from witwin.channel.kernels import materials as material_contracts
from witwin.channel.montecarlo import bdpt as bdpt_solver
from witwin.channel.kernels import montecarlo
from witwin.channel.montecarlo.bdpt import _BDPTTopologyOptions
from witwin.channel.propagation import geometry
from witwin.channel.kernels import geometry as geometry_kernels
from witwin.channel import runtime


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

_OWNER_NAMES = (
    "_bdpt_mis_mode_id",
    "_validate_bdpt_connection_samples",
    "_validate_bdpt_subpath_state",
    "bdpt_accumulate_connection_samples",
    "bdpt_compact_connection_samples",
    "bdpt_concat_connection_samples",
    "bdpt_connection_variance",
    "bdpt_count_valid_connection_samples",
    "bdpt_empty_subpath_state",
    "bdpt_endpoint_connection_samples",
    "bdpt_endpoint_connection_visibility_inputs",
    "bdpt_endpoint_subpath_state",
    "bdpt_filter_connection_samples",
    "bdpt_launch_state",
    "bdpt_mis_weights",
    "bdpt_reflected_light_subpath_state",
    "bdpt_subpath_intersection_inputs",
    "bdpt_transmitted_light_subpath_state",
)

_MAP_OWNER_NAMES = (
    "bdpt_apply_los_visibility",
    "bdpt_component_map_buffer",
    "bdpt_finalize_component_maps",
    "bdpt_finalize_point_components",
    "bdpt_host_vec3_tensor",
    "bdpt_los_component_maps",
    "bdpt_los_component_maps_from_matrix",
    "bdpt_los_export",
    "bdpt_los_visibility_inputs",
    "bdpt_point_component_power",
    "bdpt_receiver_grid_points",
    "bdpt_store_component_map",
    "bdpt_store_point_component_column",
    "bdpt_store_scaled_component_map",
    "bdpt_transmitter_tensors",
)

_SAMPLING_OWNER_NAMES = (
    "bdpt_pack_vec3",
    "bdpt_reflection_launch_inputs",
    "bdpt_sample_directions",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_bdpt_paths_is_the_single_object_owner(name: str):
    owner = getattr(montecarlo, name)

    assert owner.__module__ == montecarlo.__name__


def test_bdpt_paths_uses_canonical_dependencies():
    # The raw accessor is not a solver dependency: every probe goes through
    # runtime.required_symbol, and ci/check_import_graph.py rejects a solver
    # that imports ``native_extension`` at all.
    assert not hasattr(montecarlo, "native_extension")
    assert montecarlo._required_native_op is runtime.required_symbol
    assert montecarlo.validate_cuda_tensor is runtime.validate_cuda_tensor
    assert montecarlo._validate_layer_csr is material_contracts._validate_layer_csr
    assert montecarlo._BDPT_INTERSECTION_FIELDS is geometry.BDPT_INTERSECTION_FIELDS
    assert geometry.BDPT_INTERSECTION_FIELDS is geometry_kernels._BDPT_INTERSECTION_FIELDS
    assert "BDPT_INTERSECTION_FIELDS" not in getattr(geometry_kernels, "__all__", ())
    assert montecarlo._BDPT_SUBPATH_SCHEMA
    assert montecarlo._BDPT_CONNECTION_SCHEMA


@pytest.mark.parametrize("name", _MAP_OWNER_NAMES)
def test_bdpt_maps_is_the_single_object_owner(name: str):
    owner = getattr(montecarlo, name)

    assert owner.__module__ == montecarlo.__name__


def test_bdpt_maps_uses_canonical_runtime_dependencies():
    assert not hasattr(montecarlo, "native_extension")
    assert montecarlo._required_native_op is runtime.required_symbol
    assert montecarlo.validate_cuda_tensor is runtime.validate_cuda_tensor


def test_bdpt_zero_matrix_has_one_neutral_owner():
    owner = runtime.bdpt_zero_matrix

    assert owner.__module__ == runtime.__name__
    assert montecarlo.bdpt_zero_matrix is owner
    assert not hasattr(core_scene, "bdpt_zero_matrix")


@pytest.mark.parametrize("name", _SAMPLING_OWNER_NAMES)
def test_bdpt_sampling_is_the_single_object_owner(name: str):
    owner = getattr(montecarlo, name)

    assert owner.__module__ == montecarlo.__name__


def test_bdpt_sampling_uses_canonical_runtime_dependencies():
    assert montecarlo._required_native_op is runtime.required_symbol
    assert montecarlo.validate_cuda_tensor is runtime.validate_cuda_tensor


def test_bdpt_solver_uses_canonical_sampling_owners():
    for name in (
        "bdpt_reflection_launch_inputs",
        "bdpt_sample_directions",
    ):
        assert getattr(bdpt_solver, name) is getattr(montecarlo, name)


def test_bdpt_host_vec3_resolves_canonical_transmitter_helper():
    assert montecarlo.bdpt_host_vec3_tensor.__globals__ is montecarlo.__dict__
    assert (
        montecarlo.bdpt_host_vec3_tensor.__globals__["bdpt_transmitter_tensors"]
        is montecarlo.bdpt_transmitter_tensors
    )


def test_bdpt_callers_use_canonical_map_owners():
    assert bdpt_solver.bdpt_host_vec3_tensor is montecarlo.bdpt_host_vec3_tensor
    assert bdpt_solver.bdpt_receiver_grid_points is montecarlo.bdpt_receiver_grid_points
    assert bdpt_solver.bdpt_transmitter_tensors is montecarlo.bdpt_transmitter_tensors
    assert bdpt_solver.bdpt_component_map_buffer is montecarlo.bdpt_component_map_buffer
    assert bdpt_solver.bdpt_store_component_map is montecarlo.bdpt_store_component_map
    assert "mc_component_map_buffer" not in bdpt_solver.__dict__
    assert "mc_store_component_map" not in bdpt_solver.__dict__


def test_bdpt_public_solve_lazy_import_preserves_identity_and_pickle():
    # The solver collapsed into one module, so there is no longer a
    # ``.solver`` submodule to import lazily. What the pickle contract needs
    # survives the collapse: the public ``solve`` is defined in the module a
    # caller imports, so it round-trips by qualified name.
    code = (
        "import importlib, pickle, sys; "
        "sys.meta_path=[finder for finder in sys.meta_path "
        "if '_witwin_channel_editable' not in type(finder).__module__]; "
        "package=importlib.import_module('witwin.channel.montecarlo.bdpt'); "
        "assert 'witwin.channel.montecarlo.bdpt.solver' not in sys.modules; "
        "from witwin.channel.montecarlo.bdpt import solve; "
        "assert package.solve is solve; "
        "assert solve.__module__ == 'witwin.channel.montecarlo.bdpt'; "
        "assert pickle.loads(pickle.dumps(solve)) is solve"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    core_root = str(REPOSITORY_ROOT.parent / "core-radar-architecture-stage1")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (core_root, source_root, environment.get("PYTHONPATH"))
        if value
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_bdpt_topology_options_preserves_path_depth_cap():
    with pytest.raises(
        RuntimeError,
        match="path reflection/transmission support max_depth <= 5",
    ):
        _BDPTTopologyOptions(
            max_depth=6,
            components=frozenset({"reflection"}),
        )

    options = _BDPTTopologyOptions(
        max_depth=6,
        components=frozenset({"diffraction"}),
    )
    assert options.max_depth == 6
