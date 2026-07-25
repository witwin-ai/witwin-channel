from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from witwin.core import scene as core_scene
from witwin.channel.materials.kernels import contracts as material_contracts
from witwin.channel.montecarlo.bdpt import endpoints
from witwin.channel.montecarlo.bdpt import kernels
from witwin.channel.montecarlo.bdpt import solver as bdpt_solver
from witwin.channel.montecarlo.bdpt.kernels import maps, paths, sampling
from witwin.channel.montecarlo.bdpt.solver import _BDPTTopologyOptions
from witwin.channel.propagation import geometry
from witwin.channel.propagation.geometry.kernels import bridge
from witwin.channel.runtime import native_buffers, symbols, tensor_contracts


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
    owner = getattr(paths, name)

    assert owner.__module__ == paths.__name__
    assert not hasattr(kernels, name)


def test_bdpt_paths_uses_canonical_dependencies():
    assert paths.native_extension is symbols.native_extension
    assert paths._required_native_op is symbols.required_symbol
    assert paths.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert paths._validate_layer_csr is material_contracts._validate_layer_csr
    assert paths._BDPT_INTERSECTION_FIELDS is geometry.BDPT_INTERSECTION_FIELDS
    assert geometry.BDPT_INTERSECTION_FIELDS is bridge._BDPT_INTERSECTION_FIELDS
    assert "BDPT_INTERSECTION_FIELDS" not in getattr(geometry, "__all__", ())
    assert paths._BDPT_SUBPATH_SCHEMA
    assert paths._BDPT_CONNECTION_SCHEMA


@pytest.mark.parametrize("name", _MAP_OWNER_NAMES)
def test_bdpt_maps_is_the_single_object_owner(name: str):
    owner = getattr(maps, name)

    assert owner.__module__ == maps.__name__
    assert not hasattr(kernels, name)


def test_bdpt_maps_uses_canonical_runtime_dependencies():
    assert maps.native_extension is symbols.native_extension
    assert maps._required_native_op is symbols.required_symbol
    assert maps.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor


def test_bdpt_zero_matrix_has_one_neutral_owner():
    owner = native_buffers.bdpt_zero_matrix

    assert owner.__module__ == native_buffers.__name__
    assert maps.bdpt_zero_matrix is owner
    assert not hasattr(core_scene, "bdpt_zero_matrix")


@pytest.mark.parametrize("name", _SAMPLING_OWNER_NAMES)
def test_bdpt_sampling_is_the_single_object_owner(name: str):
    owner = getattr(sampling, name)

    assert owner.__module__ == sampling.__name__
    assert not hasattr(kernels, name)


def test_bdpt_sampling_uses_canonical_runtime_dependencies():
    assert sampling._required_native_op is symbols.required_symbol
    assert sampling.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor


def test_bdpt_solver_uses_canonical_sampling_owners():
    for name in (
        "bdpt_reflection_launch_inputs",
        "bdpt_sample_directions",
    ):
        assert getattr(bdpt_solver, name) is getattr(sampling, name)


def test_bdpt_host_vec3_resolves_canonical_transmitter_helper():
    assert maps.bdpt_host_vec3_tensor.__globals__ is maps.__dict__
    assert (
        maps.bdpt_host_vec3_tensor.__globals__["bdpt_transmitter_tensors"]
        is maps.bdpt_transmitter_tensors
    )


def test_bdpt_callers_use_canonical_map_owners():
    assert endpoints.bdpt_host_vec3_tensor is maps.bdpt_host_vec3_tensor
    assert endpoints.bdpt_receiver_grid_points is maps.bdpt_receiver_grid_points
    assert endpoints.bdpt_transmitter_tensors is maps.bdpt_transmitter_tensors
    assert bdpt_solver.bdpt_component_map_buffer is maps.bdpt_component_map_buffer
    assert bdpt_solver.bdpt_store_component_map is maps.bdpt_store_component_map
    assert "mc_component_map_buffer" not in bdpt_solver.__dict__
    assert "mc_store_component_map" not in bdpt_solver.__dict__


def test_bdpt_public_solve_lazy_import_preserves_identity_and_pickle():
    code = (
        "import importlib, pickle, sys; "
        "sys.meta_path=[finder for finder in sys.meta_path "
        "if '_witwin_channel_editable' not in type(finder).__module__]; "
        "package=importlib.import_module('witwin.channel.montecarlo.bdpt'); "
        "assert 'witwin.channel.montecarlo.bdpt.solver' not in sys.modules; "
        "from witwin.channel.montecarlo.bdpt import solve; "
        "solver=importlib.import_module('witwin.channel.montecarlo.bdpt.solver'); "
        "assert solve is solver.solve; "
        "assert package.solve is solver.solve; "
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
