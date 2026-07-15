from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from ci import check_ops_migration as migration
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.materials.kernels import contracts as material_contracts
from witwin.channel_native.montecarlo.bdpt import connections
from witwin.channel_native.montecarlo.bdpt import kernels
from witwin.channel_native.montecarlo.bdpt import solver as bdpt_solver
from witwin.channel_native.montecarlo.bdpt.kernels import maps, paths
from witwin.channel_native.montecarlo.bdpt.solver import _BDPTTopologyOptions
from witwin.channel_native.propagation import geometry
from witwin.channel_native.propagation.geometry.kernels import bridge
from witwin.channel_native.runtime import symbols, tensor_contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"

_OWNER_NAMES = (
    "_bdpt_mis_mode_id",
    "_validate_bdpt_connection_samples",
    "_validate_bdpt_subpath_state",
    "bdpt_accumulate_connection_samples",
    "bdpt_compact_connection_samples",
    "bdpt_concat_connection_samples",
    "bdpt_connection_variance",
    "bdpt_count_valid_connection_samples",
    "bdpt_diffraction_connection_samples_from_tape",
    "bdpt_diffraction_point_connection_samples",
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
    "bdpt_zero_matrix",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_bdpt_paths_is_the_single_object_owner(name: str):
    owner = getattr(paths, name)

    assert owner.__module__ == paths.__name__
    assert getattr(ops, name) is owner
    assert not hasattr(kernels, name)


def test_bdpt_paths_preserves_all_frozen_body_contracts():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{paths.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert {definition.terminal_name for definition in definitions} == set(
        _OWNER_NAMES
    )
    for definition in definitions:
        contract = contracts[definition.terminal_name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_bdpt_paths_uses_canonical_dependencies():
    assert paths.native_extension is symbols.native_extension
    assert paths._required_native_op is symbols.required_symbol
    assert paths.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert paths._validate_layer_csr is material_contracts._validate_layer_csr
    assert paths._BDPT_INTERSECTION_FIELDS is geometry.BDPT_INTERSECTION_FIELDS
    assert geometry.BDPT_INTERSECTION_FIELDS is bridge._BDPT_INTERSECTION_FIELDS
    assert "BDPT_INTERSECTION_FIELDS" not in getattr(geometry, "__all__", ())
    assert ops._BDPT_SUBPATH_SCHEMA is paths._BDPT_SUBPATH_SCHEMA
    assert ops._BDPT_CONNECTION_SCHEMA is paths._BDPT_CONNECTION_SCHEMA


@pytest.mark.parametrize("name", _MAP_OWNER_NAMES)
def test_bdpt_maps_is_the_single_object_owner(name: str):
    owner = getattr(maps, name)

    assert owner.__module__ == maps.__name__
    assert getattr(ops, name) is owner
    assert not hasattr(kernels, name)


def test_bdpt_maps_preserves_all_frozen_body_contracts():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{maps.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert {definition.terminal_name for definition in definitions} == set(
        _MAP_OWNER_NAMES
    )
    for definition in definitions:
        contract = contracts[definition.terminal_name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_bdpt_maps_uses_canonical_runtime_dependencies():
    assert maps.native_extension is symbols.native_extension
    assert maps._required_native_op is symbols.required_symbol
    assert maps.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor


def test_bdpt_host_vec3_resolves_canonical_transmitter_helper():
    assert maps.bdpt_host_vec3_tensor.__globals__ is maps.__dict__
    assert (
        maps.bdpt_host_vec3_tensor.__globals__["bdpt_transmitter_tensors"]
        is maps.bdpt_transmitter_tensors
    )


def test_bdpt_callers_use_canonical_map_owners():
    assert connections.bdpt_host_vec3_tensor is maps.bdpt_host_vec3_tensor
    assert connections.bdpt_receiver_grid_points is maps.bdpt_receiver_grid_points
    assert connections.bdpt_transmitter_tensors is maps.bdpt_transmitter_tensors
    assert bdpt_solver.bdpt_component_map_buffer is maps.bdpt_component_map_buffer
    assert bdpt_solver.bdpt_store_component_map is maps.bdpt_store_component_map
    assert "mc_component_map_buffer" not in bdpt_solver.__dict__
    assert "mc_store_component_map" not in bdpt_solver.__dict__


def test_bdpt_public_solve_lazy_import_preserves_identity_and_pickle():
    code = (
        "import importlib, pickle, sys; "
        "package=importlib.import_module('witwin.channel_native.montecarlo.bdpt'); "
        "assert 'witwin.channel_native.montecarlo.bdpt.solver' not in sys.modules; "
        "from witwin.channel_native.montecarlo.bdpt import solve; "
        "solver=importlib.import_module('witwin.channel_native.montecarlo.bdpt.solver'); "
        "assert solve is solver.solve; "
        "assert package.solve is solver.solve; "
        "assert pickle.loads(pickle.dumps(solve)) is solve"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, environment.get("PYTHONPATH"))
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


def test_core_kernels_from_import_preserves_ops_module_identity():
    code = (
        "import importlib; "
        "from witwin.channel_native.core import kernels; "
        "from witwin.channel_native.core.kernels import ops; "
        "canonical=importlib.import_module('witwin.channel_native.core.kernels.ops'); "
        "assert ops is canonical; assert kernels.ops is canonical"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, environment.get("PYTHONPATH"))
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


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.core.kernels import ops; "
            "from witwin.channel_native.montecarlo.bdpt.kernels import paths"
        ),
        (
            "from witwin.channel_native.montecarlo.bdpt.kernels import paths; "
            "from witwin.channel_native.core.kernels import ops"
        ),
    ),
)
def test_bdpt_paths_import_order_preserves_facade_identity(imports: str):
    names = repr(_OWNER_NAMES)
    code = (
        f"{imports}; "
        f"names={names}; "
        "assert all(getattr(ops, name) is getattr(paths, name) for name in names)"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, environment.get("PYTHONPATH"))
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


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.core.kernels import ops; "
            "from witwin.channel_native.montecarlo.bdpt.kernels import maps"
        ),
        (
            "from witwin.channel_native.montecarlo.bdpt.kernels import maps; "
            "from witwin.channel_native.core.kernels import ops"
        ),
    ),
)
def test_bdpt_maps_import_order_preserves_facade_identity(imports: str):
    names = repr(_MAP_OWNER_NAMES)
    code = (
        f"{imports}; "
        f"names={names}; "
        "assert all(getattr(ops, name) is getattr(maps, name) for name in names)"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, environment.get("PYTHONPATH"))
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
