from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from ci import check_ops_migration as migration
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.materials.kernels import contracts as material_contracts
from witwin.channel_native.montecarlo.bdpt import kernels
from witwin.channel_native.montecarlo.bdpt.kernels import paths
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
