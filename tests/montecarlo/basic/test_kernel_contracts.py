from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ci import check_ops_migration as migration
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.montecarlo.basic import kernels
from witwin.channel_native.montecarlo.basic.kernels import maps, sampling
from witwin.channel_native.propagation import topology
from witwin.channel_native.propagation.topology.kernels import blocks as topology_blocks
from witwin.channel_native.runtime import (
    autograd_contracts,
    symbols,
    tensor_contracts,
    torch_compat,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"

_OWNER_NAMES = (
    "mc_diffraction_state_pack",
    "mc_diffraction_state_wi",
    "mc_pack_vec3",
    "mc_receiver_grid_points",
    "mc_reflection_launch_inputs",
    "mc_sample_directions",
    "mc_transmitter_tensors",
)

_MAP_OWNER_NAMES = (
    "_McLosPathGainAdFunction",
    "mc_los_path_gain_ad",
    "mc_los_path_gain_backward",
    "mc_los_path_gain_jvp",
)

_MAP_CONTRACT_IDS = (
    "_McLosPathGainAdFunction.backward",
    "_McLosPathGainAdFunction.forward",
    "_McLosPathGainAdFunction.jvp",
    "_McLosPathGainAdFunction.setup_context",
    "mc_los_path_gain_ad",
    "mc_los_path_gain_backward",
    "mc_los_path_gain_jvp",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_mc_basic_sampling_is_the_single_object_owner(name: str):
    owner = getattr(sampling, name)

    assert owner.__module__ == sampling.__name__
    assert getattr(ops, name) is owner
    assert not hasattr(kernels, name)


def test_mc_basic_sampling_preserves_all_frozen_body_contracts():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{sampling.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert {definition.terminal_name for definition in definitions} == set(_OWNER_NAMES)
    for definition in definitions:
        contract = contracts[definition.terminal_name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_mc_basic_sampling_uses_canonical_runtime_dependencies():
    assert sampling.native_extension is symbols.native_extension
    assert sampling.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor


@pytest.mark.parametrize("name", _MAP_OWNER_NAMES)
def test_mc_basic_maps_is_the_single_object_owner(name: str):
    owner = getattr(maps, name)

    assert owner.__module__ == maps.__name__
    assert getattr(ops, name) is owner
    assert not hasattr(kernels, name)


def test_mc_basic_maps_preserves_exactly_the_frozen_los_ad_contracts():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{maps.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert {definition.terminal_name for definition in definitions} == set(
        _MAP_CONTRACT_IDS
    )
    for definition in definitions:
        contract = contracts[definition.terminal_name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_mc_basic_maps_uses_canonical_runtime_dependencies():
    assert maps.native_extension is symbols.native_extension
    assert maps.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert maps.torch_compat is torch_compat
    assert maps._ad_frequency_grad is autograd_contracts._ad_frequency_grad
    assert maps._ad_frequency_tangent is autograd_contracts._ad_frequency_tangent
    assert maps._ad_frequency_value is autograd_contracts._ad_frequency_value
    assert maps._ad_geometry_tangent is autograd_contracts._ad_geometry_tangent
    assert maps._ad_native_tensor is autograd_contracts._ad_native_tensor
    assert maps._ad_reject_fixed_inputs is autograd_contracts._ad_reject_fixed_inputs
    assert (
        maps._ad_reject_fixed_tangents
        is autograd_contracts._ad_reject_fixed_tangents
    )
    assert ops._LIGHT_SPEED_M_PER_S_AD is maps._LIGHT_SPEED_M_PER_S_AD


def test_mc_basic_maps_los_ad_methods_resolve_canonical_siblings():
    function = maps._McLosPathGainAdFunction

    assert function.forward.__globals__ is maps.__dict__
    assert function.setup_context.__globals__ is maps.__dict__
    assert inspect.unwrap(function.backward).__globals__ is maps.__dict__
    assert function.jvp.__globals__ is maps.__dict__
    assert function.forward.__globals__["path_los_export"] is maps.path_los_export
    assert (
        inspect.unwrap(function.backward).__globals__["mc_los_path_gain_backward"]
        is maps.mc_los_path_gain_backward
    )
    assert (
        function.jvp.__globals__["mc_los_path_gain_jvp"]
        is maps.mc_los_path_gain_jvp
    )
    assert (
        maps.mc_los_path_gain_ad.__globals__["_McLosPathGainAdFunction"] is function
    )


def test_mc_basic_maps_uses_package_level_los_export_same_object_alias():
    assert topology.path_los_export is topology_blocks.path_los_export
    assert maps.path_los_export is topology.path_los_export
    assert "path_los_export" not in topology.__all__


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.core.kernels import ops; "
            "from witwin.channel_native.montecarlo.basic.kernels import sampling"
        ),
        (
            "from witwin.channel_native.montecarlo.basic.kernels import sampling; "
            "from witwin.channel_native.core.kernels import ops"
        ),
    ),
)
def test_mc_basic_sampling_import_order_preserves_facade_identity(imports: str):
    names = repr(_OWNER_NAMES)
    code = (
        f"{imports}; "
        f"names={names}; "
        "assert all(getattr(ops, name) is getattr(sampling, name) for name in names)"
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
            "from witwin.channel_native.montecarlo.basic.kernels import maps"
        ),
        (
            "from witwin.channel_native.montecarlo.basic.kernels import maps; "
            "from witwin.channel_native.core.kernels import ops"
        ),
    ),
)
def test_mc_basic_maps_import_order_preserves_facade_identity(imports: str):
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
