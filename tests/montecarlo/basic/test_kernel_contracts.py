from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from ci import check_ops_migration as migration
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.montecarlo.basic import kernels
from witwin.channel_native.montecarlo.basic.kernels import sampling
from witwin.channel_native.runtime import symbols, tensor_contracts


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

    assert len(definitions) == 7
    for definition in definitions:
        contract = contracts[definition.terminal_name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_mc_basic_sampling_uses_canonical_runtime_dependencies():
    assert sampling.native_extension is symbols.native_extension
    assert sampling.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor


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
