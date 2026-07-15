from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ci import check_import_graph
from ci import check_ops_migration as migration
from witwin.channel_native.core import path_topology
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.montecarlo.basic.kernels import sampling as mc_sampling
from witwin.channel_native.propagation import topology
from witwin.channel_native.propagation.topology.kernels import (
    sampling as topology_sampling,
)
from witwin.channel_native.runtime import symbols, tensor_contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"
ALLOWLIST_PATH = REPOSITORY_ROOT / "ci" / "import_graph_allowlist.json"
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"


def test_topology_sampling_is_the_single_object_owner():
    owner = topology_sampling.mc_sample_directions

    assert owner.__module__ == topology_sampling.__name__
    assert topology.mc_sample_directions is owner
    assert mc_sampling.mc_sample_directions is owner
    assert ops.mc_sample_directions is owner
    assert path_topology.mc_sample_directions is owner
    assert topology.__all__ == ["mc_sample_directions"]


def test_topology_sampling_preserves_frozen_body_and_signature_contract():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contract = next(
        entry
        for entry in manifest["contracts"]
        if entry["id"] == "mc_sample_directions"
    )
    qualified_name = f"{topology_sampling.__name__}.mc_sample_directions"
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name == qualified_name
    ]

    assert len(definitions) == 1
    definition = definitions[0]
    assert definition.signature == contract["signature"]
    assert definition.body_sha256 == contract["body_sha256"]
    assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_topology_sampling_uses_canonical_runtime_dependencies():
    assert topology_sampling.native_extension is symbols.native_extension
    assert (
        topology_sampling.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    )


def test_manifest_names_topology_sampling_as_the_canonical_owner():
    manifest = migration.load_manifest(MANIFEST_PATH)

    assert manifest["canonical_owners"]["mc_sample_directions"] == (
        "witwin.channel_native.propagation.topology.kernels.sampling."
        "mc_sample_directions"
    )


def test_multibounce_discovery_uses_the_canonical_sampling_owner():
    owner = topology_sampling.mc_sample_directions

    assert (
        path_topology._discovered_group_chains.__globals__["mc_sample_directions"]
        is owner
    )
    assert not hasattr(path_topology, "ops")


def test_ops_003_is_retired_through_the_public_topology_seam():
    edges = check_import_graph.collect_import_edges(PACKAGE_ROOT)
    package = "witwin.channel_native"
    mc_source = f"{package}.montecarlo.basic.kernels.sampling"
    public_target = f"{package}.propagation.topology"
    canonical_target = f"{package}.propagation.topology.kernels.sampling"

    assert any(
        edge.source == mc_source and edge.target == public_target for edge in edges
    )
    assert not any(
        edge.source == mc_source and edge.target == canonical_target for edge in edges
    )
    assert not any(
        edge.source == f"{package}.core.path_topology"
        and edge.target == f"{package}.core.kernels.ops"
        for edge in edges
    )

    allowlist = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    direct_ops = allowlist["debts"]["direct_core_kernels_ops"]
    assert "ops-003" not in direct_ops["allowed"]
    assert any(entry["id"] == "ops-003" for entry in direct_ops["baseline"])


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.propagation.topology.kernels import "
            "sampling as topology_sampling; "
            "from witwin.channel_native.propagation import topology; "
            "from witwin.channel_native.montecarlo.basic.kernels import "
            "sampling as mc_sampling; "
            "from witwin.channel_native.core.kernels import ops; "
            "from witwin.channel_native.core import path_topology"
        ),
        (
            "from witwin.channel_native.montecarlo.basic.kernels import "
            "sampling as mc_sampling; "
            "from witwin.channel_native.propagation import topology; "
            "from witwin.channel_native.core.kernels import ops; "
            "from witwin.channel_native.core import path_topology; "
            "from witwin.channel_native.propagation.topology.kernels import "
            "sampling as topology_sampling"
        ),
        (
            "from witwin.channel_native.core import path_topology; "
            "from witwin.channel_native.core.kernels import ops; "
            "from witwin.channel_native.montecarlo.basic.kernels import "
            "sampling as mc_sampling; "
            "from witwin.channel_native.propagation import topology; "
            "from witwin.channel_native.propagation.topology.kernels import "
            "sampling as topology_sampling"
        ),
    ),
)
def test_topology_sampling_import_order_preserves_facade_identity(imports: str):
    code = (
        f"{imports}; "
        "owner=topology_sampling.mc_sample_directions; "
        "assert topology.mc_sample_directions is owner; "
        "assert mc_sampling.mc_sample_directions is owner; "
        "assert ops.mc_sample_directions is owner; "
        "assert path_topology.mc_sample_directions is owner"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
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
