from __future__ import annotations

from pathlib import Path

from ci import check_import_graph
from witwin.channel.montecarlo.basic.kernels import sampling as mc_sampling
from witwin.channel.propagation.enumerated import reflection
from witwin.channel.propagation import topology
from witwin.channel.propagation.topology.kernels import (
    sampling as topology_sampling,
)
from witwin.channel import runtime


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel"


def test_topology_sampling_is_the_single_object_owner():
    owner = topology_sampling.mc_sample_directions

    assert owner.__module__ == topology_sampling.__name__
    assert topology.mc_sample_directions is owner
    assert mc_sampling.mc_sample_directions is owner
    assert topology.__all__ == ["mc_sample_directions"]


def test_topology_sampling_uses_canonical_runtime_dependencies():
    assert topology_sampling.native_extension is runtime.native_extension
    assert (
        topology_sampling.validate_cuda_tensor is runtime.validate_cuda_tensor
    )


def test_multibounce_discovery_uses_the_canonical_sampling_owner():
    owner = topology_sampling.mc_sample_directions

    assert (
        reflection._discovered_group_chains.__globals__["mc_sample_directions"]
        is owner
    )


def test_sampling_dependency_uses_the_public_topology_seam():
    edges = check_import_graph.collect_import_edges(PACKAGE_ROOT)
    package = "witwin.channel"
    mc_source = f"{package}.montecarlo.basic.kernels.sampling"
    public_target = f"{package}.propagation.topology"
    canonical_target = f"{package}.propagation.topology.kernels.sampling"

    assert any(
        edge.source == mc_source and edge.target == public_target for edge in edges
    )
    assert not any(
        edge.source == mc_source and edge.target == canonical_target for edge in edges
    )
