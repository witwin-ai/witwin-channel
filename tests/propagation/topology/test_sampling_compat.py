# Copyright Xingyu Chen.
# Tests sampling compat.

from __future__ import annotations

from pathlib import Path

from ci import check_import_graph
from witwin.channel.kernels import montecarlo as mc_sampling
from witwin.channel.interactions import reflection
from witwin.channel.propagation import topology
from witwin.channel.kernels import topology as topology_kernels
from witwin.channel import runtime


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "witwin" / "channel"


def test_topology_sampling_is_the_single_object_owner():
    owner = topology_kernels.mc_sample_directions

    assert owner.__module__ == topology_kernels.__name__
    # Neither the propagation topology owner nor the Monte Carlo facades re-export it.
    assert "mc_sample_directions" not in vars(topology)
    assert "mc_sample_directions" not in vars(mc_sampling)


def test_topology_sampling_uses_canonical_runtime_dependencies():
    assert topology_kernels._required_native_op is runtime.required_symbol
    assert not hasattr(topology_kernels, "native_extension")
    assert (
        topology_kernels.validate_cuda_tensor is runtime.validate_cuda_tensor
    )


def test_multibounce_discovery_uses_the_canonical_sampling_owner():
    owner = topology_kernels.mc_sample_directions

    assert (
        reflection._discovered_group_chains.__globals__["mc_sample_directions"]
        is owner
    )


def test_monte_carlo_facades_do_not_import_the_propagation_layer():
    edges = check_import_graph.collect_import_edges(PACKAGE_ROOT)
    package = "witwin.channel"
    mc_source = f"{package}.kernels.montecarlo"

    assert not any(
        edge.source == mc_source and edge.target.startswith(f"{package}.propagation")
        for edge in edges
    )
