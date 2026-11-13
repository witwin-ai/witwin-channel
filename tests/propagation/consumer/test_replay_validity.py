# Copyright Xingyu Chen.
# Tests that fixed-topology replay applies the reflection validity rule exactly as discovery does.

from __future__ import annotations

import pytest
import torch

from witwin.channel.propagation.consumer import (
    EndpointBatch,
    FixedTopologyRequest,
    PropagationRequest,
    evaluate,
    prepare_fixed_topology,
    reevaluate,
)
from witwin.channel.scene import compile
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

FREQUENCY_HZ = 77.0e9
# The wall x = 4, |y| <= 1.2, |z| <= 3: diagonal 6.46 m, so the minimum segment is 64.6 um.
WALL_X_M = 4.0
SOURCE = (0.0, 0.0, 0.0)
NEAR_M = 5.0e-5


def _structure(vertices, faces, material, structure_id: int) -> Structure:
    return Structure(
        geometry=Mesh(
            vertices=torch.tensor(vertices, dtype=torch.float32),
            faces=torch.tensor(faces),
            recenter=False,
            fill_mode="surface",
            topology_diagnostics=False,
        ),
        material=material,
        structure_id=structure_id,
        material_id=structure_id,
        assignment_id=structure_id,
        surface_id=structure_id,
    )


def _wall_scene(material):
    wall = _structure(
        [[WALL_X_M, -1.2, -3.0], [WALL_X_M, 1.2, -3.0], [WALL_X_M, 1.2, 3.0], [WALL_X_M, -1.2, 3.0]],
        [[0, 1, 2], [0, 2, 3]],
        material,
        1,
    )
    return compile(Scene(structures=(wall,)), reference_frequency_hz=FREQUENCY_HZ)


def _corner_scene():
    """Walls x = 0 (y in [0, 4]) and y = 0 (x in [0, 4]) meeting on the z axis."""

    concrete = PhysicalMaterial(name="concrete", eps_r=5.24, sigma_e=0.0462)
    a = _structure([[0, 0, -2], [0, 4, -2], [0, 4, 2], [0, 0, 2]], [[0, 1, 2], [0, 2, 3]], concrete, 1)
    b = _structure([[0, 0, -2], [4, 0, -2], [4, 0, 2], [0, 0, 2]], [[0, 1, 2], [0, 2, 3]], concrete, 2)
    return compile(Scene(structures=(a, b)), reference_frequency_hz=FREQUENCY_HZ)


def _batch(stable_id: int, position, power: float | None = None) -> EndpointBatch:
    return EndpointBatch(
        stable_ids=torch.tensor([stable_id], device="cuda"),
        positions_m=torch.tensor([position], device="cuda", dtype=torch.float32),
        polarizations=torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
        powers_w=None if power is None else torch.tensor([power], device="cuda"),
    )


def _discover(compiled, source, sink, max_depth: int = 1):
    request = PropagationRequest(
        sources=_batch(1, source, 1.0), sinks=_batch(2, sink), reference_frequency_hz=FREQUENCY_HZ,
        components=frozenset({"los", "reflection"}), max_depth=max_depth, response="scalar_transport",
        topology_mode="discover", ad_mode="none",
    )
    return evaluate(compiled, request).paths


def _replay(compiled, frozen, source, sink):
    return reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=_batch(1, source, 1.0), sinks=_batch(2, sink), reference_frequency_hz=FREQUENCY_HZ,
            topology=frozen, response="scalar_transport", ad_mode="none",
        ),
    )


@pytest.mark.parametrize("frozen_m,replayed_m", [(1.0e-2, NEAR_M), (NEAR_M, 1.0e-2), (1.0e-2, 0.0)])
def test_discovery_and_replay_agree_near_a_wall(frozen_m: float, replayed_m: float) -> None:
    compiled = _wall_scene(PhysicalMaterial(name="concrete", eps_r=5.24, sigma_e=0.0462))
    frozen = prepare_fixed_topology(_discover(compiled, SOURCE, (WALL_X_M - frozen_m, 0.6, 0.0)).topology)
    sink = (WALL_X_M - replayed_m, 0.6, 0.0)
    discovered = _discover(compiled, SOURCE, sink)
    replayed = _replay(compiled, frozen, SOURCE, sink)
    assert sorted(discovered.topology.depth.tolist()) == [0, 1]
    assert replayed.paths.topology.depth.tolist() == frozen.topology.depth.tolist()
    assert replayed.row_valid is None or bool(replayed.row_valid.all())
    torch.testing.assert_close(
        replayed.paths.transport.coefficient.sum(), discovered.transport.coefficient.sum(), rtol=1.0e-4, atol=0.0
    )


def test_replay_rejects_a_chain_whose_interactions_meet_on_the_shared_edge() -> None:
    """Frozen where its two reflection points are 1.1 m apart, the corner chain [x = 0, y = 0] is
    replayed at a sink that puts both points on the shared edge: discovery drops it, so must replay."""

    compiled = _corner_scene()
    source = (1.0, 1.0, 0.0)
    frozen_paths = _discover(compiled, source, (3.0, 1.0, 0.0), max_depth=2)
    assert 2 in frozen_paths.topology.depth.tolist()
    frozen = prepare_fixed_topology(frozen_paths.topology)
    edge_sink = (2.0, 2.0, 0.0)
    discovered = _discover(compiled, source, edge_sink, max_depth=2)
    assert 2 not in discovered.topology.depth.tolist()
    replayed = _replay(compiled, frozen, source, edge_sink)
    depth = frozen.topology.depth.tolist()
    assert replayed.row_valid is not None
    valid = replayed.row_valid.tolist()
    assert [v for d, v in zip(depth, valid) if d == 2] == [False]
    assert all(v for d, v in zip(depth, valid) if d < 2)
