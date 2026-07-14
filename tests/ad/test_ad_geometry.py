"""AD-2 fixed-winner geometry: the torch reconstruction must be the native winner.

The differentiable field kernels only get correct geometry gradients if the
torch image-source reconstruction reproduces the hit points RayDN actually
discovered. These tests pin that parity directly (reconstruction vs the
topology tensors), independently of any gradient, so a drift in the native
discovery or in the reconstruction shows up here instead of as a silently
wrong derivative downstream.
"""

from __future__ import annotations

import pytest
import torch

from tests.ad.test_solver_material_frequency_ad import (
    _reflection_scene,
    _transmission_scene,
)
from witwin.channel_native.core.ad_geometry import (
    receiver_positions_ad,
    scene_vertex_table,
    specular_chain_geometry,
    straight_chain_geometry,
    transmitter_positions_ad,
)
from witwin.channel_native.core.path_topology import (
    export_topology,
    receiver_positions_and_layout,
    transmitter_tensors,
)
from witwin.channel_native.deterministic import Config as DeterministicConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for geometry AD"
)

_POSITION_ATOL = 1.0e-4
_NORMAL_ATOL = 1.0e-4


def _live_endpoints(scene, device):
    native_tx, _power = transmitter_tensors(scene, device=device)
    native_rx, _layout = receiver_positions_and_layout(scene, device=device)
    return (
        transmitter_positions_ad(scene, native_tx, device=device),
        receiver_positions_ad(scene, native_rx, device=device),
    )


def _topology(scene, components: frozenset[str], max_depth: int = 1):
    return export_topology(
        scene,
        DeterministicConfig(
            max_depth=max_depth, components=components, export_paths=True
        ),
    )


def test_specular_reconstruction_matches_native_hit_points():
    scene = _reflection_scene()
    device = torch.device("cuda")
    topology = _topology(scene, frozenset({"reflection"}))
    compiled = scene.compile()
    rows = torch.nonzero(topology.component_id == 1, as_tuple=False).reshape(-1)
    assert int(rows.shape[0]) > 0

    tx_positions, rx_positions = _live_endpoints(scene, device)
    source = tx_positions[topology.tx_id[rows].to(dtype=torch.int64)]
    target = rx_positions[topology.rx_id[rows].to(dtype=torch.int64)]
    positions, normals = specular_chain_geometry(
        scene_vertex_table(scene, compiled),
        compiled.geometry.faces,
        topology.primitive_sequence[rows, :1].to(dtype=torch.int64),
        source,
        target,
    )

    native_positions = topology.interaction_positions[rows, :1]
    native_normals = topology.interaction_normals[rows, :1]
    assert torch.allclose(positions, native_positions, atol=_POSITION_ATOL)
    # The field kernels flip normals against the incident ray, so only the
    # plane orientation is pinned, not the sign the discovery happened to emit.
    alignment = (normals * native_normals).sum(-1).abs()
    assert torch.allclose(alignment, torch.ones_like(alignment), atol=_NORMAL_ATOL)


def test_straight_reconstruction_matches_native_wall_crossings():
    scene = _transmission_scene()
    device = torch.device("cuda")
    topology = _topology(scene, frozenset({"transmission"}))
    compiled = scene.compile()
    rows = torch.nonzero(topology.component_id == 5, as_tuple=False).reshape(-1)
    assert int(rows.shape[0]) > 0

    tx_positions, rx_positions = _live_endpoints(scene, device)
    source = tx_positions[topology.tx_id[rows].to(dtype=torch.int64)]
    target = rx_positions[topology.rx_id[rows].to(dtype=torch.int64)]
    width = int(topology.interaction_positions.shape[1])
    slots = torch.arange(width, device=device).reshape(1, -1)
    event_valid = slots < topology.depth[rows].to(dtype=torch.int64).reshape(-1, 1)
    positions, _normals = straight_chain_geometry(
        scene_vertex_table(scene, compiled),
        compiled.geometry.faces,
        topology.primitive_sequence[rows].to(dtype=torch.int64),
        event_valid,
        source,
        target,
    )

    native_positions = topology.interaction_positions[rows]
    mask = event_valid.unsqueeze(-1)
    deviation = ((positions - native_positions).abs() * mask).max()
    assert float(deviation) <= _POSITION_ATOL


def test_reconstruction_carries_gradients_to_scene_leaves():
    scene = _reflection_scene()
    device = torch.device("cuda")
    topology = _topology(scene, frozenset({"reflection"}))
    compiled = scene.compile()
    rows = torch.nonzero(topology.component_id == 1, as_tuple=False).reshape(-1)

    vertices = scene.structures[0].vertices.to(device=device).requires_grad_(True)
    scene = scene.with_structure_vertices(0, vertices)
    tx = scene.transmitters[0].position.to(device=device).requires_grad_(True)
    native_tx, _power = transmitter_tensors(scene, device=device)
    native_rx, _layout = receiver_positions_and_layout(scene, device=device)
    source = torch.stack([tx])[topology.tx_id[rows].to(dtype=torch.int64)]
    target = receiver_positions_ad(scene, native_rx, device=device)[
        topology.rx_id[rows].to(dtype=torch.int64)
    ]
    del native_tx

    positions, normals = specular_chain_geometry(
        scene_vertex_table(scene, scene.compile()),
        compiled.geometry.faces,
        topology.primitive_sequence[rows, :1].to(dtype=torch.int64),
        source,
        target,
    )
    (positions.sum() + normals.sum()).backward()
    assert tx.grad is not None and float(tx.grad.abs().sum()) > 0.0
    assert vertices.grad is not None and float(vertices.grad.abs().sum()) > 0.0
