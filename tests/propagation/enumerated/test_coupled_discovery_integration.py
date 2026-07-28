from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.channel.interactions import coupled
from witwin.channel.interactions import coupled as discovery_coupled


def _fake_coupled_inputs(monkeypatch):
    records = SimpleNamespace(
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        face_normals=torch.tensor([[0.0, 0.0, 1.0]]),
    )
    handle = object()
    handle_calls: list[object] = []

    def require_resource():
        handle_calls.append(handle)
        return handle

    rayd = SimpleNamespace(
        available=True,
        edge_records=lambda: records,
        require_resource=require_resource,
    )
    compiled = SimpleNamespace(
        rayd=rayd,
        geometry=SimpleNamespace(face_surface_id=torch.tensor([0])),
        assignments=SimpleNamespace(
            face_material_id=torch.tensor([41], dtype=torch.int32)
        ),
    )
    scene = SimpleNamespace(structures=[object()], metadata={})
    monkeypatch.setattr(
        coupled.geometry_kernels,
        "deterministic_normalize_vec3",
        lambda values, *, eps: values,
    )
    monkeypatch.setattr(
        coupled.topology_kernels,
        "deterministic_face_anchor_points",
        lambda vertices, faces: vertices[faces[:, 0].to(dtype=torch.int64)],
    )
    monkeypatch.setattr(coupled, "_ensure_topology_fields", lambda block: block)
    monkeypatch.setattr(
        coupled,
        "concatenate_path_blocks",
        lambda _blocks, *, device: {
            "valid": torch.empty((0,), device=device, dtype=torch.bool)
        },
    )
    groups = {
        "representative_faces": torch.tensor([0], dtype=torch.int32),
        "surface_group_id": torch.tensor([0], dtype=torch.int32),
        "surface_group_size": torch.tensor([1], dtype=torch.int32),
        "surface_group_members": torch.tensor([0], dtype=torch.int32),
    }
    monkeypatch.setattr(coupled, "_cached_coplanar_face_groups", lambda *_args: groups)
    edge_position = torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    edge_direction = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    edge_scalar = torch.tensor([0.0, 0.0])
    monkeypatch.setattr(
        coupled,
        "_cached_diffraction_edge_geometry",
        lambda _rayd: (
            torch.tensor([True, True]),
            edge_position,
            edge_direction,
            edge_scalar,
            edge_scalar,
            torch.tensor([1.0, 1.0]),
            edge_direction,
            edge_direction,
            torch.tensor([0, 0], dtype=torch.int32),
            torch.tensor([0, 0], dtype=torch.int32),
            edge_scalar,
        ),
    )
    monkeypatch.setattr(
        coupled.topology_kernels,
        "mc_selected_edge_indices",
        lambda _selected: torch.tensor([0, 1], dtype=torch.int32),
    )
    tx_positions = torch.tensor([[0.0, -2.0, 1.0], [1.0, -2.0, 1.0]])
    rx_positions = torch.tensor([[0.0, 2.0, 5.0], [1.0, 2.0, 5.0]])
    return scene, compiled, tx_positions, rx_positions, handle_calls


def test_consumer_uses_lazy_chunk_requests_and_preserves_launch_accounting(monkeypatch):
    scene, compiled, tx_positions, rx_positions, handle_calls = _fake_coupled_inputs(
        monkeypatch
    )
    prepare = discovery_coupled.prepare_coupled_candidate_plan
    monkeypatch.setattr(
        coupled,
        "prepare_coupled_candidate_plan",
        # Force both streams to chunk at 3 so the union launch/candidate
        # accounting is exercised across multiple chunks. In production the D->D
        # stream uses a larger dd_chunk_size (ADR-013 G-H) and collapses to one
        # launch per block; this test pins it small on purpose.
        lambda **kwargs: prepare(**kwargs, chunk_size=3, dd_chunk_size=3),
    )
    queries = []

    def query_geometry(query):
        queries.append(query)
        return SimpleNamespace(
            valid=torch.zeros((query.face_id.shape[0],), dtype=torch.bool)
        )

    monkeypatch.setattr(coupled, "query_coupled_geometry", query_geometry)

    # cid 7 (ADR-013 D->D) shares the same plan and receiver block, so the slice
    # worker also streams the ordered edge-pair candidates. Stub its geometry the
    # same way and record the launches so the union launch/candidate accounting
    # is asserted, not silently absorbed.
    dd_queries = []

    def query_dd_geometry(query):
        dd_queries.append(query)
        return SimpleNamespace(
            valid=torch.zeros((query.edge1_id.shape[0],), dtype=torch.bool)
        )

    monkeypatch.setattr(coupled, "query_coupled_dd_geometry", query_dd_geometry)

    block, launch_count, candidate_count = (
        coupled._coupled_reflection_diffraction_topology_order2(
            scene,
            compiled,
            tx_positions,
            rx_positions,
            candidate_limit=24,
        )
    )

    # base=tx*rx*groups*edges=8, dd_base=tx*rx*edges*(edges-1)=8. With chunk_size
    # 3 the base stream chunks to 3,3,2 (x2 reverse -> 6 launches, 16 candidates)
    # and the D->D stream chunks to 3,3,2 (3 launches, 8 candidates). The union
    # budget is base*2 + dd_base = 24 (ADR-013 D1).
    assert launch_count == 9
    assert candidate_count == 24
    # The R->D/D->R loop requests the native handle once per chunk (3); the D->D
    # loop hoists the handle out of its chunk loop and requests it once (ADR-013
    # G-H), so the total is 4 rather than one-per-DD-chunk.
    assert len(handle_calls) == 4
    assert [query.reverse for query in queries] == [False, True] * 3
    assert [int(query.face_id.shape[0]) for query in queries] == [3, 3, 3, 3, 2, 2]
    assert [int(query.edge1_id.shape[0]) for query in dd_queries] == [3, 3, 2]
    assert int(torch.count_nonzero(block["valid"])) == 0
    for offset in range(0, len(queries), 2):
        rd = queries[offset]
        dr = queries[offset + 1]
        for name in ("source", "receiver", "face_id", "edge_id"):
            rd_tensor = getattr(rd, name)
            dr_tensor = getattr(dr, name)
            assert dr_tensor is rd_tensor
            assert dr_tensor.data_ptr() == rd_tensor.data_ptr()
            assert dr_tensor.stride() == rd_tensor.stride()


def test_consumer_candidate_guard_precedes_handle_and_geometry_launch(monkeypatch):
    scene, compiled, tx_positions, rx_positions, _handle_calls = _fake_coupled_inputs(
        monkeypatch
    )
    monkeypatch.setattr(
        compiled.rayd,
        "require_resource",
        lambda: pytest.fail("native handle requested before candidate guard"),
    )
    monkeypatch.setattr(
        coupled,
        "query_coupled_geometry",
        lambda _query: pytest.fail("geometry launched before candidate guard"),
    )

    # base*2 + dd_base = 8*2 + 8 = 24 union candidates (ADR-013 D1); the guard
    # must fire on the union count before any handle or geometry launch.
    with pytest.raises(
        RuntimeError,
        match="requires 24 candidates, exceeding coupled_candidate_limit=15",
    ):
        coupled._coupled_reflection_diffraction_topology_order2(
            scene,
            compiled,
            tx_positions,
            rx_positions,
            candidate_limit=15,
        )
