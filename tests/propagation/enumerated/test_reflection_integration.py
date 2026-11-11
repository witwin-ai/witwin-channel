# Copyright Xingyu Chen.
# Tests reflection integration.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.channel.interactions import reflection
from witwin.channel.interactions.reflection import ReflectionEpcRequest
from witwin.channel.kernels import fields as field_kernels


def _forbidden(name):
    def call(*args, **kwargs):
        raise AssertionError(name)

    return call


def _fixture(monkeypatch, events: list[str], *, selected: bool = True):
    vertices = torch.zeros((3, 3))
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    normals = torch.ones((1, 3))
    records = SimpleNamespace(vertices=vertices, faces=faces, face_normals=normals)
    rayd = SimpleNamespace(
        available=True,
        edge_records=lambda: records,
        require_resource=lambda: object(),
    )
    compiled = SimpleNamespace(rayd=rayd)
    scene = SimpleNamespace(structures=[object()])
    tx_positions = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tx_power = torch.tensor([2.0, 3.0], dtype=torch.float64)
    rx_positions = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    face_material_id = torch.tensor([7], dtype=torch.int32)

    monkeypatch.setattr(
        reflection.geometry_kernels, "deterministic_normalize_vec3", lambda value, eps: normals
    )
    monkeypatch.setattr(
        reflection.topology_kernels,
        "deterministic_face_anchor_points",
        lambda *args: torch.zeros((1, 3)),
    )
    monkeypatch.setattr(reflection, "face_material_id_table", lambda *a, **k: face_material_id)
    monkeypatch.setattr(reflection, "face_geometry_id_table", lambda *a, **k: torch.tensor([0]))
    monkeypatch.setattr(
        reflection,
        "_cached_coplanar_face_groups",
        lambda *a, **k: {
            "group_count": 1,
            "representative_faces": torch.tensor([0]),
            "face_group_id": torch.tensor([0]),
            "surface_group_id": torch.tensor([0]),
            "surface_group_size": torch.tensor([1]),
            "surface_group_members": torch.tensor([[0]]),
        },
    )
    monkeypatch.setattr(reflection, "prepare_reflection_order1_plan", lambda **k: object())
    monkeypatch.setattr(reflection, "prepare_reflection_multibounce_plan", lambda **k: object())
    # The discovery stage evaluates no fields: the field stage owns every reflection row.
    for name in ("deterministic_pack_complex", "field_reflection_sequence"):
        monkeypatch.setattr(field_kernels, name, _forbidden(name))
    monkeypatch.setattr(
        reflection.topology_kernels, "deterministic_topology_base_fields", _forbidden("base")
    )

    def compact(**kwargs):
        events.append("compact")
        assert kwargs["face_material_id"] is face_material_id
        count = 1 if selected else 0
        return {"valid": torch.ones(count, dtype=torch.bool), "kwargs": kwargs}

    monkeypatch.setattr(
        reflection.topology_kernels, "deterministic_reflection_topology_compact", compact
    )
    monkeypatch.setattr(
        reflection, "_ensure_topology_fields", lambda block, **k: events.append("ensure") or block
    )
    monkeypatch.setattr(
        reflection,
        "concatenate_path_blocks",
        lambda blocks, **k: events.append("concat") or {"blocks": tuple(blocks)},
    )
    return scene, compiled, tx_positions, tx_power, rx_positions


def _request(depth: int = 1) -> tuple[ReflectionEpcRequest, dict[str, object]]:
    epc_inputs = {
        "tx_batch": object(),
        "rx_batch": object(),
        "sequence_batch": object(),
        "direct_plane_points": object(),
        "direct_plane_normals": object(),
        "tx_indices": object(),
        "rx_indices": object(),
    }
    return ReflectionEpcRequest(depth, epc_inputs), epc_inputs


def _epc_geometry(events):
    def fake_query(query):
        events.append(f"epc{query.max_bounces}")
        return reflection.ReflectionEpcGeometry(
            visible=torch.ones(1, dtype=torch.bool),
            path_length_m=torch.ones(1),
            resolved_prim_ids=torch.zeros((1, query.max_bounces), dtype=torch.int32),
            surface_group_ids=torch.zeros((1, query.max_bounces), dtype=torch.int32),
            hit_positions=torch.zeros((1, query.max_bounces, 3)),
            normals=torch.ones((1, query.max_bounces, 3)),
        )

    return fake_query


def test_order1_maps_batched_request_to_query_and_topology_compact(monkeypatch):
    events: list[str] = []
    args = _fixture(monkeypatch, events)
    request, epc_inputs = _request()
    queries = []

    def fake_query(query):
        queries.append(query)
        return _epc_geometry(events)(query)

    monkeypatch.setattr(reflection, "query_reflection_epc", fake_query)
    monkeypatch.setattr(
        reflection, "iter_reflection_order1_epc_requests", lambda *a, **k: iter([request])
    )
    result, launches = reflection._reflection_topology_order1(*args, frequency_hz=3.0e9)
    assert launches == 1
    query = queries[0]
    assert query.source is epc_inputs["tx_batch"]
    assert query.receiver is epc_inputs["rx_batch"]
    assert query.expected_prim_ids is epc_inputs["sequence_batch"]
    assert query.max_bounces == 1 and query.visibility_ignore_mode == 1
    (block,) = result["blocks"]
    assert block["kwargs"]["tx_indices"] is epc_inputs["tx_indices"]
    assert block["kwargs"]["rx_indices"] is epc_inputs["rx_indices"]
    assert events == ["epc1", "compact", "ensure", "concat", "ensure"]


def test_order1_passes_every_transmitter_to_one_iterator(monkeypatch):
    events: list[str] = []
    args = _fixture(monkeypatch, events)
    seen = {}

    def iterator(plan, *, tx_positions, **kwargs):
        seen["tx"] = tx_positions
        return iter(())

    monkeypatch.setattr(reflection, "iter_reflection_order1_epc_requests", iterator)
    reflection._reflection_topology_order1(*args, frequency_hz=1.0)
    assert seen["tx"] is args[2]


def test_empty_compact_appends_no_block(monkeypatch):
    events: list[str] = []
    args = _fixture(monkeypatch, events, selected=False)
    monkeypatch.setattr(reflection, "query_reflection_epc", _epc_geometry(events))
    monkeypatch.setattr(
        reflection, "iter_reflection_order1_epc_requests", lambda *a, **k: iter([_request()[0]])
    )
    result, launches = reflection._reflection_topology_order1(*args, frequency_hz=1.0)
    assert launches == 1
    assert result["blocks"] == ()


def test_multibounce_counts_trace_and_epc_launches_and_candidates(monkeypatch):
    events: list[str] = []
    args = _fixture(monkeypatch, events)
    monkeypatch.setattr(reflection, "query_reflection_epc", _epc_geometry(events))
    monkeypatch.setattr(
        reflection,
        "_discovered_group_chains",
        lambda *a, **k: events.append("trace") or torch.zeros((1, 3)),
    )

    def iterator(plan, *, trace_group_chains, record_candidate_count, **kwargs):
        trace_group_chains(kwargs["tx_positions"], face_group_id=torch.tensor([0]), max_depth=3)
        record_candidate_count(5)
        yield _request(2)[0]
        record_candidate_count(7)
        yield _request(3)[0]

    monkeypatch.setattr(reflection, "iter_reflection_multibounce_epc_requests", iterator)
    result, launches, candidates = reflection._reflection_topology_multibounce(
        *args, frequency_hz=1.0, min_depth=2, max_depth=3, max_paths=None
    )
    assert launches == 3
    assert candidates == 12
    assert len(result["blocks"]) == 2
    assert events[:3] == ["trace", "epc2", "compact"]


def test_multibounce_trace_error_propagates(monkeypatch):
    events: list[str] = []
    args = _fixture(monkeypatch, events)
    monkeypatch.setattr(
        reflection,
        "_discovered_group_chains",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("trace")),
    )

    def iterator(plan, *, trace_group_chains, **kwargs):
        trace_group_chains(kwargs["tx_positions"], face_group_id=torch.tensor([0]), max_depth=2)
        yield _request(2)[0]

    monkeypatch.setattr(reflection, "iter_reflection_multibounce_epc_requests", iterator)
    with pytest.raises(RuntimeError, match="trace"):
        reflection._reflection_topology_multibounce(
            *args, frequency_hz=1.0, min_depth=2, max_depth=2, max_paths=None
        )
