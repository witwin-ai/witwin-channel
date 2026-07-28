from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.channel.interactions import reflection
from witwin.channel.kernels import fields as field_kernels
from witwin.channel.interactions import (
    reflection as geometry_reflection,
)
from witwin.channel.interactions.reflection import (
    ReflectionMultibounceEpcRequest,
    ReflectionOrder1EpcRequest,
)


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
    compiled = SimpleNamespace(
        rayd=rayd,
        assignments=SimpleNamespace(face_material_id=torch.tensor([0])),
        geometry=SimpleNamespace(face_surface_id=torch.tensor([0])),
    )
    scene = SimpleNamespace(structures=[object()])
    tx_positions = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tx_power = torch.tensor([2.0, 3.0], dtype=torch.float64)
    rx_positions = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tri_a = torch.zeros((1, 3))

    monkeypatch.setattr(
        reflection.geometry_kernels,
        "deterministic_normalize_vec3",
        lambda value, eps: normals,
    )
    monkeypatch.setattr(
        reflection.topology_kernels,
        "deterministic_face_anchor_points",
        lambda *args: tri_a,
    )
    material = tuple(torch.ones(1) for _ in range(5))
    monkeypatch.setattr(reflection, "face_material_tensors", lambda *a, **k: material)
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

    chosen = (
        torch.tensor([0], dtype=torch.int32)
        if selected
        else torch.empty(0, dtype=torch.int32)
    )
    selected_rows = {
        "selected_faces": chosen,
        "tx_keep": torch.zeros((1, 3)),
        "rx_keep": torch.ones((1, 3)),
        "selected_points": torch.zeros((1, 3)),
        "selected_normals": torch.ones((1, 3)),
        "tx_power": torch.ones(1),
        "eps_r": torch.ones(1),
        "sigma_e": torch.ones(1),
        "mu_r": torch.ones(1),
        "gain": torch.ones(1),
        "material_id": torch.zeros(1, dtype=torch.int32),
        "selected_rx_id": torch.zeros(1, dtype=torch.int32),
    }
    epc = (
        torch.ones(1, dtype=torch.bool),
        None,
        chosen,
        None,
        torch.zeros((1, 3)),
        torch.ones((1, 3)),
    )
    monkeypatch.setattr(
        reflection.geometry_kernels,
        "rayd_reflection_epc_paths_forward",
        lambda *a: events.append("epc") or epc,
    )

    def compact(**kwargs):
        events.append("compact")
        assert kwargs["tx_power"].dtype == torch.float32
        assert kwargs["tx_power"].is_contiguous()
        return selected_rows

    monkeypatch.setattr(
        reflection.topology_kernels,
        "deterministic_reflection_order1_compact",
        compact,
    )

    def forbidden(name):
        def call(*args, **kwargs):
            raise AssertionError(name)

        return call

    if selected:
        monkeypatch.setattr(
            field_kernels,
            "deterministic_reflection_field",
            lambda **k: (
                events.append("field")
                or {
                    "path_gain": torch.ones(1),
                    "field_real": torch.ones(1),
                    "field_imag": torch.zeros(1),
                    "path_length_m": torch.ones(1),
                    "delay_s": torch.ones(1),
                }
            ),
        )
        monkeypatch.setattr(
            field_kernels,
            "deterministic_pack_complex",
            lambda *a: events.append("pack") or torch.ones(1, dtype=torch.complex64),
        )

        def base(**kwargs):
            events.append("base_fields")
            assert kwargs["rx_id"] is selected_rows["selected_rx_id"]
            return {"valid": torch.ones(1, dtype=torch.bool)}

        monkeypatch.setattr(
            reflection.topology_kernels,
            "deterministic_topology_base_fields",
            base,
        )
    else:
        monkeypatch.setattr(
            field_kernels, "deterministic_reflection_field", forbidden("field")
        )
        monkeypatch.setattr(field_kernels, "deterministic_pack_complex", forbidden("pack"))
        monkeypatch.setattr(
            reflection.topology_kernels,
            "deterministic_topology_base_fields",
            forbidden("base_fields"),
        )

    def ensure(block, **kwargs):
        events.append("ensure/block" if kwargs else "final ensure")
        return block | kwargs

    monkeypatch.setattr(reflection, "_ensure_topology_fields", ensure)
    monkeypatch.setattr(
        reflection,
        "concatenate_path_blocks",
        lambda blocks, **k: events.append("concat") or {"blocks": tuple(blocks)},
    )
    return scene, compiled, tx_positions, tx_power, rx_positions


def _request(tx_index: int, tx: torch.Tensor):
    token = object()
    epc_inputs = {
        "tx_batch": token,
        "rx_batch": object(),
        "sequence_batch": object(),
        "direct_plane_points": object(),
        "direct_plane_normals": object(),
        "rx_indices": object(),
    }
    return ReflectionOrder1EpcRequest(tx_index, tx, epc_inputs), epc_inputs


def test_order1_maps_lazy_request_to_typed_query_without_copies(monkeypatch):
    events: list[str] = []
    args = _fixture(monkeypatch, events)
    surface_group_id = torch.tensor([0], dtype=torch.int32)
    surface_group_size = torch.tensor([1], dtype=torch.int32)
    surface_group_members = torch.arange(6, dtype=torch.int32).reshape(3, 2).t()
    monkeypatch.setattr(
        reflection,
        "_cached_coplanar_face_groups",
        lambda *a, **k: {
            "group_count": 1,
            "representative_faces": torch.tensor([0]),
            "face_group_id": torch.tensor([0]),
            "surface_group_id": surface_group_id,
            "surface_group_size": surface_group_size,
            "surface_group_members": surface_group_members,
        },
    )
    monkeypatch.setattr(
        reflection,
        "prepare_reflection_order1_plan",
        lambda **k: object(),
    )
    request, epc_inputs = _request(0, args[2][0])
    monkeypatch.setattr(
        reflection,
        "iter_reflection_order1_epc_requests",
        lambda *a, **k: iter([request]),
    )
    queries = []

    def fake_query(query):
        events.append("epc")
        queries.append(query)
        return geometry_reflection.ReflectionEpcGeometry(
            visible=torch.ones(1, dtype=torch.bool),
            path_length_m=torch.ones(1),
            resolved_prim_ids=torch.zeros(1, dtype=torch.int32),
            surface_group_ids=torch.zeros(1, dtype=torch.int32),
            hit_positions=torch.zeros((1, 3)),
            normals=torch.ones((1, 3)),
        )

    monkeypatch.setattr(reflection, "query_reflection_epc", fake_query)

    _, launches = reflection._reflection_topology_order1(
        *args,
        frequency_hz=3.0e9,
    )

    assert launches == 1
    assert len(queries) == 1
    query = queries[0]
    assert isinstance(query, geometry_reflection.ReflectionEpcQuery)
    assert query.source is epc_inputs["tx_batch"]
    assert query.receiver is epc_inputs["rx_batch"]
    assert query.expected_prim_ids is epc_inputs["sequence_batch"]
    assert query.direct_plane_points is epc_inputs["direct_plane_points"]
    assert query.direct_plane_normals is epc_inputs["direct_plane_normals"]
    assert query.surface_group_id is surface_group_id
    assert query.surface_group_size is surface_group_size
    assert query.surface_group_members is surface_group_members
    assert query.surface_group_members.stride() == surface_group_members.stride()
    assert query.max_bounces == 1
    assert query.visibility_ignore_mode == 1


def test_two_lazy_requests_resume_only_after_each_export(monkeypatch):
    events: list[str] = []
    args = _fixture(monkeypatch, events)
    monkeypatch.setattr(
        reflection,
        "prepare_reflection_order1_plan",
        lambda **k: events.append("prepare") or object(),
    )
    requests = [_request(0, args[2][0]), _request(1, args[2][1])]

    def iterator(*a, **k):
        events.append("iterator yield1")
        yield requests[0][0]
        events.append("iterator resume/yield2")
        yield requests[1][0]
        events.append("iterator resume/end")

    monkeypatch.setattr(reflection, "iter_reflection_order1_epc_requests", iterator)
    _, launches = reflection._reflection_topology_order1(*args, frequency_hz=3.0e9)
    assert launches == 2
    assert events == [
        "prepare",
        "iterator yield1",
        "epc",
        "compact",
        "field",
        "pack",
        "base_fields",
        "ensure/block",
        "iterator resume/yield2",
        "epc",
        "compact",
        "field",
        "pack",
        "base_fields",
        "ensure/block",
        "iterator resume/end",
        "concat",
        "final ensure",
    ]


def test_selected_empty_short_circuits_without_placeholder_block(monkeypatch):
    events: list[str] = []
    args = _fixture(monkeypatch, events, selected=False)
    monkeypatch.setattr(
        reflection, "prepare_reflection_order1_plan", lambda **k: object()
    )
    request, _ = _request(0, args[2][0])
    monkeypatch.setattr(
        reflection,
        "iter_reflection_order1_epc_requests",
        lambda *a, **k: iter([request]),
    )
    result, launches = reflection._reflection_topology_order1(*args, frequency_hz=1.0)
    assert launches == 1
    assert result["blocks"] == ()
    assert events == ["epc", "compact", "concat", "final ensure"]


def test_trace_callback_counts_only_success_and_propagates_error(monkeypatch):
    events: list[str] = []
    args = _fixture(monkeypatch, events, selected=False)
    monkeypatch.setattr(
        reflection, "prepare_reflection_order1_plan", lambda **k: object()
    )

    def iterator(plan, *, trace_group_chains, **kwargs):
        trace_group_chains(args[2][0], face_group_id=torch.tensor([0]), max_depth=1)
        return iter(())

    monkeypatch.setattr(reflection, "iter_reflection_order1_epc_requests", iterator)
    monkeypatch.setattr(
        reflection, "_discovered_group_chains", lambda *a, **k: torch.zeros((1, 1))
    )
    _, launches = reflection._reflection_topology_order1(*args, frequency_hz=1.0)
    assert launches == 1

    monkeypatch.setattr(
        reflection,
        "_discovered_group_chains",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("trace")),
    )
    with pytest.raises(RuntimeError, match="trace"):
        reflection._reflection_topology_order1(*args, frequency_hz=1.0)


def _patch_multibounce_selected(monkeypatch, events, *, selected=True):
    count = 1 if selected else 0
    chosen = torch.zeros((count, 2), dtype=torch.int32)
    rows = {
        "selected_sequences": chosen,
        "selected_tx": torch.zeros((count, 3)),
        "selected_rx": torch.ones((count, 3)),
        "selected_hits": torch.zeros((count, 2, 3)),
        "selected_normals": torch.ones((count, 2, 3)),
        "tx_power": torch.ones(count),
        "eps_r": torch.ones((count, 2)),
        "sigma_e": torch.ones((count, 2)),
        "mu_r": torch.ones((count, 2)),
        "gain": torch.ones((count, 2)),
        "selected_rx_id": torch.zeros(count, dtype=torch.int32),
        "first_face": torch.zeros(count, dtype=torch.int32),
        "first_hit": torch.zeros((count, 3)),
        "first_normal": torch.ones((count, 3)),
        "material_id": torch.zeros(count, dtype=torch.int32),
        "material_sequence": chosen,
    }
    monkeypatch.setattr(
        reflection.topology_kernels,
        "deterministic_reflection_sequence_compact",
        lambda **k: events.append("compact") or rows,
    )
    monkeypatch.setattr(
        reflection.topology_kernels,
        "deterministic_topology_base_fields",
        lambda **k: (
            events.append("base_fields")
            or {"valid": torch.ones(count, dtype=torch.bool)}
        ),
    )
    if selected:
        monkeypatch.setattr(
            field_kernels,
            "deterministic_reflection_sequence_field",
            lambda **k: (
                events.append("field")
                or {
                    "path_gain": torch.ones(1),
                    "field_real": torch.ones(1),
                    "field_imag": torch.zeros(1),
                    "path_length_m": torch.ones(1),
                    "delay_s": torch.ones(1),
                }
            ),
        )


def _multibounce_request(depth, tx_index, tx):
    request, epc_inputs = _request(tx_index, tx)
    return ReflectionMultibounceEpcRequest(depth, tx_index, tx, epc_inputs), epc_inputs


def test_multibounce_lazy_mixed_callbacks_and_final_counts(monkeypatch):
    events = []
    args = _fixture(monkeypatch, events)
    _patch_multibounce_selected(monkeypatch, events)
    monkeypatch.setattr(
        reflection,
        "prepare_reflection_multibounce_plan",
        lambda **k: events.append("prepare") or object(),
    )
    requests = [
        _multibounce_request(2, 0, args[2][0]),
        _multibounce_request(3, 1, args[2][1]),
    ]
    monkeypatch.setattr(
        reflection,
        "_discovered_group_chains",
        lambda *a, **k: events.append("trace") or torch.zeros((1, 3)),
    )

    def iterator(plan, *, trace_group_chains, record_candidate_count, **kwargs):
        trace_group_chains(args[2][0], face_group_id=torch.tensor([0]), max_depth=3)
        record_candidate_count(5)
        events.append("yield1")
        yield requests[0][0]
        events.append("resume/yield2")
        yield requests[1][0]
        events.append("resume/end")

    monkeypatch.setattr(
        reflection, "iter_reflection_multibounce_epc_requests", iterator
    )
    result = reflection._reflection_topology_multibounce(
        *args, min_depth=2, max_depth=3, frequency_hz=1.0, max_paths=None
    )
    assert result[1:] == (3, 5)
    assert len(result[0]["blocks"]) == 2
    assert events.index("resume/yield2") > events.index("ensure/block")
    assert events[-3:] == ["resume/end", "concat", "final ensure"]


def test_multibounce_selected_empty_and_candidate_only_no_yield(monkeypatch):
    events = []
    args = _fixture(monkeypatch, events, selected=False)
    _patch_multibounce_selected(monkeypatch, events, selected=False)
    monkeypatch.setattr(
        reflection, "prepare_reflection_multibounce_plan", lambda **k: object()
    )
    request, _ = _multibounce_request(2, 0, args[2][0])

    def iterator(plan, *, record_candidate_count, **kwargs):
        record_candidate_count(7)
        yield request

    monkeypatch.setattr(
        reflection, "iter_reflection_multibounce_epc_requests", iterator
    )
    result = reflection._reflection_topology_multibounce(
        *args, min_depth=2, max_depth=2, frequency_hz=1.0, max_paths=None
    )
    assert result[1:] == (1, 7)
    assert result[0]["blocks"] == ()
    assert events == ["epc", "compact", "concat", "final ensure"]


def test_multibounce_trace_only_counts_success_and_propagates(monkeypatch):
    events = []
    args = _fixture(monkeypatch, events, selected=False)
    monkeypatch.setattr(
        reflection, "prepare_reflection_multibounce_plan", lambda **k: object()
    )

    def iterator(plan, *, trace_group_chains, **kwargs):
        trace_group_chains(args[2][0], face_group_id=torch.tensor([0]), max_depth=2)
        return iter(())

    monkeypatch.setattr(
        reflection, "iter_reflection_multibounce_epc_requests", iterator
    )
    monkeypatch.setattr(
        reflection, "_discovered_group_chains", lambda *a, **k: torch.zeros((1, 2))
    )
    result = reflection._reflection_topology_multibounce(
        *args, min_depth=2, max_depth=2, frequency_hz=1.0, max_paths=None
    )
    assert result[1:] == (1, 0)
    monkeypatch.setattr(
        reflection,
        "_discovered_group_chains",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("trace")),
    )
    with pytest.raises(RuntimeError, match="trace"):
        reflection._reflection_topology_multibounce(
            *args, min_depth=2, max_depth=2, frequency_hz=1.0, max_paths=None
        )
