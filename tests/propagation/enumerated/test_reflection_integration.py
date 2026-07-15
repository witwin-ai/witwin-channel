from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.channel_native.propagation.enumerated import reflection
from witwin.channel_native.propagation.fields.kernels import deterministic as fields
from witwin.channel_native.propagation.topology.discovery.reflection import (
    ReflectionOrder1EpcRequest,
)


def _fixture(monkeypatch, events: list[str], *, selected: bool = True):
    vertices = torch.zeros((3, 3))
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    normals = torch.ones((1, 3))
    records = SimpleNamespace(vertices=vertices, faces=faces, face_normals=normals)
    raydn = SimpleNamespace(
        available=True,
        edge_records=lambda: records,
        require_handle=lambda: object(),
    )
    compiled = SimpleNamespace(
        raydn=raydn,
        assignments=SimpleNamespace(face_material_id=torch.tensor([0])),
        geometry=SimpleNamespace(face_surface_id=torch.tensor([0])),
    )
    scene = SimpleNamespace(structures=[object()])
    tx_positions = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tx_power = torch.tensor([2.0, 3.0], dtype=torch.float64)
    rx_positions = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tri_a = torch.zeros((1, 3))

    monkeypatch.setattr(
        reflection.geometry_primitives,
        "deterministic_normalize_vec3",
        lambda value, eps: normals,
    )
    monkeypatch.setattr(
        reflection.topology_construction,
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
        reflection.geometry_bridge,
        "raydn_reflection_epc_paths_forward",
        lambda *a: events.append("epc") or epc,
    )

    def compact(**kwargs):
        events.append("compact")
        assert kwargs["tx_power"].dtype == torch.float32
        assert kwargs["tx_power"].is_contiguous()
        return selected_rows

    monkeypatch.setattr(
        reflection.topology_compaction,
        "deterministic_reflection_order1_compact",
        compact,
    )

    def forbidden(name):
        def call(*args, **kwargs):
            raise AssertionError(name)

        return call

    if selected:
        monkeypatch.setattr(
            fields,
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
            fields,
            "deterministic_pack_complex",
            lambda *a: events.append("pack") or torch.ones(1, dtype=torch.complex64),
        )

        def base(**kwargs):
            events.append("base_fields")
            assert kwargs["rx_id"] is selected_rows["selected_rx_id"]
            return {"valid": torch.ones(1, dtype=torch.bool)}

        monkeypatch.setattr(
            reflection.topology_construction,
            "deterministic_topology_base_fields",
            base,
        )
    else:
        monkeypatch.setattr(
            fields, "deterministic_reflection_field", forbidden("field")
        )
        monkeypatch.setattr(fields, "deterministic_pack_complex", forbidden("pack"))
        monkeypatch.setattr(
            reflection.topology_construction,
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
