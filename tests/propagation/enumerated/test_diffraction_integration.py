# Copyright Xingyu Chen.
# Tests diffraction integration.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.channel.interactions import diffraction
from witwin.channel.kernels import fields as field_kernels
# The discovery, geometry and enumerated halves of diffraction are one module
# now; the alias is kept so the geometry-facing assertions still read as such.
from witwin.channel.interactions import (
    diffraction as geometry_diffraction,
)


def _states() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([7], dtype=torch.int32),
        torch.zeros((1, 3)),
        torch.ones((1, 3)),
        torch.zeros(1),
        torch.ones(1),
        torch.zeros((1, 3)),
        torch.ones((1, 3)),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.ones(1),
        torch.zeros((1, 3)),
        torch.ones(1),
    )


def _states_with_rows(rows: int) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.expand((rows, *tensor.shape[1:])) for tensor in _states())


def _visible_plan(
    states: tuple[torch.Tensor, ...], active: torch.Tensor,
) -> geometry_diffraction.DiffractionVisibleStatePlan:
    named = geometry_diffraction.name_diffraction_states(states)
    return geometry_diffraction.DiffractionVisibleStatePlan(
        edge_index=named.edge_index,
        edge_position=named.edge_position,
        edge_direction=named.edge_direction,
        edge_t_min=named.edge_t_min,
        edge_t_max=named.edge_t_max,
        n0=named.n0,
        n1=named.n1,
        prim0=named.prim0,
        prim1=named.prim1,
        exterior_angle=named.exterior_angle,
        source=named.source,
        source_power=named.source_power,
        active=active,
    )


def _minimal_scene_and_compiled() -> tuple[object, object]:
    rayd = SimpleNamespace(available=True, require_resource=lambda: 41)
    scene = SimpleNamespace(
        structures=[object()],
        metadata={"mitsuba": {"merge_shapes": True}},
        transmitters=[SimpleNamespace(polarization=torch.tensor([0.0, 0.0, 1.0]))],
    )
    return scene, SimpleNamespace(rayd=rayd)


def _patch_empty_result_assembly(monkeypatch) -> None:
    monkeypatch.setattr(
        diffraction,
        "concatenate_path_blocks",
        lambda blocks, **_kwargs: {"blocks": tuple(blocks)},
    )
    monkeypatch.setattr(
        diffraction,
        "_ensure_topology_fields",
        lambda block, **kwargs: block | kwargs,
    )


def test_empty_states_call_native_planner_but_not_exporter(monkeypatch) -> None:
    scene, compiled = _minimal_scene_and_compiled()
    empty_states = _states_with_rows(0)
    planner_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        diffraction,
        "face_material_tensors",
        lambda *_args, **_kwargs: tuple(torch.ones(1) for _ in range(5)),
    )
    monkeypatch.setattr(
        diffraction,
        "_deterministic_diffraction_states",
        lambda *_args, **_kwargs: empty_states,
    )

    def plan(rayd, states, tx):
        planner_calls.append((rayd, states, tx))
        return _visible_plan(states, torch.empty(0, dtype=torch.bool))

    monkeypatch.setattr(diffraction, "plan_tx_visible_diffraction_states", plan)
    monkeypatch.setattr(
        diffraction,
        "query_diffraction_order1",
        lambda _query: pytest.fail("empty state capacity must not call exporter"),
    )
    _patch_empty_result_assembly(monkeypatch)

    block, launch_count, _field = diffraction._diffraction_topology_order1(
        scene,
        compiled,
        torch.zeros((1, 3)),
        torch.ones(1),
        torch.ones((2, 3)),
        frequency_hz=3.0e9,
    )

    assert len(planner_calls) == 1
    assert planner_calls[0][1] is empty_states
    assert launch_count == 0
    assert block["blocks"] == ()


@pytest.mark.parametrize(
    "active",
    (torch.tensor([False, False]), torch.tensor([True, False])),
    ids=("all-invisible", "sparse"),
)
def test_nonempty_mask_keeps_capacity_n_for_exporter(monkeypatch, active) -> None:
    scene, compiled = _minimal_scene_and_compiled()
    states = _states_with_rows(2)
    plan = _visible_plan(states, active)
    queries: list[geometry_diffraction.DiffractionOrder1Query] = []
    monkeypatch.setattr(
        diffraction,
        "face_material_tensors",
        lambda *_args, **_kwargs: tuple(torch.ones(1) for _ in range(5)),
    )
    monkeypatch.setattr(
        diffraction,
        "_deterministic_diffraction_states",
        lambda *_args, **_kwargs: states,
    )
    monkeypatch.setattr(
        diffraction,
        "plan_tx_visible_diffraction_states",
        lambda *_args: plan,
    )

    def export(query):
        queries.append(query)
        empty = torch.empty(0)
        return geometry_diffraction.DiffractionOrder1Geometry(
            valid=torch.empty(0, dtype=torch.bool),
            rx_id=torch.empty(0, dtype=torch.int32),
            depth=torch.empty(0, dtype=torch.int32),
            edge_id=torch.empty(0, dtype=torch.int32),
            delay_s=empty,
            x_re=empty,
            x_im=empty,
            y_re=empty,
            y_im=empty,
            z_re=empty,
            z_im=empty,
            interaction_position=torch.empty((0, 3)),
        )

    monkeypatch.setattr(diffraction, "query_diffraction_order1", export)
    monkeypatch.setattr(
        diffraction.topology_kernels,
        "deterministic_diffraction_order1_compact",
        lambda **_kwargs: {"rx_id": torch.empty(0, dtype=torch.int32)},
    )
    _patch_empty_result_assembly(monkeypatch)

    _block, launch_count, _field = diffraction._diffraction_topology_order1(
        scene,
        compiled,
        torch.zeros((1, 3)),
        torch.ones(1),
        torch.ones((2, 3)),
        frequency_hz=3.0e9,
    )

    assert launch_count == 1
    assert len(queries) == 1
    assert queries[0].active is plan.active
    assert queries[0].states is plan
    assert queries[0].state_count == 2
    assert queries[0].capacity == 4


def test_diffraction_lazy_event_field_and_export_order(monkeypatch):
    events: list[str] = []
    tx_positions = torch.zeros((1, 3))
    tx_power = torch.ones(1)
    rx_positions = torch.ones((2, 3))
    raw_states = _states()
    material = tuple(torch.ones(1) for _ in range(5))
    rayd = SimpleNamespace(
        available=True,
        require_resource=lambda: events.append("handle") or 41,
    )
    scene = SimpleNamespace(
        structures=[object()],
        metadata={"mitsuba": {"merge_shapes": True}},
        # order-1 diffraction now threads the real transmitter
        # polarization (transmitter_polarizations reads scene.transmitters).
        transmitters=[SimpleNamespace(polarization=torch.tensor([0.0, 0.0, 1.0]))],
    )
    compiled = SimpleNamespace(rayd=rayd)

    def fake_materials(*args, **kwargs):
        events.append("materials")
        return material

    monkeypatch.setattr(diffraction, "face_material_tensors", fake_materials)
    original_prepare = diffraction.prepare_diffraction_order1_plan

    def fake_prepare(**kwargs):
        events.append("prepare")
        return original_prepare(**kwargs)

    monkeypatch.setattr(diffraction, "prepare_diffraction_order1_plan", fake_prepare)
    original_tx_requests = diffraction.iter_diffraction_tx_requests

    def fake_tx_requests(*args, **kwargs):
        for request in original_tx_requests(*args, **kwargs):
            events.append("yield tx")
            yield request
            events.append("resume tx")

    monkeypatch.setattr(
        diffraction,
        "iter_diffraction_tx_requests",
        fake_tx_requests,
    )

    def fake_states(rayd_arg, tx, power, tx_index, *, preserve_imported_edges):
        events.append("states")
        assert rayd_arg is rayd
        assert tx.data_ptr() == tx_positions[0].data_ptr()
        assert power is tx_power
        assert tx_index == 0
        assert preserve_imported_edges is True
        return raw_states

    monkeypatch.setattr(
        diffraction,
        "_deterministic_diffraction_states",
        fake_states,
    )

    active = torch.ones(1, dtype=torch.bool)

    def fake_visibility_plan(rayd_arg, states, tx):
        events.append("visibility")
        assert rayd_arg is rayd
        assert states is raw_states
        assert tx.data_ptr() == tx_positions[0].data_ptr()
        named = geometry_diffraction.name_diffraction_states(states)
        return geometry_diffraction.DiffractionVisibleStatePlan(
            edge_index=named.edge_index,
            edge_position=named.edge_position,
            edge_direction=named.edge_direction,
            edge_t_min=named.edge_t_min,
            edge_t_max=named.edge_t_max,
            n0=named.n0,
            n1=named.n1,
            prim0=named.prim0,
            prim1=named.prim1,
            exterior_angle=named.exterior_angle,
            source=named.source,
            source_power=named.source_power,
            active=active,
        )

    monkeypatch.setattr(
        diffraction,
        "plan_tx_visible_diffraction_states",
        fake_visibility_plan,
    )
    original_rx_requests = diffraction.iter_diffraction_rx_chunk_requests

    def fake_rx_requests(*args, **kwargs):
        for request in original_rx_requests(*args, **kwargs):
            events.append("yield rx")
            yield request
            events.append("resume rx")

    monkeypatch.setattr(
        diffraction,
        "iter_diffraction_rx_chunk_requests",
        fake_rx_requests,
    )
    raw_geometry = geometry_diffraction.DiffractionOrder1Geometry(
        valid=torch.ones(1, dtype=torch.bool),
        rx_id=torch.zeros(1, dtype=torch.int32),
        depth=torch.ones(1, dtype=torch.int32),
        edge_id=torch.tensor([7], dtype=torch.int32),
        delay_s=torch.tensor([0.1]),
        x_re=torch.tensor([1.0]),
        x_im=torch.tensor([0.0]),
        y_re=torch.tensor([2.0]),
        y_im=torch.tensor([0.0]),
        z_re=torch.tensor([3.0]),
        z_im=torch.tensor([0.0]),
        interaction_position=torch.zeros((1, 3)),
    )

    def fake_geometry(query):
        events.append("geometry")
        assert isinstance(query, geometry_diffraction.DiffractionOrder1Query)
        assert query.handle == 41
        assert query.tx_position.is_contiguous()
        assert query.rx_positions.data_ptr() == rx_positions.data_ptr()
        assert query.rx_positions.stride() == rx_positions.stride()
        assert query.active is active
        assert query.states.edge_index is raw_states[0]
        assert query.material_eta_r is material[0]
        assert query.material_sigma is material[1]
        assert query.material_mu_r is material[2]
        assert query.material_gain is material[3]
        assert query.material_valid is material[4]
        assert query.state_count == 1
        assert query.capacity == 2
        return raw_geometry

    monkeypatch.setattr(diffraction, "query_diffraction_order1", fake_geometry)
    compacted = {
        "rx_id": raw_geometry.rx_id,
        "depth": raw_geometry.depth,
        "edge_id": raw_geometry.edge_id,
        "delay_s": raw_geometry.delay_s,
        "x_re": raw_geometry.x_re,
        "x_im": raw_geometry.x_im,
        "y_re": raw_geometry.y_re,
        "y_im": raw_geometry.y_im,
        "z_re": raw_geometry.z_re,
        "z_im": raw_geometry.z_im,
        "interaction_position": raw_geometry.interaction_position,
    }

    def fake_compact(**kwargs):
        events.append("compact")
        assert kwargs["valid"] is raw_geometry.valid
        assert kwargs["rx_id"] is raw_geometry.rx_id
        assert kwargs["interaction_position"] is raw_geometry.interaction_position
        return compacted

    monkeypatch.setattr(
        diffraction.topology_kernels,
        "deterministic_diffraction_order1_compact",
        fake_compact,
    )

    def fake_field(**kwargs):
        events.append("field")
        assert kwargs["x_re"] is compacted["x_re"]
        return {
            "path_gain": torch.tensor([14.0]),
            "field_real": torch.tensor([4.0]),
            "field_imag": torch.tensor([5.0]),
        }

    monkeypatch.setattr(
        field_kernels,
        "deterministic_diffraction_vector_field",
        fake_field,
    )
    monkeypatch.setattr(
        field_kernels,
        "deterministic_pack_complex",
        lambda *args: events.append("pack") or torch.complex(args[0], args[1]),
    )
    monkeypatch.setattr(
        field_kernels,
        "deterministic_delay_to_path_length",
        lambda delay: events.append("delay") or delay * 300.0,
    )

    def fake_base(**kwargs):
        events.append("base")
        assert kwargs["rx_id"] is compacted["rx_id"]
        assert kwargs["path_gain"].item() == 14.0
        return {"valid": torch.ones(1, dtype=torch.bool)}

    monkeypatch.setattr(
        diffraction.topology_kernels,
        "deterministic_topology_base_fields",
        fake_base,
    )

    def fake_ensure(block, **kwargs):
        events.append("ensure/block" if kwargs else "final ensure")
        return block | kwargs

    monkeypatch.setattr(diffraction, "_ensure_topology_fields", fake_ensure)
    monkeypatch.setattr(
        diffraction,
        "concatenate_path_blocks",
        lambda blocks, **kwargs: (
            events.append("concat") or {"blocks": tuple(blocks)}
        ),
    )

    block, launch_count, vector_field = diffraction._diffraction_topology_order1(
        scene,
        compiled,
        tx_positions,
        tx_power,
        rx_positions,
        frequency_hz=3.0e9,
    )

    assert launch_count == 1
    assert len(block["blocks"]) == 1
    torch.testing.assert_close(
        vector_field[0, 0],
        torch.tensor([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
    )
    torch.testing.assert_close(vector_field[0, 1], torch.zeros(3, dtype=torch.complex64))
    assert events == [
        "materials",
        "handle",
        "prepare",
        "yield tx",
        "states",
        "visibility",
        "yield rx",
        "geometry",
        "compact",
        "field",
        "pack",
        "delay",
        "base",
        "ensure/block",
        "resume rx",
        "resume tx",
        "concat",
        "final ensure",
    ]