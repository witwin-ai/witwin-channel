# Copyright Xingyu Chen.
# Tests diffraction integration.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.channel.interactions import diffraction


def _states(rows: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.full((rows,), 7, dtype=torch.int32),
        torch.zeros((rows, 3)),
        torch.ones((rows, 3)),
        torch.zeros(rows),
        torch.ones(rows),
        torch.zeros((rows, 3)),
        torch.ones((rows, 3)),
        torch.zeros(rows, dtype=torch.int32),
        torch.ones(rows, dtype=torch.int32),
        torch.ones(rows),
        torch.zeros((rows, 3)),
        torch.ones(rows),
    )


def _visible_plan(states, active):
    named = diffraction.name_diffraction_states(states)
    return diffraction.DiffractionVisibleStatePlan(
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


def _export(query):
    rows = int(query.capacity)
    rx_count = int(query.rx_positions.shape[0])
    lane = torch.arange(rows, dtype=torch.int32)
    state = lane % query.state_count
    return diffraction.DiffractionOrder1Geometry(
        valid=torch.ones(rows, dtype=torch.bool),
        tx_id=query.state_source_index[state.long()],
        rx_id=(lane // query.state_count) % rx_count,
        depth=torch.ones(rows, dtype=torch.int32),
        edge_id=torch.full((rows,), 7, dtype=torch.int32),
        delay_s=torch.ones(rows),
        x_re=torch.ones(rows),
        x_im=torch.zeros(rows),
        y_re=torch.zeros(rows),
        y_im=torch.zeros(rows),
        z_re=torch.zeros(rows),
        z_im=torch.zeros(rows),
        interaction_position=torch.zeros((rows, 3)),
    )


def _fixture(monkeypatch, events, *, states_per_tx: int, tx_count: int = 2):
    rayd = SimpleNamespace(available=True, require_resource=lambda: 41)
    scene = SimpleNamespace(
        structures=[object()],
        metadata={"mitsuba": {"merge_shapes": True}},
        transmitters=[SimpleNamespace(polarization=torch.tensor([0.0, 0.0, 1.0]))] * tx_count,
    )
    compiled = SimpleNamespace(rayd=rayd)
    monkeypatch.setattr(
        diffraction, "face_material_tensors", lambda *a, **k: tuple(torch.ones(1) for _ in range(5))
    )
    monkeypatch.setattr(diffraction, "selected_diffraction_edges", lambda *a, **k: (None, None))
    monkeypatch.setattr(
        diffraction,
        "transmitter_polarizations_f32",
        lambda *a, **k: torch.tensor([[0.0, 0.0, 1.0]]).repeat(tx_count, 1),
    )
    total = states_per_tx * tx_count
    states = _states(total)
    source_index = torch.arange(tx_count, dtype=torch.int32).repeat_interleave(states_per_tx)

    def batched(edges, tx_positions, tx_power):
        events.append(f"pack{int(tx_positions.shape[0])}")
        return states, source_index

    monkeypatch.setattr(diffraction, "_batched_diffraction_states", batched)

    def plan(rayd_arg, states_arg, tx, *, per_state_source=False):
        events.append(f"visibility:{per_state_source}")
        assert states_arg is states
        return _visible_plan(states_arg, torch.ones(total, dtype=torch.bool))

    monkeypatch.setattr(diffraction, "plan_tx_visible_diffraction_states", plan)
    queries = []

    def export(query):
        events.append("export")
        queries.append(query)
        assert query.state_source_index is source_index
        assert query.tx_position.shape == (tx_count, 3)
        return _export(query)

    monkeypatch.setattr(diffraction, "query_diffraction_order1", export)
    compacts = []
    native_compact = diffraction.topology_kernels.deterministic_diffraction_topology_compact

    def compact(**kwargs):
        events.append("compact")
        compacts.append(kwargs)
        count = int(kwargs["valid"].sum())
        return {
            "valid": torch.ones(count, dtype=torch.bool),
            "tx_id": kwargs["tx_id"][kwargs["valid"]],
            "rx_id": kwargs["rx_id"][kwargs["valid"]] + kwargs["rx_offset"],
            "field_xyz": torch.ones((count, 3), dtype=torch.complex64),
        }

    assert callable(native_compact)
    monkeypatch.setattr(diffraction.topology_kernels, "deterministic_diffraction_topology_compact", compact)
    monkeypatch.setattr(diffraction, "_ensure_topology_fields", lambda block, **k: block)
    monkeypatch.setattr(
        diffraction, "concatenate_path_blocks", lambda blocks, **k: {"blocks": tuple(blocks)}
    )
    return scene, compiled, queries, compacts


def test_every_transmitter_is_packed_planned_and_exported_in_one_call_per_stage(monkeypatch):
    events: list[str] = []
    scene, compiled, queries, compacts = _fixture(monkeypatch, events, states_per_tx=3)
    block, launches, vector_field = diffraction._diffraction_topology_order1(
        scene, compiled, torch.zeros((2, 3)), torch.ones(2), torch.ones((4, 3)), frequency_hz=3.0e9
    )
    assert events == ["pack2", "visibility:True", "export", "compact"]
    assert launches == 1
    assert queries[0].state_count == 6 and queries[0].capacity == 24
    (rows,) = block["blocks"]
    assert sorted(set(rows["tx_id"].tolist())) == [0, 1]
    # Every (tx, rx) pair received three unit vector-field contributions.
    torch.testing.assert_close(
        vector_field[:, :, 0], torch.full((2, 4), 3.0, dtype=torch.complex64)
    )


def test_receiver_chunks_shift_receiver_ids(monkeypatch):
    monkeypatch.setattr(diffraction, "_MULTIBOUNCE_PAIR_CHUNK_SIZE", 12)
    events: list[str] = []
    scene, compiled, queries, compacts = _fixture(monkeypatch, events, states_per_tx=3)
    block, launches, _ = diffraction._diffraction_topology_order1(
        scene, compiled, torch.zeros((2, 3)), torch.ones(2), torch.ones((4, 3)), frequency_hz=3.0e9
    )
    assert launches == 2
    assert [call["rx_offset"] for call in compacts] == [0, 2]
    assert [int(query.rx_positions.shape[0]) for query in queries] == [2, 2]
    assert sorted(set(torch.cat([rows["rx_id"] for rows in block["blocks"]]).tolist())) == [0, 1, 2, 3]


def test_no_visible_state_capacity_calls_no_exporter(monkeypatch):
    events: list[str] = []
    scene, compiled, _queries, _compacts = _fixture(monkeypatch, events, states_per_tx=0)
    monkeypatch.setattr(
        diffraction,
        "query_diffraction_order1",
        lambda _query: pytest.fail("empty state capacity must not call the exporter"),
    )
    block, launches, _ = diffraction._diffraction_topology_order1(
        scene, compiled, torch.zeros((2, 3)), torch.ones(2), torch.ones((4, 3)), frequency_hz=3.0e9
    )
    assert launches == 0
    assert block["blocks"] == ()
