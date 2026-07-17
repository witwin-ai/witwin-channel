from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from witwin.channel_native.propagation.geometry import diffraction


class _Raydn:
    def __init__(self):
        self.records = object()
        self.edge_record_calls = 0
        self.handle_calls = 0

    def edge_records(self):
        self.edge_record_calls += 1
        return self.records

    def require_handle(self):
        self.handle_calls += 1
        return 29


def _edge_raw() -> tuple[torch.Tensor, ...]:
    raw = [torch.tensor([index], dtype=torch.float32) for index in range(11)]
    raw[2] = torch.arange(6, dtype=torch.float32).reshape(3, 2).t()
    return tuple(raw)


def _state_raw() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([0], dtype=torch.int32),
        torch.zeros((1, 3)),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.zeros((1, 3)),
        torch.ones((1, 3)),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1.5]),
        torch.zeros((1, 3)),
        torch.ones(1),
    )


def test_edge_query_names_cached_and_imported_raw_geometry_without_copies(
    monkeypatch,
):
    raydn = _Raydn()
    raw = _edge_raw()
    calls = []

    def fake_cached(actual_raydn):
        calls.append(("cached", actual_raydn))
        return raw

    def fake_imported(records):
        calls.append(("imported", records))
        return raw

    monkeypatch.setattr(
        diffraction,
        "_cached_diffraction_edge_geometry",
        fake_cached,
    )
    monkeypatch.setattr(diffraction, "_diffraction_edge_geometry", fake_imported)

    cached = diffraction.query_diffraction_edges(
        raydn,
        preserve_imported_edges=False,
    )
    imported = diffraction.query_diffraction_edges(
        raydn,
        preserve_imported_edges=True,
    )

    assert calls == [("cached", raydn), ("imported", raydn.records)]
    assert raydn.edge_record_calls == 1
    assert [field.name for field in fields(cached)] == [
        "selected",
        "edge_position",
        "edge_direction",
        "edge_length",
        "line_min",
        "line_max",
        "n0",
        "n1",
        "face0",
        "face1",
        "exterior_angle",
    ]
    assert cached.edge_direction is raw[2]
    assert imported.edge_direction is raw[2]
    assert cached.edge_direction.stride() == raw[2].stride()


def test_state_geometry_names_raw_pack_by_identity():
    raw = _state_raw()

    states = diffraction.name_diffraction_states(raw)

    assert [field.name for field in fields(states)] == [
        "edge_index",
        "edge_position",
        "edge_direction",
        "edge_t_min",
        "edge_t_max",
        "n0",
        "n1",
        "prim0",
        "prim1",
        "exterior_angle",
        "source",
        "source_power",
    ]
    assert all(
        value is raw[index]
        for index, value in enumerate(
            (
                states.edge_index,
                states.edge_position,
                states.edge_direction,
                states.edge_t_min,
                states.edge_t_max,
                states.n0,
                states.n1,
                states.prim0,
                states.prim1,
                states.exterior_angle,
                states.source,
                states.source_power,
            )
        )
    )


def test_tx_visibility_uses_all_prefilter_fractions_in_order(monkeypatch):
    raydn = _Raydn()
    states = _state_raw()
    sampled_x = []

    def fake_visibility(handle, starts, points, active):
        assert handle == 29
        assert starts.is_contiguous()
        assert points.is_contiguous()
        assert active is None
        sampled_x.append(float(points[0, 0]))
        return (torch.ones(1, dtype=torch.bool),)

    monkeypatch.setattr(
        diffraction.geometry_bridge,
        "raydn_visibility_forward",
        fake_visibility,
    )

    result = diffraction._tx_visible_diffraction_states(
        raydn,
        states,
        torch.zeros(3),
    )

    assert result is states
    assert raydn.handle_calls == 4
    assert sampled_x == pytest.approx(
        [0.02, 1.0 / 3.0, 2.0 / 3.0, 0.98]
    )


def test_order1_query_consumes_raw_native_tuple_and_preserves_state_identity(
    monkeypatch,
):
    states = diffraction.name_diffraction_states(_state_raw())
    raw = tuple(torch.tensor([float(index)]) for index in range(18))
    material = tuple(torch.tensor([float(index)]) for index in range(5))
    rx_positions = torch.arange(6, dtype=torch.float32).reshape(3, 2).t()
    calls = []

    def fake_forward(*args):
        calls.append(args)
        return raw

    monkeypatch.setattr(
        diffraction.geometry_bridge,
        "raydn_diffraction_paths_order1_forward",
        fake_forward,
    )
    tx_polarization = torch.tensor([[0.0, 0.0, 1.0]])
    query = diffraction.DiffractionOrder1Query(
        handle=31,
        tx_position=torch.zeros((1, 3)),
        tx_polarization=tx_polarization,
        rx_positions=rx_positions,
        active=None,
        states=states,
        material_eta_r=material[0],
        material_sigma=material[1],
        material_mu_r=material[2],
        material_gain=material[3],
        material_valid=material[4],
        state_count=1,
        capacity=2,
        wavelength=0.1,
    )

    result = diffraction.query_diffraction_order1(query)

    assert len(calls) == 1
    # tx_polarization is threaded in as argument index 2 (R5 fix), shifting the
    # state and receiver arguments down by one slot.
    assert calls[0][2] is tx_polarization
    assert calls[0][5] is states.edge_index
    assert calls[0][6] is states.edge_position
    assert calls[0][16] is states.source_power
    assert calls[0][3] is rx_positions
    assert calls[0][3].stride() == rx_positions.stride()
    assert result.valid is raw[1]
    assert result.rx_id is raw[3]
    assert result.depth is raw[4]
    assert result.edge_id is raw[5]
    assert result.delay_s is raw[8]
    assert result.x_re is raw[9]
    assert result.z_im is raw[14]
    assert result.interaction_position is raw[15]
    with pytest.raises(FrozenInstanceError):
        result.valid = torch.ones_like(result.valid)
