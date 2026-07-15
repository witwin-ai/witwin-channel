from __future__ import annotations

from types import SimpleNamespace

import torch

from witwin.channel_native.propagation.enumerated import diffraction
from witwin.channel_native.propagation.fields.kernels import (
    deterministic as field_kernels,
)
from witwin.channel_native.propagation.geometry import (
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


def test_diffraction_lazy_event_field_and_export_order(monkeypatch):
    events: list[str] = []
    tx_positions = torch.zeros((1, 3))
    tx_power = torch.ones(1)
    rx_positions = torch.ones((2, 3))
    raw_states = _states()
    material = tuple(torch.ones(1) for _ in range(5))
    raydn = SimpleNamespace(
        available=True,
        require_handle=lambda: events.append("handle") or 41,
    )
    scene = SimpleNamespace(
        structures=[object()],
        metadata={"mitsuba": {"merge_shapes": True}},
    )
    compiled = SimpleNamespace(raydn=raydn)

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

    def fake_states(raydn_arg, tx, power, tx_index, *, preserve_imported_edges):
        events.append("states")
        assert raydn_arg is raydn
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

    def fake_visibility(raydn_arg, states, tx):
        events.append("visibility")
        assert raydn_arg is raydn
        assert states is raw_states
        assert tx.data_ptr() == tx_positions[0].data_ptr()
        return states

    monkeypatch.setattr(
        diffraction,
        "_tx_visible_diffraction_states",
        fake_visibility,
    )
    original_name_states = diffraction.name_diffraction_states

    def fake_name_states(states):
        events.append("name states")
        return original_name_states(states)

    monkeypatch.setattr(diffraction, "name_diffraction_states", fake_name_states)
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
        diffraction.topology_compaction,
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
        diffraction.topology_construction,
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
        "name states",
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
