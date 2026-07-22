from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.channel.core.objects import ReceiverGrid
from witwin.channel.deterministic import accumulation, solver
from witwin.channel.propagation.enumerated import scattering as enumerated
from witwin.channel.propagation.geometry import endpoints


_ENDPOINT_NAMES = (
    "ReceiverLayout",
    "transmitter_tensors",
    "receiver_positions_and_layout",
    "apply_receiver_layout",
)


def test_endpoint_helpers_have_canonical_owners():
    for name in _ENDPOINT_NAMES:
        owner = getattr(endpoints, name)

        assert owner.__module__ == endpoints.__name__

    assert accumulation.ReceiverLayout is endpoints.ReceiverLayout
    assert accumulation.apply_receiver_layout is endpoints.apply_receiver_layout
    assert solver.apply_receiver_layout is endpoints.apply_receiver_layout
    assert (
        solver.receiver_positions_and_layout is endpoints.receiver_positions_and_layout
    )
    assert enumerated.transmitter_tensors is endpoints.transmitter_tensors
    assert (
        enumerated.receiver_positions_and_layout
        is endpoints.receiver_positions_and_layout
    )


def test_receiver_layout_preserves_point_and_grid_storage_contracts():
    point_values = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    point_layout = endpoints.ReceiverLayout("point", 6)

    assert endpoints.apply_receiver_layout(point_values, point_layout) is point_values

    grid_layout = endpoints.ReceiverLayout("grid", 6, (2, 3))
    grid_values = endpoints.apply_receiver_layout(point_values, grid_layout)
    expected = point_values.reshape(2, 2, 3).transpose(1, 2).contiguous()

    assert grid_values.shape == (2, 3, 2)
    assert grid_values.is_contiguous()
    assert torch.equal(grid_values, expected)


def test_receiver_layout_preserves_validation_errors():
    values = torch.zeros((1, 1))

    with pytest.raises(ValueError, match="^grid layout requires grid_shape$"):
        endpoints.ReceiverLayout("grid", 1).apply(values)
    with pytest.raises(
        ValueError, match="^receiver layout kind is not accepted: invalid$"
    ):
        endpoints.ReceiverLayout("invalid", 1).apply(values)


def test_endpoint_helpers_delegate_to_raw_scene_tensor_owners(monkeypatch):
    device = torch.device("cpu")
    tx_positions = torch.arange(3, dtype=torch.float32).reshape(1, 3)
    tx_power = torch.ones((1,), dtype=torch.float32)
    rx_positions = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    calls: list[tuple[str, object, object]] = []

    def fake_transmitter_positions(scene, *, device):
        calls.append(("tx", scene, device))
        return tx_positions, tx_power

    def fake_receiver_positions(scene, *, device, reference):
        calls.append(("rx", scene, reference))
        assert reference is tx_positions
        return rx_positions

    monkeypatch.setattr(
        endpoints, "_native_transmitter_positions", fake_transmitter_positions
    )
    monkeypatch.setattr(
        endpoints, "_native_receiver_positions", fake_receiver_positions
    )

    scene = SimpleNamespace(receivers=[object()])
    exported_positions, exported_power = endpoints.transmitter_tensors(
        scene, device=device
    )
    point_positions, point_layout = endpoints.receiver_positions_and_layout(
        scene, device=device
    )

    assert exported_positions is tx_positions
    assert exported_power is tx_power
    assert point_positions is rx_positions
    assert point_layout == endpoints.ReceiverLayout("point", 6)

    grid = ReceiverGrid(
        origin=torch.zeros(3),
        x_axis=torch.tensor((1.0, 0.0, 0.0)),
        y_axis=torch.tensor((0.0, 1.0, 0.0)),
        shape=(2, 3),
        spacing=(1.0, 1.0),
    )
    grid_scene = SimpleNamespace(receivers=[grid])
    grid_positions, grid_layout = endpoints.receiver_positions_and_layout(
        grid_scene, device=device
    )

    assert grid_positions is rx_positions
    assert grid_layout == endpoints.ReceiverLayout("grid", 6, (2, 3))
    assert calls == [
        ("tx", scene, device),
        ("tx", scene, device),
        ("rx", scene, tx_positions),
        ("tx", grid_scene, device),
        ("rx", grid_scene, tx_positions),
    ]


def test_empty_receiver_layout_skips_raw_endpoint_owners(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("empty receiver layout must not materialize endpoints")

    monkeypatch.setattr(endpoints, "transmitter_tensors", unexpected)
    monkeypatch.setattr(endpoints, "_native_receiver_positions", unexpected)

    positions, layout = endpoints.receiver_positions_and_layout(
        SimpleNamespace(receivers=[]), device=torch.device("cpu")
    )

    assert positions.shape == (0, 3)
    assert positions.dtype == torch.float32
    assert layout == endpoints.ReceiverLayout("point", 0)
