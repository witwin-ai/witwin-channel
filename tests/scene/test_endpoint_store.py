# Copyright Xingyu Chen.
# Tests the batched, cached endpoint store bound to every solver scene.

from __future__ import annotations

import math

import pytest
import torch

from tests.support.core_world import make_receiver, make_receiver_grid, make_transmitter
from witwin.channel.scene import endpoints as scene_endpoints
from witwin.channel.scene.endpoints import (
    _ReceiverGridView,
    _ReceiverPointView,
    _TransmitterView,
    _world_polarizations,
    endpoint_token,
)
from witwin.core import Scene

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _oriented_views(device: str) -> list:
    generator = torch.Generator().manual_seed(7)
    views = []
    for index in range(40):
        position = (torch.rand(3, generator=generator) * 50.0).to(device)
        kind = index % 4
        orientation = None
        if kind == 1:
            orientation = (torch.rand(3, generator=generator) * 12.0 - 6.0).to(device)
        elif kind == 2:
            quaternion = torch.randn(4, generator=generator)
            orientation = (quaternion / quaternion.norm()).to(device)
        polarization = (
            None if index % 3 == 0 else torch.randn(3, generator=generator).to(device)
        )
        pattern = "horizontal" if index % 5 == 0 else "isotropic"
        endpoint = (make_transmitter if index % 2 else make_receiver)(
            position, orientation=orientation, polarization=polarization, pattern=pattern
        )
        view_type = _TransmitterView if index % 2 else _ReceiverPointView
        views.append(view_type(endpoint))
    return views


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=requires_cuda)],
)
def test_batched_polarizations_match_per_endpoint_bits(device: str) -> None:
    views = _oriented_views(device)
    batched = _world_polarizations(views)
    expected = torch.stack([view.polarization for view in views])
    assert batched.dtype == expected.dtype
    assert torch.equal(batched.view(torch.int32), expected.view(torch.int32))


def test_batched_polarizations_refuse_zero_vectors() -> None:
    view = _ReceiverPointView(make_receiver(torch.zeros(3), polarization=torch.zeros(3)))
    with pytest.raises(ValueError, match="polarization must be finite and non-zero"):
        _world_polarizations([view])


def test_endpoint_token_tracks_in_place_mutation_and_grad_flag() -> None:
    position = torch.tensor([1.0, 2.0, 3.0])
    scene = Scene(endpoints=[make_transmitter(position), make_receiver(torch.zeros(3))])
    before = endpoint_token(scene)
    assert endpoint_token(scene) == before
    position.add_(1.0)
    mutated = endpoint_token(scene)
    assert mutated != before
    position.requires_grad_(True)
    assert endpoint_token(scene) != mutated


def _bind(scene: Scene):
    from witwin.channel.scene.compiler import compile as compile_scene

    compiled = compile_scene(scene, reference_frequency_hz=3.5e9)
    return compiled, scene_endpoints.bind_solver_scene(compiled)


def test_binding_is_host_only_and_builds_no_store(monkeypatch: pytest.MonkeyPatch) -> None:
    from witwin.channel.scene.compiler import compile as compile_scene

    scene = Scene(
        endpoints=[
            make_transmitter(torch.tensor([0.0, 0.0, 10.0])),
            make_receiver(torch.tensor([3.0, 4.0, 1.5])),
        ]
    )
    compiled = compile_scene(scene, reference_frequency_hz=3.5e9)

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("binding started CUDA or tensor work")

    monkeypatch.setattr(scene_endpoints.torch.cuda, "is_available", unexpected)
    monkeypatch.setattr(scene_endpoints.torch.cuda, "current_device", unexpected)
    monkeypatch.setattr(scene_endpoints, "build_endpoint_store", unexpected)
    solver_scene = scene_endpoints.bind_solver_scene(compiled)
    assert solver_scene.endpoint_store is None
    assert compiled._endpoint_store_cache is None


@requires_cuda
def test_store_is_built_once_per_endpoint_token() -> None:
    receivers = [make_receiver(torch.tensor([10.0 + i, 5.0, 1.5])) for i in range(6)]
    scene = Scene(endpoints=[make_transmitter(torch.tensor([0.0, 0.0, 10.0])), *receivers])
    compiled, first = _bind(scene)
    device = torch.device("cuda")
    store = scene_endpoints.endpoint_store(first, device=device)
    assert scene_endpoints.endpoint_store(first, device=device) is store
    second = scene_endpoints.bind_solver_scene(compiled)
    assert second.endpoint_store is store
    assert scene_endpoints.endpoint_store(second, device=device) is store
    assert second.receivers is first.receivers
    assert store.rx_positions.shape == (6, 3)
    assert torch.equal(
        store.rx_positions.cpu(), torch.stack([receiver.position for receiver in receivers])
    )
    assert torch.equal(store.tx_power.cpu(), torch.ones(1))


@requires_cuda
def test_grid_polarization_is_one_broadcast_row() -> None:
    grid = make_receiver_grid(
        origin=torch.tensor([0.5, 0.5, 1.5]),
        x_axis=torch.tensor((1.0, 0.0, 0.0)),
        y_axis=torch.tensor((0.0, 1.0, 0.0)),
        shape=(64, 32),
        spacing=(1.0, 1.0),
    )
    scene = Scene(endpoints=[make_transmitter(torch.tensor([0.0, 0.0, 10.0])), grid])
    _, solver_scene = _bind(scene)
    rows = scene_endpoints.receiver_polarization_rows(solver_scene, device=torch.device("cuda"))
    assert rows.shape == (64 * 32, 3)
    assert rows.stride() == (0, 1)
    assert isinstance(solver_scene.receivers[0], _ReceiverGridView)
    expected = solver_scene.receivers[0].polarization.to(torch.float32)
    assert torch.equal(rows[0].cpu(), expected)
    full = scene_endpoints.receiver_polarizations_f32(solver_scene, device=torch.device("cuda"))
    assert full.is_contiguous() and torch.equal(full, rows)


@requires_cuda
def test_differentiable_endpoint_leaves_stay_on_the_graph_and_are_not_cached() -> None:
    position = torch.tensor([0.0, 0.0, 10.0], requires_grad=True)
    scene = Scene(
        endpoints=[make_transmitter(position), make_receiver(torch.tensor([3.0, 4.0, 1.5]))]
    )
    compiled, solver_scene = _bind(scene)
    store = scene_endpoints.endpoint_store(solver_scene, device=torch.device("cuda"))
    assert store.differentiable
    assert scene_endpoints.endpoint_store(solver_scene, device=torch.device("cuda")) is store
    assert compiled._endpoint_store_cache is None
    assert scene_endpoints.bind_solver_scene(compiled).endpoint_store is None
    store.tx_positions.sum().backward()
    assert torch.equal(position.grad, torch.ones(3))
    assert not math.isnan(float(store.tx_power.sum()))
