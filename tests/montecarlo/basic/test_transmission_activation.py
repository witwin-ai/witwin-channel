from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from witwin.channel_native.montecarlo.events import transmission
from witwin.channel_native.montecarlo.basic import rayd_components
from witwin.channel_native.propagation.models.penetration import (
    SegmentPenetrationPolicy,
)
from witwin.channel_native.runtime.capacity import CapacityFailureState


def _failure_state() -> CapacityFailureState:
    state = object.__new__(CapacityFailureState)
    object.__setattr__(state, "bits", torch.zeros(1, dtype=torch.int32))
    return state


def test_straight_transmission_chains_dispatches_one_target_inset_primal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origins = torch.zeros((6, 3))
    targets = torch.ones((6, 3))
    state = _failure_state()
    expected = object()
    calls: list[tuple[object, ...]] = []

    def forward(*args: object, **kwargs: object) -> object:
        calls.append((*args, kwargs))
        return expected

    monkeypatch.setattr(transmission, "rayd_segment_penetration_forward", forward)
    actual = transmission.straight_transmission_chains(
        object(),
        origins,
        targets,
        vertices=None,
        max_depth=3,
        scene_diagonal=7.5,
        failure_state=state,
        ad=False,
    )

    assert actual is expected
    assert len(calls) == 1
    call = calls[0]
    assert call[1] is origins
    assert call[2] is targets
    assert call[3] is None
    kwargs = call[4]
    assert kwargs["input_active_any"] is True
    assert kwargs["hit_capacity"] == 3
    assert kwargs["policy"] is SegmentPenetrationPolicy.MonteCarloTargetInset
    assert kwargs["scene_diagonal"] == 7.5
    assert kwargs["failure_state"] is state


def test_straight_transmission_chains_ad_keeps_state_and_vertices_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertices = torch.zeros((4, 3), requires_grad=True)
    origins = torch.zeros((0, 3), requires_grad=True)
    targets = torch.zeros((0, 3), requires_grad=True)
    state = _failure_state()
    expected = object()
    calls: list[tuple[object, ...]] = []

    def ad(*args: object, **kwargs: object) -> object:
        calls.append((*args, kwargs))
        return expected

    monkeypatch.setattr(transmission, "rayd_segment_penetration_ad", ad)
    actual = transmission.straight_transmission_chains(
        object(),
        origins,
        targets,
        vertices=vertices,
        max_depth=0,
        scene_diagonal=0.0,
        failure_state=state,
        ad=True,
    )

    assert actual is expected
    assert len(calls) == 1
    call = calls[0]
    assert call[1] is vertices
    assert call[2] is origins
    assert call[3] is targets
    assert call[4] is None
    kwargs = call[5]
    assert kwargs["input_active_any"] is False
    assert kwargs["policy"] is SegmentPenetrationPolicy.MonteCarloTargetInset
    assert kwargs["failure_state"] is state


def test_straight_transmission_chains_ad_requires_live_vertices() -> None:
    with pytest.raises(TypeError, match="live scene vertices"):
        transmission.straight_transmission_chains(
            object(),
            torch.zeros((1, 3)),
            torch.ones((1, 3)),
            vertices=None,
            max_depth=1,
            scene_diagonal=1.0,
            failure_state=_failure_state(),
            ad=True,
        )


def test_straight_transmission_chains_has_no_python_march_or_fallback() -> None:
    source = inspect.getsource(transmission.straight_transmission_chains)
    assert source.count("rayd_segment_penetration_forward(") == 1
    assert source.count("rayd_segment_penetration_ad(") == 1
    for forbidden in (
        "for ",
        "while ",
        ".item(",
        ".cpu(",
        ".numpy(",
        "torch.nonzero",
        "rayd_intersect_forward",
        "except ",
    ):
        assert forbidden not in source


def test_transmission_component_map_flattens_pair_major_and_shares_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tx = torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    rx = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    polarization = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    base_power = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    scaled_power = torch.arange(6, dtype=torch.float32) + 10.0
    state = _failure_state()
    compiled = SimpleNamespace(montecarlo_penetration_scene_diagonal_m=8.5)
    scene = SimpleNamespace(
        structures=[object()],
        frequency=3.5e9,
        compile=lambda: compiled,
    )
    grid = SimpleNamespace(shape=(1, 3))
    rayd = SimpleNamespace(available=True)
    bundle = {
        "material_id": torch.zeros(1, dtype=torch.int32),
        "geometry_mode_id": torch.zeros(1, dtype=torch.int32),
        "layer_offset": torch.zeros(1, dtype=torch.int32),
        "layer_count": torch.ones(1, dtype=torch.int32),
        "layer_thickness_m": torch.ones(1),
        "layer_eps_r": torch.ones(1),
        "layer_sigma_e": torch.zeros(1),
        "layer_mu_r": torch.ones(1),
    }
    penetration = SimpleNamespace(
        valid=torch.ones((6, 2), dtype=torch.bool),
        num_hits=torch.ones(6, dtype=torch.int32),
        reached_target=torch.ones(6, dtype=torch.bool),
        direction=torch.ones((6, 3)),
        normal=torch.ones((6, 2, 3)),
        global_primitive_id=torch.zeros((6, 2), dtype=torch.int32),
    )
    traversal_calls: list[tuple[object, ...]] = []
    product_calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        rayd_components,
        "_grid_los_matrix",
        lambda *args, **kwargs: base_power,
    )
    monkeypatch.setattr(
        rayd_components,
        "transmitter_positions",
        lambda *args, **kwargs: (tx, torch.ones(2)),
    )
    monkeypatch.setattr(
        rayd_components,
        "receiver_grid_points",
        lambda *args, **kwargs: rx,
    )
    monkeypatch.setattr(
        rayd_components,
        "face_material_field_bundle",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        rayd_components,
        "transmitter_polarizations",
        lambda *args, **kwargs: polarization,
    )

    def trace(*args: object, **kwargs: object) -> object:
        traversal_calls.append((*args, kwargs))
        return penetration

    def wall_product(*args: object, **kwargs: object) -> object:
        product_calls.append((*args, kwargs))
        return SimpleNamespace(scaled_power=scaled_power)

    monkeypatch.setattr(rayd_components, "straight_transmission_chains", trace)
    monkeypatch.setattr(rayd_components, "mc_transmission_wall_product", wall_product)
    monkeypatch.setattr(
        rayd_components,
        "mc_los_component_maps_from_matrix",
        lambda matrix, **kwargs: matrix,
    )

    result = rayd_components.transmission_component_map(
        scene,
        rayd,
        grid,
        max_depth=2,
        device=torch.device("cpu"),
        failure_state=state,
    )

    assert len(traversal_calls) == 1
    traversal = traversal_calls[0]
    torch.testing.assert_close(
        traversal[1], tx.repeat_interleave(3, dim=0), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(traversal[2], rx.repeat(2, 1), rtol=0.0, atol=0.0)
    traversal_kwargs = traversal[3]
    assert traversal_kwargs["max_depth"] == 2
    assert traversal_kwargs["scene_diagonal"] == 8.5
    assert traversal_kwargs["failure_state"] is state
    assert traversal_kwargs["ad"] is False

    assert len(product_calls) == 1
    product = product_calls[0]
    assert product[0] is penetration.valid
    assert product[1] is penetration.num_hits
    assert product[2] is penetration.reached_target
    torch.testing.assert_close(
        product[14], polarization.repeat_interleave(3, dim=0), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(product[15], base_power.reshape(-1), rtol=0.0, atol=0.0)
    assert product[16] is state
    assert product[17]["frequency_hz"] == 3.5e9
    torch.testing.assert_close(result, scaled_power.reshape(2, 3), rtol=0.0, atol=0.0)
