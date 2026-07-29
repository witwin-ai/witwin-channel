# Copyright Xingyu Chen.
# Tests kernel bridge.

from __future__ import annotations

import pytest
import torch

from witwin.channel.kernels import geometry
from witwin.channel import runtime


_CANONICAL_FUNCTION_NAMES = (
    "coupled_rd_geometry_forward",
    "diffraction_tx_visible_state_plan",
    "rayd_diffraction_paths_order1_forward",
    "rayd_diffraction_sample_tape_forward",
    "rayd_intersect_forward",
    "rayd_reflection_epc_paths_forward",
    "rayd_trace_reflections_forward",
    "rayd_visibility_forward",
)


@pytest.mark.parametrize("name", _CANONICAL_FUNCTION_NAMES)
def test_geometry_bridge_is_the_single_body_owner(name: str):
    owner = getattr(geometry, name)

    assert owner.__module__ == geometry.__name__


def test_geometry_bridge_uses_canonical_runtime_and_scene_dependencies():
    assert geometry._required_native_op is runtime.required_symbol
    assert geometry.validate_cuda_tensor is runtime.validate_cuda_tensor
    assert "_rayd_resource" not in geometry.__dict__
    assert geometry._rayd_scene_resource is runtime._rayd_scene_resource


def test_intersection_returns_the_named_tensor_contract(monkeypatch: pytest.MonkeyPatch):
    ray_o = torch.zeros((2, 3), dtype=torch.float32)
    ray_d = torch.ones((2, 3), dtype=torch.float32)
    ray_tmax = torch.empty((0,), dtype=torch.float32)
    outputs = (
        torch.ones((2,), dtype=torch.float32),
        torch.zeros((2, 3), dtype=torch.float32),
        torch.zeros((2, 3), dtype=torch.float32),
        torch.zeros((2, 3), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.float32),
        torch.zeros((2, 3), dtype=torch.float32),
        torch.zeros((2,), dtype=torch.int32),
        torch.zeros((2,), dtype=torch.int32),
        torch.zeros((2,), dtype=torch.int32),
        torch.zeros((2,), dtype=torch.int32),
    )
    native_calls: list[tuple[object, ...]] = []

    def required_symbol(name: str):
        assert name == "rayd_intersect_forward"

        def intersect(*args: object) -> tuple[torch.Tensor, ...]:
            native_calls.append(args)
            return outputs

        return intersect

    monkeypatch.setattr(geometry, "_required_native_op", required_symbol)
    monkeypatch.setattr(geometry, "validate_cuda_tensor", lambda *_args, **_kwargs: None)

    resource = object()
    result = geometry.rayd_intersect_forward(
        resource, ray_o, ray_d, ray_tmax, None, flags=5
    )

    assert tuple(result) == (
        "t",
        "p",
        "n",
        "geo_n",
        "uv",
        "barycentric",
        "shape_id",
        "prim_id",
        "local_prim_id",
        "global_prim_id",
    )
    assert all(actual is expected for actual, expected in zip(result.values(), outputs))
    assert native_calls == [(resource, ray_o, ray_d, ray_tmax, None, 5)]


def test_diffraction_visibility_plan_dispatches_only_native_numerical_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 2
    tx = torch.zeros(3)
    edge_index = torch.arange(rows, dtype=torch.int32)
    edge_position = torch.zeros((rows, 3))
    edge_direction = torch.ones((rows, 3))
    edge_t_min = torch.zeros(rows)
    edge_t_max = torch.ones(rows)
    passthrough_float3 = torch.zeros((rows, 6))[:, ::2]
    prim0 = torch.zeros(rows, dtype=torch.int32)
    prim1 = torch.ones(rows, dtype=torch.int32)
    exterior_angle = torch.ones(rows)
    source = torch.full((rows, 3), 7.0)
    source_power = torch.ones(rows)
    active = torch.tensor([True, False])
    native_calls: list[tuple[object, ...]] = []
    validations: list[tuple[str, bool]] = []

    def validate(name: str, _tensor: torch.Tensor, **kwargs: object) -> None:
        validations.append((name, bool(kwargs.get("require_contiguous", True))))

    def required_symbol(name: str):
        assert name == "diffraction_tx_visible_state_plan"

        def plan(*args: object) -> torch.Tensor:
            native_calls.append(args)
            return active

        return plan

    monkeypatch.setattr(geometry, "validate_cuda_tensor", validate)
    monkeypatch.setattr(geometry, "_required_native_op", required_symbol)

    result = geometry.diffraction_tx_visible_state_plan(
        object(),
        tx,
        edge_index,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        passthrough_float3,
        passthrough_float3,
        prim0,
        prim1,
        exterior_angle,
        source,
        source_power,
    )

    assert result is active
    assert native_calls == [
        (
            native_calls[0][0],
            tx,
            edge_position,
            edge_direction,
            edge_t_min,
            edge_t_max,
        )
    ]
    assert all(argument is not source for argument in native_calls[0])
    validation_contract = dict(validations)
    for name in ("tx", "edge_position", "edge_direction", "edge_t_min", "edge_t_max"):
        assert validation_contract[name] is True
    for name in (
        "edge_index",
        "n0",
        "n1",
        "prim0",
        "prim1",
        "exterior_angle",
        "source",
        "source_power",
    ):
        assert validation_contract[name] is False


def test_diffraction_visibility_plan_rejects_capacity_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = geometry._DIFFRACTION_STATE_CAPACITY + 1
    float3 = torch.empty((1, 3), device="meta").expand(rows, 3)
    float1 = torch.empty((1,), device="meta").expand(rows)
    int1 = torch.empty((1,), dtype=torch.int32, device="meta").expand(rows)
    monkeypatch.setattr(geometry, "validate_cuda_tensor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        geometry,
        "_required_native_op",
        lambda _name: pytest.fail("native dispatch must not be reached"),
    )

    with pytest.raises(ValueError, match="exceeds 4194304"):
        geometry.diffraction_tx_visible_state_plan(
            object(),
            torch.empty((3,), device="meta"),
            int1,
            float3,
            float3,
            float1,
            float1,
            float3,
            float3,
            int1,
            int1,
            float1,
            float3,
            float1,
        )


def test_diffraction_visibility_plan_accepts_exact_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = geometry._DIFFRACTION_STATE_CAPACITY
    float3 = torch.empty((1, 3), device="meta").expand(rows, 3)
    float1 = torch.empty((1,), device="meta").expand(rows)
    int1 = torch.empty((1,), dtype=torch.int32, device="meta").expand(rows)
    active = torch.empty((rows,), dtype=torch.bool, device="meta")
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(geometry, "validate_cuda_tensor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(geometry, "_rayd_scene_resource", lambda resource: resource)
    monkeypatch.setattr(
        geometry,
        "_required_native_op",
        lambda _name: lambda *args: calls.append(args) or active,
    )

    result = geometry.diffraction_tx_visible_state_plan(
        object(),
        torch.empty((3,), device="meta"),
        int1,
        float3,
        float3,
        float1,
        float1,
        float3,
        float3,
        int1,
        int1,
        float1,
        float3,
        float1,
    )

    assert result is active
    assert len(calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_diffraction_visibility_plan_accepts_passthrough_views_and_rejects_numerical_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 2
    tx = torch.zeros(3, device="cuda")
    passthrough_int = torch.zeros((rows, 2), dtype=torch.int32, device="cuda")[:, 0]
    passthrough_float = torch.zeros((rows, 2), device="cuda")[:, 0]
    edge_index = passthrough_int
    edge_position = torch.zeros((rows, 3), device="cuda")
    edge_direction = torch.ones((rows, 3), device="cuda")
    edge_t_min = torch.zeros(rows, device="cuda")
    edge_t_max = torch.ones(rows, device="cuda")
    passthrough_float3 = torch.zeros((rows, 6), device="cuda")[:, ::2]
    prim0 = passthrough_int
    prim1 = passthrough_int
    exterior_angle = passthrough_float
    source = passthrough_float3
    source_power = passthrough_float
    active = torch.ones(rows, dtype=torch.bool, device="cuda")
    monkeypatch.setattr(geometry, "_rayd_scene_resource", lambda resource: resource)
    monkeypatch.setattr(
        geometry, "_required_native_op", lambda _name: lambda *_args: active
    )

    assert (
        geometry.diffraction_tx_visible_state_plan(
            object(),
            tx,
            edge_index,
            edge_position,
            edge_direction,
            edge_t_min,
            edge_t_max,
            passthrough_float3,
            passthrough_float3,
            prim0,
            prim1,
            exterior_angle,
            source,
            source_power,
        )
        is active
    )

    numerical = [tx, edge_position, edge_direction, edge_t_min, edge_t_max]
    noncontiguous = [
        torch.zeros(6, device="cuda")[::2],
        torch.zeros((rows, 6), device="cuda")[:, ::2],
        torch.zeros((rows, 6), device="cuda")[:, ::2],
        torch.zeros(rows * 2, device="cuda")[::2],
        torch.zeros(rows * 2, device="cuda")[::2],
    ]
    for index, (name, view) in enumerate(
        zip(
            ("tx", "edge_position", "edge_direction", "edge_t_min", "edge_t_max"),
            noncontiguous,
            strict=True,
        )
    ):
        invalid_numerical = numerical.copy()
        invalid_numerical[index] = view
        with pytest.raises(ValueError, match=f"{name} must be contiguous"):
            geometry.diffraction_tx_visible_state_plan(
                object(),
                invalid_numerical[0],
                edge_index,
                invalid_numerical[1],
                invalid_numerical[2],
                invalid_numerical[3],
                invalid_numerical[4],
                passthrough_float3,
                passthrough_float3,
                prim0,
                prim1,
                exterior_angle,
                source,
                source_power,
            )