from __future__ import annotations

import pytest
import torch

from witwin.channel_native.propagation.geometry.kernels import bridge as ops
from witwin.channel_native.propagation.geometry import kernels
from witwin.channel_native.propagation.geometry.kernels import bridge
from witwin.channel_native.runtime import native_resources, symbols, tensor_contracts


_CANONICAL_FUNCTION_NAMES = (
    "bdpt_diffraction_accumulation_forward",
    "coupled_rd_geometry_forward",
    "rayd_diffraction_paths_order1_forward",
    "rayd_intersect_forward",
    "rayd_reflection_epc_paths_forward",
    "rayd_trace_reflections_forward",
    "rayd_visibility_forward",
)


@pytest.mark.parametrize("name", _CANONICAL_FUNCTION_NAMES)
def test_geometry_bridge_is_the_single_body_owner(name: str):
    owner = getattr(bridge, name)

    assert owner.__module__ == bridge.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


def test_geometry_bridge_uses_canonical_runtime_and_scene_dependencies():
    assert bridge._required_native_op is symbols.required_symbol
    assert bridge.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert "_rayd_resource" not in bridge.__dict__
    assert bridge._rayd_scene_resource is native_resources._rayd_scene_resource


def test_intersection_returns_the_named_tensor_contract(
    monkeypatch: pytest.MonkeyPatch,
):
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

    monkeypatch.setattr(bridge, "_required_native_op", required_symbol)
    monkeypatch.setattr(bridge, "validate_cuda_tensor", lambda *_args, **_kwargs: None)

    resource = object()
    result = bridge.rayd_intersect_forward(
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
