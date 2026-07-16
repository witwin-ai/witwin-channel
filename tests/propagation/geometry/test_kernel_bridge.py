from __future__ import annotations

import pytest
import torch

from witwin.channel_native.core.kernels import ops
from witwin.channel_native.propagation.geometry import kernels
from witwin.channel_native.propagation.geometry.kernels import bridge
from witwin.channel_native.runtime import native_handles, symbols, tensor_contracts


_CANONICAL_FUNCTION_NAMES = (
    "bdpt_diffraction_accumulation_forward",
    "bdpt_diffraction_discover_edges",
    "bdpt_diffraction_discover_edges_counted",
    "bdpt_intersect_forward",
    "bdpt_reflection_accumulation_forward",
    "bdpt_visibility_forward",
    "raydn_coupled_rd_geometry_forward",
    "raydn_diffraction_paths_order1_forward",
    "raydn_reflection_epc_paths_forward",
    "raydn_trace_reflections_forward",
)

_ALIASES = (
    ("raydn_visibility_forward", "bdpt_visibility_forward"),
    (
        "raydn_reflection_accumulation_forward",
        "bdpt_reflection_accumulation_forward",
    ),
    ("raydn_diffraction_discover_edges", "bdpt_diffraction_discover_edges"),
    (
        "raydn_diffraction_discover_edges_counted",
        "bdpt_diffraction_discover_edges_counted",
    ),
    (
        "raydn_diffraction_accumulation_forward",
        "bdpt_diffraction_accumulation_forward",
    ),
)


@pytest.mark.parametrize("name", _CANONICAL_FUNCTION_NAMES)
def test_geometry_bridge_is_the_single_body_owner(name: str):
    owner = getattr(bridge, name)

    assert owner.__module__ == bridge.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


@pytest.mark.parametrize(("alias", "canonical"), _ALIASES)
def test_raydn_aliases_remain_same_object(alias: str, canonical: str):
    owner = getattr(bridge, canonical)

    assert getattr(bridge, alias) is owner
    assert getattr(kernels, alias) is owner
    assert getattr(ops, alias) is owner


def test_geometry_bridge_uses_canonical_runtime_and_scene_dependencies():
    assert bridge._required_native_op is symbols.required_symbol
    assert bridge.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert "_raydn_module_handle" not in bridge.__dict__
    assert bridge._raydn_scene_handle_id is native_handles._raydn_scene_handle_id


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
        assert name == "bdpt_intersect_forward"

        def intersect(*args: object) -> tuple[torch.Tensor, ...]:
            native_calls.append(args)
            return outputs

        return intersect

    monkeypatch.setattr(bridge, "_required_native_op", required_symbol)
    monkeypatch.setattr(bridge, "validate_cuda_tensor", lambda *_args, **_kwargs: None)

    result = bridge.bdpt_intersect_forward(17, ray_o, ray_d, ray_tmax, None, flags=5)

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
    assert native_calls == [(17, ray_o, ray_d, ray_tmax, None, 5)]
