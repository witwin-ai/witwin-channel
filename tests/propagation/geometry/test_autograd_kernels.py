from __future__ import annotations

import inspect

import pytest

from witwin.channel_native.propagation.geometry.kernels import autograd as ops
from witwin.channel_native.propagation.geometry import kernels
from witwin.channel_native.propagation.geometry.kernels import autograd, primitives
from witwin.channel_native.runtime import (
    autograd_contracts,
    native_resources,
    symbols,
    tensor_contracts,
)
from witwin.channel_native.runtime import torch_compat


_OWNER_NAMES = (
    "_RaydFaceNormalsAdFunction",
    "_RaydIntersectAdFunction",
    "_RaydReflectionEpcPathsAdFunction",
    "_RaydTraceReflectionsAdFunction",
    "_epc_paths_frozen_winner_checks",
    "rayd_face_normals_ad",
    "rayd_intersect_ad",
    "rayd_intersect_backward",
    "rayd_intersect_jvp",
    "rayd_reflection_epc_paths_ad",
    "rayd_reflection_epc_paths_backward",
    "rayd_reflection_epc_paths_jvp",
    "rayd_scene_face_normals_backward",
    "rayd_scene_face_normals_jvp",
    "rayd_trace_reflections_ad",
    "rayd_trace_reflections_backward",
    "rayd_trace_reflections_forward_tape",
    "rayd_trace_reflections_jvp",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_geometry_autograd_is_the_single_object_owner(name: str):
    owner = getattr(autograd, name)

    assert owner.__module__ == autograd.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


def test_geometry_autograd_uses_canonical_runtime_and_scene_dependencies():
    assert autograd._required_native_op is symbols.required_symbol
    assert autograd.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert autograd.torch_compat is torch_compat
    assert "_rayd_resource" not in autograd.__dict__
    assert autograd._rayd_scene_resource is native_resources._rayd_scene_resource
    assert (
        autograd.deterministic_normalize_vec3 is primitives.deterministic_normalize_vec3
    )
    for name in (
        "_ad_active_ctx",
        "_ad_check_active",
        "_ad_check_optional_grad",
        "_ad_check_rows",
        "_ad_check_tangent_vec3",
        "_ad_checked_tangent",
        "_ad_native_tangent_or_none",
        "_ad_native_tensor",
    ):
        assert getattr(autograd, name) is getattr(autograd_contracts, name)


def test_autograd_methods_resolve_companions_in_the_canonical_owner():
    assert (
        autograd._RaydFaceNormalsAdFunction.forward.__globals__[
            "deterministic_normalize_vec3"
        ]
        is primitives.deterministic_normalize_vec3
    )
    assert (
        inspect.unwrap(autograd._RaydFaceNormalsAdFunction.backward).__globals__[
            "rayd_scene_face_normals_backward"
        ]
        is autograd.rayd_scene_face_normals_backward
    )
    assert (
        autograd._RaydFaceNormalsAdFunction.jvp.__globals__[
            "rayd_scene_face_normals_jvp"
        ]
        is autograd.rayd_scene_face_normals_jvp
    )
    assert (
        inspect.unwrap(autograd._RaydIntersectAdFunction.backward).__globals__[
            "rayd_intersect_backward"
        ]
        is autograd.rayd_intersect_backward
    )
    assert (
        autograd._RaydIntersectAdFunction.jvp.__globals__["rayd_intersect_jvp"]
        is autograd.rayd_intersect_jvp
    )
    assert (
        inspect.unwrap(autograd._RaydTraceReflectionsAdFunction.backward).__globals__[
            "rayd_trace_reflections_backward"
        ]
        is autograd.rayd_trace_reflections_backward
    )
    assert (
        autograd._RaydTraceReflectionsAdFunction.jvp.__globals__[
            "rayd_trace_reflections_jvp"
        ]
        is autograd.rayd_trace_reflections_jvp
    )
    assert (
        inspect.unwrap(
            autograd._RaydReflectionEpcPathsAdFunction.backward
        ).__globals__["rayd_reflection_epc_paths_backward"]
        is autograd.rayd_reflection_epc_paths_backward
    )
    assert (
        autograd._RaydReflectionEpcPathsAdFunction.jvp.__globals__[
            "rayd_reflection_epc_paths_jvp"
        ]
        is autograd.rayd_reflection_epc_paths_jvp
    )
