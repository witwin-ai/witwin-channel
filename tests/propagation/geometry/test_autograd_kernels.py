from __future__ import annotations

import inspect

import pytest

from witwin.channel.kernels import geometry
from witwin.channel import runtime


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
    owner = getattr(geometry, name)

    assert owner.__module__ == geometry.__name__


def test_geometry_autograd_uses_canonical_runtime_and_scene_dependencies():
    assert geometry._required_native_op is runtime.required_symbol
    assert geometry.validate_cuda_tensor is runtime.validate_cuda_tensor
    assert geometry.disable_functorch is runtime.disable_functorch
    assert "_rayd_resource" not in geometry.__dict__
    assert geometry._rayd_scene_resource is runtime._rayd_scene_resource
    assert (
        geometry.deterministic_normalize_vec3.__module__ == geometry.__name__
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
        assert getattr(geometry, name) is getattr(runtime, name)


def test_autograd_methods_resolve_companions_in_the_canonical_owner():
    assert (
        geometry._RaydFaceNormalsAdFunction.forward.__globals__[
            "deterministic_normalize_vec3"
        ]
        is geometry.deterministic_normalize_vec3
    )
    assert (
        inspect.unwrap(geometry._RaydFaceNormalsAdFunction.backward).__globals__[
            "rayd_scene_face_normals_backward"
        ]
        is geometry.rayd_scene_face_normals_backward
    )
    assert (
        geometry._RaydFaceNormalsAdFunction.jvp.__globals__[
            "rayd_scene_face_normals_jvp"
        ]
        is geometry.rayd_scene_face_normals_jvp
    )
    assert (
        inspect.unwrap(geometry._RaydIntersectAdFunction.backward).__globals__[
            "rayd_intersect_backward"
        ]
        is geometry.rayd_intersect_backward
    )
    assert (
        geometry._RaydIntersectAdFunction.jvp.__globals__["rayd_intersect_jvp"]
        is geometry.rayd_intersect_jvp
    )
    assert (
        inspect.unwrap(geometry._RaydTraceReflectionsAdFunction.backward).__globals__[
            "rayd_trace_reflections_backward"
        ]
        is geometry.rayd_trace_reflections_backward
    )
    assert (
        geometry._RaydTraceReflectionsAdFunction.jvp.__globals__[
            "rayd_trace_reflections_jvp"
        ]
        is geometry.rayd_trace_reflections_jvp
    )
    assert (
        inspect.unwrap(
            geometry._RaydReflectionEpcPathsAdFunction.backward
        ).__globals__["rayd_reflection_epc_paths_backward"]
        is geometry.rayd_reflection_epc_paths_backward
    )
    assert (
        geometry._RaydReflectionEpcPathsAdFunction.jvp.__globals__[
            "rayd_reflection_epc_paths_jvp"
        ]
        is geometry.rayd_reflection_epc_paths_jvp
    )
