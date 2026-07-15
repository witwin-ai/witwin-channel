from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ci import check_ops_migration as migration
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.propagation.geometry import kernels
from witwin.channel_native.propagation.geometry.kernels import autograd, primitives
from witwin.channel_native.runtime import autograd_contracts, symbols, tensor_contracts
from witwin.channel_native.runtime import torch_compat
from witwin.channel_native.scene import native_handles


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"

_OWNER_NAMES = (
    "_RaydnFaceNormalsAdFunction",
    "_RaydnIntersectAdFunction",
    "_RaydnReflectionEpcPathsAdFunction",
    "_RaydnTraceReflectionsAdFunction",
    "_epc_paths_frozen_winner_checks",
    "raydn_face_normals_ad",
    "raydn_intersect_ad",
    "raydn_intersect_backward",
    "raydn_intersect_jvp",
    "raydn_reflection_epc_paths_ad",
    "raydn_reflection_epc_paths_backward",
    "raydn_reflection_epc_paths_jvp",
    "raydn_scene_face_normals_backward",
    "raydn_scene_face_normals_jvp",
    "raydn_trace_reflections_ad",
    "raydn_trace_reflections_backward",
    "raydn_trace_reflections_forward_tape",
    "raydn_trace_reflections_jvp",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_geometry_autograd_is_the_single_object_owner(name: str):
    owner = getattr(autograd, name)

    assert owner.__module__ == autograd.__name__
    assert getattr(kernels, name) is owner
    assert getattr(ops, name) is owner


def test_geometry_autograd_preserves_all_frozen_body_contracts():
    manifest = migration.load_manifest(MANIFEST_PATH)
    contracts = {entry["id"]: entry for entry in manifest["contracts"]}
    prefix = f"{autograd.__name__}."
    definitions = [
        item
        for item in migration.scan_definitions(REPOSITORY_ROOT)
        if item.qualified_name.startswith(prefix)
    ]

    assert len(definitions) == 30
    for definition in definitions:
        contract = contracts[definition.terminal_name]
        assert definition.signature == contract["signature"]
        assert definition.body_sha256 == contract["body_sha256"]
        assert definition.normalized_ast_sha256 == contract["normalized_ast_sha256"]


def test_geometry_autograd_uses_canonical_runtime_and_scene_dependencies():
    assert autograd._required_native_op is symbols.required_symbol
    assert autograd.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert autograd.torch_compat is torch_compat
    assert autograd._raydn_module_handle is native_handles._raydn_module_handle
    assert autograd._raydn_scene_handle_id is native_handles._raydn_scene_handle_id
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
        autograd._RaydnFaceNormalsAdFunction.forward.__globals__[
            "deterministic_normalize_vec3"
        ]
        is primitives.deterministic_normalize_vec3
    )
    assert (
        inspect.unwrap(autograd._RaydnFaceNormalsAdFunction.backward).__globals__[
            "raydn_scene_face_normals_backward"
        ]
        is autograd.raydn_scene_face_normals_backward
    )
    assert (
        autograd._RaydnFaceNormalsAdFunction.jvp.__globals__[
            "raydn_scene_face_normals_jvp"
        ]
        is autograd.raydn_scene_face_normals_jvp
    )
    assert (
        inspect.unwrap(autograd._RaydnIntersectAdFunction.backward).__globals__[
            "raydn_intersect_backward"
        ]
        is autograd.raydn_intersect_backward
    )
    assert (
        autograd._RaydnIntersectAdFunction.jvp.__globals__["raydn_intersect_jvp"]
        is autograd.raydn_intersect_jvp
    )
    assert (
        inspect.unwrap(
            autograd._RaydnTraceReflectionsAdFunction.backward
        ).__globals__[
            "raydn_trace_reflections_backward"
        ]
        is autograd.raydn_trace_reflections_backward
    )
    assert (
        autograd._RaydnTraceReflectionsAdFunction.jvp.__globals__[
            "raydn_trace_reflections_jvp"
        ]
        is autograd.raydn_trace_reflections_jvp
    )
    assert (
        inspect.unwrap(
            autograd._RaydnReflectionEpcPathsAdFunction.backward
        ).__globals__[
            "raydn_reflection_epc_paths_backward"
        ]
        is autograd.raydn_reflection_epc_paths_backward
    )
    assert (
        autograd._RaydnReflectionEpcPathsAdFunction.jvp.__globals__[
            "raydn_reflection_epc_paths_jvp"
        ]
        is autograd.raydn_reflection_epc_paths_jvp
    )
