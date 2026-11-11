# Copyright Xingyu Chen.
# Tests that per-face material tables stay device resident on the compiled scene.

from __future__ import annotations

import pytest
import torch

from tests.support.scenes import single_wall_reflection_scene
from witwin.channel.materials import (
    face_geometry_id_table,
    face_material_field_bundle,
    face_material_tensors,
    face_material_thickness,
)
from witwin.channel.scene import compile as compile_scene


def _compiled():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for device material tables")
    return compile_scene(single_wall_reflection_scene(), reference_frequency_hz=3.0e9)


def test_tables_are_built_once_per_compiled_scene_and_device():
    compiled = _compiled()
    device = torch.device("cuda")
    first = face_material_field_bundle(compiled, device=device)
    second = face_material_field_bundle(compiled, device=device)
    assert first is not second
    assert all(first[name] is second[name] for name in first)
    assert all(value.is_cuda for value in first.values())
    assert face_material_tensors(compiled, device=device)[0] is face_material_tensors(
        compiled, device=device
    )[0]
    assert face_material_thickness(compiled, device=device) is face_material_thickness(
        compiled, device=device
    )
    surface = face_geometry_id_table(compiled, "face_surface_id", device=device)
    assert surface is face_geometry_id_table(compiled, "face_surface_id", device=device)
    assert surface.dtype == torch.int64 and surface.is_cuda
    with pytest.raises(ValueError, match="unknown per-face geometry id table"):
        face_geometry_id_table(compiled, "face_normals", device=device)


def test_second_lookup_issues_no_host_to_device_copy():
    compiled = _compiled()
    device = torch.device("cuda")
    face_material_field_bundle(compiled, device=device)
    face_material_tensors(compiled, device=device)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        face_material_field_bundle(compiled, device=device)
        face_material_tensors(compiled, device=device)
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def test_in_place_store_mutation_rebuilds_the_table():
    compiled = _compiled()
    device = torch.device("cuda")
    before = face_material_field_bundle(compiled, device=device)["eps_r"].clone()
    compiled.materials.eps_r.mul_(2.0)
    after = face_material_field_bundle(compiled, device=device)["eps_r"]
    torch.testing.assert_close(after, 2.0 * before)


def test_autograd_store_leaf_is_never_cached():
    compiled = _compiled()
    device = torch.device("cuda")
    compiled.materials.eps_r.requires_grad_(True)
    try:
        first = face_material_field_bundle(compiled, device=device)["eps_r"]
        second = face_material_field_bundle(compiled, device=device)["eps_r"]
        assert first.grad_fn is not None and second.grad_fn is not None
        assert first is not second
        first.sum().backward()
        assert compiled.materials.eps_r.grad is not None
    finally:
        compiled.materials.eps_r.requires_grad_(False)
        compiled.materials.eps_r.grad = None
