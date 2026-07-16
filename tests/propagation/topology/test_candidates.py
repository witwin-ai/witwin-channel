from __future__ import annotations


import pytest

from witwin.channel_native.propagation.topology import kernels
from witwin.channel_native.propagation.topology.kernels import blocks, candidates
from witwin.channel_native.runtime import native_handles, symbols, tensor_contracts
from witwin.channel_native.scene import native_handles as scene_native_handles
from witwin.channel_native.scene.kernels import rayd_scene


_OWNER_NAMES = (
    "path_diffraction_paths_order1",
    "path_reflection_candidates",
)


@pytest.mark.parametrize("name", _OWNER_NAMES)
def test_topology_candidates_is_the_single_object_owner(name: str):
    owner = getattr(candidates, name)

    assert owner.__module__ == candidates.__name__
    assert getattr(kernels, name) is owner


def test_topology_candidates_use_only_canonical_dependencies():
    assert candidates._required_native_op is symbols.required_symbol
    assert candidates.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert candidates._validate_path_block is blocks._validate_path_block
    assert (
        candidates._validate_path_reflection_candidates
        is blocks._validate_path_reflection_candidates
    )
    assert "_raydn_module_handle" not in candidates.__dict__
    assert candidates._raydn_scene_handle_id is native_handles._raydn_scene_handle_id
    for name in _OWNER_NAMES:
        assert getattr(candidates, name).__globals__ is candidates.__dict__
    assert "ops" not in candidates.__dict__


def test_scene_handle_normalizer_is_a_same_object_reexport():
    assert native_handles.__all__ == ["_raydn_scene_handle_id"]
    assert scene_native_handles._raydn_scene_handle_id is (
        native_handles._raydn_scene_handle_id
    )
    assert rayd_scene._raydn_scene_handle_id is native_handles._raydn_scene_handle_id
