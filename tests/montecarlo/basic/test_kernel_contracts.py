from __future__ import annotations

import inspect

import pytest

from witwin.channel.kernels import montecarlo
from witwin.channel.propagation import topology
from witwin.channel.kernels import topology as topology_kernels
from witwin.channel import runtime


_SAMPLING_OWNER_NAMES = (
    "mc_diffraction_state_pack",
    "mc_diffraction_state_wi",
    "mc_reflection_launch_inputs",
)

_NATIVE_BUFFER_OWNER_NAMES = (
    "mc_pack_vec3",
    "mc_receiver_grid_points",
    "mc_transmitter_tensors",
)

_FROZEN_NATIVE_BUFFER_OWNER_NAMES = (
    "bdpt_zero_matrix",
    *_NATIVE_BUFFER_OWNER_NAMES,
)

_OWNER_NAMES = _SAMPLING_OWNER_NAMES + _NATIVE_BUFFER_OWNER_NAMES

_CAPACITY_OWNER_NAMES = (
    "_McCapacityFailureComponentMapsSanitizeFunction",
    "mc_capacity_failure_component_maps_sanitize",
)

_MAP_OWNER_NAMES = (
    "_McDiffractionMapAdFunction",
    "_McFinalizeComponentMapsAdFunction",
    "_McLosGridMapsAdFunction",
    "_McLosPathGainAdFunction",
    "_McReflectionMapAdFunction",
    "mc_apply_los_visibility",
    "mc_component_map_buffer",
    "mc_finalize_component_maps",
    "mc_finalize_component_maps_ad",
    "mc_los_component_maps",
    "mc_los_component_maps_adjoint",
    "mc_los_component_maps_from_matrix",
    "mc_los_grid_maps_ad",
    "mc_los_path_gain_ad",
    "mc_los_path_gain_backward",
    "mc_los_path_gain_jvp",
    "mc_los_visibility_inputs",
    "mc_point_component_power",
    "mc_reflection_ad_max_depth",
    "mc_slab_reflection_accumulate",
    "mc_slab_reflection_accumulate_ad",
    "mc_slab_reflection_accumulate_backward",
    "mc_slab_reflection_accumulate_jvp",
    "mc_utd_diffraction_tape_accumulate",
    "mc_utd_diffraction_tape_accumulate_ad",
    "mc_utd_diffraction_tape_accumulate_backward",
    "mc_utd_diffraction_tape_accumulate_jvp",
    "mc_store_component_map",
    "mc_store_scaled_component_map",
    "mc_zero_matrix",
)

_CAPACITY_CONTRACT_IDS = (
    "_McCapacityFailureComponentMapsSanitizeFunction.backward",
    "_McCapacityFailureComponentMapsSanitizeFunction.forward",
    "_McCapacityFailureComponentMapsSanitizeFunction.jvp",
    "_McCapacityFailureComponentMapsSanitizeFunction.setup_context",
    "mc_capacity_failure_component_maps_sanitize",
)

_MAP_CONTRACT_IDS = (
    "_McDiffractionMapAdFunction.backward",
    "_McDiffractionMapAdFunction.forward",
    "_McDiffractionMapAdFunction.jvp",
    "_McDiffractionMapAdFunction.setup_context",
    "_McFinalizeComponentMapsAdFunction.backward",
    "_McFinalizeComponentMapsAdFunction.forward",
    "_McFinalizeComponentMapsAdFunction.jvp",
    "_McFinalizeComponentMapsAdFunction.setup_context",
    "_McLosGridMapsAdFunction.backward",
    "_McLosGridMapsAdFunction.forward",
    "_McLosGridMapsAdFunction.jvp",
    "_McLosGridMapsAdFunction.setup_context",
    "_McLosPathGainAdFunction.backward",
    "_McLosPathGainAdFunction.forward",
    "_McLosPathGainAdFunction.jvp",
    "_McLosPathGainAdFunction.setup_context",
    "_McReflectionMapAdFunction.backward",
    "_McReflectionMapAdFunction.forward",
    "_McReflectionMapAdFunction.jvp",
    "_McReflectionMapAdFunction.setup_context",
    "mc_apply_los_visibility",
    "mc_component_map_buffer",
    "mc_finalize_component_maps",
    "mc_finalize_component_maps_ad",
    "mc_los_component_maps",
    "mc_los_component_maps_adjoint",
    "mc_los_component_maps_from_matrix",
    "mc_los_grid_maps_ad",
    "mc_los_path_gain_ad",
    "mc_los_path_gain_backward",
    "mc_los_path_gain_jvp",
    "mc_los_visibility_inputs",
    "mc_point_component_power",
    "mc_reflection_ad_max_depth",
    "mc_slab_reflection_accumulate",
    "mc_slab_reflection_accumulate_ad",
    "mc_slab_reflection_accumulate_backward",
    "mc_slab_reflection_accumulate_jvp",
    "mc_utd_diffraction_tape_accumulate",
    "mc_utd_diffraction_tape_accumulate_ad",
    "mc_utd_diffraction_tape_accumulate_backward",
    "mc_utd_diffraction_tape_accumulate_jvp",
    "mc_store_component_map",
    "mc_store_scaled_component_map",
    "mc_zero_matrix",
)


@pytest.mark.parametrize("name", _SAMPLING_OWNER_NAMES)
def test_mc_basic_sampling_is_the_single_object_owner(name: str):
    owner = getattr(montecarlo, name)

    assert owner.__module__ == montecarlo.__name__


@pytest.mark.parametrize("name", _NATIVE_BUFFER_OWNER_NAMES)
def test_runtime_native_buffers_is_the_neutral_single_object_owner(name: str):
    owner = getattr(runtime, name)

    assert owner.__module__ == runtime.__name__
    assert getattr(montecarlo, name) is owner


def test_mc_basic_sampling_uses_canonical_runtime_dependencies():
    # The raw accessor is not a solver dependency: every probe goes through
    # runtime.required_symbol, and ci/check_import_graph.py rejects a solver
    # that imports ``native_extension`` at all.
    assert not hasattr(montecarlo, "native_extension")
    assert montecarlo.required_symbol is runtime.required_symbol
    assert montecarlo.validate_cuda_tensor is runtime.validate_cuda_tensor


def test_runtime_native_buffers_uses_canonical_runtime_dependencies():
    assert runtime.native_extension is runtime.native_extension
    assert runtime.validate_cuda_tensor is runtime.validate_cuda_tensor


@pytest.mark.parametrize("name", _CAPACITY_OWNER_NAMES)
def test_mc_basic_capacity_is_the_single_object_owner(name: str):
    owner = getattr(montecarlo, name)

    assert owner.__module__ == montecarlo.__name__


def test_mc_basic_capacity_uses_canonical_runtime_dependencies():
    assert montecarlo._required_native_op is runtime.required_symbol
    assert montecarlo.validate_cuda_tensor is runtime.validate_cuda_tensor
    assert montecarlo.disable_functorch is runtime.disable_functorch
    assert (
        montecarlo._ad_native_tangent_or_none
        is runtime._ad_native_tangent_or_none
    )
    assert montecarlo._ad_native_tensor is runtime._ad_native_tensor


@pytest.mark.parametrize("name", _MAP_OWNER_NAMES)
def test_mc_basic_maps_is_the_single_object_owner(name: str):
    owner = getattr(montecarlo, name)

    assert owner.__module__ == montecarlo.__name__


def test_mc_basic_maps_uses_canonical_runtime_dependencies():
    assert not hasattr(montecarlo, "native_extension")
    assert montecarlo._required_native_op is runtime.required_symbol
    assert montecarlo.validate_cuda_tensor is runtime.validate_cuda_tensor
    assert montecarlo.disable_functorch is runtime.disable_functorch
    assert montecarlo._ad_frequency_grad is runtime._ad_frequency_grad
    assert montecarlo._ad_frequency_tangent is runtime._ad_frequency_tangent
    assert montecarlo._ad_frequency_value is runtime._ad_frequency_value
    assert montecarlo._ad_geometry_tangent is runtime._ad_geometry_tangent
    assert (
        montecarlo._ad_native_tangent_or_none is runtime._ad_native_tangent_or_none
    )
    assert montecarlo._ad_native_tensor is runtime._ad_native_tensor
    assert montecarlo._ad_reject_fixed_inputs is runtime._ad_reject_fixed_inputs
    assert (
        montecarlo._ad_reject_fixed_tangents is runtime._ad_reject_fixed_tangents
    )
    assert montecarlo._LIGHT_SPEED_M_PER_S_AD == 299_792_458.0
    assert montecarlo._MC_FINALIZE_FIELDS


def test_mc_basic_maps_los_ad_methods_resolve_canonical_siblings():
    function = montecarlo._McLosPathGainAdFunction

    assert function.forward.__globals__ is montecarlo.__dict__
    assert function.setup_context.__globals__ is montecarlo.__dict__
    assert inspect.unwrap(function.backward).__globals__ is montecarlo.__dict__
    assert function.jvp.__globals__ is montecarlo.__dict__
    assert function.forward.__globals__["path_los_export"] is montecarlo.path_los_export
    assert (
        inspect.unwrap(function.backward).__globals__["mc_los_path_gain_backward"]
        is montecarlo.mc_los_path_gain_backward
    )
    assert function.jvp.__globals__["mc_los_path_gain_jvp"] is montecarlo.mc_los_path_gain_jvp
    assert montecarlo.mc_los_path_gain_ad.__globals__["_McLosPathGainAdFunction"] is function


def test_mc_basic_capacity_map_sanitizer_resolves_native_siblings():
    function = montecarlo._McCapacityFailureComponentMapsSanitizeFunction

    assert function.forward.__globals__ is montecarlo.__dict__
    assert function.setup_context.__globals__ is montecarlo.__dict__
    assert inspect.unwrap(function.backward).__globals__ is montecarlo.__dict__
    assert function.jvp.__globals__ is montecarlo.__dict__
    assert (
        function.forward.__globals__[
            "_mc_capacity_failure_component_maps_sanitize_native"
        ]
        is montecarlo._mc_capacity_failure_component_maps_sanitize_native
    )
    assert (
        inspect.unwrap(function.backward).__globals__[
            "_mc_capacity_failure_component_maps_sanitize_backward_native"
        ]
        is montecarlo._mc_capacity_failure_component_maps_sanitize_backward_native
    )
    assert (
        function.jvp.__globals__[
            "_mc_capacity_failure_component_maps_sanitize_jvp_native"
        ]
        is montecarlo._mc_capacity_failure_component_maps_sanitize_jvp_native
    )
    assert (
        montecarlo.mc_capacity_failure_component_maps_sanitize.__globals__[
            "_McCapacityFailureComponentMapsSanitizeFunction"
        ]
        is function
    )


def test_mc_basic_maps_finalize_ad_methods_resolve_canonical_siblings():
    function = montecarlo._McFinalizeComponentMapsAdFunction

    assert function.forward.__globals__ is montecarlo.__dict__
    assert function.setup_context.__globals__ is montecarlo.__dict__
    assert inspect.unwrap(function.backward).__globals__ is montecarlo.__dict__
    assert function.jvp.__globals__ is montecarlo.__dict__
    assert (
        function.forward.__globals__["mc_finalize_component_maps"]
        is montecarlo.mc_finalize_component_maps
    )
    assert (
        function.jvp.__globals__["mc_finalize_component_maps"]
        is montecarlo.mc_finalize_component_maps
    )
    assert (
        montecarlo.mc_finalize_component_maps_ad.__globals__[
            "_McFinalizeComponentMapsAdFunction"
        ]
        is function
    )


def test_mc_basic_maps_grid_ad_methods_resolve_canonical_siblings():
    function = montecarlo._McLosGridMapsAdFunction

    assert function.forward.__globals__ is montecarlo.__dict__
    assert function.setup_context.__globals__ is montecarlo.__dict__
    assert inspect.unwrap(function.backward).__globals__ is montecarlo.__dict__
    assert function.jvp.__globals__ is montecarlo.__dict__
    assert (
        function.forward.__globals__["mc_los_component_maps_from_matrix"]
        is montecarlo.mc_los_component_maps_from_matrix
    )
    assert (
        function.forward.__globals__["mc_apply_los_visibility"]
        is montecarlo.mc_apply_los_visibility
    )
    assert (
        inspect.unwrap(function.backward).__globals__["mc_los_component_maps_adjoint"]
        is montecarlo.mc_los_component_maps_adjoint
    )
    assert (
        function.jvp.__globals__["mc_los_component_maps_from_matrix"]
        is montecarlo.mc_los_component_maps_from_matrix
    )
    assert (
        function.jvp.__globals__["mc_apply_los_visibility"]
        is montecarlo.mc_apply_los_visibility
    )
    assert montecarlo.mc_los_grid_maps_ad.__globals__["_McLosGridMapsAdFunction"] is function


@pytest.mark.parametrize(
    ("class_name", "forward_name", "backward_name", "jvp_name", "entry_name"),
    (
        (
            "_McReflectionMapAdFunction",
            "mc_slab_reflection_accumulate",
            "mc_slab_reflection_accumulate_backward",
            "mc_slab_reflection_accumulate_jvp",
            "mc_slab_reflection_accumulate_ad",
        ),
        (
            "_McDiffractionMapAdFunction",
            "mc_utd_diffraction_tape_accumulate",
            "mc_utd_diffraction_tape_accumulate_backward",
            "mc_utd_diffraction_tape_accumulate_jvp",
            "mc_utd_diffraction_tape_accumulate_ad",
        ),
    ),
)
def test_mc_basic_maps_component_ad_methods_resolve_canonical_siblings(
    class_name: str,
    forward_name: str,
    backward_name: str,
    jvp_name: str,
    entry_name: str,
):
    function = getattr(montecarlo, class_name)

    assert function.forward.__globals__ is montecarlo.__dict__
    assert function.setup_context.__globals__ is montecarlo.__dict__
    assert inspect.unwrap(function.backward).__globals__ is montecarlo.__dict__
    assert function.jvp.__globals__ is montecarlo.__dict__
    assert function.forward.__globals__[forward_name] is getattr(montecarlo, forward_name)
    assert inspect.unwrap(function.backward).__globals__[backward_name] is getattr(
        montecarlo, backward_name
    )
    assert function.jvp.__globals__[jvp_name] is getattr(montecarlo, jvp_name)
    assert getattr(montecarlo, entry_name).__globals__[class_name] is function


def test_mc_basic_maps_uses_package_level_los_export_same_object_alias():
    assert topology.path_los_export is topology_kernels.path_los_export
    assert montecarlo.path_los_export is topology.path_los_export
    assert "path_los_export" not in topology.__all__
