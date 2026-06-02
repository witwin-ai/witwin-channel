"""Pair-level regression checks for K1 forward/reverse AD kernels."""

from __future__ import annotations

import drjit as dr
import pytest
import witwin as wt

import witwin.channel.trace.diffraction.field as diffraction_field_module
from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from tests.support.bin._multipath_benchmark import create_grad_multipath_case
from witwin.channel import Field
from witwin.channel.trace import compute_reflection_field
from witwin.channel.trace.diffraction.builders import _prepare_diffraction_state_arrays
from witwin.channel.trace.diffraction.field import (
    _accumulate_edge_state_dispatch_chunks_totals_impl,
    _accumulate_edge_states_to_receivers_totals_hybrid_backward,
    _partition_visible_pair_dispatch_chunks,
)
from witwin.channel.trace.diffraction.kernels.utd_accumulate_slang import (
    accumulate_edge_state_totals_slang_backward,
    accumulate_edge_state_totals_slang_forward_jvp,
    slang_utd_accumulate_available,
)


FREQUENCY = 1e9
WAVELENGTH = 299792458.0 / FREQUENCY
WAVENUMBER = 2.0 * dr.pi / WAVELENGTH
TX_POLARIZATION = (0.0, 1.0, 0.0)


def _scalar_max_abs(value) -> float:
    return float(dr.slice(dr.max(dr.abs(value))))


def _complex_max_abs(value) -> float:
    return max(_scalar_max_abs(value.real), _scalar_max_abs(value.imag))


def _zero_vector_totals(width: int):
    zeros = dr.zeros(wt.Float, width)
    return {
        "x": wt.Complex2f(zeros, zeros),
        "y": wt.Complex2f(zeros, zeros),
        "z": wt.Complex2f(zeros, zeros),
    }


def _scalar_output_grads(width: int, *, receiver_idx: int | None = None):
    real = dr.zeros(wt.Float, width)
    if receiver_idx is None:
        real = dr.full(wt.Float, 1.0, width)
    else:
        dr.scatter(real, wt.Float(1.0), wt.UInt32(receiver_idx))
    imag = dr.zeros(wt.Float, width)
    zero_vector = _zero_vector_totals(width)
    return (
        wt.Complex2f(real, imag),
        wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width)),
        zero_vector,
        zero_vector,
    )


def _vector_output_grads(width: int, *, receiver_idx: int | None = None):
    zeros = dr.zeros(wt.Float, width)
    if receiver_idx is None:
        real = dr.full(wt.Float, 1.0, width)
    else:
        real = dr.zeros(wt.Float, width)
        dr.scatter(real, wt.Float(1.0), wt.UInt32(receiver_idx))
    direct_vector = {
        "x": wt.Complex2f(real, zeros),
        "y": wt.Complex2f(zeros, zeros),
        "z": wt.Complex2f(zeros, zeros),
    }
    zero_vector = _zero_vector_totals(width)
    return (
        wt.Complex2f(zeros, zeros),
        wt.Complex2f(zeros, zeros),
        direct_vector,
        zero_vector,
    )


def _mixed_output_grads(width: int, *, receiver_idx: int | None = None):
    scalar = _scalar_output_grads(width, receiver_idx=receiver_idx)
    vector = _vector_output_grads(width, receiver_idx=receiver_idx)
    return (
        scalar[0],
        scalar[1],
        vector[2],
        vector[3],
    )


def _small_scene_state_case():
    scene = build_test_scene(
        box_geometry(center=(-2.0, -2.0, 1.5), size=2.0),
        box_geometry(center=(2.0, 1.5, 1.5), size=2.0),
    )
    field = Field(bounds=((-6.0, 6.0), (-6.0, 6.0)), size=(16, 16))
    coords = field.get_coordinates()
    tx = wt.Point3f(0.0, -5.0, 1.5)
    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=64,
        max_reflections=2,
        mode="2d",
        reflection_coef=0.82,
        tx_polarization=TX_POLARIZATION,
        return_per_bounce=False,
        grid_data=coords,
    )
    _, _, state_arrays, _ = _prepare_diffraction_state_arrays(
        tx,
        1.5,
        scene,
        WAVELENGTH,
        WAVENUMBER,
        reflection_detail,
        None,
        64,
        2,
        0.82,
        "2d",
        1,
        tx_polarization=TX_POLARIZATION,
    )
    rx_pos = wt.Point3f(coords["X"], coords["Y"], wt.Float(1.5))
    return scene, state_arrays, rx_pos


def _source_pos_x_tangents(state_arrays, rx_pos):
    n_states = int(state_arrays["n_states"])
    state_tangents = {}
    for key, value in state_arrays.items():
        if key == "n_states":
            continue
        if isinstance(value, wt.Point3f):
            state_tangents[key] = wt.Point3f(
                dr.zeros(wt.Float, n_states),
                dr.zeros(wt.Float, n_states),
                dr.zeros(wt.Float, n_states),
            )
        elif isinstance(value, wt.Complex2f):
            state_tangents[key] = wt.Complex2f(
                dr.zeros(wt.Float, n_states),
                dr.zeros(wt.Float, n_states),
            )
        else:
            state_tangents[key] = dr.zeros(type(value), n_states)
    state_tangents["source_pos"] = wt.Point3f(
        dr.full(wt.Float, 1.0, n_states),
        dr.zeros(wt.Float, n_states),
        dr.zeros(wt.Float, n_states),
    )
    rx_tangents = wt.Point3f(
        dr.zeros(wt.Float, dr.width(rx_pos.x)),
        dr.zeros(wt.Float, dr.width(rx_pos.y)),
        dr.zeros(wt.Float, dr.width(rx_pos.z)),
    )
    return state_tangents, rx_tangents


@pytest.mark.gpu
@pytest.mark.skipif(not slang_utd_accumulate_available(), reason="slangtorch UTD accumulation module is unavailable")
def test_scalar_safe_backward_matches_replay_on_small_scene():
    scene, state_arrays, rx_pos = _small_scene_state_case()
    safe_chunks, _ = _partition_visible_pair_dispatch_chunks(state_arrays, rx_pos, scene)
    assert safe_chunks
    dispatch_chunks = safe_chunks[:1]
    output_grads = _scalar_output_grads(dr.width(rx_pos.x))

    slang_state_grads, slang_rx_grads = accumulate_edge_state_totals_slang_backward(
        state_arrays,
        rx_pos,
        output_grads,
        WAVENUMBER,
        scene=scene,
        wavelength=WAVELENGTH,
        material_detail=None,
        dispatch_chunks=dispatch_chunks,
    )

    detached_state = dr.detach(state_arrays)
    detached_rx = dr.detach(rx_pos)
    dr.enable_grad(detached_state, detached_rx)
    outputs = _accumulate_edge_state_dispatch_chunks_totals_impl(
        detached_state,
        detached_rx,
        dispatch_chunks,
        WAVENUMBER,
        wavelength=WAVELENGTH,
        material_detail=None,
    )
    dr.set_grad(outputs, output_grads)
    replay_state_grads, replay_rx_grads = dr.backward_to(
        detached_state,
        detached_rx,
        flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad,
    )

    assert _scalar_max_abs(slang_state_grads["source_pos"].x - replay_state_grads["source_pos"].x) < 1e-3
    assert _scalar_max_abs(slang_state_grads["source_pos"].y - replay_state_grads["source_pos"].y) < 1e-3
    assert _scalar_max_abs(slang_state_grads["edge_pos"].x - replay_state_grads["edge_pos"].x) < 1e-3
    assert _scalar_max_abs(slang_rx_grads.x - replay_rx_grads.x) < 1e-3
    assert bool(dr.all(dr.isfinite(slang_state_grads["source_pos"].x)))
    assert bool(dr.all(dr.isfinite(slang_rx_grads.x)))


@pytest.mark.gpu
@pytest.mark.skipif(not slang_utd_accumulate_available(), reason="slangtorch UTD accumulation module is unavailable")
def test_forward_jvp_matches_replay_on_small_scene():
    scene, state_arrays, rx_pos = _small_scene_state_case()
    safe_chunks, _ = _partition_visible_pair_dispatch_chunks(state_arrays, rx_pos, scene)
    assert safe_chunks
    dispatch_chunks = safe_chunks[:1]
    state_tangents, rx_tangents = _source_pos_x_tangents(state_arrays, rx_pos)

    slang_outputs = accumulate_edge_state_totals_slang_forward_jvp(
        state_arrays,
        rx_pos,
        state_tangents,
        rx_tangents,
        WAVENUMBER,
        scene=scene,
        wavelength=WAVELENGTH,
        material_detail=None,
        dispatch_chunks=dispatch_chunks,
    )

    detached_state = dr.detach(state_arrays)
    detached_rx = dr.detach(rx_pos)
    dr.enable_grad(detached_state["source_pos"], detached_rx)
    dr.set_grad(detached_state["source_pos"], state_tangents["source_pos"])
    dr.set_grad(detached_rx, rx_tangents)
    outputs = _accumulate_edge_state_dispatch_chunks_totals_impl(
        detached_state,
        detached_rx,
        dispatch_chunks,
        WAVENUMBER,
        wavelength=WAVELENGTH,
        material_detail=None,
    )
    replay_outputs = dr.forward_to(
        outputs,
        flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad,
    )

    assert _complex_max_abs(slang_outputs[0] - replay_outputs[0]) < 3e-3
    assert _complex_max_abs(slang_outputs[1] - replay_outputs[1]) < 3e-3
    assert bool(dr.all(dr.isfinite(slang_outputs[0].real)))
    assert bool(dr.all(dr.isfinite(slang_outputs[1].real)))


@pytest.mark.gpu
@pytest.mark.skipif(not slang_utd_accumulate_available(), reason="slangtorch UTD accumulation module is unavailable")
def test_scalar_safe_backward_stabilizes_known_bad_benchmark_pair():
    case = create_grad_multipath_case()
    _, _, state_arrays, _ = _prepare_diffraction_state_arrays(
        case.tx_pos,
        1.5,
        case.scene,
        case.tracer.wavelength,
        case.tracer.k,
        None,
        None,
        case.tracer.reflection_n_rays,
        case.tracer.reflection_max_bounces,
        case.tracer.reflection_coef,
        "2d",
        1,
        tx_polarization=(1.0, 0.0, 0.0),
    )
    field = case.monitor.to_field(1.0)
    coords = field.get_coordinates()
    rx_pos = wt.Point3f(coords["X"], coords["Y"], wt.Float(1.5))
    dispatch_chunks = [(wt.UInt32([0]), wt.UInt32([7170]))]
    output_grads = _scalar_output_grads(dr.width(rx_pos.x), receiver_idx=7170)

    state_grads, rx_grads = accumulate_edge_state_totals_slang_backward(
        state_arrays,
        rx_pos,
        output_grads,
        case.tracer.k,
        scene=case.scene,
        wavelength=case.tracer.wavelength,
        material_detail=None,
        dispatch_chunks=dispatch_chunks,
    )

    assert bool(dr.all(dr.isfinite(state_grads["source_pos"].x)))
    assert bool(dr.all(dr.isfinite(state_grads["source_pos"].y)))
    assert bool(dr.all(dr.isfinite(rx_grads.x)))
    assert bool(dr.all(dr.isfinite(rx_grads.y)))


@pytest.mark.gpu
@pytest.mark.skipif(not slang_utd_accumulate_available(), reason="slangtorch UTD accumulation module is unavailable")
def test_scalar_backward_matches_replay_on_unsafe_benchmark_chunk():
    case = create_grad_multipath_case()
    field = case.monitor.to_field(1.0)
    coords = field.get_coordinates()
    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=case.tx_pos,
        scene=case.scene,
        wavelength=case.tracer.wavelength,
        k=case.tracer.k,
        n_rays=case.tracer.reflection_n_rays,
        max_reflections=case.tracer.reflection_max_bounces,
        mode="2d",
        reflection_coef=case.tracer.reflection_coef,
        tx_polarization=case.tracer.tx_polarization,
        reflection_relative_permittivity=case.tracer.reflection_relative_permittivity,
        reflection_conductivity=case.tracer.reflection_conductivity,
        reflection_material=case.tracer.reflection_material,
        return_per_bounce=False,
        grid_data=coords,
    )
    _, _, state_arrays, _ = _prepare_diffraction_state_arrays(
        case.tx_pos,
        1.5,
        case.scene,
        case.tracer.wavelength,
        case.tracer.k,
        reflection_detail,
        None,
        case.tracer.reflection_n_rays,
        case.tracer.reflection_max_bounces,
        case.tracer.reflection_coef,
        "2d",
        2,
        tx_polarization=case.tracer.tx_polarization,
    )
    rx_pos = wt.Point3f(coords["X"], coords["Y"], wt.Float(1.5))
    _, unsafe_chunks = _partition_visible_pair_dispatch_chunks(state_arrays, rx_pos, case.scene)
    assert unsafe_chunks
    dispatch_chunks = unsafe_chunks[:1]
    output_grads = _scalar_output_grads(dr.width(rx_pos.x), receiver_idx=7170)

    slang_state_grads, slang_rx_grads = accumulate_edge_state_totals_slang_backward(
        state_arrays,
        rx_pos,
        output_grads,
        case.tracer.k,
        scene=case.scene,
        wavelength=case.tracer.wavelength,
        material_detail=None,
        dispatch_chunks=dispatch_chunks,
    )

    detached_state = dr.detach(state_arrays)
    detached_rx = dr.detach(rx_pos)
    dr.enable_grad(detached_state, detached_rx)
    outputs = _accumulate_edge_state_dispatch_chunks_totals_impl(
        detached_state,
        detached_rx,
        dispatch_chunks,
        case.tracer.k,
        wavelength=case.tracer.wavelength,
        material_detail=None,
    )
    dr.set_grad(outputs, output_grads)
    replay_state_grads, replay_rx_grads = dr.backward_to(
        detached_state,
        detached_rx,
        flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad,
    )

    assert _scalar_max_abs(slang_state_grads["source_pos"].x - replay_state_grads["source_pos"].x) < 1e-4
    assert _scalar_max_abs(slang_state_grads["edge_pos"].x - replay_state_grads["edge_pos"].x) < 1e-4
    assert _scalar_max_abs(slang_rx_grads.x - replay_rx_grads.x) < 1e-4
    assert bool(dr.all(dr.isfinite(slang_state_grads["source_pos"].x)))
    assert bool(dr.all(dr.isfinite(slang_rx_grads.x)))


@pytest.mark.gpu
@pytest.mark.skip(reason="Temporarily skipped: full-suite execution aborts in the Torch DLPack bridge on this benchmark path")
@pytest.mark.skipif(not slang_utd_accumulate_available(), reason="slangtorch UTD accumulation module is unavailable")
def test_hybrid_backward_splits_scalar_and_vector_grads_on_unsafe_benchmark_scene():
    case = create_grad_multipath_case()
    field = case.monitor.to_field(1.0)
    coords = field.get_coordinates()
    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=case.tx_pos,
        scene=case.scene,
        wavelength=case.tracer.wavelength,
        k=case.tracer.k,
        n_rays=case.tracer.reflection_n_rays,
        max_reflections=case.tracer.reflection_max_bounces,
        mode="2d",
        reflection_coef=case.tracer.reflection_coef,
        tx_polarization=case.tracer.tx_polarization,
        reflection_relative_permittivity=case.tracer.reflection_relative_permittivity,
        reflection_conductivity=case.tracer.reflection_conductivity,
        reflection_material=case.tracer.reflection_material,
        return_per_bounce=False,
        grid_data=coords,
    )
    _, _, state_arrays, _ = _prepare_diffraction_state_arrays(
        case.tx_pos,
        1.5,
        case.scene,
        case.tracer.wavelength,
        case.tracer.k,
        reflection_detail,
        None,
        case.tracer.reflection_n_rays,
        case.tracer.reflection_max_bounces,
        case.tracer.reflection_coef,
        "2d",
        2,
        tx_polarization=case.tracer.tx_polarization,
    )
    rx_pos = wt.Point3f(coords["X"], coords["Y"], wt.Float(1.5))
    output_grads = _mixed_output_grads(dr.width(rx_pos.x), receiver_idx=7170)

    def _replay_should_not_run(*args, **kwargs):
        raise AssertionError("mixed benchmark backward should stay on the pure-kernel CUDA path")

    original_replay = diffraction_field_module._backward_dispatch_chunks_replay
    diffraction_field_module._backward_dispatch_chunks_replay = _replay_should_not_run
    try:
        hybrid_state_grads, hybrid_rx_grads = _accumulate_edge_states_to_receivers_totals_hybrid_backward(
            state_arrays,
            rx_pos,
            output_grads,
            case.tracer.k,
            scene=case.scene,
            wavelength=case.tracer.wavelength,
            material_detail=None,
        )
    finally:
        diffraction_field_module._backward_dispatch_chunks_replay = original_replay

    safe_chunks, unsafe_chunks = _partition_visible_pair_dispatch_chunks(state_arrays, rx_pos, case.scene)
    dispatch_chunks = safe_chunks + unsafe_chunks
    detached_state = dr.detach(state_arrays)
    detached_rx = dr.detach(rx_pos)
    dr.enable_grad(detached_state, detached_rx)
    outputs = _accumulate_edge_state_dispatch_chunks_totals_impl(
        detached_state,
        detached_rx,
        dispatch_chunks,
        case.tracer.k,
        wavelength=case.tracer.wavelength,
        material_detail=None,
    )
    dr.set_grad(outputs, output_grads)
    replay_state_grads, replay_rx_grads = dr.backward_to(
        detached_state,
        detached_rx,
        flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad,
    )

    assert bool(dr.all(dr.isfinite(hybrid_state_grads["source_pos"].x)))
    assert bool(dr.all(dr.isfinite(hybrid_rx_grads.x)))
    assert _scalar_max_abs(hybrid_state_grads["source_pos"].x - replay_state_grads["source_pos"].x) < 6e-3
    assert _scalar_max_abs(hybrid_state_grads["edge_pos"].x - replay_state_grads["edge_pos"].x) < 6e-3
    assert _scalar_max_abs(hybrid_rx_grads.x - replay_rx_grads.x) < 6e-3


@pytest.mark.gpu
@pytest.mark.skipif(not slang_utd_accumulate_available(), reason="slangtorch UTD accumulation module is unavailable")
def test_full_backward_kernel_smoke_matches_replay_on_small_scene():
    scene, state_arrays, rx_pos = _small_scene_state_case()
    safe_chunks, _ = _partition_visible_pair_dispatch_chunks(state_arrays, rx_pos, scene)
    assert safe_chunks
    dispatch_chunks = safe_chunks[:1]
    output_grads = _vector_output_grads(dr.width(rx_pos.x))

    slang_state_grads, slang_rx_grads = accumulate_edge_state_totals_slang_backward(
        state_arrays,
        rx_pos,
        output_grads,
        WAVENUMBER,
        scene=scene,
        wavelength=WAVELENGTH,
        material_detail=None,
        dispatch_chunks=dispatch_chunks,
    )

    assert bool(dr.all(dr.isfinite(slang_state_grads["source_pos"].x)))
    assert bool(dr.all(dr.isfinite(slang_state_grads["edge_pos"].x)))
    assert bool(dr.all(dr.isfinite(slang_rx_grads.x)))
    assert _scalar_max_abs(slang_state_grads["source_pos"].x) > 1e-6
    assert _scalar_max_abs(slang_state_grads["edge_pos"].x) > 1e-6
    assert _scalar_max_abs(slang_rx_grads.x) > 1e-6
