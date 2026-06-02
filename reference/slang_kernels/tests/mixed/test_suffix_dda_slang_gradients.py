"""Guard differentiable suffix traces away from the Slang DDA fast path."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import drjit as dr
import numpy as np
import pytest
import witwin as wt

from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import DiffractionExecutionConfig, Field, to_numpy
from witwin.channel.trace import compute_reflection_field
from witwin.channel.trace.diffraction import (
    _prepare_diffraction_state_arrays,
)
from witwin.channel.trace.diffraction.suffix import trace_reflected_suffix_from_edge_states
from witwin.channel.trace.diffraction.kernels.dda_slang import slang_dda_available


FREQUENCY = 1e9
WAVELENGTH = 299792458.0 / FREQUENCY
WAVENUMBER = 2.0 * math.pi / WAVELENGTH
TX_POLARIZATION = (1.0, 0.0, 0.0)
CUBE1_BASE_CENTER = (-2.5, -3.0, 1.5)
CUBE2_CENTER = (2.0, 0.5, 1.5)
CUBE3_CENTER = (-0.5, 3.5, 1.5)
CUBE_SIZE = 2.0
TX_POS = (0.0, -5.0, 1.5)
TRACE_BOUNDS = ((-6.0, 6.0), (-6.0, 6.0))
CALC_HEIGHT = 1.5
REFLECTION_COEF = 0.8
MAX_REFLECTIONS = 3
MAX_DIFFRACTIONS = 2
GRID_SIZE = 24
REFLECTION_N_RAYS = 256
FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad


def _build_main_scene():
    cube1 = box_drjit_geometry(center=CUBE1_BASE_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    cube2 = box_drjit_geometry(center=CUBE2_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    cube3 = box_drjit_geometry(center=CUBE3_CENTER, size=CUBE_SIZE, rotation=None).to_mesh()
    return build_test_scene(cube1, cube2, cube3)


def _trace_suffix_tx_x_gradient(*, execution: DiffractionExecutionConfig, enable_grad: bool = True) -> np.ndarray:
    scene = _build_main_scene()
    field = Field(bounds=TRACE_BOUNDS, size=(GRID_SIZE, GRID_SIZE))
    coords = field.get_coordinates()
    tx_x = wt.Float(TX_POS[0])
    if enable_grad:
        dr.enable_grad(tx_x)
        dr.set_grad(tx_x, 1.0)
    tx = wt.Point3f(tx_x, TX_POS[1], TX_POS[2])

    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=CALC_HEIGHT,
        tx_pos=tx,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        n_rays=REFLECTION_N_RAYS,
        max_reflections=MAX_REFLECTIONS,
        mode="2d",
        reflection_coef=REFLECTION_COEF,
        tx_polarization=TX_POLARIZATION,
        return_per_bounce=False,
        grid_data=coords,
    )
    _, _, state_arrays, _ = _prepare_diffraction_state_arrays(
        tx_pos=tx,
        rx_z=CALC_HEIGHT,
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        reflection_detail=reflection_detail,
        material_detail=None,
        reflection_n_rays=REFLECTION_N_RAYS,
        reflection_max_bounces=MAX_REFLECTIONS,
        reflection_coef=REFLECTION_COEF,
        reflection_mode="2d",
        max_diffractions=MAX_DIFFRACTIONS,
        tx_polarization=TX_POLARIZATION,
    )
    from witwin.channel.config import ReflectionSuffixConfig
    suffix, _ = trace_reflected_suffix_from_edge_states(
        state_arrays=state_arrays,
        suffix=ReflectionSuffixConfig(
            n_rays=REFLECTION_N_RAYS,
            max_bounces=MAX_REFLECTIONS,
            coef=REFLECTION_COEF,
            mode="2d",
            detail=reflection_detail,
            grid=field,
            grid_data=coords,
            rx_z=CALC_HEIGHT,
        ),
        scene=scene,
        wavelength=WAVELENGTH,
        k=WAVENUMBER,
        execution=execution,
    )

    if enable_grad:
        dr.forward_to(suffix, flags=FLAGS)
        grad_real = to_numpy(dr.grad(suffix.real)).reshape(GRID_SIZE, GRID_SIZE)
        grad_imag = to_numpy(dr.grad(suffix.imag)).reshape(GRID_SIZE, GRID_SIZE)
        return np.sqrt(grad_real * grad_real + grad_imag * grad_imag)

    suffix_real = to_numpy(suffix.real).reshape(GRID_SIZE, GRID_SIZE)
    suffix_imag = to_numpy(suffix.imag).reshape(GRID_SIZE, GRID_SIZE)
    return np.sqrt(suffix_real * suffix_real + suffix_imag * suffix_imag)


@pytest.mark.gpu
@pytest.mark.skipif(not slang_dda_available(), reason="slangtorch DDA module is unavailable")
def test_suffix_slang_rejects_differentiable_inputs():
    with pytest.raises(RuntimeError, match="does not support active gradients"):
        _trace_suffix_tx_x_gradient(
            execution=DiffractionExecutionConfig(
                suffix_dda="slang",
                accumulate_primal="drjit",
                accumulate_jvp="drjit_replay",
                accumulate_backward="drjit_replay",
            )
        )


@pytest.mark.gpu
def test_suffix_symbolic_rejects_differentiable_inputs():
    with pytest.raises(RuntimeError, match="requires non-differentiable suffix inputs"):
        _trace_suffix_tx_x_gradient(
            execution=DiffractionExecutionConfig(
                suffix_dda="symbolic",
                accumulate_primal="drjit",
                accumulate_jvp="drjit_replay",
                accumulate_backward="drjit_replay",
            )
        )


@pytest.mark.gpu
def test_suffix_symbolic_and_evaluated_produce_finite_forward_fields():
    symbolic = _trace_suffix_tx_x_gradient(
        execution=DiffractionExecutionConfig(
            suffix_dda="symbolic",
            accumulate_primal="drjit",
            accumulate_jvp="drjit_replay",
            accumulate_backward="drjit_replay",
        ),
        enable_grad=False,
    )
    evaluated = _trace_suffix_tx_x_gradient(
        execution=DiffractionExecutionConfig(
            suffix_dda="evaluated",
            accumulate_primal="drjit",
            accumulate_jvp="drjit_replay",
            accumulate_backward="drjit_replay",
        ),
        enable_grad=False,
    )

    assert float(np.linalg.norm(symbolic.ravel())) > 1e-8
    assert float(np.linalg.norm(evaluated.ravel())) > 1e-8
    assert np.isfinite(symbolic).all()
    assert np.isfinite(evaluated).all()
