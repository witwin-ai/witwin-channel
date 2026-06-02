"""Parity checks for the experimental Slang UTD accumulation bridge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import drjit as dr
import pytest
import witwin as wt

from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import DiffractionExecutionConfig, Field, compute_diffraction_field
from witwin.channel.utils.polarization import diffraction_edge_basis, jones_from_vector, path_basis
from witwin.channel.trace import compute_reflection_field
from witwin.channel.trace.diffraction.field import _edge_state_field_to_targets
from witwin.channel.trace.diffraction.kernels.utd_accumulate_slang import (
    accumulate_edge_state_totals_slang,
    slang_utd_accumulate_available,
)


FREQUENCY = 1e9
WAVELENGTH = 299792458.0 / FREQUENCY
WAVENUMBER = 2.0 * dr.pi / WAVELENGTH
TX_POLARIZATION = (0.0, 1.0, 0.0)


def _build_scene():
    cube1 = box_geometry(center=(-2.0, -2.0, 1.5), size=2.0)
    cube2 = box_geometry(center=(2.0, 1.5, 1.5), size=2.0)
    return build_test_scene(cube1, cube2)


def _field_power(field):
    return float(dr.sum(field.real * field.real + field.imag * field.imag)[0])


def _field_delta(lhs, rhs):
    return wt.Complex2f(lhs.real - rhs.real, lhs.imag - rhs.imag)


def _field_sum(lhs, rhs):
    return wt.Complex2f(lhs.real + rhs.real, lhs.imag + rhs.imag)


def _assert_field_close(lhs, rhs, *, atol_power=1e-6, rtol_power=1e-5):
    delta_power = _field_power(_field_delta(lhs, rhs))
    ref_power = max(_field_power(rhs), 1e-20)
    assert delta_power < max(atol_power, ref_power * rtol_power)


def _run_case(custom_op_enabled: bool, *, return_per_edge: bool = False):
    scene = _build_scene()
    field = Field(bounds=((-6.0, 6.0), (-6.0, 6.0)), size=(16, 16))
    coords = field.get_coordinates()
    tx = wt.Point3f(0.0, -5.0, 1.5)
    execution = (
        DiffractionExecutionConfig(
            accumulate_primal="custom_op_partitioned",
            accumulate_jvp="drjit_replay",
            accumulate_backward="drjit_replay",
            suffix_dda="symbolic",
        )
        if custom_op_enabled
        else DiffractionExecutionConfig.strict_drjit()
    )

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

    _, _, _, dif_components = compute_diffraction_field(
        coords["X"],
        coords["Y"],
        1.5,
        tx,
        scene,
        WAVELENGTH,
        WAVENUMBER,
        reflection_detail=reflection_detail,
        max_diffractions=1,
        reflection_n_rays=64,
        reflection_max_bounces=2,
        reflection_coef=0.82,
        reflection_mode="2d",
        grid=field,
        grid_data=coords,
        return_components=True,
        return_per_edge=return_per_edge,
        tx_polarization=TX_POLARIZATION,
        execution=execution,
    )
    return dif_components


def _synthetic_jones_operator_state():
    source_pos = wt.Point3f(-1.5, 0.2, 0.0)
    edge_pos = wt.Point3f(0.0, 0.0, 0.0)
    edge_dir = wt.Vector3f(0.0, 0.0, 1.0)
    n0 = wt.Vector3f(1.0, 0.0, 0.0)
    nn = wt.Vector3f(0.0, 1.0, 0.0)
    target_pos = wt.Point3f(0.3, 1.7, 0.4)
    incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    zero = wt.Complex2f(0.0, 0.0)
    state_arrays = {
        "n_states": 1,
        "edge_idx": wt.UInt32(0),
        "edge_pos": edge_pos,
        "edge_dir": edge_dir,
        "n0": n0,
        "nn": nn,
        "wedge_n": wt.Float(1.5),
        "adjacent_face0": wt.Int32(0),
        "adjacent_face1": wt.Int32(1),
        "source_pos": source_pos,
        "incident_field": zero,
        "incident_normal_derivative": zero,
        "r0": zero,
        "rn": zero,
        "incident_vector_x": zero,
        "incident_vector_y": zero,
        "incident_vector_z": zero,
        "incident_normal_derivative_vector_x": zero,
        "incident_normal_derivative_vector_y": zero,
        "incident_normal_derivative_vector_z": zero,
        "incident_jones_u": wt.Complex2f(1.0, 0.0),
        "incident_jones_v": zero,
        "incident_derivative_jones_u": zero,
        "incident_derivative_jones_v": zero,
        "incident_basis_u": incident_basis["u"],
        "incident_basis_v": incident_basis["v"],
        "incident_basis_k": incident_basis["k"],
        "face0_operator_m00": wt.Complex2f(1.0, 0.0),
        "face0_operator_m01": zero,
        "face0_operator_m10": zero,
        "face0_operator_m11": wt.Complex2f(-1.0, 0.0),
        "face1_operator_m00": zero,
        "face1_operator_m01": zero,
        "face1_operator_m10": zero,
        "face1_operator_m11": zero,
        "is_direct_tx": wt.Bool(True),
        "source_type_code": wt.UInt32(0),
        "prefix_reflection_depth": wt.UInt32(0),
        "intermediate_reflection_depth": wt.UInt32(0),
        "suffix_reflection_depth": wt.UInt32(0),
        "approximation_mode_code": wt.UInt32(0),
        "order": wt.UInt32(1),
        "path_edge_idx_0": wt.Int32(0),
        "path_reflection_depth_0": wt.UInt32(0),
    }
    return state_arrays, target_pos


@pytest.mark.gpu
@pytest.mark.skipif(not slang_utd_accumulate_available(), reason="slangtorch UTD accumulation module is unavailable")
def test_utd_accumulate_slang_matches_default():
    baseline = _run_case(custom_op_enabled=False)
    fused = _run_case(custom_op_enabled=True)

    _assert_field_close(fused["a_direct"], baseline["a_direct"])
    _assert_field_close(fused["a_multi"], baseline["a_multi"])
    _assert_field_close(
        _field_sum(fused["a_direct"], fused["a_multi"]),
        _field_sum(baseline["a_direct"], baseline["a_multi"]),
    )
    assert baseline["solver_metadata"]["execution"] == DiffractionExecutionConfig.strict_drjit().to_dict()
    assert fused["solver_metadata"]["execution"] == DiffractionExecutionConfig(
        accumulate_primal="custom_op_partitioned",
        accumulate_jvp="drjit_replay",
        accumulate_backward="drjit_replay",
        suffix_dda="symbolic",
    ).to_dict()


@pytest.mark.gpu
@pytest.mark.skipif(not slang_utd_accumulate_available(), reason="slangtorch UTD accumulation module is unavailable")
def test_utd_accumulate_slang_keeps_face_operator_vector_response():
    state_arrays, rx_pos = _synthetic_jones_operator_state()
    expected_field, expected_vector = _edge_state_field_to_targets(
        state_arrays,
        rx_pos,
        WAVENUMBER,
        return_vector=True,
        wavelength=WAVELENGTH,
        material_detail=None,
    )
    direct_total, multi_total, direct_vector_total, multi_vector_total = accumulate_edge_state_totals_slang(
        state_arrays,
        rx_pos,
        WAVENUMBER,
        wavelength=WAVELENGTH,
        material_detail=None,
    )
    outgoing_basis = diffraction_edge_basis(rx_pos - state_arrays["edge_pos"], state_arrays["edge_dir"], outgoing=True)
    direct_outgoing_jones = jones_from_vector(direct_vector_total, outgoing_basis)
    expected_outgoing_jones = jones_from_vector(expected_vector, outgoing_basis)

    assert _field_power(direct_total) < 1e-12
    assert _field_power(multi_total) < 1e-12
    assert _field_power(wt.Complex2f(direct_total.real - expected_field.real, direct_total.imag - expected_field.imag)) < 1e-12
    assert _field_power(multi_vector_total["x"]) + _field_power(multi_vector_total["y"]) + _field_power(multi_vector_total["z"]) < 1e-12
    assert _field_power(wt.Complex2f(direct_vector_total["x"].real - expected_vector["x"].real, direct_vector_total["x"].imag - expected_vector["x"].imag)) < 1e-10
    assert _field_power(wt.Complex2f(direct_vector_total["y"].real - expected_vector["y"].real, direct_vector_total["y"].imag - expected_vector["y"].imag)) < 1e-10
    assert _field_power(wt.Complex2f(direct_vector_total["z"].real - expected_vector["z"].real, direct_vector_total["z"].imag - expected_vector["z"].imag)) < 1e-10
    assert _field_power(direct_outgoing_jones["u"]) > 1e-12 or _field_power(direct_outgoing_jones["v"]) > 1e-12
    assert _field_power(wt.Complex2f(direct_outgoing_jones["u"].real - expected_outgoing_jones["u"].real, direct_outgoing_jones["u"].imag - expected_outgoing_jones["u"].imag)) < 1e-10
    assert _field_power(wt.Complex2f(direct_outgoing_jones["v"].real - expected_outgoing_jones["v"].real, direct_outgoing_jones["v"].imag - expected_outgoing_jones["v"].imag)) < 1e-10


@pytest.mark.gpu
def test_custom_op_partitioned_is_isolated_by_default(monkeypatch):
    monkeypatch.delenv("WITWIN_CHANNEL_ENABLE_EXPERIMENTAL_SLANG", raising=False)
    with pytest.raises(RuntimeError, match="isolated experimental Slang/custom-op diffraction path"):
        _run_case(custom_op_enabled=True, return_per_edge=True)
