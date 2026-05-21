import numpy as np

from witwin.channel.montecarlo import types as mc_wt
from witwin.channel.core.grid import Grid, GridSpec
import witwin.channel.montecarlo.integrators.basic as mc_basic
from witwin.channel.montecarlo.trace.diffraction import DiffractionStates
from witwin.channel.montecarlo.trace.postprocessing import ShadowBoundary


def _single_edge_state() -> DiffractionStates:
    material = {
        "eta_r": mc_wt.Float([5.0]),
        "sigma": mc_wt.Float([0.0]),
        "gain": mc_wt.Float([0.8]),
        "use_fresnel": mc_wt.Bool([True]),
    }
    return DiffractionStates(
        edge_index=mc_wt.Int32([0]),
        edge_pos=mc_wt.Point3f(
            mc_wt.Float([0.0]),
            mc_wt.Float([0.0]),
            mc_wt.Float([1.0]),
        ),
        edge_dir=mc_wt.Vector3f(
            mc_wt.Float([0.0]),
            mc_wt.Float([0.0]),
            mc_wt.Float([1.0]),
        ),
        n0=mc_wt.Vector3f(
            mc_wt.Float([1.0]),
            mc_wt.Float([0.0]),
            mc_wt.Float([0.0]),
        ),
        nn=mc_wt.Vector3f(
            mc_wt.Float([0.0]),
            mc_wt.Float([1.0]),
            mc_wt.Float([0.0]),
        ),
        wedge_n=mc_wt.Float([1.5]),
        edge_line_min=mc_wt.Float([-1.0]),
        edge_line_max=mc_wt.Float([1.0]),
        source_pos=mc_wt.Point3f(
            mc_wt.Float([-2.0]),
            mc_wt.Float([0.0]),
            mc_wt.Float([1.0]),
        ),
        adjacent_face0=mc_wt.Int32([0]),
        adjacent_face1=mc_wt.Int32([1]),
        face0_material=material,
        face1_material=material,
    )


def test_cell_center_shadow_boundary_weights_are_finite_and_bounded() -> None:
    grid = Grid.from_spec(
        GridSpec(
            axis="z",
            position=1.0,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(64, 64),
        )
    )
    weights = ShadowBoundary._accumulate_state_weights(
        states=_single_edge_state(),
        grid=grid,
        k=2.0 * np.pi / 0.085,
    )

    incident = np.asarray(
        weights["incident_shadow_boundary_weight"],
        dtype=np.float32,
    )
    reflection = np.asarray(
        weights["reflection_shadow_boundary_weight"],
        dtype=np.float32,
    )
    incident_response = np.asarray(
        weights["incident_transition_response_real"],
        dtype=np.float32,
    ) + 1j * np.asarray(
        weights["incident_transition_response_imag"],
        dtype=np.float32,
    )
    reflection_response = np.asarray(
        weights["reflection_transition_response_real"],
        dtype=np.float32,
    ) + 1j * np.asarray(
        weights["reflection_transition_response_imag"],
        dtype=np.float32,
    )

    assert np.all(np.isfinite(incident))
    assert np.all(np.isfinite(reflection))
    assert np.all(np.isfinite(incident_response))
    assert np.all(np.isfinite(reflection_response))
    assert np.min(incident) >= 0.0
    assert np.max(incident) <= 1.0
    assert np.min(reflection) >= 0.0
    assert np.max(reflection) <= 1.0
    assert np.max(incident) > 0.5
    assert np.max(reflection) > 0.5
    assert np.max(np.abs(incident_response)) > 0.0
    assert np.max(np.abs(reflection_response)) > 0.0


def test_shadow_boundary_weights_respect_finite_edge_line_bounds() -> None:
    grid = Grid.from_spec(
        GridSpec(
            axis="z",
            position=1.0,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(64, 64),
        )
    )
    finite_state = _single_edge_state()
    zero_length_state = _single_edge_state()
    zero_length_state.edge_line_min = mc_wt.Float([0.0])
    zero_length_state.edge_line_max = mc_wt.Float([0.0])

    finite_weights = ShadowBoundary._accumulate_state_weights(
        states=finite_state,
        grid=grid,
        k=2.0 * np.pi / 0.085,
    )
    zero_length_weights = ShadowBoundary._accumulate_state_weights(
        states=zero_length_state,
        grid=grid,
        k=2.0 * np.pi / 0.085,
    )

    finite_incident = np.asarray(
        finite_weights["incident_shadow_boundary_weight"],
        dtype=np.float32,
    )
    zero_incident = np.asarray(
        zero_length_weights["incident_shadow_boundary_weight"],
        dtype=np.float32,
    )
    zero_reflection = np.asarray(
        zero_length_weights["reflection_shadow_boundary_weight"],
        dtype=np.float32,
    )
    zero_response = np.asarray(
        zero_length_weights["incident_transition_response_real"],
        dtype=np.float32,
    ) + 1j * np.asarray(
        zero_length_weights["incident_transition_response_imag"],
        dtype=np.float32,
    )

    assert np.max(finite_incident) > 0.5
    assert np.all(zero_incident == 0.0)
    assert np.all(zero_reflection == 0.0)
    assert np.all(zero_response == 0.0)


def test_finite_edge_shadow_boundary_support_tapers_near_segment_endpoints() -> None:
    target_pos = mc_wt.Point3f(
        mc_wt.Float([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        mc_wt.Float([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mc_wt.Float([-2.2, -2.0, -1.8, 0.0, 1.8, 2.0, 2.2]),
    )
    _, finite_scale, finite_support_weight = ShadowBoundary._finite_edge_factor(
        source_pos=mc_wt.Point3f(-2.0, 0.0, 0.0),
        target_pos=target_pos,
        edge_pos=mc_wt.Point3f(0.0, 0.0, 0.0),
        edge_dir=mc_wt.Vector3f(0.0, 0.0, 1.0),
        edge_line_min=mc_wt.Float([-1.0]),
        edge_line_max=mc_wt.Float([1.0]),
        k=2.0 * np.pi / 0.085,
    )

    support = np.asarray(finite_support_weight, dtype=np.float32)
    scale = np.asarray(finite_scale, dtype=np.float32)

    assert np.all(support >= 0.0)
    assert np.all(support <= 1.0)
    assert support[0] < support[1] < support[2] <= support[3]
    assert support[6] < support[5] < support[4] <= support[3]
    assert 0.0 < support[1] < 1.0
    assert 0.0 < support[5] < 1.0
    assert scale[0] > 0.0
    assert scale[-1] > 0.0


def test_power_domain_smoothing_reduces_incident_shadow_step() -> None:
    smoothed = mc_basic._empty_radio_map(4)
    smoothed["incoherent"]["los"] = mc_wt.Float([1.0, 1.0, 0.0, 0.0])
    smoothed["incoherent"]["diffraction"] = mc_wt.Float([0.25, 0.25, 0.25, 0.25])
    smoothed["incoherent"]["continued_incident_power"] = mc_wt.Float(
        [1.0, 1.0, 1.0, 1.0]
    )
    smoothed["incoherent"]["incident_shadow_boundary_weight"] = mc_wt.Float(
        [1.0, 1.0, 1.0, 1.0]
    )

    mc_basic._finalize_component_totals(
        smoothed,
        shadow_boundary_mode="utd_power_smoothing",
    )

    raw = np.asarray(smoothed["incoherent"]["raw_total"], dtype=np.float32)
    total = np.asarray(smoothed["incoherent"]["total"], dtype=np.float32)
    correction = np.asarray(
        smoothed["incoherent"]["shadow_boundary_correction"],
        dtype=np.float32,
    )

    assert np.max(np.abs(np.diff(total))) < np.max(np.abs(np.diff(raw)))
    assert correction[0] < 0.0
    assert correction[-1] > 0.0
    assert np.all(total >= 0.0)


def test_geometry_shadow_weight_smooths_without_sampled_proxy_power() -> None:
    smoothed = mc_basic._empty_radio_map(4)
    smoothed["incoherent"]["los"] = mc_wt.Float([1.0, 1.0, 0.0, 0.0])
    smoothed["incoherent"]["diffraction"] = mc_wt.Float([0.02, 0.02, 0.02, 0.02])
    smoothed["incoherent"]["continued_incident_power"] = mc_wt.Float(
        [1.0, 1.0, 1.0, 1.0]
    )
    smoothed["incoherent"]["incident_shadow_boundary_weight"] = mc_wt.Float(
        [0.0, 0.5, 0.0, 0.0]
    )

    mc_basic._finalize_component_totals(
        smoothed,
        shadow_boundary_mode="utd_power_smoothing",
    )

    raw = np.asarray(smoothed["incoherent"]["raw_total"], dtype=np.float32)
    total = np.asarray(smoothed["incoherent"]["total"], dtype=np.float32)
    correction = np.asarray(
        smoothed["incoherent"]["shadow_boundary_correction"],
        dtype=np.float32,
    )
    sampled_proxy = np.asarray(
        smoothed["incoherent"]["diffraction_incident_transition_power"],
        dtype=np.float32,
    )

    assert np.max(np.abs(np.diff(total))) < np.max(np.abs(np.diff(raw)))
    assert np.all(sampled_proxy == 0.0)
    assert correction[1] < 0.0
    assert np.all(np.isfinite(correction))


def test_power_domain_smoothing_reduces_reflection_shadow_step() -> None:
    smoothed = mc_basic._empty_radio_map(4)
    smoothed["incoherent"]["reflection"] = mc_wt.Float([0.64, 0.64, 0.0, 0.0])
    smoothed["incoherent"]["diffraction"] = mc_wt.Float([0.16, 0.16, 0.16, 0.16])
    smoothed["incoherent"]["reflection_shadow_boundary_weight"] = mc_wt.Float(
        [1.0, 1.0, 1.0, 1.0]
    )

    mc_basic._finalize_component_totals(
        smoothed,
        shadow_boundary_mode="utd_power_smoothing",
    )
    no_smoothing = mc_basic._empty_radio_map(4)
    no_smoothing["incoherent"]["reflection"] = smoothed["incoherent"]["reflection"]
    no_smoothing["incoherent"]["diffraction"] = smoothed["incoherent"]["diffraction"]
    mc_basic._finalize_component_totals(
        no_smoothing,
        shadow_boundary_mode="none",
    )

    total = np.asarray(smoothed["incoherent"]["total"], dtype=np.float32)
    raw_total = np.asarray(no_smoothing["incoherent"]["total"], dtype=np.float32)
    correction = np.asarray(
        smoothed["incoherent"]["shadow_boundary_correction"],
        dtype=np.float32,
    )

    assert np.max(np.abs(np.diff(total))) < np.max(np.abs(np.diff(raw_total)))
    assert np.all(correction <= 0.0)
    np.testing.assert_allclose(
        np.asarray(no_smoothing["incoherent"]["raw_total"], dtype=np.float32),
        raw_total,
    )


def test_shadow_boundary_correction_does_not_use_hard_reflection_fallback() -> None:
    smoothed = mc_basic._empty_radio_map(4)
    smoothed["incoherent"]["reflection"] = mc_wt.Float([0.0, 0.0, 0.64, 0.64])
    smoothed["incoherent"]["reflection_shadow_boundary_weight"] = mc_wt.Float(
        [0.0, 0.25, 0.25, 0.0]
    )

    mc_basic._finalize_component_totals(
        smoothed,
        shadow_boundary_mode="utd_power_smoothing",
    )

    total = np.asarray(smoothed["incoherent"]["total"], dtype=np.float32)
    correction = np.asarray(
        smoothed["incoherent"]["shadow_boundary_correction"],
        dtype=np.float32,
    )

    np.testing.assert_allclose(total[1], 0.0, atol=1.0e-7)
    np.testing.assert_allclose(correction[1], 0.0, atol=1.0e-7)
    assert total[2] < 0.64
    assert correction[2] < 0.0
