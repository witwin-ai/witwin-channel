"""Regression tests for first-order reflection-prefix diffraction states."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import witwin as wt

import drjit as dr

from tests._scene_helpers import box_geometry, build_scene as build_test_scene, prism_geometry
from witwin.channel import Field
from witwin.channel.trace.diffraction import (
    _build_state_audit,
    _build_reflection_first_order_state_arrays,
)
from witwin.channel.trace import compute_reflection_field
def build_scene():
    return build_test_scene(
        box_geometry(center=(-2.5, 0.0, 1.5), size=2.0),
        prism_geometry(n_sides=3, center=(2.5, 0.0, 1.5), radius=1.0, height=3.0, rotation=0.0),
    )

def test_reflection_prefix_states_suppress_incident_slope():
    scene = build_scene()
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    wavelength = 299792458.0 / 1e9
    k = 2.0 * dr.pi / wavelength
    field = Field(bounds=((-6, 6), (-6, 6)), size=(20, 20))
    coords = field.get_coordinates()
    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=wt.Point3f(0.0, -5.0, 1.5),
        scene=scene,
        wavelength=wavelength,
        k=k,
        n_rays=512,
        max_reflections=1,
        reflection_coef=0.7,
        grid_data=coords,
    )

    assert reflection_detail["source_paths_per_bounce"][0]["n_paths"] > 0
    assert "path_prim_idx_0" in reflection_detail["source_paths_per_bounce"][0]
    assert int(np.asarray(reflection_detail["source_paths_per_bounce"][0]["discovery_count"])[0]) > 0

    states = _build_reflection_first_order_state_arrays(
        reflection_detail, scene, edge_data, wavelength, k, None, history_size=2
    )

    assert states["n_states"] > 0
    slope_power = float(dr.sum(
        states["incident_normal_derivative"].real * states["incident_normal_derivative"].real +
        states["incident_normal_derivative"].imag * states["incident_normal_derivative"].imag
    )[0])
    assert slope_power == 0.0, f"Expected zero reflected incident slope, got {slope_power:.6e}"

    audit = _build_state_audit(states, edge_data)
    assert all(label == "reflection_prefix" for label in audit["source_type"])
    assert all(depth == 1 for depth in np.asarray(audit["prefix_reflection_depth"]))
    assert all(depth == 0 for depth in np.asarray(audit["suffix_reflection_depth"]))
    assert all(label == "approx_sampled_reflection_prefix" for label in audit["approximation_mode"])
    assert all(label == "S -> R -> D" for label in audit["path_sequence"])


