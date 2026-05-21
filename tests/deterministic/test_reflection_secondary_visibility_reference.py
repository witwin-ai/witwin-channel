from __future__ import annotations

import drjit as dr
import numpy as np
import pytest

import witwin.channel as wt
from witwin.channel.core.runtime import Rx, Tx, Wave
from witwin.channel.core.scene import EdgePolicy, Scene
from witwin.channel.deterministic.kernels.reflection import native_impl
from witwin.channel.deterministic.reflection import epc
from witwin.channel.deterministic.reflection.detail import build_trace_detail
from witwin.channel.deterministic.reflection.paths import enumerate_first_bounce_surface_paths
from witwin.core import Material, Mesh, Structure


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _bool_scalar(value) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _vector_power(vec) -> float:
    total = dr.zeros(wt.Float, dr.width(vec["x"].real))
    for axis in ("x", "y", "z"):
        total += vec[axis].real * vec[axis].real + vec[axis].imag * vec[axis].imag
    return _scalar(total)


def _edge_policy() -> EdgePolicy:
    return EdgePolicy(
        edge_diffraction=True,
        boundary_edge_policy="half_plane",
        edge_selection_mode="all_edges",
    )


def _reflector_mesh() -> Mesh:
    return Mesh(
        vertices=(
            (-1.0, 0.0, -1.0),
            (1.0, 0.0, -1.0),
            (-1.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
        ),
        faces=((0, 1, 3), (0, 3, 2)),
        device="cpu",
    )


def _blocker_mesh() -> Mesh:
    return Mesh(
        vertices=(
            (-0.1, -0.5, -1.0),
            (0.8, -0.5, -1.0),
            (-0.1, -0.5, 1.0),
            (0.8, -0.5, 1.0),
        ),
        faces=((0, 3, 1), (0, 2, 3)),
        recenter=False,
        device="cpu",
    )


def _scene(*, blocker: bool) -> Scene:
    structures = [
        Structure(
            geometry=_reflector_mesh(),
            material=Material(eps_r=4.0, sigma_e=0.0),
            name="reflector",
        )
    ]
    if blocker:
        structures.append(
            Structure(
                geometry=_blocker_mesh(),
                material=Material(eps_r=4.0, sigma_e=0.0),
                name="blocker",
            )
        )
    return Scene(structures=structures, device="cpu")


def _chain(scene: Scene, *, rx_x: float, secondary_mode: str, transition_mode: str = "f_weight_reference"):
    edge_policy = _edge_policy()
    scene.diffraction_edge_count(edge_policy=edge_policy)
    tx = Tx(position=(0.0, -1.0, 0.0), polarization=(1.0, 0.0, 0.0))
    wave = Wave(wavelength=0.1)
    paths = enumerate_first_bounce_surface_paths(tx=tx, tri_data=scene._triangle_runtime())
    detail = build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="secondary-visibility-test",
        reflection_gain=1.0,
        source_paths_per_bounce=(paths,),
        reflection_transition_mode=transition_mode,
        reflection_f_weight_boundary_radius_wavelengths=20.0,
        reflection_f_weight_max_edges_per_slot=1,
        reflection_secondary_visibility_mode=secondary_mode,
    )
    return epc.chain_to_target(
        paths=paths,
        path_idx=wt.UInt32(0),
        target_pos=wt.Point3f(rx_x, -1.0, 0.0),
        scene=scene,
        target_adjacent_faces=(),
        reflection_detail=detail,
        wave=wave,
        tx=tx,
        return_endpoints=True,
    )


def test_secondary_visibility_clear_segment_matches_primary_only() -> None:
    scene = _scene(blocker=False)

    primary_valid, primary_vec, _ = _chain(scene, rx_x=0.99, secondary_mode="hard")
    secondary_valid, secondary_vec, _ = _chain(scene, rx_x=0.99, secondary_mode="f_weight")

    assert _bool_scalar(primary_valid) is True
    assert _bool_scalar(secondary_valid) is True
    assert _vector_power(secondary_vec) == pytest.approx(_vector_power(primary_vec), rel=1e-6, abs=1e-12)


def test_secondary_visibility_replaces_hard_receiver_segment_kill_near_silhouette() -> None:
    scene = _scene(blocker=True)

    hard_valid, hard_vec, _ = _chain(scene, rx_x=0.99, secondary_mode="hard")
    secondary_valid, secondary_vec, _ = _chain(scene, rx_x=0.99, secondary_mode="f_weight")

    assert _bool_scalar(hard_valid) is False
    assert _vector_power(hard_vec) == pytest.approx(0.0, abs=1e-12)
    assert _bool_scalar(secondary_valid) is True
    assert _vector_power(secondary_vec) > 0.0


def test_secondary_visibility_is_independent_of_primary_transition_mode() -> None:
    scene = _scene(blocker=True)

    valid, vec, _ = _chain(scene, rx_x=0.99, secondary_mode="f_weight", transition_mode="hard")

    assert _bool_scalar(valid) is True
    assert _vector_power(vec) > 0.0


def test_secondary_visibility_rejects_unimplemented_native_cuda_path() -> None:
    scene = _scene(blocker=True)
    edge_policy = _edge_policy()
    scene.diffraction_edge_count(edge_policy=edge_policy)
    tx = Tx(position=(0.0, -1.0, 0.0), polarization=(1.0, 0.0, 0.0))
    rx = Rx(positions=[(0.99, -1.0, 0.0)])
    wave = Wave(wavelength=0.1)
    paths = enumerate_first_bounce_surface_paths(tx=tx, tri_data=scene._triangle_runtime())
    detail = build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="secondary-visibility-test",
        reflection_gain=1.0,
        source_paths_per_bounce=(paths,),
        reflection_transition_mode="f_weight_native",
        reflection_f_weight_boundary_radius_wavelengths=20.0,
        reflection_f_weight_max_edges_per_slot=1,
        reflection_secondary_visibility_mode="f_weight",
    )

    with pytest.raises(RuntimeError, match="native CUDA secondary-visibility"):
        native_impl.reflection_accumulate_forward(
            rx=rx,
            tx=tx,
            scene=scene,
            wave=wave,
            source_paths_per_bounce=[paths],
            reflection_detail=detail,
        )
