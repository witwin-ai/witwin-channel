from __future__ import annotations

import drjit as dr
import numpy as np
import pytest

import witwin.channel as wt
from witwin.channel.core.runtime import Tx, Wave
from witwin.channel.core.scene import EdgePolicy, Scene
from witwin.channel.deterministic.kernels.reflection import native_impl
from witwin.channel.deterministic.reflection import epc
from witwin.channel.deterministic.reflection.detail import build_trace_detail
from witwin.channel.deterministic.reflection.paths import (
    accumulate_paths_exact,
    enumerate_first_bounce_surface_paths,
)
from witwin.core import Material, Mesh, Structure


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _bool_scalar(value) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _open_wall_scene() -> Scene:
    mesh = Mesh(
        vertices=(
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 3.0),
            (1.0, 0.0, 3.0),
        ),
        faces=((0, 1, 3), (0, 3, 2)),
        device="cpu",
    )
    return Scene(
        structures=[Structure(geometry=mesh, material=Material(eps_r=4.0, sigma_e=0.0), name="open_wall")],
        device="cpu",
    )


def _two_face_wedge_scene() -> Scene:
    mesh = Mesh(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        faces=((0, 1, 2), (0, 3, 1)),
        device="cpu",
    )
    return Scene(
        structures=[Structure(geometry=mesh, material=Material(eps_r=4.0, sigma_e=0.0), name="wedge")],
        device="cpu",
    )


def _edge_policy() -> EdgePolicy:
    return EdgePolicy(
        edge_diffraction=True,
        boundary_edge_policy="half_plane",
        edge_selection_mode="all_edges",
    )


def _chain(
    scene: Scene,
    *,
    mode: str,
    target_x: float = 0.0,
    target_y: float = -1.0,
    target_z: float = 0.0,
    tx_position=(0.0, -1.0, 0.0),
    path_idx=0,
):
    tx = Tx(position=tx_position, polarization=(1.0, 0.0, 0.0))
    wave = Wave(wavelength=0.1)
    paths = enumerate_first_bounce_surface_paths(tx=tx, tri_data=scene._triangle_runtime())
    detail = build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="test",
        reflection_gain=1.0,
        source_paths_per_bounce=(paths,),
        reflection_transition_mode=mode,
        reflection_f_weight_boundary_radius_wavelengths=2.0,
        reflection_f_weight_max_edges_per_slot=1,
    )
    return epc.chain_to_target(
        paths=paths,
        path_idx=wt.UInt32(path_idx),
        target_pos=wt.Point3f(target_x, target_y, target_z),
        scene=scene,
        target_adjacent_faces=(),
        reflection_detail=detail,
        wave=wave,
        tx=tx,
        return_endpoints=True,
    )


def _source_paths(scene: Scene, tx: Tx):
    return enumerate_first_bounce_surface_paths(tx=tx, tri_data=scene._triangle_runtime())


def _detail(*, paths, mode: str, radius: float = 2.0):
    return build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="test",
        reflection_gain=1.0,
        source_paths_per_bounce=(paths,),
        reflection_transition_mode=mode,
        reflection_f_weight_boundary_radius_wavelengths=radius,
        reflection_f_weight_max_edges_per_slot=1,
    )


def _vector_power(vec) -> float:
    return _scalar(_vector_power_array(vec))


def _vector_power_array(vec):
    total = dr.zeros(wt.Float, dr.width(vec["x"].real))
    for axis in ("x", "y", "z"):
        total += vec[axis].real * vec[axis].real + vec[axis].imag * vec[axis].imag
    return total


def test_reference_f_weight_matches_hard_far_from_boundary() -> None:
    scene = _open_wall_scene()
    scene.diffraction_edge_count(edge_policy=_edge_policy())

    hard_valid, hard_vec, _ = _chain(scene, mode="hard", target_x=0.0)
    ref_valid, ref_vec, _ = _chain(scene, mode="f_weight_reference", target_x=0.0)

    assert _bool_scalar(hard_valid) is True
    assert _bool_scalar(ref_valid) is True
    assert _vector_power(ref_vec) == pytest.approx(_vector_power(hard_vec), rel=1e-6, abs=1e-8)


def test_reference_f_weight_attenuates_primary_reflection_near_boundary() -> None:
    scene = _open_wall_scene()
    scene.diffraction_edge_count(edge_policy=_edge_policy())

    hard_valid, hard_vec, _ = _chain(scene, mode="hard", target_x=1.8)
    ref_valid, ref_vec, _ = _chain(scene, mode="f_weight_reference", target_x=1.8)

    assert _bool_scalar(hard_valid) is True
    assert _bool_scalar(ref_valid) is True
    assert 0.0 < _vector_power(ref_vec) < _vector_power(hard_vec)


def test_reference_f_weight_emits_adjacent_residual_after_crossing_wedge_edge() -> None:
    scene = _two_face_wedge_scene()
    scene.diffraction_edge_count(edge_policy=_edge_policy())

    ref_valid, ref_vec, _ = _chain(
        scene,
        mode="f_weight_reference",
        tx_position=(0.0, -0.51, 0.0),
        target_x=0.0,
        target_y=-0.51,
        target_z=0.0,
        path_idx=0,
    )

    assert _bool_scalar(ref_valid) is True
    assert _vector_power(ref_vec) > 0.0


def test_reference_f_weight_near_boundary_gradient_matches_finite_difference() -> None:
    scene = _open_wall_scene()
    scene.diffraction_edge_count(edge_policy=_edge_policy())

    target_x = wt.Float(1.8)
    dr.enable_grad(target_x)
    _, vec, _ = _chain(scene, mode="f_weight_reference", target_x=target_x)
    loss = _vector_power_array(vec)
    dr.backward(loss)
    ad_grad = _scalar(dr.grad(target_x))

    h = 1.0e-3
    _, vec_plus, _ = _chain(scene, mode="f_weight_reference", target_x=1.8 + h)
    _, vec_minus, _ = _chain(scene, mode="f_weight_reference", target_x=1.8 - h)
    fd_grad = (_vector_power(vec_plus) - _vector_power(vec_minus)) / (2.0 * h)

    assert np.isfinite(ad_grad)
    assert ad_grad == pytest.approx(fd_grad, rel=5e-2, abs=5e-4)


def test_accumulation_applies_reference_f_weight_not_just_valid_mask() -> None:
    scene = _open_wall_scene()
    scene.diffraction_edge_count(edge_policy=_edge_policy())
    tx = Tx(position=(0.0, -1.0, 0.0), polarization=(1.0, 0.0, 0.0))
    wave = Wave(wavelength=0.1)
    paths = _source_paths(scene, tx)
    rx = type("RxStub", (), {
        "positions": wt.Point3f(1.8, -1.0, 0.0),
        "polarization": None,
        "effective_polarization": lambda self, tx_arg: tx_arg.polarization,
    })()

    hard_vec = accumulate_paths_exact(
        rx=rx,
        tx=tx,
        scene=scene,
        wave=wave,
        source_paths_per_bounce=(paths,),
        reflection_detail=_detail(paths=paths, mode="hard"),
    )[0]
    ref_vec = accumulate_paths_exact(
        rx=rx,
        tx=tx,
        scene=scene,
        wave=wave,
        source_paths_per_bounce=(paths,),
        reflection_detail=_detail(paths=paths, mode="f_weight_reference"),
    )[0]

    assert 0.0 < _vector_power(ref_vec) < _vector_power(hard_vec)


def test_f_weight_native_matches_reference_primary_transition() -> None:
    scene = _open_wall_scene()
    scene.diffraction_edge_count(edge_policy=_edge_policy())

    ref_valid, ref_vec, _ = _chain(scene, mode="f_weight_reference", target_x=1.8)
    native_valid, native_vec, _ = _chain(scene, mode="f_weight_native", target_x=1.8)

    assert _bool_scalar(native_valid) == _bool_scalar(ref_valid)
    assert _vector_power(native_vec) == pytest.approx(_vector_power(ref_vec), rel=1e-6, abs=1e-10)


def test_f_weight_native_accumulation_matches_reference_replay() -> None:
    scene = _open_wall_scene()
    scene.diffraction_edge_count(edge_policy=_edge_policy())
    tx = Tx(position=(0.0, -1.0, 0.0), polarization=(1.0, 0.0, 0.0))
    wave = Wave(wavelength=0.1)
    paths = _source_paths(scene, tx)
    rx = type("RxStub", (), {
        "positions": wt.Point3f(1.8, -1.0, 0.0),
        "polarization": None,
        "effective_polarization": lambda self, tx_arg: tx_arg.polarization,
    })()

    ref_vec = accumulate_paths_exact(
        rx=rx,
        tx=tx,
        scene=scene,
        wave=wave,
        source_paths_per_bounce=(paths,),
        reflection_detail=_detail(paths=paths, mode="f_weight_reference"),
    )[0]
    native_vec = accumulate_paths_exact(
        rx=rx,
        tx=tx,
        scene=scene,
        wave=wave,
        source_paths_per_bounce=(paths,),
        reflection_detail=_detail(paths=paths, mode="f_weight_native"),
    )[0]

    assert _vector_power(native_vec) == pytest.approx(_vector_power(ref_vec), rel=1e-6, abs=1e-12)


def test_f_weight_native_accumulation_uses_native_fast_path_far_from_boundary(monkeypatch) -> None:
    scene = _open_wall_scene()
    scene.diffraction_edge_count(edge_policy=_edge_policy())
    tx = Tx(position=(0.0, -1.0, 0.0), polarization=(1.0, 0.0, 0.0))
    wave = Wave(wavelength=0.1)
    paths = _source_paths(scene, tx)
    rx = type("RxStub", (), {
        "positions": wt.Point3f(0.0, -1.0, 0.0),
        "polarization": None,
        "effective_polarization": lambda self, tx_arg: tx_arg.polarization,
    })()
    calls = {"native": 0, "replay": 0}
    native_accumulate = native_impl._accumulate_reflection_f_weight_chunk_arrays
    replay_accumulate = native_impl._reflection_accumulate_f_weight_reference

    def native_spy(**kwargs):
        calls["native"] += 1
        return native_accumulate(**kwargs)

    def replay_spy(**kwargs):
        calls["replay"] += 1
        return replay_accumulate(**kwargs)

    monkeypatch.setattr(native_impl, "_accumulate_reflection_f_weight_chunk_arrays", native_spy)
    monkeypatch.setattr(native_impl, "_reflection_accumulate_f_weight_reference", replay_spy)

    accumulate_paths_exact(
        rx=rx,
        tx=tx,
        scene=scene,
        wave=wave,
        source_paths_per_bounce=(paths,),
        reflection_detail=_detail(paths=paths, mode="f_weight_native"),
    )

    assert calls["native"] > 0
    assert calls["replay"] == 0


def test_f_weight_native_accumulation_does_not_replay_near_boundary(monkeypatch) -> None:
    scene = _open_wall_scene()
    scene.diffraction_edge_count(edge_policy=_edge_policy())
    tx = Tx(position=(0.0, -1.0, 0.0), polarization=(1.0, 0.0, 0.0))
    wave = Wave(wavelength=0.1)
    paths = _source_paths(scene, tx)
    rx = type("RxStub", (), {
        "positions": wt.Point3f(1.8, -1.0, 0.0),
        "polarization": None,
        "effective_polarization": lambda self, tx_arg: tx_arg.polarization,
    })()

    def fail_reference_replay(**kwargs):
        raise AssertionError("f_weight_native must not replay reference F-weight pairs")

    monkeypatch.setattr(native_impl, "_scatter_f_weight_reference_pairs", fail_reference_replay)
    monkeypatch.setattr(native_impl, "_reflection_accumulate_f_weight_reference", fail_reference_replay)

    native_vec = accumulate_paths_exact(
        rx=rx,
        tx=tx,
        scene=scene,
        wave=wave,
        source_paths_per_bounce=(paths,),
        reflection_detail=_detail(paths=paths, mode="f_weight_native"),
    )[0]

    assert _vector_power(native_vec) > 0.0


def test_f_weight_native_accumulation_matches_reference_adjacent_residual(monkeypatch) -> None:
    scene = _two_face_wedge_scene()
    scene.diffraction_edge_count(edge_policy=_edge_policy())
    tx = Tx(position=(0.0, -0.51, 0.0), polarization=(1.0, 0.0, 0.0))
    wave = Wave(wavelength=0.1)
    paths = _source_paths(scene, tx)
    rx = type("RxStub", (), {
        "positions": wt.Point3f(0.0, -0.51, 0.0),
        "polarization": None,
        "effective_polarization": lambda self, tx_arg: tx_arg.polarization,
    })()

    ref_vec = accumulate_paths_exact(
        rx=rx,
        tx=tx,
        scene=scene,
        wave=wave,
        source_paths_per_bounce=(paths,),
        reflection_detail=_detail(paths=paths, mode="f_weight_reference"),
    )[0]

    def fail_reference_replay(**kwargs):
        raise AssertionError("f_weight_native adjacent residual must be CUDA-side")

    monkeypatch.setattr(native_impl, "_scatter_f_weight_reference_pairs", fail_reference_replay)
    native_vec = accumulate_paths_exact(
        rx=rx,
        tx=tx,
        scene=scene,
        wave=wave,
        source_paths_per_bounce=(paths,),
        reflection_detail=_detail(paths=paths, mode="f_weight_native"),
    )[0]

    assert _vector_power(native_vec) == pytest.approx(_vector_power(ref_vec), rel=1e-5, abs=1e-10)


def test_f_weight_native_accumulation_jvp_matches_finite_difference() -> None:
    scene = _open_wall_scene()
    scene.diffraction_edge_count(edge_policy=_edge_policy())
    tx = Tx(position=(0.0, -1.0, 0.0), polarization=(1.0, 0.0, 0.0))
    wave = Wave(wavelength=0.1)
    paths = _source_paths(scene, tx)

    target_x = wt.Float(1.8)
    dr.enable_grad(target_x)
    rx = type("RxStub", (), {
        "positions": wt.Point3f(target_x, -1.0, 0.0),
        "polarization": None,
        "effective_polarization": lambda self, tx_arg: tx_arg.polarization,
    })()
    vec = accumulate_paths_exact(
        rx=rx,
        tx=tx,
        scene=scene,
        wave=wave,
        source_paths_per_bounce=(paths,),
        reflection_detail=_detail(paths=paths, mode="f_weight_native"),
    )[0]
    loss = _vector_power_array(vec)
    dr.set_grad(target_x, 1.0)
    ad_grad = _scalar(
        dr.forward_to(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    )

    h = 1.0e-3
    rx_plus = type("RxStub", (), {
        "positions": wt.Point3f(1.8 + h, -1.0, 0.0),
        "polarization": None,
        "effective_polarization": lambda self, tx_arg: tx_arg.polarization,
    })()
    rx_minus = type("RxStub", (), {
        "positions": wt.Point3f(1.8 - h, -1.0, 0.0),
        "polarization": None,
        "effective_polarization": lambda self, tx_arg: tx_arg.polarization,
    })()
    vec_plus = accumulate_paths_exact(
        rx=rx_plus,
        tx=tx,
        scene=scene,
        wave=wave,
        source_paths_per_bounce=(paths,),
        reflection_detail=_detail(paths=paths, mode="f_weight_native"),
    )[0]
    vec_minus = accumulate_paths_exact(
        rx=rx_minus,
        tx=tx,
        scene=scene,
        wave=wave,
        source_paths_per_bounce=(paths,),
        reflection_detail=_detail(paths=paths, mode="f_weight_native"),
    )[0]
    fd_grad = (_vector_power(vec_plus) - _vector_power(vec_minus)) / (2.0 * h)

    assert np.isfinite(ad_grad)
    assert ad_grad == pytest.approx(fd_grad, rel=5e-2, abs=5e-4)
