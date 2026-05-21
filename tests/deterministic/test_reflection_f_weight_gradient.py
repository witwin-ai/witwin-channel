from __future__ import annotations

from dataclasses import replace

import drjit as dr
from drjit.opt import Adam
import numpy as np
import pytest

from examples.deterministic_radiomap_three_cubes import ThreeCubeExperiment
import witwin.channel as wt
from witwin.channel.core.runtime import Tx, Wave
from witwin.channel.core.scene import EdgePolicy, Mesh as DrJitMesh, Scene
from witwin.channel.deterministic.reflection.detail import build_trace_detail
from witwin.channel.deterministic.reflection.paths import (
    accumulate_paths_exact,
    enumerate_first_bounce_surface_paths,
)
from witwin.core import Material, Structure


pytestmark = [pytest.mark.gpu, pytest.mark.acceptance]

WAVELENGTH = 0.1
FD_STEP = WAVELENGTH / 1000.0


def _edge_policy() -> EdgePolicy:
    return EdgePolicy(
        edge_diffraction=True,
        boundary_edge_policy="half_plane",
        edge_selection_mode="all_edges",
    )


def _double_slit_scene(*, strip_x=0.0, strip_yaw=0.0) -> Scene:
    base_x = wt.Float([-1.05, -0.25, -1.05, -0.25, 0.25, 1.05, 0.25, 1.05])
    base_y = wt.Float([0.0] * 8)
    base_z = wt.Float([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    yaw = wt.Float(strip_yaw)
    c = dr.cos(yaw)
    s = dr.sin(yaw)
    vertices = wt.Point3f(base_x * c - base_y * s + wt.Float(strip_x), base_x * s + base_y * c, base_z)
    mesh = DrJitMesh(
        vertices=vertices,
        faces=((0, 1, 3), (0, 3, 2), (4, 5, 7), (4, 7, 6)),
    )
    scene = Scene(
        structures=[
            Structure(
                geometry=mesh,
                material=Material(eps_r=4.0, sigma_e=0.0),
                name="double_slit_strips",
            )
        ],
        device="cpu",
    )
    scene.diffraction_edge_count(edge_policy=_edge_policy())
    return scene


def _receiver_line(n: int = 41):
    xs = np.linspace(-1.45, 1.45, n, dtype=np.float32)
    return type("RxStub", (), {
        "positions": wt.Point3f(xs.tolist(), [-1.0] * n, [0.5] * n),
        "polarization": None,
        "effective_polarization": lambda self, tx_arg: tx_arg.polarization,
    })()


def _detail(*, paths, mode: str):
    return build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="double-slit-acceptance",
        reflection_gain=1.0,
        source_paths_per_bounce=(paths,),
        reflection_transition_mode=mode,
        reflection_f_weight_boundary_radius_wavelengths=2.0,
        reflection_f_weight_max_edges_per_slot=1,
    )


def _vector_power(vec):
    total = dr.zeros(wt.Float, dr.width(vec["x"].real))
    for axis in ("x", "y", "z"):
        total += vec[axis].real * vec[axis].real + vec[axis].imag * vec[axis].imag
    return total


def _double_slit_power(*, tx_x=0.0, strip_x=0.0, strip_yaw=0.0, mode: str = "f_weight_native"):
    scene = _double_slit_scene(strip_x=strip_x, strip_yaw=strip_yaw)
    tx = Tx(position=wt.Point3f(tx_x, -1.0, 0.5), polarization=(1.0, 0.0, 0.0))
    wave = Wave(wavelength=WAVELENGTH)
    paths = enumerate_first_bounce_surface_paths(tx=tx, tri_data=scene._triangle_runtime())
    vec = accumulate_paths_exact(
        rx=_receiver_line(),
        tx=tx,
        scene=scene,
        wave=wave,
        source_paths_per_bounce=(paths,),
        reflection_detail=_detail(paths=paths, mode=mode),
    )[0]
    return _vector_power(vec)


def _as_array(value) -> np.ndarray:
    dr.eval(value)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _psnr(candidate: np.ndarray, reference: np.ndarray) -> float:
    mse = float(np.mean((candidate - reference) ** 2))
    if mse == 0.0:
        return float("inf")
    data_range = max(float(np.ptp(reference)), float(np.max(np.abs(reference))), 1.0e-12)
    return 20.0 * np.log10(data_range / np.sqrt(mse))


def _ssim_1d(candidate: np.ndarray, reference: np.ndarray) -> float:
    data_range = max(
        float(np.max([candidate.max(), reference.max()]) - np.min([candidate.min(), reference.min()])),
        1.0e-12,
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = float(np.mean(candidate))
    mu_y = float(np.mean(reference))
    var_x = float(np.mean((candidate - mu_x) ** 2))
    var_y = float(np.mean((reference - mu_y) ** 2))
    cov_xy = float(np.mean((candidate - mu_x) * (reference - mu_y)))
    return ((2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    )


def _native_jvp(parameter: str, value: float) -> np.ndarray:
    variable = wt.Float(value)
    dr.enable_grad(variable)
    kwargs = {"tx_x": 0.0, "strip_x": 0.0, "strip_yaw": 0.0}
    kwargs[parameter] = variable
    power = _double_slit_power(**kwargs, mode="f_weight_native")
    dr.set_grad(variable, 1.0)
    return _as_array(dr.forward_to(power, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad))


def _finite_difference(parameter: str, value: float, step: float = FD_STEP) -> np.ndarray:
    kwargs_plus = {"tx_x": 0.0, "strip_x": 0.0, "strip_yaw": 0.0}
    kwargs_minus = dict(kwargs_plus)
    kwargs_plus[parameter] = value + step
    kwargs_minus[parameter] = value - step
    plus = _as_array(_double_slit_power(**kwargs_plus, mode="f_weight_native"))
    minus = _as_array(_double_slit_power(**kwargs_minus, mode="f_weight_native"))
    return (plus - minus) / (2.0 * step)


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("tx_x", 0.08),
        ("strip_x", 0.04),
        ("strip_yaw", 0.035),
    ),
)
def test_double_slit_native_gradient_map_matches_finite_difference(parameter: str, value: float) -> None:
    jvp = _native_jvp(parameter, value)
    fd = _finite_difference(parameter, value)

    assert np.all(np.isfinite(jvp))
    assert np.all(np.isfinite(fd))
    assert np.linalg.norm(fd, ord=2) > 1.0e-8
    assert _ssim_1d(jvp, fd) >= 0.99
    assert _psnr(jvp, fd) >= 30.0


def test_double_slit_reference_tx_position_recovery_converges() -> None:
    target = dr.detach(dr.sqrt(_double_slit_power(tx_x=0.0, mode="f_weight_reference") + 1.0e-18))
    opt = Adam(lr=0.02)
    opt["tx_x"] = wt.Float(WAVELENGTH)

    for _ in range(80):
        pred = dr.sqrt(_double_slit_power(tx_x=opt["tx_x"], mode="f_weight_reference") + 1.0e-18)
        loss = dr.mean((pred - target) * (pred - target))
        dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        opt.step()

    assert abs(float(np.asarray(opt["tx_x"]).reshape(-1)[0])) < 0.02


def test_three_cube_native_matches_reference_across_reflection_orders() -> None:
    for max_bounces in (1, 2, 3):
        experiment = ThreeCubeExperiment(
            grid_shape=(8, 8),
            forward_num_samples=48,
            gradient_num_samples=48,
            max_bounces=max_bounces,
            max_diffraction_order=0,
            shadow_boundary_correction=False,
            seed=7,
        )
        base_config = experiment.forward_config
        tuning_kwargs = {
            "reflection_f_weight_boundary_radius_wavelengths": 0.25,
            "reflection_f_weight_max_edges_per_slot": 1,
        }
        reference_config = replace(
            base_config,
            tuning=replace(
                base_config.tuning,
                reflection_transition_mode="f_weight_reference",
                **tuning_kwargs,
            ),
        )
        native_config = replace(
            base_config,
            tuning=replace(
                base_config.tuning,
                reflection_transition_mode="f_weight_native",
                **tuning_kwargs,
            ),
        )

        reference = np.asarray(
            experiment._solve(config=reference_config).squeeze_tx(0).path_gain,
            dtype=np.float64,
        )
        native_result = experiment._solve(config=native_config).squeeze_tx(0)
        native = np.asarray(native_result.path_gain, dtype=np.float64)

        metadata = native_result.metadata.get("runtime_backends", {}).get("reflection_transition", {})
        assert metadata.get("resolved_backend") == "native_cuda_f_weight"
        np.testing.assert_allclose(native, reference, rtol=1.0e-5, atol=1.0e-12)


def test_three_cube_native_interior_forward_parity_matches_hard_mode() -> None:
    experiment = ThreeCubeExperiment(
        grid_shape=(32, 32),
        forward_num_samples=128,
        gradient_num_samples=128,
        max_bounces=2,
        max_diffraction_order=0,
        shadow_boundary_correction=False,
        seed=7,
    )
    base_config = experiment.forward_config
    hard_config = replace(
        base_config,
        tuning=replace(base_config.tuning, reflection_transition_mode="hard"),
    )
    native_config = replace(
        base_config,
        tuning=replace(
            base_config.tuning,
            reflection_transition_mode="f_weight_native",
            reflection_f_weight_boundary_radius_wavelengths=1.0e-9,
            reflection_f_weight_max_edges_per_slot=1,
        ),
    )

    hard = np.asarray(
        experiment._solve(config=hard_config).squeeze_tx(0).path_gain,
        dtype=np.float64,
    )
    native_result = experiment._solve(config=native_config).squeeze_tx(0)
    native = np.asarray(native_result.path_gain, dtype=np.float64)

    metadata = native_result.metadata.get("runtime_backends", {}).get("reflection_transition", {})
    assert metadata.get("resolved_backend") == "native_cuda_f_weight"
    relative_error = np.abs(native - hard) / np.maximum(np.abs(hard), 1.0e-20)
    assert float(np.max(relative_error)) <= 1.0e-5
