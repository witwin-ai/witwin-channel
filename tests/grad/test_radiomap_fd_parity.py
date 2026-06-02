"""FD-vs-AD parity gate for radiomap solver gradients.

This module is the tracked contract for `plans/29-radiomap-differentiability-parity-plan.md`.
Unlike the existing radiomap gradient tests (which only assert finiteness, non-zero
magnitude, and JVP/VJP self-consistency), every check here compares autodiff against
central finite differences on `result.path_gain[tx=0]`.

Cells that are known not to validate today are marked `xfail(strict=True)` so the gap
is encoded rather than hidden. As tasks 2-4 of plan 29 land, those marks flip to passing:

- deterministic reverse-mode reflection (Task 2): `ReflectionAccumulate::backward()` is
  unimplemented.
- diffraction parity (Task 4): UTD discontinuity handling does not FD-match for motion.
"""

from __future__ import annotations

import numpy as np
import pytest
import drjit as dr

import witwin.channel as wt
from witwin.channel.core.scene import EdgePolicy, ReceiverGrid, Scene, Transmitter
from witwin.channel.core.scene import Mesh as DrJitMesh
from witwin.core import Box, Material, Structure
from witwin.channel.deterministic import Config as DetConfig, Tuning as DetTuning, solve as det_solve
from witwin.channel.montecarlo import Config as MCConfig, IntegratorOptions, solve as mc_solve

pytestmark = pytest.mark.gpu

FREQUENCY = 1.0e9
FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad
FD_STEP = 1.0e-2
FD_RTOL = 5.0e-2


def _clear_ad_state() -> None:
    try:
        dr.clear_grad()
    except TypeError:
        pass
    dr.sync_thread()


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _sum(value) -> float:
    return float(np.sum(np.asarray(value, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Scene builders. Each accepts a scalar (python float for FD, wt.Float for AD).
# ---------------------------------------------------------------------------


def _cube_scene(*, tx_x, eps_r) -> Scene:
    scene = Scene(
        structures=[
            Structure(
                name="cube",
                geometry=Box(position=(0.0, 0.0, 1.5), size=(1.0, 1.0, 1.0), device="cuda"),
                material=Material(eps_r=4.0, sigma_e=0.0),
            )
        ],
        transmitters=[Transmitter("tx", wt.Point3f(tx_x, -3.0, 1.5))],
        receivers=[
            ReceiverGrid(
                "rm",
                axis="z",
                position=1.5,
                bounds=((-0.5, 0.5), (1.5, 2.5)),
                grid_shape=(2, 2),
            )
        ],
        frequency=FREQUENCY,
        device="cuda",
    )
    scene.structure("cube").set_material_parameters(eps_r=eps_r)
    return scene


def _open_wall_geometry_scene(wall_y) -> Scene:
    vertices = wt.Point3f(
        wt.Float([-1.0, 1.0, -1.0, 1.0]),
        dr.repeat(wt.Float(wall_y), 4),
        wt.Float([0.0, 0.0, 3.0, 3.0]),
    )
    mesh = DrJitMesh(vertices=vertices, faces=((0, 1, 3), (0, 3, 2)))
    return Scene(
        structures=[
            Structure(geometry=mesh, material=Material(eps_r=4.0, sigma_e=0.0), name="open_wall")
        ],
        transmitters=[Transmitter("tx", wt.Point3f(-0.5, -1.0, 1.5))],
        receivers=[
            ReceiverGrid(
                "rm",
                axis="z",
                position=1.5,
                bounds=((0.0, 1.0), (-1.5, -0.5)),
                grid_shape=(2, 2),
            )
        ],
        frequency=FREQUENCY,
        device="cuda",
    )


# ---------------------------------------------------------------------------
# Deterministic configs per component.
# ---------------------------------------------------------------------------


def _det_los_config() -> DetConfig:
    return DetConfig(num_samples=8, max_bounces=0, max_diffraction_order=0, shadow_boundary_correction=False)


def _det_diffraction_config() -> DetConfig:
    return DetConfig(
        num_samples=8,
        max_bounces=0,
        max_diffraction_order=1,
        edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
        shadow_boundary_correction=False,
        tuning=DetTuning(enable_rd_diffraction=False, diffraction_state_budget=16, inserted_reflection_state_budget=8),
    )


def _det_reflection_config() -> DetConfig:
    return DetConfig(num_samples=16, max_bounces=1, max_diffraction_order=0, shadow_boundary_correction=False)


def _det_loss(loss_param_name, value, config) -> float:
    """Forward-only scalar loss = sum(path_gain) for finite differences."""
    kwargs = {"tx_x": 0.0, "eps_r": 4.0}
    kwargs[loss_param_name] = value
    result = det_solve(scene=_cube_scene(**kwargs), transmitter="tx", receiver="rm", config=config)
    return _sum(result.path_gain)


def _det_fd(loss_param_name, x0, config) -> float:
    plus = _det_loss(loss_param_name, x0 + FD_STEP, config)
    minus = _det_loss(loss_param_name, x0 - FD_STEP, config)
    return (plus - minus) / (2.0 * FD_STEP)


def _det_jvp(loss_param_name, x0, config) -> float:
    _clear_ad_state()
    param = wt.Float(x0)
    dr.enable_grad(param)
    dr.set_grad(param, 1.0)
    kwargs = {"tx_x": 0.0, "eps_r": 4.0}
    kwargs[loss_param_name] = param
    result = det_solve(scene=_cube_scene(**kwargs), transmitter="tx", receiver="rm", config=config)
    jvp = dr.forward_to(result.path_gain, flags=FLAGS)
    return _sum(jvp)


def _det_vjp(loss_param_name, x0, config) -> float:
    _clear_ad_state()
    param = wt.Float(x0)
    dr.enable_grad(param)
    kwargs = {"tx_x": 0.0, "eps_r": 4.0}
    kwargs[loss_param_name] = param
    result = det_solve(scene=_cube_scene(**kwargs), transmitter="tx", receiver="rm", config=config)
    dr.backward(dr.sum(result.path_gain), flags=FLAGS)
    return _scalar(dr.grad(param))


# ---------------------------------------------------------------------------
# Smooth cells: expected to pass today.
# ---------------------------------------------------------------------------


def test_det_los_tx_position_jvp_matches_fd() -> None:
    config = _det_los_config()
    fd = _det_fd("tx_x", 0.0, config)
    ad = _det_jvp("tx_x", 0.0, config)
    assert ad == pytest.approx(fd, rel=FD_RTOL, abs=1.0e-12)


def test_det_los_tx_position_vjp_matches_fd() -> None:
    config = _det_los_config()
    fd = _det_fd("tx_x", 0.0, config)
    ad = _det_vjp("tx_x", 0.0, config)
    assert ad == pytest.approx(fd, rel=FD_RTOL, abs=1.0e-12)


def test_det_diffraction_material_jvp_matches_fd() -> None:
    config = _det_diffraction_config()
    fd = _det_fd("eps_r", 4.0, config)
    ad = _det_jvp("eps_r", 4.0, config)
    assert ad == pytest.approx(fd, rel=FD_RTOL, abs=1.0e-10)


def test_det_diffraction_material_vjp_matches_fd() -> None:
    config = _det_diffraction_config()
    fd = _det_fd("eps_r", 4.0, config)
    ad = _det_vjp("eps_r", 4.0, config)
    assert ad == pytest.approx(fd, rel=FD_RTOL, abs=1.0e-10)


# ---------------------------------------------------------------------------
# Known-failing cells: tracked via xfail until plan 29 tasks 2 and 4 land.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="plan 29 Task 2: reflection JVP does not FD-match (discontinuity handling)")
def test_det_reflection_tx_position_jvp_matches_fd() -> None:
    config = _det_reflection_config()
    fd = _det_fd("tx_x", 0.0, config)
    ad = _det_jvp("tx_x", 0.0, config)
    assert ad == pytest.approx(fd, rel=FD_RTOL, abs=1.0e-12)


@pytest.mark.xfail(strict=True, reason="plan 29 Task 2: ReflectionAccumulate::backward() is unimplemented")
def test_det_reflection_tx_position_vjp_matches_fd() -> None:
    config = _det_reflection_config()
    fd = _det_fd("tx_x", 0.0, config)
    ad = _det_vjp("tx_x", 0.0, config)
    assert ad == pytest.approx(fd, rel=FD_RTOL, abs=1.0e-12)


def test_det_reflection_geometry_jvp_matches_fd() -> None:
    config = _det_reflection_config()

    def loss(value) -> float:
        result = det_solve(scene=_open_wall_geometry_scene(value), transmitter="tx", receiver="rm", config=config)
        return _sum(result.path_gain)

    fd = (loss(0.0 + FD_STEP) - loss(0.0 - FD_STEP)) / (2.0 * FD_STEP)

    _clear_ad_state()
    wall_y = wt.Float(0.0)
    dr.enable_grad(wall_y)
    dr.set_grad(wall_y, 1.0)
    result = det_solve(scene=_open_wall_geometry_scene(wall_y), transmitter="tx", receiver="rm", config=config)
    ad = _sum(dr.forward_to(result.path_gain, flags=FLAGS))

    assert ad == pytest.approx(fd, rel=FD_RTOL, abs=1.0e-12)


# ---------------------------------------------------------------------------
# Monte Carlo cells.
# ---------------------------------------------------------------------------


def _mc_config(*, max_bounces, max_diffraction_order) -> MCConfig:
    return MCConfig(
        num_samples=64,
        max_bounces=max_bounces,
        max_diffraction_order=max_diffraction_order,
        integrator_options=IntegratorOptions(
            integrator="basic", samples_per_tx=4096, accumulation_backend="auto", seed=7
        ),
    )


def _mc_loss(tx_x, config) -> float:
    result = mc_solve(scene=_cube_scene(tx_x=tx_x, eps_r=4.0), transmitter="tx", receiver="rm", config=config)
    return _sum(result.path_gain)


def test_mc_los_tx_position_vjp_matches_fd() -> None:
    config = _mc_config(max_bounces=0, max_diffraction_order=0)
    fd = (_mc_loss(0.0 + FD_STEP, config) - _mc_loss(0.0 - FD_STEP, config)) / (2.0 * FD_STEP)

    _clear_ad_state()
    tx_x = wt.Float(0.0)
    dr.enable_grad(tx_x)
    result = mc_solve(
        scene=_cube_scene(tx_x=tx_x, eps_r=4.0),
        transmitter="tx",
        receiver="rm",
        config=config,
    )
    dr.backward(dr.sum(result.path_gain), flags=FLAGS)
    ad = _scalar(dr.grad(tx_x))

    assert ad == pytest.approx(fd, rel=FD_RTOL, abs=1.0e-12)


def test_mc_reflection_tx_position_vjp_matches_fd() -> None:
    config = _mc_config(max_bounces=1, max_diffraction_order=0)
    fd = (_mc_loss(0.0 + FD_STEP, config) - _mc_loss(0.0 - FD_STEP, config)) / (2.0 * FD_STEP)

    _clear_ad_state()
    tx_x = wt.Float(0.0)
    dr.enable_grad(tx_x)
    result = mc_solve(
        scene=_cube_scene(tx_x=tx_x, eps_r=4.0),
        transmitter="tx",
        receiver="rm",
        config=config,
    )
    dr.backward(dr.sum(result.path_gain), flags=FLAGS)
    ad = _scalar(dr.grad(tx_x))

    assert ad == pytest.approx(fd, rel=FD_RTOL, abs=1.0e-12)
