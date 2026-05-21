from __future__ import annotations

import numpy as np
import pytest
import drjit as dr

import witwin.channel as wc
from witwin.channel.core.scene import Mesh as DrJitMesh
from witwin.core import Material, Mesh, Structure


FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _clear_ad_state() -> None:
    try:
        dr.clear_grad()
    except TypeError:
        pass


def _los_tau_loss(tx_x) -> object:
    scene = wc.Scene(frequency=3.5e9, device="cuda")
    scene.add(wc.Transmitter("tx0", wc.Point3f(tx_x, 0.0, 0.0)))
    scene.add(wc.Receiver("rx0", (10.0, 0.0, 0.0)))
    result = wc.path.solve(
        scene=scene,
        transmitter="tx0",
        receiver="rx0",
        config=wc.path.Config(
            max_bounces=0,
            max_diffraction_order=0,
            max_num_paths=1,
            return_geometry=False,
        ),
    )
    return dr.sum(dr.select(result.valid, result.tau, wc.Float(0.0)))


def _multi_endpoint_los_tau_loss(tx0_x, tx1_x) -> object:
    scene = wc.Scene(frequency=3.5e9, device="cuda")
    scene.add(wc.Transmitter("tx0", wc.Point3f(tx0_x, 0.0, 0.0)))
    scene.add(wc.Transmitter("tx1", wc.Point3f(tx1_x, 1.0, 0.0)))
    scene.add(wc.Receiver("rx0", (10.0, 0.0, 0.0)))
    scene.add(wc.Receiver("rx1", (-2.0, 4.0, 0.0)))
    result = wc.path.solve(
        scene=scene,
        transmitter=["tx0", "tx1"],
        receiver=["rx0", "rx1"],
        config=wc.path.Config(
            max_bounces=0,
            max_diffraction_order=0,
            max_num_paths=1,
            return_geometry=False,
        ),
    )
    return dr.sum(dr.select(result.valid, result.tau, wc.Float(0.0)))


def _open_wall_reflection_scene(tx_x, *, eps_r=4.0) -> wc.Scene:
    mesh = Mesh(
        vertices=(
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 3.0),
            (1.0, 0.0, 3.0),
        ),
        faces=((0, 1, 3), (0, 3, 2)),
        device="cuda",
    )
    scene = wc.Scene(
        structures=[
            Structure(
                geometry=mesh,
                material=Material(eps_r=4.0, sigma_e=0.0),
                name="open_wall",
            )
        ],
        frequency=3.5e9,
        device="cuda",
    )
    scene.structure("open_wall").set_material_parameters(eps_r=eps_r)
    scene.add(wc.Transmitter("tx0", wc.Point3f(tx_x, -1.0, 1.5)))
    scene.add(wc.Receiver("rx0", (0.5, -1.0, 1.5)))
    return scene


def _open_wall_reflection_result(tx_x, *, eps_r=4.0):
    return wc.path.solve(
        scene=_open_wall_reflection_scene(tx_x, eps_r=eps_r),
        transmitter="tx0",
        receiver="rx0",
        config=wc.path.Config(
            num_samples=16,
            max_bounces=1,
            max_diffraction_order=0,
            max_num_paths=4,
            return_geometry=False,
        ),
    )


def _open_wall_tau_loss(tx_x) -> object:
    result = _open_wall_reflection_result(tx_x)
    return dr.sum(dr.select(result.valid, result.tau, wc.Float(0.0)))


def _open_wall_coeff_power_loss(tx_x) -> object:
    result = _open_wall_reflection_result(tx_x)
    power = result.a.real * result.a.real + result.a.imag * result.a.imag
    return dr.sum(dr.select(result.valid, power, wc.Float(0.0)))


def _open_wall_material_coeff_power_loss(eps_r) -> object:
    result = _open_wall_reflection_result(-0.5, eps_r=eps_r)
    power = result.a.real * result.a.real + result.a.imag * result.a.imag
    return dr.sum(dr.select(result.valid, power, wc.Float(0.0)))


def _open_wall_geometry_tau_loss(wall_y) -> object:
    vertices = wc.Point3f(
        wc.Float([-1.0, 1.0, -1.0, 1.0]),
        dr.repeat(wc.Float(wall_y), 4),
        wc.Float([0.0, 0.0, 3.0, 3.0]),
    )
    mesh = DrJitMesh(vertices=vertices, faces=((0, 1, 3), (0, 3, 2)))
    scene = wc.Scene(
        structures=[
            Structure(
                geometry=mesh,
                material=Material(eps_r=4.0, sigma_e=0.0),
                name="open_wall",
            )
        ],
        frequency=3.5e9,
        device="cuda",
    )
    scene.add(wc.Transmitter("tx0", wc.Point3f(-0.5, -1.0, 1.5)))
    scene.add(wc.Receiver("rx0", (0.5, -1.0, 1.5)))
    result = wc.path.solve(
        scene=scene,
        transmitter="tx0",
        receiver="rx0",
        config=wc.path.Config(
            num_samples=16,
            max_bounces=1,
            max_diffraction_order=0,
            max_num_paths=4,
            return_geometry=False,
        ),
    )
    return dr.sum(dr.select(result.valid, result.tau, wc.Float(0.0)))


@pytest.mark.gpu
def test_path_solver_los_tau_tx_position_gradient_matches_fd() -> None:
    _clear_ad_state()
    tx_x = wc.Float(0.0)
    dr.enable_grad(tx_x)

    loss = _los_tau_loss(tx_x)
    dr.backward(loss, flags=FLAGS)
    ad_grad = _scalar(dr.grad(tx_x))

    step = 1.0e-2
    plus = _scalar(_los_tau_loss(step))
    minus = _scalar(_los_tau_loss(-step))
    fd_grad = (plus - minus) / (2.0 * step)

    assert ad_grad == pytest.approx(fd_grad, rel=5.0e-3, abs=1.0e-12)


@pytest.mark.gpu
def test_path_solver_multi_tx_rx_los_gradients_match_fd() -> None:
    _clear_ad_state()
    tx0_x = wc.Float(0.0)
    tx1_x = wc.Float(1.5)
    dr.enable_grad(tx0_x, tx1_x)

    loss = _multi_endpoint_los_tau_loss(tx0_x, tx1_x)
    dr.backward(loss, flags=FLAGS)
    ad_tx0 = _scalar(dr.grad(tx0_x))
    ad_tx1 = _scalar(dr.grad(tx1_x))

    step = 1.0e-2
    fd_tx0 = (
        _scalar(_multi_endpoint_los_tau_loss(step, 1.5))
        - _scalar(_multi_endpoint_los_tau_loss(-step, 1.5))
    ) / (2.0 * step)
    fd_tx1 = (
        _scalar(_multi_endpoint_los_tau_loss(0.0, 1.5 + step))
        - _scalar(_multi_endpoint_los_tau_loss(0.0, 1.5 - step))
    ) / (2.0 * step)

    assert ad_tx0 == pytest.approx(fd_tx0, rel=5.0e-3, abs=1.0e-12)
    assert ad_tx1 == pytest.approx(fd_tx1, rel=5.0e-3, abs=1.0e-12)


@pytest.mark.gpu
def test_path_solver_reflection_tau_tx_position_gradient_matches_fd() -> None:
    _clear_ad_state()
    tx_x = wc.Float(-0.5)
    dr.enable_grad(tx_x)

    result = _open_wall_reflection_result(tx_x)
    assert int(np.asarray(result.num_paths).reshape(-1)[0]) == 2
    assert np.asarray(result.types).reshape(-1).tolist()[1] == int(wc.path.InteractionType.REFLECTION)

    loss = dr.sum(dr.select(result.valid, result.tau, wc.Float(0.0)))
    dr.backward(loss, flags=FLAGS)
    ad_grad = _scalar(dr.grad(tx_x))

    step = 1.0e-2
    plus = _scalar(_open_wall_tau_loss(-0.5 + step))
    minus = _scalar(_open_wall_tau_loss(-0.5 - step))
    fd_grad = (plus - minus) / (2.0 * step)

    assert ad_grad == pytest.approx(fd_grad, rel=5.0e-3, abs=1.0e-12)


@pytest.mark.gpu
def test_path_solver_reflection_coeff_power_tx_position_gradient_matches_fd() -> None:
    _clear_ad_state()
    tx_x = wc.Float(-0.5)
    dr.enable_grad(tx_x)

    loss = _open_wall_coeff_power_loss(tx_x)
    dr.backward(loss, flags=FLAGS)
    ad_grad = _scalar(dr.grad(tx_x))

    step = 1.0e-2
    plus = _scalar(_open_wall_coeff_power_loss(-0.5 + step))
    minus = _scalar(_open_wall_coeff_power_loss(-0.5 - step))
    fd_grad = (plus - minus) / (2.0 * step)

    assert ad_grad == pytest.approx(fd_grad, rel=5.0e-3, abs=1.0e-10)


@pytest.mark.gpu
def test_path_solver_reflection_coeff_power_material_gradient_matches_fd() -> None:
    _clear_ad_state()
    eps_r = wc.Float(4.0)
    dr.enable_grad(eps_r)

    loss = _open_wall_material_coeff_power_loss(eps_r)
    dr.backward(loss, flags=FLAGS)
    ad_grad = _scalar(dr.grad(eps_r))

    step = 1.0e-2
    plus = _scalar(_open_wall_material_coeff_power_loss(4.0 + step))
    minus = _scalar(_open_wall_material_coeff_power_loss(4.0 - step))
    fd_grad = (plus - minus) / (2.0 * step)

    assert ad_grad == pytest.approx(fd_grad, rel=5.0e-3, abs=1.0e-10)


@pytest.mark.gpu
def test_path_solver_reflection_tau_surface_geometry_gradient_matches_fd() -> None:
    _clear_ad_state()
    wall_y = wc.Float(0.0)
    dr.enable_grad(wall_y)

    loss = _open_wall_geometry_tau_loss(wall_y)
    dr.backward(loss, flags=FLAGS)
    ad_grad = _scalar(dr.grad(wall_y))

    step = 1.0e-2
    plus = _scalar(_open_wall_geometry_tau_loss(step))
    minus = _scalar(_open_wall_geometry_tau_loss(-step))
    fd_grad = (plus - minus) / (2.0 * step)

    assert ad_grad == pytest.approx(fd_grad, rel=5.0e-3, abs=1.0e-12)
