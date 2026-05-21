"""Deterministic material-parameter AD integration coverage."""

from __future__ import annotations

import drjit as dr
import numpy as np
import pytest
import witwin.channel as wt

from witwin.channel.core.scene import EdgePolicy, ReceiverGrid, Scene as ChannelScene, Transmitter
from witwin.core import Box, Material, Structure
from witwin.channel.deterministic import Config, FieldSpec, Tuning, solve, solve_field

pytestmark = pytest.mark.gpu

FREQUENCY = 1.0e9
FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad


def _clear_ad_state() -> None:
    try:
        dr.clear_grad()
    except TypeError:
        pass
    dr.sync_thread()


def _cube1_scene(cube1_eps) -> ChannelScene:
    scene = ChannelScene(
        structures=[
            Structure(
                name="cube1",
                geometry=Box(
                    position=(0.0, 0.0, 1.5),
                    size=(1.0, 1.0, 1.0),
                    device="cuda",
                ),
                material=Material(eps_r=4.0, sigma_e=0.0),
            )
        ],
        transmitters=[
            Transmitter("tx", wt.Point3f(0.0, -3.0, 1.5)),
        ],
        receivers=[
            ReceiverGrid(
                "rm",
                axis="z",
                position=1.5,
                bounds=((-0.5, 0.5), (1.5, 2.5)),
                grid_shape=(1, 1),
            ),
        ],
        frequency=FREQUENCY,
        device="cuda",
    )
    scene.diffraction_edge_count(edge_policy=EdgePolicy(edge_selection_mode="all_edges"))
    scene.structure("cube1").set_material_parameters(eps_r=cube1_eps)
    return scene


def _config() -> Config:
    return Config(
        num_samples=8,
        max_bounces=0,
        max_diffraction_order=1,
        edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
        shadow_boundary_correction=False,
        tuning=Tuning(
            enable_rd_diffraction=False,
            shadow_support_cutoff_db=25.0,
            diffraction_state_budget=16,
            inserted_reflection_state_budget=8,
        ),
    )


def _l1(value) -> float:
    return float(np.sum(np.abs(np.asarray(value, dtype=np.float64))))


def _scalar_grad(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _solve_for_cube1_eps(cube1_eps):
    return solve(
        scene=_cube1_scene(cube1_eps),
        transmitter="tx",
        receiver="rm",
        config=_config(),
    )


def test_solve_multi_tx_positions_keep_independent_gradients():
    _clear_ad_state()
    tx0_x = wt.Float(-1.0)
    tx1_x = wt.Float(1.5)
    dr.enable_grad(tx0_x, tx1_x)
    scene = ChannelScene(
        structures=[
            Structure(
                name="distant_block",
                geometry=Box(
                    position=(10.0, 10.0, 10.0),
                    size=(0.25, 0.25, 0.25),
                    device="cuda",
                ),
                material=Material(eps_r=4.0, sigma_e=0.0),
            ),
        ],
        transmitters=[
            Transmitter("tx0", wt.Point3f(tx0_x, 0.0, 1.0)),
            Transmitter("tx1", wt.Point3f(tx1_x, 0.0, 1.0)),
        ],
        receivers=[
            ReceiverGrid(
                "rm",
                axis="z",
                position=1.0,
                bounds=((-0.25, 0.25), (-0.25, 0.25)),
                grid_shape=(1, 1),
            ),
        ],
        frequency=FREQUENCY,
        device="cuda",
    )
    result = solve(
        scene=scene,
        transmitter=["tx0", "tx1"],
        receiver="rm",
        config=Config(
            num_samples=4,
            max_bounces=0,
            max_diffraction_order=0,
            shadow_boundary_correction=False,
        ),
    )

    dr.backward(dr.sum(result.path_gain), flags=FLAGS)
    tx0_grad = _scalar_grad(dr.grad(tx0_x))
    tx1_grad = _scalar_grad(dr.grad(tx1_x))

    assert np.isfinite(tx0_grad)
    assert np.isfinite(tx1_grad)
    assert abs(tx0_grad) > 0.0
    assert abs(tx1_grad) > 0.0
    assert tx0_grad * tx1_grad < 0.0


def _solve_field_for_cube1_eps(cube1_eps):
    return solve_field(
        scene=_cube1_scene(cube1_eps),
        frequency=FREQUENCY,
        tx_pos=wt.Point3f(0.0, -3.0, 1.5),
        field=FieldSpec(
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(1, 1),
            ray_mode="2d",
        ),
        config=_config(),
    )


def test_solve_cube1_eps_material_forward_backward_and_diffraction_component():
    _clear_ad_state()
    cube1_eps = wt.Float(4.0)
    dr.enable_grad(cube1_eps)
    result = _solve_for_cube1_eps(cube1_eps)

    dr.set_grad(cube1_eps, 1.0)
    diffraction_jvp, path_gain_jvp = dr.forward_to(
        result.components["diffraction"],
        result.path_gain,
        flags=FLAGS,
    )

    assert np.all(np.isfinite(np.asarray(path_gain_jvp, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(diffraction_jvp, dtype=np.float64)))
    assert _l1(result.components["diffraction"]) > 0.0
    assert _l1(diffraction_jvp) > 0.0
    assert _l1(path_gain_jvp) > 0.0
    jvp_sum = float(np.sum(np.asarray(path_gain_jvp, dtype=np.float64)))

    _clear_ad_state()
    cube1_eps = wt.Float(4.0)
    dr.enable_grad(cube1_eps)
    result = _solve_for_cube1_eps(cube1_eps)
    dr.backward(dr.sum(result.path_gain), flags=FLAGS)
    vjp = _scalar_grad(dr.grad(cube1_eps))

    assert np.isfinite(vjp)
    assert abs(vjp) > 0.0
    assert vjp == pytest.approx(jvp_sum, rel=5.0e-2, abs=1.0e-12)


def test_solve_field_cube1_eps_material_forward_backward_and_diffraction_power():
    _clear_ad_state()
    cube1_eps = wt.Float(4.0)
    dr.enable_grad(cube1_eps)
    result = _solve_field_for_cube1_eps(cube1_eps)

    dr.set_grad(cube1_eps, 1.0)
    diffraction_jvp, total_jvp = dr.forward_to(
        result.power["diffraction"],
        result.power["total"],
        flags=FLAGS,
    )

    assert np.all(np.isfinite(np.asarray(total_jvp, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(diffraction_jvp, dtype=np.float64)))
    assert _l1(result.power["diffraction"]) > 0.0
    assert _l1(diffraction_jvp) > 0.0
    assert _l1(total_jvp) > 0.0
    jvp_sum = float(np.sum(np.asarray(total_jvp, dtype=np.float64)))

    _clear_ad_state()
    cube1_eps = wt.Float(4.0)
    dr.enable_grad(cube1_eps)
    result = _solve_field_for_cube1_eps(cube1_eps)
    dr.backward(dr.sum(result.power["total"]), flags=FLAGS)
    vjp = _scalar_grad(dr.grad(cube1_eps))

    assert np.isfinite(vjp)
    assert abs(vjp) > 0.0
    assert vjp == pytest.approx(jvp_sum, rel=5.0e-2, abs=1.0e-12)
