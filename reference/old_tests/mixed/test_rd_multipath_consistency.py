"""Regression test for multi-bounce reflection-diffraction consistency."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import witwin as wt
import pytest

import drjit as dr

from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import FieldMonitor, Tracer
FREQ = 1e9
N_RAYS = 640
MAX_BOUNCES = 3
REFLECTION_COEF = 0.8
GRID_SIZE = 32
RANGE_XY = (-6, 6)
CALC_HEIGHT = 1.5
BASE_TX = (0.0, -5.0, 1.5)

FD_EPS = 1e-3
REL_TOL_PERCENT = 12.0
FD_NEAR_ZERO = 1e-4
NEAR_ZERO_ABS_TOL = 1e-4

MONITOR = FieldMonitor(
    "rd_plane",
    axis="z",
    position=CALC_HEIGHT,
    bounds=(RANGE_XY, RANGE_XY),
    grid_size=GRID_SIZE,
)


def build_scene(**scene_kwargs):
    cube1 = box_geometry(center=(-2.5, -3.0, 1.5), size=2.0)
    cube2 = box_geometry(center=(2.5, 1.0, 1.5), size=2.0)
    return build_test_scene(cube1, cube2, **scene_kwargs)


def build_tracer(scene, enable_rd):
    return Tracer(
        frequency=FREQ,
        scene=scene,
        reflection_n_rays=N_RAYS,
        reflection_max_bounces=MAX_BOUNCES,
        reflection_coef=REFLECTION_COEF,
        enable_rd_diffraction=enable_rd,
    )


def total_power_loss(tracer, x, y, z):
    tx = wt.Point3f(wt.Float(x), wt.Float(y), wt.Float(z))
    result = tracer.trace(
        tx,
        monitor=MONITOR,
        verbose=False,
    )
    a_tot = result.primary.field.total
    return float(dr.sum(a_tot.real * a_tot.real + a_tot.imag * a_tot.imag)[0])


def compute_tx_gradients_ad_fd(tracer, base_tx=BASE_TX, eps=FD_EPS):
    base_x, base_y, base_z = base_tx

    # AD
    tx_x = wt.Float(base_x)
    tx_y = wt.Float(base_y)
    tx_z = wt.Float(base_z)
    dr.enable_grad(tx_x)
    dr.enable_grad(tx_y)
    dr.enable_grad(tx_z)
    tx = wt.Point3f(tx_x, tx_y, tx_z)

    result = tracer.trace(
        tx,
        monitor=MONITOR,
        verbose=False,
    )
    a_tot = result.primary.field.total
    loss = dr.sum(a_tot.real * a_tot.real + a_tot.imag * a_tot.imag)
    dr.backward(loss)

    ad = np.array(
        [
            float(dr.grad(tx_x)[0]) if dr.width(dr.grad(tx_x)) > 0 else 0.0,
            float(dr.grad(tx_y)[0]) if dr.width(dr.grad(tx_y)) > 0 else 0.0,
            float(dr.grad(tx_z)[0]) if dr.width(dr.grad(tx_z)) > 0 else 0.0,
        ],
        dtype=np.float64,
    )

    # FD (central difference)
    fd = np.array(
        [
            (total_power_loss(tracer, base_x + eps, base_y, base_z)
             - total_power_loss(tracer, base_x - eps, base_y, base_z)) / (2 * eps),
            (total_power_loss(tracer, base_x, base_y + eps, base_z)
             - total_power_loss(tracer, base_x, base_y - eps, base_z)) / (2 * eps),
            (total_power_loss(tracer, base_x, base_y, base_z + eps)
             - total_power_loss(tracer, base_x, base_y, base_z - eps)) / (2 * eps),
        ],
        dtype=np.float64,
    )

    return ad, fd


def assert_rd_toggle_behavior(tracer_off, tracer_on):
    tx = wt.Point3f(*BASE_TX)

    res_off = tracer_off.trace(
        tx,
        monitor=MONITOR,
        verbose=False,
    )
    off_field = res_off.primary.field
    rd_power_off = float(
        dr.sum(
            off_field.diffraction_mixed.real * off_field.diffraction_mixed.real
            + off_field.diffraction_mixed.imag * off_field.diffraction_mixed.imag
        )[0]
    )
    assert rd_power_off < 1e-20, f"a_dif_mixed should be ~0 when disabled, got {rd_power_off:.6e}"

    tot_real_ref = off_field.los.real + off_field.reflection.real + off_field.diffraction.real
    tot_imag_ref = off_field.los.imag + off_field.reflection.imag + off_field.diffraction.imag
    err_real = np.max(np.abs((off_field.total.real - tot_real_ref).numpy()))
    err_imag = np.max(np.abs((off_field.total.imag - tot_imag_ref).numpy()))
    assert err_real < 1e-8 and err_imag < 1e-8, (
        f"a_tot composition mismatch when RD disabled (real={err_real:.3e}, imag={err_imag:.3e})"
    )

    res_on = tracer_on.trace(
        tx,
        monitor=MONITOR,
        verbose=False,
    )
    on_field = res_on.primary.field
    rd_power_on = float(
        dr.sum(
            on_field.diffraction_mixed.real * on_field.diffraction_mixed.real
            + on_field.diffraction_mixed.imag * on_field.diffraction_mixed.imag
        )[0]
    )
    assert rd_power_on > 1e-20, f"a_dif_mixed should be non-zero when enabled, got {rd_power_on:.6e}"

    print(f"[OK] RD toggle check: off={rd_power_off:.3e}, on={rd_power_on:.3e}")


def assert_tx_gradient_consistency(ad, fd):
    axis_names = ["tx_x", "tx_y", "tx_z"]

    for i, axis_name in enumerate(axis_names):
        ad_val = float(ad[i])
        fd_val = float(fd[i])
        if abs(fd_val) <= FD_NEAR_ZERO:
            abs_err = abs(ad_val - fd_val)
            assert abs_err <= NEAR_ZERO_ABS_TOL, (
                f"{axis_name}: FD~0 but abs error too large ({abs_err:.6e})"
            )
            print(f"[OK] {axis_name}: FD~0, abs_err={abs_err:.3e}")
        else:
            rel_err = abs(ad_val - fd_val) / abs(fd_val) * 100.0
            assert rel_err < REL_TOL_PERCENT, (
                f"{axis_name}: relative error {rel_err:.3f}% exceeds {REL_TOL_PERCENT:.1f}%"
            )
            print(f"[OK] {axis_name}: rel_err={rel_err:.3f}%")


def main():
    print("=" * 70)
    print("R-D Multipath Consistency Test")
    print("=" * 70)
    print(f"Config: n_rays={N_RAYS}, grid={GRID_SIZE}, eps={FD_EPS}")

    scene = build_scene()
    tracer_off = build_tracer(scene, enable_rd=False)
    tracer_on = build_tracer(scene, enable_rd=True)

    print("\n[1] Checking RD toggle behavior...")
    assert_rd_toggle_behavior(tracer_off, tracer_on)

    print("\n[2] Checking AD/FD consistency for tx gradients (RD enabled)...")
    ad, fd = compute_tx_gradients_ad_fd(tracer_on, BASE_TX, FD_EPS)
    print(f"AD gradients: {ad}")
    print(f"FD gradients: {fd}")
    assert_tx_gradient_consistency(ad, fd)

    print("\n[OK] All checks passed.")


@pytest.mark.gpu
def test_rd_toggle_behavior_preserves_nonzero_mixed_diffraction():
    scene = build_scene()
    tracer_off = build_tracer(scene, enable_rd=False)
    tracer_on = build_tracer(scene, enable_rd=True)
    assert scene._wedge_backend_kind == "rayd"
    assert scene._rayd_scene is not None
    assert_rd_toggle_behavior(tracer_off, tracer_on)


@pytest.mark.gpu
def test_rd_toggle_behavior_with_default_rayd_wedge_runtime():
    scene = build_scene()
    tracer_off = build_tracer(scene, enable_rd=False)
    tracer_on = build_tracer(scene, enable_rd=True)
    assert scene._wedge_backend_kind == "rayd"
    assert_rd_toggle_behavior(tracer_off, tracer_on)


@pytest.mark.gpu
def test_rd_toggle_behavior_with_default_rayd_query_and_wedge_runtime():
    scene = build_scene()
    tracer_off = build_tracer(scene, enable_rd=False)
    tracer_on = build_tracer(scene, enable_rd=True)
    assert scene._wedge_backend_kind == "rayd"
    assert scene._rayd_scene is not None
    assert_rd_toggle_behavior(tracer_off, tracer_on)


if __name__ == "__main__":
    main()


