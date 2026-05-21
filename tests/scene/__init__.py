from __future__ import annotations

import math

import drjit as dr
import pytest
import witwin.channel as wt
import witwin.channel as wc
from witwin.channel.core.physics.polarization import receiver_tangential
from witwin.channel.core.physics.wave_math import complex_relative_permittivity, fresnel_reflection


def test_planar_array_dual_pol_slots_and_differentiable_spacing():
    spacing = wt.Float(0.5)
    dr.enable_grad(spacing)

    array = wc.PlanarArray(
        num_rows=2,
        num_cols=3,
        vertical_spacing=spacing,
        horizontal_spacing=0.25,
        pattern="dipole",
        polarization="VH",
    )

    assert array.num_elements == 6
    assert array.num_polarization_slots == 2
    assert array.num_ant == 12
    assert array.pattern == "dipole"

    # Element y-positions depend on vertical spacing and must keep AD.
    loss = dr.sum(array.element_positions.y * array.element_positions.y)
    dr.backward(loss)
    assert float(dr.grad(spacing)[0]) != 0.0


def test_antenna_array_accepts_complex_polarization_slots_and_orientations():
    yaw = wt.Float([0.0, 0.1])
    dr.enable_grad(yaw)
    element_orientations = wt.Point3f(yaw, wt.Float([0.0, 0.0]), wt.Float([0.0, 0.0]))

    array = wc.AntennaArray(
        element_positions=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
        element_orientations=element_orientations,
        polarization=[(1.0 / math.sqrt(2.0), 1.0j / math.sqrt(2.0), 0.0)],
        pattern="iso",
    )

    assert array.num_elements == 2
    assert array.num_polarization_slots == 1
    assert array.num_ant == 2
    assert array.polarization_slots[0][1] == 1.0j / math.sqrt(2.0)
    assert array.element_orientations is element_orientations

    loss = dr.sum(array.element_orientations.x * array.element_orientations.x)
    dr.backward(loss)
    assert float(dr.grad(yaw)[1]) != 0.0


def test_cross_polarization_is_two_local_slant_slots():
    array = wc.AntennaArray(
        element_positions=[(0.0, 0.0, 0.0)],
        polarization="cross",
    )

    assert array.num_ant == 2
    plus, minus = array.polarization_slots
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert plus == pytest.approx((inv_sqrt2, inv_sqrt2, 0.0))
    assert minus == pytest.approx((inv_sqrt2, -inv_sqrt2, 0.0))


def test_receiver_tangential_normalizes_complex_jones_polarization():
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    rx = receiver_tangential((inv_sqrt2, 1.0j * inv_sqrt2, 0.0), axis="z")
    power = (
        rx["x"].real * rx["x"].real
        + rx["x"].imag * rx["x"].imag
        + rx["y"].real * rx["y"].real
        + rx["y"].imag * rx["y"].imag
    )

    assert float(power[0]) == pytest.approx(1.0)
    assert float(rx["x"].real[0]) == pytest.approx(inv_sqrt2)
    assert float(rx["y"].imag[0]) == pytest.approx(inv_sqrt2)


def test_endpoint_arrays_override_scene_defaults():
    scene_tx_array = wc.ULA(num_elements=2, spacing=0.5)
    endpoint_tx_array = wc.ULA(num_elements=3, spacing=0.25)
    scene_rx_array = wc.PlanarArray(num_rows=1, num_cols=2)
    scene = wc.Scene(
        transmitters=[wc.Transmitter(name="tx", position=(0.0, 0.0, 0.0), array=endpoint_tx_array)],
        receivers=[wc.Receiver(name="rx", position=(1.0, 0.0, 0.0))],
        tx_array=scene_tx_array,
        rx_array=scene_rx_array,
        device="cpu",
    )

    assert scene.transmitter_array("tx") is endpoint_tx_array
    assert scene.receiver_array("rx") is scene_rx_array


def test_scene_frequency_owns_late_bound_itu_material_runtime():
    concrete = wc.Material.from_itu("concrete")
    scene = wc.Scene(
        structures=[
            wc.Structure(
                name="wall",
                geometry=wc.Box(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), device="cuda"),
                material=concrete,
            )
        ],
        frequency=2.4e9,
        device="cuda",
    )

    low = scene.triangle_material(wt.Int32(0))
    low_eps = float(low["eps_r"][0])
    low_sigma = float(low["sigma_e"][0])

    scene.frequency = 28.0e9
    high = scene.triangle_material(wt.Int32(0))
    high_eps = float(high["eps_r"][0])
    high_sigma = float(high["sigma_e"][0])

    assert concrete.itu_descriptor == ("concrete", None)
    assert (high_eps, high_sigma) != pytest.approx((low_eps, low_sigma))
    assert float(high["mu_r"][0]) == pytest.approx(1.0)


def test_fresnel_reflection_uses_relative_permeability():
    eta = complex_relative_permittivity(wt.Float(4.0), wt.Float(0.0), wt.Float(2.0 * math.pi * 1.0e9))

    r_te_mu1, r_tm_mu1 = fresnel_reflection(wt.Float(0.7), eta, mu_r=wt.Float(1.0))
    r_te_mu2, r_tm_mu2 = fresnel_reflection(wt.Float(0.7), eta, mu_r=wt.Float(2.0))

    assert float(dr.abs(r_te_mu1 - r_te_mu2)[0]) > 1.0e-6
    assert float(dr.abs(r_tm_mu1 - r_tm_mu2)[0]) > 1.0e-6


def test_scene_material_runtime_keeps_eps_mu_sigma_gradients():
    eps_r = wt.Float(4.0)
    mu_r = wt.Float(1.2)
    sigma_e = wt.Float(0.01)
    dr.enable_grad(eps_r, mu_r, sigma_e)
    scene = wc.Scene(
        structures=[
            wc.Structure(
                name="wall",
                geometry=wc.Box(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), device="cuda"),
                material=wc.Material(eps_r=4.0, sigma_e=0.0),
            )
        ],
        frequency=3.5e9,
        device="cuda",
    )

    scene.structure("wall").set_material_parameters(eps_r=eps_r, mu_r=mu_r, sigma_e=sigma_e)
    material = scene.triangle_material(wt.Int32(0))
    loss = dr.sum(material["eps_r"] + 2.0 * material["mu_r"] + 3.0 * material["sigma_e"])
    dr.backward(loss)

    assert float(dr.grad(eps_r)[0]) != 0.0
    assert float(dr.grad(mu_r)[0]) != 0.0
    assert float(dr.grad(sigma_e)[0]) != 0.0
