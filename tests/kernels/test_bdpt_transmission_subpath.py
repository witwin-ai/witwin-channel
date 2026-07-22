"""Shooting-context specular transmission subpath op (contract section 4).

Vacuum invariants: the exit point lies on the original ray, the combined
Jones/phase factor is exactly 1, and the throughput amplitude proxy is
unchanged; the component mask gains the transmission bit and the event is the
specular-transmission delta event.
"""

import math

import pytest
import torch

from witwin.channel.montecarlo.bdpt.kernels import paths as ops
from witwin.channel.materials.kernels import functional as material_functional

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)


def _csr(materials: list[list[tuple[float, float, float, float]]]) -> dict[str, torch.Tensor]:
    offsets: list[int] = []
    counts: list[int] = []
    thickness: list[float] = []
    eps_r: list[float] = []
    sigma_e: list[float] = []
    mu_r: list[float] = []
    for layers in materials:
        offsets.append(len(thickness))
        counts.append(len(layers))
        for layer in layers:
            thickness.append(layer[0])
            eps_r.append(layer[1])
            sigma_e.append(layer[2])
            mu_r.append(layer[3])
    return {
        "layer_offset": torch.tensor(offsets, device="cuda", dtype=torch.int32),
        "layer_count": torch.tensor(counts, device="cuda", dtype=torch.int32),
        "layer_thickness_m": torch.tensor(thickness, device="cuda", dtype=torch.float32),
        "layer_eps_r": torch.tensor(eps_r, device="cuda", dtype=torch.float32),
        "layer_sigma_e": torch.tensor(sigma_e, device="cuda", dtype=torch.float32),
        "layer_mu_r": torch.tensor(mu_r, device="cuda", dtype=torch.float32),
    }


def _light_state(
    direction: list[float], field: list[float], *, component_mask: int = 3
) -> dict[str, torch.Tensor]:
    return {
        "origin": torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "direction": torch.tensor([direction], device="cuda", dtype=torch.float32),
        "throughput_real": torch.tensor([2.0], device="cuda", dtype=torch.float32),
        "throughput_imag": torch.tensor([0.5], device="cuda", dtype=torch.float32),
        "pdf_forward": torch.tensor([0.25], device="cuda", dtype=torch.float32),
        "pdf_reverse": torch.tensor([0.0], device="cuda", dtype=torch.float32),
        "depth": torch.tensor([1], device="cuda", dtype=torch.int32),
        "component_mask": torch.tensor(
            [component_mask], device="cuda", dtype=torch.int32
        ),
        "primitive_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "edge_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "tx_id": torch.tensor([4], device="cuda", dtype=torch.int32),
        "rx_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "grid_linear_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "valid": torch.tensor([True], device="cuda", dtype=torch.bool),
        "path_length": torch.tensor([1.5], device="cuda", dtype=torch.float32),
        "field_real": torch.tensor([field], device="cuda", dtype=torch.float32),
        "field_imag": torch.zeros((1, 3), device="cuda", dtype=torch.float32),
        "source_power": torch.tensor([4.0], device="cuda", dtype=torch.float32),
        "event_type": torch.tensor([1], device="cuda", dtype=torch.int32),
    }


def _intersection(t: float, point: list[float]) -> dict[str, torch.Tensor]:
    return {
        "t": torch.tensor([t], device="cuda", dtype=torch.float32),
        "p": torch.tensor([point], device="cuda", dtype=torch.float32),
        "n": torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "geo_n": torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "uv": torch.empty((1, 2), device="cuda", dtype=torch.float32),
        "barycentric": torch.empty((1, 3), device="cuda", dtype=torch.float32),
        "shape_id": torch.tensor([0], device="cuda", dtype=torch.int32),
        "prim_id": torch.tensor([0], device="cuda", dtype=torch.int32),
        "local_prim_id": torch.tensor([0], device="cuda", dtype=torch.int32),
        "global_prim_id": torch.tensor([0], device="cuda", dtype=torch.int32),
    }


def test_vacuum_wall_transmission_keeps_ray_field_and_throughput():
    frequency = 3.0e9
    theta = math.radians(30.0)
    direction = [math.sin(theta), 0.0, -math.cos(theta)]
    thickness = 0.2
    hit_t = 1.0 / math.cos(theta)
    hit_point = [math.tan(theta), 0.0, 0.0]
    # y-hat is the exact s direction for this wall/ray pair.
    light = _light_state(direction, [0.0, 1.0, 0.0])

    transmitted = ops.bdpt_transmitted_light_subpath_state(
        light,
        _intersection(hit_t, hit_point),
        face_material_id=torch.tensor([0], device="cuda", dtype=torch.int32),
        **_csr([[(thickness, 1.0, 0.0, 1.0)]]),
        frequency_hz=frequency,
    )

    # Vacuum: theta_l == theta_i, so the exit point lies ON the original ray
    # at interior chord distance thickness/cos(theta).
    chord = thickness / math.cos(theta)
    expected_exit = torch.tensor(
        [
            [
                hit_point[0] + chord * direction[0],
                hit_point[1] + chord * direction[1],
                hit_point[2] + chord * direction[2],
            ]
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        transmitted["origin"].cpu(), expected_exit, rtol=1.0e-5, atol=1.0e-6
    )
    torch.testing.assert_close(
        transmitted["direction"].cpu(),
        torch.tensor([direction], dtype=torch.float32),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    # Combined t_stack * exp(+j*k0*jump) * exp(-j*k_par*dx_par) factor is
    # exactly 1 for a vacuum layer: the field passes through unchanged.
    torch.testing.assert_close(
        transmitted["field_real"].cpu(),
        light["field_real"].cpu(),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        transmitted["field_imag"].cpu(),
        torch.zeros((1, 3), dtype=torch.float32),
        rtol=0.0,
        atol=1.0e-5,
    )
    # Vacuum T_eff == 1: throughput amplitude proxy is unchanged.
    torch.testing.assert_close(
        transmitted["throughput_real"].cpu(),
        torch.tensor([2.0], dtype=torch.float32),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        transmitted["throughput_imag"].cpu(),
        torch.tensor([0.5], dtype=torch.float32),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    assert transmitted["component_mask"].item() == (3 | 8)
    assert transmitted["event_type"].item() == 2
    assert transmitted["depth"].item() == 2
    assert transmitted["valid"].item() is True
    assert transmitted["primitive_id"].item() == 0
    assert transmitted["tx_id"].item() == 4
    # path_length grows by the surface hit distance plus the interior jump.
    assert transmitted["path_length"].item() == pytest.approx(
        1.5 + hit_t + chord, rel=1.0e-5
    )
    # Delta event: pdfs carry the unchanged non-delta proposal density.
    assert transmitted["pdf_forward"].item() == pytest.approx(0.25)
    assert transmitted["pdf_reverse"].item() == pytest.approx(0.25)
    # Source power is carried through untouched.
    assert transmitted["source_power"].item() == pytest.approx(4.0)


def test_lossy_wall_scales_throughput_by_sqrt_power_transmittance():
    frequency = 3.0e9
    materials = [[(0.1, 4.0, 0.05, 1.0)]]
    light = _light_state([0.0, 0.0, -1.0], [1.0, 0.0, 0.0])

    transmitted = ops.bdpt_transmitted_light_subpath_state(
        light,
        _intersection(1.0, [0.0, 0.0, 0.0]),
        face_material_id=torch.tensor([0], device="cuda", dtype=torch.int32),
        **_csr(materials),
        frequency_hz=frequency,
    )
    stack = material_functional.em_layer_stack_eval(
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        **_csr(materials),
        frequency_hz=frequency,
    )

    # Normal incidence: TE and TM power transmittances coincide, so
    # T_eff == cap_T and throughput scales by sqrt(cap_T).
    cap_t = stack["cap_T_te"][0].item()
    assert 0.0 < cap_t < 1.0
    expected = 2.0 * math.sqrt(cap_t)
    assert transmitted["throughput_real"].item() == pytest.approx(
        expected, rel=1.0e-4
    )
    assert transmitted["throughput_imag"].item() == pytest.approx(
        0.5 * math.sqrt(cap_t), rel=1.0e-4
    )
    # Straight-through exit at normal incidence: x_e = x_i - d_total*n_in.
    torch.testing.assert_close(
        transmitted["origin"].cpu(),
        torch.tensor([[0.0, 0.0, -0.1]], dtype=torch.float32),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    assert transmitted["component_mask"].item() == (3 | 8)
    assert transmitted["event_type"].item() == 2


def _sensor_state() -> dict[str, torch.Tensor]:
    return {
        "origin": torch.tensor([[5.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        "direction": torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        "throughput_real": torch.tensor([1.0], device="cuda", dtype=torch.float32),
        "throughput_imag": torch.tensor([0.0], device="cuda", dtype=torch.float32),
        "pdf_forward": torch.tensor([1.0], device="cuda", dtype=torch.float32),
        "pdf_reverse": torch.tensor([1.0], device="cuda", dtype=torch.float32),
        "depth": torch.tensor([0], device="cuda", dtype=torch.int32),
        "component_mask": torch.tensor([1], device="cuda", dtype=torch.int32),
        "primitive_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "edge_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "tx_id": torch.tensor([-1], device="cuda", dtype=torch.int32),
        "rx_id": torch.tensor([0], device="cuda", dtype=torch.int32),
        "grid_linear_id": torch.tensor([0], device="cuda", dtype=torch.int32),
        "valid": torch.tensor([True], device="cuda", dtype=torch.bool),
        "path_length": torch.tensor([0.0], device="cuda", dtype=torch.float32),
        "field_real": torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        "field_imag": torch.zeros((1, 3), device="cuda", dtype=torch.float32),
        "source_power": torch.tensor([1.0], device="cuda", dtype=torch.float32),
        "event_type": torch.tensor([0], device="cuda", dtype=torch.int32),
    }


@pytest.mark.parametrize(
    ("component_mask", "expected_component"),
    [
        (1, 0),  # los only
        (1 | 2, 1),  # reflection
        (1 | 8, 5),  # transmission
        (1 | 2 | 8, 5),  # exclusive priority: transmission beats reflection
        (1 | 4 | 8, 2),  # exclusive priority: diffraction beats transmission
        (1 | 16, 6),  # scattering
        (1 | 2 | 16, 6),  # exclusive priority: scattering beats reflection
        (1 | 4 | 16, 6),  # exclusive priority: scattering beats diffraction
        (1 | 2 | 8 | 16, 6),  # scattering wins over every other bit
    ],
)
def test_connection_component_classification_uses_exclusive_priority(
    component_mask, expected_component
):
    """Contract section 1: path_class priority is
    scattering > diffraction > transmission > reflection > los."""

    light = _light_state(
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], component_mask=component_mask
    )
    samples = ops.bdpt_endpoint_connection_samples(
        light,
        _sensor_state(),
        frequency_hz=3.0e9,
        samples_per_tx=1,
    )
    assert samples["valid"].item() is True
    assert samples["component_id"].item() == expected_component
    assert samples["topology"][0, 2].item() == expected_component


def test_invalid_material_id_invalidates_the_path():
    light = _light_state([0.0, 0.0, -1.0], [1.0, 0.0, 0.0])
    transmitted = ops.bdpt_transmitted_light_subpath_state(
        light,
        _intersection(1.0, [0.0, 0.0, 0.0]),
        face_material_id=torch.tensor([-1], device="cuda", dtype=torch.int32),
        **_csr([[(0.1, 4.0, 0.0, 1.0)]]),
        frequency_hz=3.0e9,
    )
    assert transmitted["valid"].item() is False
    assert transmitted["event_type"].item() == -1
    assert transmitted["throughput_real"].item() == 0.0
