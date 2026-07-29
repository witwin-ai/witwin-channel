# Copyright Xingyu Chen.
# Tests material abi v3.

import math
from dataclasses import dataclass

import pytest
import torch

from witwin.channel.materials import (
    DIELECTRIC_MODEL_ID,
    MATERIAL_ABI_VERSION,
    PEC_EFFECTIVE_SIGMA_E,
    face_material_field_bundle,
)
from witwin.channel.scene import compile as compile_scene
from witwin.core import (
    MaterialLayer,
    Mesh,
    PhaseScreen,
    PhysicalMaterial,
    Scene,
    Structure,
    SurfaceRoughness,
)

_EPS0 = 8.8541878128e-12
_FREQUENCY_HZ = 3.5e9


@dataclass(frozen=True)
class _DebyeDispersion:
    eps_inf: float
    delta_eps: float
    tau_s: float
    sigma_dc: float = 0.0

    def complex_eps(self, frequency_hz):
        omega = 2.0 * math.pi * frequency_hz
        return (
            self.eps_inf
            + self.delta_eps / (1.0 + 1j * omega * self.tau_s)
            - 1j * self.sigma_dc / (omega * _EPS0)
        )


def _triangle(
    material, *, name: str, z: float, phase_screen: PhaseScreen | None = None,
) -> Structure:
    vertices = torch.tensor(
        [[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]],
        dtype=torch.float32,
    )
    geometry = Mesh(
        vertices,
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    return Structure(
        geometry,
        material,
        name=name,
        phase_screen=phase_screen,
    )


def _compile(
    *materials, phase_screens: dict[int, PhaseScreen] | None = None,
    reference_frequency_hz=_FREQUENCY_HZ,
):
    screens = phase_screens or {}
    scene = Scene(
        structures=[
            _triangle(
                material,
                name=f"wall{index}",
                z=float(index),
                phase_screen=screens.get(index),
            )
            for index, material in enumerate(materials)
        ]
    )
    return compile_scene(
        scene, reference_frequency_hz=reference_frequency_hz
    )


def _three_layer_surface(**overrides) -> PhysicalMaterial:
    fields = {
        "layers": (
            MaterialLayer(thickness_m=0.02, eps_r=4.0, sigma_e=0.01),
            MaterialLayer(
                thickness_m=0.05,
                dispersion=_DebyeDispersion(
                    eps_inf=2.0,
                    delta_eps=1.5,
                    tau_s=8.0e-12,
                ),
            ),
            MaterialLayer(thickness_m=0.01, eps_r=2.2, mu_r=1.1),
        ),
        "roughness_front": SurfaceRoughness(
            rms_height_m=1.5e-3,
            correlation_length_x_m=0.03,
            correlation_length_y_m=0.05,
            principal_axis_rad=0.4,
        ),
        "name": "layered-wall",
    }
    fields.update(overrides)
    return PhysicalMaterial(**fields)


def test_core_dispersion_is_evaluated_without_legacy_material_facades():
    dispersion = _DebyeDispersion(
        eps_inf=2.0,
        delta_eps=1.5,
        tau_s=8.0e-12,
        sigma_dc=0.02,
    )
    layer = MaterialLayer(thickness_m=0.05, dispersion=dispersion)

    low = layer.evaluate_at_frequency(1.0e9)
    high = layer.evaluate_at_frequency(10.0e9)

    assert low.eps_r == pytest.approx(dispersion.complex_eps(1.0e9))
    assert high.eps_r == pytest.approx(dispersion.complex_eps(10.0e9))
    assert low.eps_r.real > high.eps_r.real
    assert low.eps_r.imag <= 0.0


def test_csr_layout_for_mixed_core_material_scene():
    compiled = _compile(
        PhysicalMaterial(eps_r=2.5, sigma_e=0.01, thickness_m=0.2),
        PhysicalMaterial.perfect_conductor(),
        _three_layer_surface(),
        PhysicalMaterial(name="concrete"),
    )
    materials = compiled.materials

    assert materials.abi_version == MATERIAL_ABI_VERSION == 3
    assert materials.layer_count.tolist() == [1, 1, 3, 1]
    assert materials.layer_offset.tolist() == [0, 1, 2, 5]
    assert materials.layer_thickness_m.shape == (6,)
    assert materials.layer_thickness_m[0] == pytest.approx(0.2)
    assert materials.layer_eps_r[0] == pytest.approx(2.5)
    assert materials.layer_sigma_e[0] == pytest.approx(0.01)
    assert float(materials.layer_sigma_e[1]) >= PEC_EFFECTIVE_SIGMA_E
    assert materials.model_id.tolist()[2] == DIELECTRIC_MODEL_ID
    assert materials.scatter_model_id.tolist() == [0, 0, 1, 0]


def test_roughness_fields_land_in_runtime_store():
    rough = _three_layer_surface()
    smooth = _three_layer_surface(roughness_front=None, name="smooth-wall")
    materials = _compile(rough, smooth).materials

    assert materials.rough_sigma_h_m.tolist() == pytest.approx(
        [1.5e-3, 0.0]
    )
    assert materials.rough_corr_x_m.tolist() == pytest.approx([0.03, 0.0])
    assert materials.rough_corr_y_m.tolist() == pytest.approx([0.05, 0.0])
    assert materials.rough_axis_rad.tolist() == pytest.approx([0.4, 0.0])
    assert materials.scatter_model_id.tolist() == [1, 0]


def test_structure_phase_screen_is_compiled_from_core_assignment():
    screen = PhaseScreen(
        height=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        height_scale_m=0.01,
        realization_id=3,
    )
    compiled = _compile(
        PhysicalMaterial(eps_r=2.0),
        _three_layer_surface(),
        phase_screens={1: screen},
    )

    assert compiled.assignments.structure_phase_screens == {1: screen}
    assert screen.height.shape == (2, 2)
    assert screen.height.dtype == torch.float32


def test_cache_token_changes_with_layer_or_roughness():
    surface = _three_layer_surface()
    base = _compile(surface).materials
    same = _compile(surface).materials
    thicker = _compile(
        _three_layer_surface(
            layers=(
                MaterialLayer(
                    thickness_m=0.03, eps_r=4.0, sigma_e=0.01
                ),
                MaterialLayer(
                    thickness_m=0.05,
                    dispersion=_DebyeDispersion(
                        eps_inf=2.0,
                        delta_eps=1.5,
                        tau_s=8.0e-12,
                    ),
                ),
                MaterialLayer(thickness_m=0.01, eps_r=2.2, mu_r=1.1),
            )
        )
    ).materials
    rougher = _compile(
        _three_layer_surface(
            roughness_front=SurfaceRoughness(
                rms_height_m=2.5e-3,
                correlation_length_x_m=0.03,
                correlation_length_y_m=0.05,
                principal_axis_rad=0.4,
            )
        )
    ).materials

    assert base.cache_token == same.cache_token
    assert base.cache_token != thicker.cache_token
    assert base.cache_token != rougher.cache_token


def test_field_bundle_exposes_material_level_csr_views():
    if not torch.cuda.is_available():
        pytest.skip("field bundle export requires CUDA")
    compiled = _compile(
        PhysicalMaterial(eps_r=2.0), _three_layer_surface()
    )

    bundle = face_material_field_bundle(
        compiled, device=torch.device("cuda")
    )

    assert bundle["layer_offset"].tolist() == [0, 1]
    assert bundle["layer_count"].tolist() == [1, 3]
    assert bundle["layer_thickness_m"].shape == (4,)
    assert bundle["scatter_model_id"].tolist() == [0, 1]
    assert bundle["material_id"].tolist() == [0, 1]