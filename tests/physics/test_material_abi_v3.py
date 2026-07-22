import math

import pytest
import torch

from witwin.channel import Scene, Structure
from witwin.channel.core.material_runtime import face_material_field_bundle
from witwin.channel.core.materials import (
    MATERIAL_ABI_VERSION,
    PEC_EFFECTIVE_SIGMA_E,
    PHYSICAL_SURFACE_MODEL_ID,
    DebyeModel,
    Dielectric,
    ITUMaterial,
    Layer,
    PerfectConductor,
    PhaseScreen,
    PhysicalSurface,
    Roughness,
    SurfaceAssignment,
    TabulatedPermittivity,
)

_EPS0 = 8.8541878128e-12
_FREQUENCY = 3.5e9


def _triangle(material, *, name: str = "wall", z: float = 0.0) -> Structure:
    return Structure(
        vertices=torch.tensor(
            [[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        material=material,
        name=name,
    )


def _compile(*materials):
    structures = [
        _triangle(material, name=f"wall{i}", z=float(i))
        for i, material in enumerate(materials)
    ]
    return Scene(
        structures=structures, transmitters=[], receivers=[], frequency=_FREQUENCY
    ).compile()


def _three_layer_surface(**overrides) -> PhysicalSurface:
    fields = {
        "layers": (
            Layer(thickness_m=0.02, eps_r=4.0, sigma_e=0.01),
            Layer(
                thickness_m=0.05,
                eps_model=DebyeModel(eps_inf=2.0, delta_eps=1.5, tau_s=8.0e-12),
            ),
            Layer(thickness_m=0.01, eps_r=2.2, mu_r=1.1),
        ),
        "roughness_front": Roughness(
            rms_height_m=1.5e-3,
            corr_length_x_m=0.03,
            corr_length_y_m=0.05,
            principal_axis_rad=0.4,
        ),
        "name": "layered-wall",
    }
    fields.update(overrides)
    return PhysicalSurface(**fields)


def test_debye_model_dispersion_differs_across_frequencies():
    model = DebyeModel(eps_inf=2.0, delta_eps=1.5, tau_s=8.0e-12, sigma_dc=0.02)
    for frequency in (1.0e9, 10.0e9):
        omega = 2.0 * math.pi * frequency
        expected = (
            2.0
            + 1.5 / (1.0 + 1j * omega * 8.0e-12)
            - 1j * 0.02 / (omega * _EPS0)
        )
        eps = model.complex_eps(frequency)
        assert eps == pytest.approx(expected, rel=1e-12)
        assert eps.imag <= 0.0  # passive under e^{+jwt}
    assert model.complex_eps(1.0e9) != model.complex_eps(10.0e9)


def test_tabulated_permittivity_interpolates_and_validates_range():
    table = TabulatedPermittivity(
        frequency_hz=(1.0e9, 2.0e9), eps_real=(4.0, 3.0), eps_imag=(-0.2, -0.4)
    )
    assert table.complex_eps(1.0e9) == pytest.approx(4.0 - 0.2j)
    assert table.complex_eps(1.5e9) == pytest.approx(3.5 - 0.3j)
    assert table.complex_eps(2.0e9) == pytest.approx(3.0 - 0.4j)
    with pytest.raises(ValueError, match="outside the tabulated range"):
        table.complex_eps(0.5e9)
    with pytest.raises(ValueError, match="eps_imag"):
        TabulatedPermittivity(
            frequency_hz=(1.0e9, 2.0e9), eps_real=(4.0, 3.0), eps_imag=(0.2, -0.4)
        )


def test_layer_parameters_fold_dispersion_loss_into_equivalent_sigma():
    layer = Layer(
        thickness_m=0.05,
        eps_model=DebyeModel(eps_inf=2.0, delta_eps=1.5, tau_s=8.0e-12),
    )
    for frequency in (1.0e9, 10.0e9):
        eps = layer.complex_eps(frequency)
        eps_r_real, sigma_equiv, mu_r = layer.parameters(frequency)
        omega = 2.0 * math.pi * frequency
        assert eps_r_real == pytest.approx(eps.real, rel=1e-12)
        assert sigma_equiv == pytest.approx(-eps.imag * omega * _EPS0, rel=1e-12)
        assert mu_r == 1.0
    low = layer.parameters(1.0e9)
    high = layer.parameters(10.0e9)
    assert low[0] > high[0]  # Debye eps' relaxes down with frequency
    assert low[1] < high[1]  # while the equivalent conductivity grows


def test_csr_layout_for_mixed_material_scene():
    compiled = _compile(
        Dielectric(eps_r=2.5, sigma_e=0.01, thickness_m=0.2),
        PerfectConductor(),
        _three_layer_surface(),
        ITUMaterial(name="concrete"),
    )
    materials = compiled.materials

    assert materials.abi_version == MATERIAL_ABI_VERSION == 3
    assert materials.layer_count.tolist() == [1, 1, 3, 1]
    assert materials.layer_offset.tolist() == [0, 1, 2, 5]
    assert materials.layer_thickness_m.shape == (6,)

    # Legacy scalar materials become one layer from their scalar parameters.
    assert materials.layer_thickness_m[0] == pytest.approx(0.2)
    assert materials.layer_eps_r[0] == pytest.approx(2.5)
    assert materials.layer_sigma_e[0] == pytest.approx(0.01)
    # PEC keeps its effective-sigma encoding inside the CSR.
    assert float(materials.layer_sigma_e[1]) >= PEC_EFFECTIVE_SIGMA_E
    # PhysicalSurface layers land in declaration order at their offset.
    surface = _three_layer_surface()
    expected = surface.layer_parameters(_FREQUENCY)
    for slot, row in enumerate(expected, start=2):
        assert materials.layer_thickness_m[slot] == pytest.approx(row[0])
        assert materials.layer_eps_r[slot] == pytest.approx(row[1], rel=1e-6)
        assert materials.layer_sigma_e[slot] == pytest.approx(row[2], rel=1e-6)
        assert materials.layer_mu_r[slot] == pytest.approx(row[3])

    assert materials.model_id.tolist()[2] == PHYSICAL_SURFACE_MODEL_ID
    assert materials.geometry_mode_id.tolist() == [0, 0, 0, 0]
    assert materials.scatter_model_id.tolist() == [0, 0, 1, 0]


def test_roughness_fields_and_scatter_model_flag_land_in_store():
    rough = _three_layer_surface()
    smooth = _three_layer_surface(roughness_front=None, name="smooth-wall")
    materials = _compile(rough, smooth).materials

    assert materials.rough_sigma_h_m.tolist() == pytest.approx([1.5e-3, 0.0])
    assert materials.rough_corr_x_m.tolist() == pytest.approx([0.03, 0.0])
    assert materials.rough_corr_y_m.tolist() == pytest.approx([0.05, 0.0])
    assert materials.rough_axis_rad.tolist() == pytest.approx([0.4, 0.0])
    assert materials.scatter_model_id.tolist() == [1, 0]


def test_legacy_scalar_view_equals_layer_zero():
    surface = _three_layer_surface()
    materials = _compile(surface).materials
    eps_r0, sigma_e0, mu_r0 = surface.layers[0].parameters(_FREQUENCY)

    assert float(materials.eps_r[0]) == pytest.approx(eps_r0, rel=1e-6)
    assert float(materials.sigma_e[0]) == pytest.approx(sigma_e0, rel=1e-6)
    assert float(materials.mu_r[0]) == pytest.approx(mu_r0, rel=1e-6)
    assert float(materials.thickness_m[0]) == pytest.approx(
        surface.layers[0].thickness_m
    )
    assert float(materials.layer_eps_r[0]) == pytest.approx(
        float(materials.eps_r[0])
    )
    params = surface.parameters(_FREQUENCY)
    assert params["legacy_scalar_approximation"] is True
    assert surface.parameters(_FREQUENCY)["model_id"] == PHYSICAL_SURFACE_MODEL_ID


def test_dielectric_as_physical_surface_matches_scalar_compile():
    dielectric = Dielectric(eps_r=3.0, sigma_e=0.02, mu_r=1.2, thickness_m=0.15)
    surface = dielectric.as_physical_surface()

    assert len(surface.layers) == 1
    assert surface.layer_parameters(_FREQUENCY) == ((0.15, 3.0, 0.02, 1.2),)

    scalar = _compile(dielectric).materials
    layered = _compile(surface).materials
    assert float(layered.layer_eps_r[0]) == float(scalar.layer_eps_r[0])
    assert float(layered.layer_sigma_e[0]) == float(scalar.layer_sigma_e[0])
    assert float(layered.layer_mu_r[0]) == float(scalar.layer_mu_r[0])
    assert float(layered.layer_thickness_m[0]) == float(scalar.layer_thickness_m[0])


def test_surface_assignment_unwraps_material_and_registers_phase_screen():
    screen = PhaseScreen(
        height=[[0.0, 1.0], [1.0, 0.0]],
        height_scale_m=0.01,
        realization_id=3,
    )
    assigned = SurfaceAssignment(material=_three_layer_surface(), phase_screen=screen)
    compiled = _compile(Dielectric(eps_r=2.0), assigned)

    assert compiled.materials.model_id.tolist() == [1, PHYSICAL_SURFACE_MODEL_ID]
    assert compiled.assignments.structure_phase_screens == {1: screen}
    height = screen.height_tensor()
    assert height.shape == (2, 2)
    assert height.dtype == torch.float32

    bare = SurfaceAssignment(material=Dielectric(eps_r=2.0))
    assert _compile(bare).assignments.structure_phase_screens == {}


def test_cache_token_changes_when_layer_or_roughness_changes():
    base = _compile(_three_layer_surface()).materials
    same = _compile(_three_layer_surface()).materials
    thicker_layer = _compile(
        _three_layer_surface(
            layers=(
                Layer(thickness_m=0.03, eps_r=4.0, sigma_e=0.01),
                Layer(
                    thickness_m=0.05,
                    eps_model=DebyeModel(eps_inf=2.0, delta_eps=1.5, tau_s=8.0e-12),
                ),
                Layer(thickness_m=0.01, eps_r=2.2, mu_r=1.1),
            )
        )
    ).materials
    rougher = _compile(
        _three_layer_surface(
            roughness_front=Roughness(
                rms_height_m=2.5e-3,
                corr_length_x_m=0.03,
                corr_length_y_m=0.05,
                principal_axis_rad=0.4,
            )
        )
    ).materials

    assert base.cache_token == same.cache_token
    assert base.cache_token != thicker_layer.cache_token
    assert base.cache_token != rougher.cache_token


def test_field_bundle_exposes_material_level_csr_views():
    if not torch.cuda.is_available():
        pytest.skip("field bundle export requires CUDA")
    compiled = _compile(Dielectric(eps_r=2.0), _three_layer_surface())
    bundle = face_material_field_bundle(compiled, device=torch.device("cuda"))

    # Material-level (M or L sized), not per-face expanded.
    assert bundle["layer_offset"].tolist() == [0, 1]
    assert bundle["layer_count"].tolist() == [1, 3]
    assert bundle["layer_thickness_m"].shape == (4,)
    assert bundle["scatter_model_id"].tolist() == [0, 1]
    assert bundle["geometry_mode_id"].tolist() == [0, 0]
    assert bundle["rough_sigma_h_m"].tolist() == pytest.approx([0.0, 1.5e-3])
    # Existing per-face keys keep their contract.
    assert bundle["material_id"].tolist() == [0, 1]
    assert bundle["eps_r"].shape == (2,)
