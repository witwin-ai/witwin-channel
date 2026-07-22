import math
import xml.etree.ElementTree as ET

import torch

from witwin.channel import Scene, Structure
from witwin.channel.core.material_runtime import face_material_field_bundle
from witwin.channel.core.materials import (
    MATERIAL_ABI_VERSION,
    DispersiveMaterial,
    PerfectConductor,
)
from witwin.channel.core.scene_loader import _material_defs, _native_material


def _triangle(material, *, name: str = "wall") -> Structure:
    return Structure(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        material=material,
        name=name,
    )


def test_dispersive_material_re_evaluates_and_invalidates_frequency_cache():
    material = DispersiveMaterial(
        eps_r_ref=4.0,
        sigma_e_ref=0.01,
        reference_frequency_hz=1.0e9,
        eps_r_exponent=-0.25,
        sigma_e_exponent=0.5,
        thickness_m=0.2,
        scattering_coefficient=0.3,
        xpd_coefficient=0.1,
        name="measured-wall",
    )
    low = Scene(
        structures=[_triangle(material)], transmitters=[], receivers=[], frequency=1.0e9
    ).compile()
    high = Scene(
        structures=[_triangle(material)], transmitters=[], receivers=[], frequency=4.0e9
    ).compile()

    assert low.materials.abi_version == MATERIAL_ABI_VERSION
    assert math.isclose(float(high.materials.eps_r[0]), 4.0 * 4.0**-0.25, rel_tol=1e-6)
    assert math.isclose(float(high.materials.sigma_e[0]), 0.02, rel_tol=1e-6)
    assert low.materials.cache_token != high.materials.cache_token
    assert low.materials.material_keys == ("0:wall:measured-wall",)
    assert low.materials.material_id.tolist() == [0]
    assert math.isclose(float(low.materials.thickness_m[0]), 0.2, rel_tol=1e-6)
    assert math.isclose(
        float(low.materials.scattering_coefficient[0]), 0.3, rel_tol=1e-6
    )
    assert math.isclose(float(low.materials.xpd_coefficient[0]), 0.1, rel_tol=1e-6)


def test_xml_material_fields_and_units_survive_frequency_evaluation():
    root = ET.fromstring(
        """
        <scene>
          <bsdf id="wall-material">
            <string name="type" value="concrete"/>
            <float name="thickness" value="0.25"/>
            <float name="scattering_coefficient" value="0.35"/>
            <float name="xpd_coefficient" value="0.15"/>
          </bsdf>
        </scene>
        """
    )
    parsed = _material_defs(root)["wall-material"]
    material = _native_material(
        str(parsed["type"]),
        3.5e9,
        thickness_m=float(parsed["thickness"]),
        scattering_coefficient=float(parsed["scattering_coefficient"]),
        xpd_coefficient=float(parsed["xpd_coefficient"]),
    )
    parameters = material.parameters(28.0e9)
    assert parameters["name"] == "concrete"
    assert parameters["thickness_m"] == 0.25
    assert parameters["scattering_coefficient"] == 0.35
    assert parameters["xpd_coefficient"] == 0.15
    assert parameters["eps_r"] == 5.24


def test_pec_is_traceable_as_explicit_model_not_inferred_from_sigma():
    scene = Scene(
        structures=[_triangle(PerfectConductor(), name="metal")],
        transmitters=[],
        receivers=[],
        frequency=3.5e9,
    )
    compiled = scene.compile()
    assert compiled.materials.model_id.tolist() == [2]
    assert compiled.materials.sigma_e.tolist() == [0.0]

    if torch.cuda.is_available():
        bundle = face_material_field_bundle(compiled, device=torch.device("cuda"))
        assert bundle["material_id"].tolist() == [0]
        assert bundle["model_id"].tolist() == [2]
        assert float(bundle["sigma_e"][0]) >= 1.0e9
