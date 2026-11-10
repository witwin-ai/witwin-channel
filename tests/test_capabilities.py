# Copyright Xingyu Chen.
# Tests capabilities.

import json

from witwin.channel import capabilities


def test_capability_manifest_is_versioned_serializable_and_defensive():
    manifest = capabilities()

    assert manifest["schema_version"] == 1
    assert manifest["components"] == [
        "los",
        "reflection",
        "diffraction",
        "transmission",
        "scattering",
    ]
    # transmission is integrated in all four solvers as of transmission; Kirchhoff
    # scattering is integrated in all four solvers as of rough scattering.
    integration = manifest["component_solver_integration"]
    assert set(integration) == {"transmission", "scattering"}
    assert integration["transmission"] == {
        "path": True,
        "deterministic": True,
        "montecarlo_basic": True,
        "montecarlo_bdpt": True,
    }
    assert integration["scattering"] == {
        "path": True,
        "deterministic": True,
        "montecarlo_basic": True,
        "montecarlo_bdpt": True,
    }
    path = manifest["solvers"]["path"]
    assert path["max_reflection_depth"] == 5
    assert path["supports_reflection_diffraction_coupling"] is True
    assert path["supports_arrays"] is True
    assert path["supports_reflection_diffraction_coupling_geometry"] is True
    assert (
        path["reflection_diffraction_coupling_topology"]
        == "one_reflection_one_diffraction_both_orders"
    )
    assert path["max_reflections_in_coupled_path"] == 1
    assert path["reflection_diffraction_coupling_candidate_limit"] == 1_000_000
    # coupled double diffraction: coupled_paths=True now enables the uniform order-2
    # compensator family including cid 7 double diffraction (D->D). The key is
    # exposed on every solver block that declares coupling support.
    assert path["coupled_double_diffraction"] is True
    assert manifest["solvers"]["deterministic"]["coupled_double_diffraction"] is True
    assert manifest["solvers"]["montecarlo_bdpt"]["coupled_double_diffraction"] is True
    assert "coupled_double_diffraction" not in manifest["solvers"]["montecarlo_basic"]
    assert manifest["supports_ad"] is True
    materials = manifest["materials"]
    assert materials["abi_version"] == 3
    assert materials["perfect_conductor_model"] == "explicit"
    assert materials["physical_surface"] is True
    assert materials["layer_csr"] is True
    assert all(
        enabled is True
        for enabled in materials["runtime_material_abi_integration"].values()
    )
    assert all(
        enabled is False for enabled in materials["event_solver_integration"].values()
    )
    json.dumps(manifest)

    manifest["components"].append("invalid")
    assert "invalid" not in capabilities()["components"]
