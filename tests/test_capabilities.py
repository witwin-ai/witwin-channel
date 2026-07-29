# Copyright Xingyu Chen.
# Tests capabilities.

import json
from pathlib import Path

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
    # transmission is integrated in all four solvers as of wave 2; Kirchhoff
    # scattering is integrated in all four solvers as of wave 3.
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
    # ADR-013 D5: coupled_paths=True now enables the uniform order-2
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


def test_replacement_inventory_has_unique_legacy_apis_and_required_fields():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "docs" / "dev" / "replacement" / "channel-api-inventory.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["schema_version"] == 1
    assert payload["scan_scope"]["production_import_references"] == 0
    apis = payload["apis"]
    assert len({entry["legacy_api"] for entry in apis}) == len(apis)
    assert all(
        {"legacy_api", "observed_calls", "priority", "native_api", "status"}.issubset(entry)
        for entry in apis
    )
    assert all(entry["observed_calls"] >= 0 for entry in apis)
    public_surface = {
        "Bool", "AntennaArray", "Box", "Complex2f", "EdgePolicy", "Float", "Grid",
        "GridSpec", "Int32", "Material", "Matrix4f", "PlanarArray", "Point2f", "Point3f",
        "RadioMapResult", "Receiver", "ReceiverGrid", "Scene", "Structure", "Transmitter",
        "ULA", "UPA", "UInt32", "Vector2f", "Vector3f", "Vector3u", "deterministic",
        "montecarlo", "path", "path.Config", "path.InteractionType", "path.PathResult",
        "path.Tuning", "path.solve", "deterministic.Config", "deterministic.FieldResult",
        "deterministic.FieldSpec", "deterministic.NativeExtension",
        "deterministic.RadioMapResult", "deterministic.Tuning",
        "deterministic.native_extension_available", "deterministic.solve",
        "deterministic.solve_field", "montecarlo.Config", "montecarlo.ComponentFilterConfig",
        "montecarlo.DiffractionExecutionConfig", "montecarlo.FilterConfig",
        "montecarlo.IntegratorOptions", "montecarlo.NativeExtension",
        "montecarlo.RadioMapResult", "montecarlo.Tuning", "montecarlo.solve",
    }
    assert public_surface.issubset({entry["legacy_api"] for entry in apis})