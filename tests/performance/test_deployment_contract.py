from __future__ import annotations

from witwin.channel_native.deployment import (
    DEPLOYMENT_ABI,
    PIPELINE_CACHE_ABI,
    PIPELINE_CACHE_IMPLEMENTED,
    pipeline_cache_key,
    runtime_diagnostics,
    sm_support,
)


def _key(**updates):
    values = {
        "geometry_version": 1,
        "material_version": 2,
        "assignment_version": 3,
        "frequency_hz": 3.5e9,
        "solver": "path",
        "build": {"cuda": "12.8", "abi": 1},
    }
    values.update(updates)
    return pipeline_cache_key(**values)


def test_pipeline_cache_key_is_stable_and_invalidates_all_contract_inputs():
    baseline = _key()

    assert baseline == _key()
    assert baseline != _key(geometry_version=2)
    assert baseline != _key(material_version=3)
    assert baseline != _key(assignment_version=4)
    assert baseline != _key(frequency_hz=3.6e9)
    assert baseline != _key(solver="bdpt")
    assert baseline != _key(build={"cuda": "12.9", "abi": 1})


def test_runtime_diagnostics_are_import_safe_and_report_sm_policy():
    diagnostics = runtime_diagnostics()

    assert diagnostics["deployment_abi"] == DEPLOYMENT_ABI
    assert PIPELINE_CACHE_ABI.endswith("v1")
    assert diagnostics["declared_sm_architectures"] == [75, 80, 86, 89, 120]
    assert diagnostics["sm_matrix_status"] == "declared_unverified"
    assert diagnostics["pipeline_cache"] == {
        "implemented": False,
        "key_abi": PIPELINE_CACHE_ABI,
    }
    assert diagnostics["wheel_smoke"]["verified"] is False
    assert PIPELINE_CACHE_IMPLEMENTED is False
    assert isinstance(diagnostics["errors"], list)
    declared = sm_support(89)
    assert declared["declared_supported"] is True
    assert declared["runtime_verified"] is False
    assert declared["status"] == "declared_unverified"
    assert declared["evidence"] == []
    assert sm_support(90)["status"] == "unsupported"
