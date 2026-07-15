from __future__ import annotations

from types import SimpleNamespace

import pytest

import witwin.channel_native.deployment as deployment
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


def test_runtime_diagnostics_are_import_safe_and_report_sm_policy(monkeypatch):
    monkeypatch.setattr(deployment, "_import_torch", _cuda_unavailable_torch)
    monkeypatch.setattr(deployment, "_import_native_build_info", lambda: dict)

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
    verified = sm_support(120)
    assert verified["declared_supported"] is True
    assert verified["runtime_verified"] is True
    assert verified["status"] == "runtime_verified"
    assert verified["evidence"] == [deployment.SM120_EVIDENCE]
    assert sm_support(90)["status"] == "not_declared"


def _cuda_unavailable_torch():
    return SimpleNamespace(
        __version__="2.10.0",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )


def test_runtime_diagnostics_records_expected_import_failures(monkeypatch):
    def missing_torch():
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(deployment, "_import_torch", missing_torch)
    monkeypatch.setattr(deployment, "_import_native_build_info", lambda: dict)

    diagnostics = runtime_diagnostics()

    assert diagnostics["errors"] == [
        "PyTorch import failed (ModuleNotFoundError): No module named 'torch'"
    ]
    assert diagnostics["native_build"] == {}


@pytest.mark.parametrize(
    "error",
    [
        OSError("dependent CUDA DLL is unavailable"),
        ImportError("native extension ABI validation failed"),
    ],
)
def test_runtime_diagnostics_records_native_library_load_failures(monkeypatch, error):
    def missing_native_library():
        raise error

    monkeypatch.setattr(deployment, "_import_torch", _cuda_unavailable_torch)
    monkeypatch.setattr(
        deployment, "_import_native_build_info", lambda: missing_native_library
    )

    diagnostics = runtime_diagnostics()

    assert diagnostics["cuda_available"] is False
    assert diagnostics["errors"] == [
        "native extension import failed; install a matching Channel Native wheel "
        "or configure the explicit developer extension override; reason "
        f"({type(error).__name__}): {error}"
    ]


def test_runtime_diagnostics_does_not_hide_cuda_probe_failures(monkeypatch):
    torch = _cuda_unavailable_torch()

    def fail_probe():
        raise RuntimeError("CUDA driver probe failed")

    torch.cuda.is_available = fail_probe
    monkeypatch.setattr(deployment, "_import_torch", lambda: torch)

    with pytest.raises(RuntimeError, match="CUDA driver probe failed"):
        runtime_diagnostics()


@pytest.mark.parametrize("error", [RuntimeError("native build metadata is malformed")])
def test_runtime_diagnostics_does_not_hide_build_info_failures(monkeypatch, error):
    def fail_build_info():
        raise error

    monkeypatch.setattr(deployment, "_import_torch", _cuda_unavailable_torch)
    monkeypatch.setattr(
        deployment, "_import_native_build_info", lambda: fail_build_info
    )

    with pytest.raises(type(error), match=str(error)):
        runtime_diagnostics()
