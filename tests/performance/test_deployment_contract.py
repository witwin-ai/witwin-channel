# Copyright Xingyu Chen.
# Tests the deployment diagnostics, the GPU support check and the doctor command.

from __future__ import annotations

from types import SimpleNamespace

import pytest

import witwin.channel.deployment as deployment
from witwin.channel import runtime
from witwin.channel.deployment import (
    DEPLOYMENT_ABI,
    PIPELINE_CACHE_ABI,
    pipeline_cache_key,
    runtime_diagnostics,
)

_RELEASE_ARCHITECTURES = [
    "70-real", "75-real", "80-real", "86-real", "89-real", "90-real", "100-real", "120-real",
    "120-virtual",
]


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


def _cuda_unavailable_torch(cuda="12.8"):
    return SimpleNamespace(
        __version__="2.10.0",
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(is_available=lambda: False),
    )


def _cuda_torch(sm: int, *, device_index: int = 0):
    return SimpleNamespace(
        __version__="2.10.0",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: device_index,
            get_device_capability=lambda index: divmod(sm, 10),
            get_device_name=lambda index: "test GPU",
            get_device_properties=lambda index: SimpleNamespace(total_memory=1024),
        ),
    )


def _fake_native(monkeypatch, architectures, *, driver=12080, optix=(True, "")):
    info = {
        "cuda_architectures": list(architectures),
        "cuda_version": "12.8",
        "torch_version": "2.10.0",
    }
    monkeypatch.setattr(deployment, "_import_native_build_info", lambda: lambda: info)
    monkeypatch.setattr(
        runtime,
        "runtime_probe",
        lambda index, probe_optix=True: {
            "cuda_driver_version": driver,
            "optix_available": optix[0],
            "optix_reason": optix[1],
        },
    )
    monkeypatch.setattr(runtime, "scene_trace_backends", lambda: [])


def test_runtime_diagnostics_drop_hard_coded_placeholders(monkeypatch):
    monkeypatch.setattr(deployment, "_import_torch", _cuda_unavailable_torch)
    _fake_native(monkeypatch, _RELEASE_ARCHITECTURES)

    diagnostics = runtime_diagnostics()

    assert diagnostics["deployment_abi"] == DEPLOYMENT_ABI
    assert PIPELINE_CACHE_ABI.endswith("v1")
    for removed in (
        "declared_sm_architectures", "sm_matrix_status", "pipeline_cache", "wheel_smoke",
        "ptx_forward_compatibility_sm",
    ):
        assert removed not in diagnostics
    assert diagnostics["errors"] == []


def test_declared_architectures_come_from_the_build(monkeypatch):
    monkeypatch.setattr(deployment, "_import_torch", lambda: _cuda_torch(90))
    _fake_native(monkeypatch, _RELEASE_ARCHITECTURES)

    diagnostics = runtime_diagnostics()

    assert diagnostics["build_sm_architectures"] == {
        "sass": [70, 75, 80, 86, 89, 90, 100, 120],
        "ptx": [120],
    }
    device = diagnostics["device"]
    assert device["sm"] == 90
    assert device["code_mode"] == "sass"
    assert device["supported"] is True
    assert device["problems"] == []


def test_runtime_diagnostics_report_the_optix_probe_and_requested_backend(monkeypatch):
    monkeypatch.setattr(deployment, "_import_torch", lambda: _cuda_torch(86, device_index=2))
    _fake_native(monkeypatch, ["86-real"], optix=(False, "disabled by RAYD_DISABLE_OPTIX"))
    monkeypatch.setenv(runtime.TRACE_BACKEND_ENV, "cuda")

    diagnostics = runtime_diagnostics()

    assert diagnostics["device"]["index"] == 2
    assert diagnostics["cuda_driver_version"] == 12080
    assert diagnostics["trace_backend"] == {
        "requested": "cuda",
        "scenes": [],
        "optix": {"available": False, "reason": "disabled by RAYD_DISABLE_OPTIX"},
    }


def test_runtime_diagnostics_rejects_negative_active_cuda_device(monkeypatch):
    monkeypatch.setattr(
        deployment, "_import_torch", lambda: _cuda_torch(120, device_index=-1)
    )

    with pytest.raises(RuntimeError, match="invalid active device index -1"):
        runtime_diagnostics()


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
    [message] = diagnostics["errors"]
    assert message.startswith(
        f"native extension import failed: ({type(error).__name__}): {error}."
    )
    assert "pip install --force-reinstall witwin-channel" in message


def test_native_load_failure_under_cpu_only_torch_names_the_cause(monkeypatch):
    def missing_native_library():
        raise ImportError("libc10_cuda.so: cannot open shared object file")

    monkeypatch.setattr(deployment, "_import_torch", lambda: _cuda_unavailable_torch(None))
    monkeypatch.setattr(
        deployment, "_import_native_build_info", lambda: missing_native_library
    )

    diagnostics = runtime_diagnostics()

    assert diagnostics["torch_cuda_build"] is False
    [message] = diagnostics["errors"]
    assert "CPU-only build" in message
    assert "pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128" in message


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


# --- GPU support rules --------------------------------------------------


@pytest.mark.parametrize(
    ("sm", "architectures", "mode"),
    [
        (86, ["86-real"], "sass"),
        (89, ["86-real"], "sass"),
        (86, ["89-real"], "unsupported"),
        (87, ["80-real", "86-real"], "unsupported"),
        (87, ["87-real"], "sass"),
        (89, ["87-real"], "unsupported"),
        (90, ["86-real"], "unsupported"),
        (121, ["120-real", "120-virtual"], "sass"),
        (130, ["120-real", "120-virtual"], "ptx-jit"),
        (61, ["70-real", "70-virtual"], "unsupported"),
    ],
)
def test_device_code_mode_follows_cubin_and_ptx_compatibility(sm, architectures, mode):
    sass, ptx = runtime.build_targets(architectures)
    assert runtime.device_code_mode(sm, sass, ptx) == mode


def test_unsupported_gpu_problem_names_the_build_targets():
    problems = runtime.device_support_problems(
        90, ["86-real", "120-real", "120-virtual"], 12080, cuda_version="12.8"
    )
    assert problems == [
        "the build has no CUDA code for sm_90 (cubins: sm_86, sm_120; PTX: compute_120)"
    ]


def test_pre_volta_gpu_is_rejected_outright():
    [problem] = runtime.device_support_problems(61, _RELEASE_ARCHITECTURES, 12080, cuda_version="12.8")
    assert "7.0 (Volta) or newer" in problem


def test_old_driver_is_rejected():
    problems = runtime.device_support_problems(86, ["86-real"], 11080, cuda_version="12.8")
    assert problems == [
        "the NVIDIA driver supports CUDA 11.8; Channel needs a driver for CUDA 12 or newer "
        "(R525+, R555+ for OptiX)"
    ]


def test_ptx_jit_needs_a_driver_for_the_build_toolkit():
    assert runtime.device_support_problems(
        130, ["120-real", "120-virtual"], 12080, cuda_version="12.8"
    ) == []
    [problem] = runtime.device_support_problems(
        130, ["120-real", "120-virtual"], 12040, cuda_version="12.8"
    )
    assert "JIT-compiling its PTX" in problem


def test_unsupported_device_message_names_gpu_sm_build_and_fix():
    message = runtime.unsupported_device_message(
        device_index=0, name="Tesla V100", sm=70, architectures=["86-real"],
        problems=["the build has no CUDA code for sm_70"],
    )
    assert "CUDA device 0 'Tesla V100' (compute capability 7.0, sm_70)" in message
    assert "86-real" in message
    assert "CHANNEL_DEV_CUDA_ARCH=70" in message
    assert runtime.SKIP_DEVICE_CHECK_ENV in message


def test_compile_guard_raises_before_native_work(monkeypatch):
    runtime.require_device_supported.cache_clear()
    monkeypatch.delenv(runtime.SKIP_DEVICE_CHECK_ENV, raising=False)
    monkeypatch.setattr(runtime.torch.version, "cuda", "12.8")
    monkeypatch.setattr(
        runtime, "build_info", lambda: {"cuda_architectures": ["120-real"], "cuda_version": "12.8"}
    )
    monkeypatch.setattr(runtime.torch.cuda, "get_device_capability", lambda index: (8, 6))
    monkeypatch.setattr(runtime.torch.cuda, "get_device_name", lambda index: "RTX A6000")
    monkeypatch.setattr(
        runtime, "runtime_probe", lambda index, probe_optix=True: {"cuda_driver_version": 12080}
    )
    try:
        with pytest.raises(runtime.UnsupportedDeviceError, match="'RTX A6000'.*sm_86"):
            runtime.require_device_supported(0)
        monkeypatch.setenv(runtime.SKIP_DEVICE_CHECK_ENV, "1")
        runtime.require_device_supported.cache_clear()
        runtime.require_device_supported(0)
    finally:
        runtime.require_device_supported.cache_clear()


def test_compile_guard_names_cpu_only_torch(monkeypatch):
    runtime.require_device_supported.cache_clear()
    monkeypatch.delenv(runtime.SKIP_DEVICE_CHECK_ENV, raising=False)
    monkeypatch.setattr(runtime.torch.version, "cuda", None)
    try:
        with pytest.raises(runtime.UnsupportedDeviceError, match="CPU-only build"):
            runtime.require_device_supported(0)
    finally:
        runtime.require_device_supported.cache_clear()


# --- Trace backend request ----------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "auto"), ("", "auto"), ("auto", "auto"), ("OptiX", "optix"), (" cuda ", "cuda")],
)
def test_requested_trace_backend_reads_the_environment(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(runtime.TRACE_BACKEND_ENV, raising=False)
    else:
        monkeypatch.setenv(runtime.TRACE_BACKEND_ENV, raw)
    assert runtime.requested_trace_backend() == expected


def test_requested_trace_backend_rejects_unknown_names(monkeypatch):
    monkeypatch.setenv(runtime.TRACE_BACKEND_ENV, "embree")
    with pytest.raises(ValueError, match="WITWIN_CHANNEL_TRACE_BACKEND must be one of"):
        runtime.requested_trace_backend()


# --- Doctor -------------------------------------------------------------


def _healthy_diagnostics(**updates):
    diagnostics = {
        "package_version": "0.6.0",
        "python": "3.12.3",
        "python_executable": "/usr/bin/python3",
        "platform": "Linux",
        "torch": "2.10.0+cu128",
        "cuda_runtime": "12.8",
        "torch_cuda_build": True,
        "cuda_available": True,
        "cuda_driver_version": 12080,
        "device": {"index": 0, "name": "RTX A6000", "sm": 86, "problems": []},
        "native_build": {"torch_version": "2.10.0", "cuda_version": "12.8",
                         "build_fingerprint": "f" * 64, "rayd_commit": "e" * 40},
        "build_sm_architectures": {"sass": [86], "ptx": []},
        "trace_backend": {"requested": "auto", "scenes": [],
                          "optix": {"available": True, "reason": ""}},
        "errors": [],
    }
    diagnostics.update(updates)
    return diagnostics


def test_doctor_reports_ok_on_a_healthy_machine():
    text, status = deployment.doctor_report(_healthy_diagnostics(), driver_release="570.86")
    assert status == 0
    assert "Verdict: OK - Channel runs here." in text
    assert "570.86" in text
    assert "sm_86" in text
    assert "f" * 64 in text


def test_doctor_warns_when_optix_is_unusable():
    trace = {"requested": "auto", "scenes": [],
             "optix": {"available": False, "reason": "driver library missing"}}
    text, status = deployment.doctor_report(
        _healthy_diagnostics(trace_backend=trace), driver_release=None
    )
    assert status == 0
    assert "Verdict: OK with warnings" in text
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics" in text


def test_doctor_fails_on_cpu_only_torch():
    diagnostics = _healthy_diagnostics(
        torch="2.10.0+cpu", cuda_runtime=None, torch_cuda_build=False, cuda_available=False
    )
    del diagnostics["device"]
    text, status = deployment.doctor_report(diagnostics, driver_release=None)
    assert status == 1
    assert "CPU-only build" in text
    assert "--index-url https://download.pytorch.org/whl/cu128" in text


def test_doctor_fails_on_an_unsupported_gpu():
    device = {"index": 0, "name": "GTX 1080", "sm": 61, "problems": ["needs 7.0"]}
    text, status = deployment.doctor_report(_healthy_diagnostics(device=device), driver_release=None)
    assert status == 1
    assert "Verdict: FAIL" in text


def test_doctor_main_prints_report_and_exit_status(monkeypatch, capsys):
    monkeypatch.setattr(deployment, "runtime_diagnostics", _healthy_diagnostics)
    monkeypatch.setattr(deployment, "_nvidia_driver_release", lambda: "570.86")
    assert deployment.main(["doctor"]) == 0
    assert "Verdict: OK" in capsys.readouterr().out


def test_doctor_main_reports_a_failed_probe_without_blaming_pytorch(monkeypatch, capsys):
    def fail() -> dict:
        raise RuntimeError("CUDA error in cudaFree(nullptr): out of memory")

    monkeypatch.setattr(deployment, "runtime_diagnostics", fail)
    monkeypatch.setattr(deployment, "_nvidia_driver_release", lambda: "570.86")
    monkeypatch.setattr(deployment, "_import_torch", lambda: SimpleNamespace(
        __version__="2.10.0+cu128", version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: True),
    ))
    assert deployment.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "runtime diagnostics failed (RuntimeError): CUDA error in cudaFree" in out
    assert "2.10.0+cu128" in out
    assert "not importable" not in out
    assert "CPU-only" not in out
