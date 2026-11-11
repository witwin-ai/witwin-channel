# Copyright Xingyu Chen.
# Reports the installed Channel build, GPU, driver and trace backend, and runs the doctor check.

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast


DEPLOYMENT_ABI = "witwin.channel.deployment.v2"
PIPELINE_CACHE_ABI = "witwin.channel.pipeline-cache.v1"


def _package_version() -> str:
    try:
        return importlib.metadata.version("witwin-channel")
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _required_torch() -> str:
    """Return the ``torch==X`` requirement of the installed wheel, or plain ``torch``."""

    try:
        requirements = importlib.metadata.requires("witwin-channel") or []
    except importlib.metadata.PackageNotFoundError:
        requirements = []
    for requirement in requirements:
        match = re.fullmatch(r"torch\s*==\s*([0-9][0-9A-Za-z.]*)", requirement.strip())
        if match is not None:
            return f"torch=={match.group(1)}"
    return "torch"


def _import_torch() -> ModuleType:
    import torch

    return cast(ModuleType, torch)


def _import_runtime() -> ModuleType:
    from witwin.channel import runtime

    return runtime


def _import_native_build_info() -> Callable[[], dict[str, object]]:
    return cast(Callable[[], dict[str, object]], _import_runtime().build_info)


def _require_active_cuda_device(torch: ModuleType) -> int:
    index = int(torch.cuda.current_device())
    if index < 0:
        raise RuntimeError(
            f"CUDA reports available but has invalid active device index {index}"
        )
    return index


def _import_error(component: str, exc: ImportError | OSError) -> str:
    return f"{component} import failed ({type(exc).__name__}): {exc}"


def _torch_runtime_diagnostics(torch: ModuleType) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "torch_cuda_build": torch.version.cuda is not None,
        "cuda_available": cuda_available,
    }
    if cuda_available and (index := _require_active_cuda_device(torch)) >= 0:
        major, minor = torch.cuda.get_device_capability(index)
        info["device"] = {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "sm": major * 10 + minor,
            "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
        }
    return info


def _native_import_error(exc: ImportError | OSError, torch: ModuleType | None) -> str:
    if torch is not None and torch.version.cuda is None:
        reason = _import_runtime().cpu_only_torch_message(str(torch.__version__))
    else:
        reason = f"({type(exc).__name__}): {exc}"
    return (
        f"native extension import failed: {reason}. A wheel install needs "
        "`pip install --force-reinstall witwin-channel` with the PyTorch build it names; a "
        "source checkout needs the developer extension override (scripts/dev_env.sh)"
    )


def _device_and_trace_diagnostics(
    diagnostics: dict[str, Any], native_build: dict[str, object],
) -> None:
    runtime = _import_runtime()
    architectures = cast(list[object], native_build.get("cuda_architectures", []))
    sass, ptx = runtime.build_targets(architectures)
    diagnostics["build_sm_architectures"] = {"sass": list(sass), "ptx": list(ptx)}
    trace: dict[str, Any] = {
        "requested": runtime.requested_trace_backend(),
        "scenes": runtime.scene_trace_backends(),
    }
    diagnostics["trace_backend"] = trace
    device = diagnostics.get("device")
    if not isinstance(device, dict):
        return
    probe = runtime.runtime_probe(int(device["index"]))
    driver_version = int(probe["cuda_driver_version"])
    diagnostics["cuda_driver_version"] = driver_version
    trace["optix"] = {
        "available": bool(probe["optix_available"]),
        "reason": str(probe["optix_reason"]).rstrip("."),
    }
    sm = int(device["sm"])
    problems = runtime.device_support_problems(
        sm, architectures, driver_version, cuda_version=str(native_build.get("cuda_version", ""))
    )
    device["code_mode"] = runtime.device_code_mode(sm, sass, ptx)
    device["supported"] = not problems
    device["problems"] = problems


def runtime_diagnostics() -> dict[str, Any]:
    """Return package, PyTorch, GPU, driver, build and trace backend diagnostics."""

    diagnostics: dict[str, Any] = {
        "deployment_abi": DEPLOYMENT_ABI,
        "package_version": _package_version(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "errors": [],
    }
    torch: ModuleType | None = None
    try:
        torch = _import_torch()
    except (ImportError, OSError) as exc:
        diagnostics["errors"].append(_import_error("PyTorch", exc))
    else:
        diagnostics.update(_torch_runtime_diagnostics(torch))

    try:
        native_build = _import_native_build_info()()
    except (ImportError, OSError) as exc:
        diagnostics["errors"].append(_native_import_error(exc, torch))
        return diagnostics
    diagnostics["native_build"] = native_build
    if "cuda_architectures" in native_build:
        _device_and_trace_diagnostics(diagnostics, native_build)
    return diagnostics


def pipeline_cache_key(
    *, geometry_version: int, material_version: int, assignment_version: int, frequency_hz: float,
    solver: str, build: dict[str, Any] | None = None,
) -> str:
    """Return a stable digest of the inputs that identify one compiled solve pipeline."""

    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    payload = {
        "abi": PIPELINE_CACHE_ABI,
        "package_version": _package_version(),
        "geometry_version": int(geometry_version),
        "material_version": int(material_version),
        "assignment_version": int(assignment_version),
        "frequency_hz": float(frequency_hz),
        "solver": str(solver),
        "build": build or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


# --- doctor -------------------------------------------------------------


def _nvidia_driver_release() -> str | None:
    """Return the NVIDIA kernel driver release (for example ``570.86``) when it can be read."""

    proc = Path("/proc/driver/nvidia/version")
    try:
        match = re.search(r"\s(\d+\.\d+(?:\.\d+)?)\s", proc.read_text(encoding="utf-8"))
    except OSError:
        match = None
    if match is not None:
        return match.group(1)
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = completed.stdout.strip().splitlines()
    return lines[0].strip() if completed.returncode == 0 and lines else None


def _cuda_version_text(version: object) -> str:
    if not isinstance(version, int) or version <= 0:
        return "unknown"
    return f"{version // 1000}.{(version % 1000) // 10}"


def _report_rows(diagnostics: dict[str, Any], driver_release: str | None) -> list[str]:
    rows: list[tuple[str, object]] = [
        ("witwin-channel", diagnostics.get("package_version")),
        ("Python", f"{diagnostics.get('python')} ({diagnostics.get('python_executable')})"),
        ("Platform", diagnostics.get("platform")),
        ("PyTorch", diagnostics.get("torch") or "not importable"),
        ("PyTorch CUDA", diagnostics.get("cuda_runtime") or "none (CPU-only build)"),
        ("NVIDIA driver", driver_release or "unknown"),
        ("Driver CUDA version", _cuda_version_text(diagnostics.get("cuda_driver_version"))),
    ]
    device = diagnostics.get("device")
    rows.append(("GPU", f"{device['index']}: {device['name']} (sm_{device['sm']})"
                 if isinstance(device, dict) else "none visible"))
    rows += _build_rows(diagnostics)
    rows += _trace_rows(diagnostics.get("trace_backend"))
    return [f"  {label:<22} {value}" for label, value in rows]


def _build_rows(diagnostics: dict[str, Any]) -> list[tuple[str, object]]:
    native = diagnostics.get("native_build")
    if not isinstance(native, dict):
        return [("Native extension", "not loaded")]
    rows: list[tuple[str, object]] = [
        ("Build torch / CUDA", f"{native.get('torch_version')} / {native.get('cuda_version')}")
    ]
    targets = diagnostics.get("build_sm_architectures")
    if isinstance(targets, dict):
        rows.append(("Build cubins", ", ".join(f"sm_{a}" for a in targets["sass"]) or "none"))
        rows.append(("Build PTX", ", ".join(f"compute_{a}" for a in targets["ptx"]) or "none"))
    rows.append(("Build fingerprint", native.get("build_fingerprint")))
    rows.append(("RayD commit", native.get("rayd_commit")))
    return rows


def _trace_rows(trace: object) -> list[tuple[str, object]]:
    if not isinstance(trace, dict):
        return []
    rows: list[tuple[str, object]] = [("Trace backend request", trace.get("requested"))]
    optix = trace.get("optix")
    if isinstance(optix, dict):
        rows.append(
            ("OptiX", "available" if optix["available"] else f"unavailable: {optix['reason']}")
        )
    scenes = trace.get("scenes") or []
    if scenes:
        rows.append(("Compiled scenes", ", ".join(
            f"device {scene['device_index']}: {scene['trace_backend']}" for scene in scenes
        )))
    return rows


def _torch_failures(diagnostics: dict[str, Any]) -> list[str]:
    torch_version = diagnostics.get("torch")
    if torch_version is None:
        return [
            f"PyTorch is not importable. Install the CUDA build: pip install {_required_torch()} "
            "--index-url https://download.pytorch.org/whl/cu128"
        ]
    if not diagnostics.get("torch_cuda_build", False):
        return [
            f"PyTorch {torch_version} is a CPU-only build. Install the CUDA build: pip install "
            f"torch=={str(torch_version).split('+')[0]} --index-url "
            "https://download.pytorch.org/whl/cu128"
        ]
    if not diagnostics.get("cuda_available", False):
        return [
            "PyTorch sees no CUDA GPU. Check `nvidia-smi`, CUDA_VISIBLE_DEVICES, and in a "
            "container run with `--gpus all`."
        ]
    return []


def _device_failures(device: object) -> list[str]:
    if not isinstance(device, dict) or not device.get("problems"):
        return []
    return [
        *(str(problem) for problem in device["problems"]),
        f"Update the NVIDIA driver, or build Channel from source for sm_{device['sm']} "
        "(see CONTRIBUTING.md).",
    ]


def _optix_findings(trace: object) -> tuple[list[str], list[str]]:
    """Return the OptiX failures and warnings."""

    if not isinstance(trace, dict) or not isinstance(trace.get("optix"), dict):
        return [], []
    optix = trace["optix"]
    if optix["available"]:
        return [], []
    if trace.get("requested") == "optix":
        return [f"WITWIN_CHANNEL_TRACE_BACKEND=optix but OptiX is unusable: {optix['reason']}"], []
    return [], [
        "OptiX is unusable, so scenes trace on the slower pure-CUDA backend "
        f"({optix['reason']}). OptiX needs driver R555 or newer; in a container set "
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics."
    ]


def doctor_report(diagnostics: dict[str, Any], *, driver_release: str | None) -> tuple[str, int]:
    """Format a doctor report; return the text and the exit status (0 ok, 1 failure)."""

    optix_failures, warnings = _optix_findings(diagnostics.get("trace_backend"))
    failures = [
        *_torch_failures(diagnostics),
        *(str(error) for error in diagnostics.get("errors", [])),
        *_device_failures(diagnostics.get("device")),
        *optix_failures,
    ]
    lines = ["witwin-channel doctor", "", *_report_rows(diagnostics, driver_release), ""]
    if failures:
        lines.append("Verdict: FAIL - Channel cannot run here.")
        lines += [f"  - {item}" for item in failures]
    elif warnings:
        lines.append("Verdict: OK with warnings - Channel runs here.")
    else:
        lines.append("Verdict: OK - Channel runs here.")
    lines += [f"  ! {item}" for item in warnings]
    return "\n".join(lines), 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``python -m witwin.channel doctor``."""

    parser = argparse.ArgumentParser(
        prog="python -m witwin.channel",
        description="Check that this machine can run witwin-channel.",
    )
    parser.add_argument("command", nargs="?", default="doctor", choices=["doctor"])
    parser.add_argument("--json", action="store_true", help="print the raw diagnostics as JSON")
    arguments = parser.parse_args(argv)
    try:
        diagnostics = runtime_diagnostics()
    except Exception as exc:  # noqa: BLE001 - the doctor reports every failure it meets.
        diagnostics = {
            "package_version": _package_version(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "errors": [f"runtime diagnostics failed ({type(exc).__name__}): {exc}"],
        }
    if arguments.json:
        print(json.dumps(diagnostics, indent=2, sort_keys=True, default=str))
        return 1 if diagnostics.get("errors") else 0
    text, status = doctor_report(diagnostics, driver_release=_nvidia_driver_release())
    print(text)
    return status


# The package root publishes ``build_info`` through this reporting facade. It is
# the same object the runtime extension loader defines, not a second wrapper.
build_info = _import_native_build_info()
