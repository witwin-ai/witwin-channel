from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any


DEPLOYMENT_ABI = "witwin.channel_native.deployment.v1"
PIPELINE_CACHE_ABI = "witwin.channel_native.pipeline-cache.v1"
DECLARED_SM_ARCHITECTURES = (75, 80, 86, 89, 120)
VERIFIED_SM_ARCHITECTURES = (120,)
PTX_FORWARD_COMPATIBILITY_SM = 120
SM120_EVIDENCE = (
    "docs/dev/baselines/0892d855b27ee851521a181f5158b0bf41091eda/"
    "static/environment.json"
)
PIPELINE_CACHE_IMPLEMENTED = False
WHEEL_SMOKE_VERIFIED = False


def _package_version() -> str:
    try:
        return importlib.metadata.version("witwin-channel-native")
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def sm_support(sm: int) -> dict[str, Any]:
    sm = int(sm)
    declared = sm in DECLARED_SM_ARCHITECTURES
    verified = sm in VERIFIED_SM_ARCHITECTURES
    return {
        "sm": sm,
        "declared_supported": declared,
        "runtime_verified": verified,
        "status": (
            "runtime_verified"
            if verified
            else "declared_unverified"
            if declared
            else "not_declared"
        ),
        "evidence": [SM120_EVIDENCE] if verified else [],
        "mode": (
            "sass+ptx"
            if sm == PTX_FORWARD_COMPATIBILITY_SM
            else "sass"
            if sm in DECLARED_SM_ARCHITECTURES
            else "not_available"
        ),
    }


def _import_torch() -> ModuleType:
    import torch

    return torch


def _torch_runtime_diagnostics(torch: ModuleType) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
    }
    if cuda_available:
        major, minor = torch.cuda.get_device_capability(0)
        sm = major * 10 + minor
        support = sm_support(sm)
        info["device"] = {
            "name": torch.cuda.get_device_name(0),
            "sm": sm,
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            **support,
        }
        info["sm_matrix_status"] = support["status"]
    return info


def _import_native_build_info() -> Callable[[], dict[str, object]]:
    from .runtime.extension import build_info

    return build_info


def _import_error(component: str, exc: ImportError | OSError) -> str:
    return f"{component} import failed ({type(exc).__name__}): {exc}"


def runtime_diagnostics() -> dict[str, Any]:
    """Return import-safe package, CUDA runtime, build, and SM declarations."""

    diagnostics: dict[str, Any] = {
        "deployment_abi": DEPLOYMENT_ABI,
        "package_version": _package_version(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "declared_sm_architectures": list(DECLARED_SM_ARCHITECTURES),
        "sm_matrix_status": "declared_unverified",
        "ptx_forward_compatibility_sm": PTX_FORWARD_COMPATIBILITY_SM,
        "pipeline_cache": {
            "implemented": PIPELINE_CACHE_IMPLEMENTED,
            "key_abi": PIPELINE_CACHE_ABI,
        },
        "wheel_smoke": {
            "verified": WHEEL_SMOKE_VERIFIED,
            "status": "not_run",
            "evidence": None,
        },
        "errors": [],
    }
    try:
        torch = _import_torch()
    except (ImportError, OSError) as exc:
        diagnostics["errors"].append(_import_error("PyTorch", exc))
    else:
        diagnostics.update(_torch_runtime_diagnostics(torch))

    try:
        build_info = _import_native_build_info()
        native_build = build_info()
    except (ImportError, OSError) as exc:
        diagnostics["errors"].append(
            "native extension import failed; install a matching Channel Native "
            "wheel or configure the explicit developer extension override; "
            f"reason ({type(exc).__name__}): {exc}"
        )
    else:
        diagnostics["native_build"] = native_build
    return diagnostics


def require_supported_runtime() -> dict[str, Any]:
    """Require CUDA and a declared build architecture, not runtime verification."""
    diagnostics = runtime_diagnostics()
    errors = list(diagnostics["errors"])
    if not diagnostics.get("cuda_available", False):
        errors.append("CUDA is unavailable; Channel Native has no CPU/ROCm backend")
    device = diagnostics.get("device")
    if isinstance(device, dict) and not device.get("declared_supported", False):
        errors.append(
            f"GPU SM {device.get('sm')} is outside the declared build SM values "
            f"{list(DECLARED_SM_ARCHITECTURES)}"
        )
    if errors:
        raise RuntimeError(
            "Channel Native runtime requirements failed: " + "; ".join(errors)
        )
    return diagnostics


def pipeline_cache_key(
    *,
    geometry_version: int,
    material_version: int,
    assignment_version: int,
    frequency_hz: float,
    solver: str,
    build: dict[str, Any] | None = None,
) -> str:
    """Return a cache-key ABI digest; no pipeline cache is implemented yet."""

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
