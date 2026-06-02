"""Shared runtime and backend-report helpers for benchmark scripts."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

try:
    from ._paths import REPO_ROOT
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _paths import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import witwin.channel as channel
from witwin.channel import cuda_runtime_version, native_extension_available


def _jsonable_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable_copy(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_copy(item) for item in value]
    return value


def benchmark_environment_report() -> dict[str, Any]:
    try:
        runtime_version = int(cuda_runtime_version())
    except Exception:
        runtime_version = None
    return {
        "repo_root": str(REPO_ROOT.resolve()),
        "python_executable": sys.executable,
        "channel_module_file": str(Path(channel.__file__).resolve()),
        "channel_package_root": str(Path(channel.__file__).resolve().parent),
        "backend_variant": getattr(channel, "DEFAULT_VARIANT", "rayd"),
        "runtime_backend": "rayd",
        "native_extension_available": bool(native_extension_available()),
        "cuda_runtime_version": runtime_version,
    }


def assert_native_benchmark_support(
    *,
    benchmark_name: str,
    reflection_field_backend: str,
    diffraction_execution,
) -> None:
    requested_native_paths: list[str] = []
    if str(reflection_field_backend) == "native":
        requested_native_paths.append("reflection_field_backend=native")
    if str(diffraction_execution.suffix_backend) == "native":
        requested_native_paths.append("diffraction_execution.suffix_backend=native")
    if requested_native_paths and not native_extension_available():
        requested_text = ", ".join(requested_native_paths)
        raise RuntimeError(
            f"{benchmark_name} requested native benchmark backends ({requested_text}) "
            "but witwin.channel.native_extension_available() is False."
        )


def extract_monitor_runtime_backends(payload) -> dict[str, Any]:
    metadata = {}
    if isinstance(payload, Mapping):
        metadata = payload.get("metadata", {}) or {}
    else:
        metadata = getattr(payload, "metadata", {}) or {}
    runtime_backends = metadata.get("runtime_backends")
    if isinstance(runtime_backends, Mapping):
        return _jsonable_copy(runtime_backends)
    return {
        "reflection": _jsonable_copy(metadata.get("reflection_backend", {})),
        "diffraction": _jsonable_copy(metadata.get("diffraction_accumulation_backend", {})),
        "suffix": _jsonable_copy(metadata.get("reflection_suffix_backend", {})),
    }


def extract_monitor_performance_timing(payload) -> dict[str, Any]:
    metadata = {}
    if isinstance(payload, Mapping):
        metadata = payload.get("metadata", {}) or {}
    else:
        metadata = getattr(payload, "metadata", {}) or {}
    performance_timing = metadata.get("performance_timing")
    if isinstance(performance_timing, Mapping):
        return _jsonable_copy(performance_timing)
    return {}


__all__ = [
    "assert_native_benchmark_support",
    "benchmark_environment_report",
    "extract_monitor_performance_timing",
    "extract_monitor_runtime_backends",
]

