from __future__ import annotations

import platform
import sys
from typing import Any

import drjit as dr


def _optional_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def benchmark_environment_report() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "drjit": str(getattr(dr, "__version__", "unknown")),
        "rayd": _optional_version("rayd"),
        "cuda_backend": "cuda" if hasattr(dr, "cuda") else "unavailable",
    }
