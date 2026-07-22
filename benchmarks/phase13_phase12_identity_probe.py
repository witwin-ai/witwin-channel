"""Runner-owned packaged-extension identity probe.

This probe performs no numerical measurement.  The parent runner retains its
stdout, independently hashes the configured extension bytes, and binds the
reported compiled identity to checkout and RayD lock state.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

from witwin.channel import build_info


SCHEMA = {
    "name": "witwin.channel.phase13-phase12-identity-probe",
    "version": 3,
}


def _driver_version() -> str:
    raw = torch.cuda.cudart().cudaDriverGetVersion()
    if isinstance(raw, tuple):
        if len(raw) != 2 or int(raw[0]) != 0:
            raise RuntimeError(f"cudaDriverGetVersion failed: {raw!r}")
        raw = raw[1]
    value = int(raw)
    major = value // 1000
    minor = (value % 1000) // 10
    return f"{major}.{minor}"


def _loaded_dependencies() -> list[str]:
    accepted = (
        "torch", "c10", "cuda", "cudart", "nvrtc", "nvjitlink", "nvptx",
        "vcruntime", "msvcp", "ucrtbase", "python",
    )
    paths: set[Path] = set()
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process = ctypes.windll.kernel32.GetCurrentProcess()
        modules = (wintypes.HMODULE * 4096)()
        needed = wintypes.DWORD()
        if not ctypes.windll.psapi.EnumProcessModules(
            process, modules, ctypes.sizeof(modules), ctypes.byref(needed)
        ):
            raise RuntimeError("EnumProcessModules failed")
        count = int(needed.value // ctypes.sizeof(wintypes.HMODULE))
        for module in modules[:count]:
            buffer = ctypes.create_unicode_buffer(32768)
            length = ctypes.windll.psapi.GetModuleFileNameExW(
                process, module, buffer, len(buffer)
            )
            if length:
                paths.add(Path(buffer.value).resolve(strict=True))
    else:
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
            raw = line.rsplit(maxsplit=1)[-1]
            path = Path(raw)
            if path.is_absolute() and path.exists():
                paths.add(path.resolve(strict=True))
    selected = [
        str(path)
        for path in paths
        if any(fragment in path.name.casefold() for fragment in accepted)
    ]
    if not selected:
        raise RuntimeError("no Torch/CUDA/runtime dependency modules were resolved")
    return sorted(selected)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("identity probe requires CUDA")
    info = build_info()
    device_index = int(torch.cuda.current_device())
    raw_uuid = getattr(torch.cuda.get_device_properties(device_index), "uuid", None)
    if raw_uuid is None:
        raise RuntimeError("CUDA device UUID is unavailable")
    device_uuid = str(raw_uuid).strip()
    if not device_uuid:
        raise RuntimeError("CUDA device UUID is empty")
    spec = importlib.util.find_spec("witwin.channel._channel_native")
    origin = None if spec is None else spec.origin
    if not isinstance(origin, str) or not origin:
        raise RuntimeError("packaged extension origin is unavailable")
    extension = Path(origin).resolve(strict=True)
    package = Path(__import__("witwin.channel", fromlist=["x"]).__file__).resolve().parent
    if not extension.is_relative_to(package):
        raise RuntimeError("identity probe refuses a non-packaged extension")
    payload = {
        "schema": SCHEMA,
        "build_info": info,
        "runtime": {
            "device_index": device_index,
            "device_uuid": device_uuid,
            "gpu_name": torch.cuda.get_device_name(device_index),
            "driver_version": _driver_version(),
            "python_version": sys.version.split()[0],
        },
        "extension_path": str(extension),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "loaded_dependencies": _loaded_dependencies(),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
