# Copyright Xingyu Chen.
# Runtime ownership for the compiled Channel extension and symbols.

"""Runtime ownership for the compiled Channel extension and symbols."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field as dataclass_field
import enum
from functools import lru_cache, wraps
import functools
import hashlib
import importlib
from importlib import machinery, metadata, resources, util
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
import weakref
from typing import Any, ParamSpec, Protocol, TypeVar, cast

import torch


# --- Supported PyTorch private runtime APIs -----------------------------


class _Interpreter(Protocol):
    def key(self) -> object: ...


def is_transform_wrapped_tensor(value: torch.Tensor) -> bool:
    """Return whether ``value`` still carries a functorch transform wrapper."""

    functorch = torch._C._functorch
    return bool(
        functorch.is_functorch_wrapped_tensor(value)
        or functorch.is_gradtrackingtensor(value)
    )


def transform_level(value: torch.Tensor) -> int:
    """Return the functorch transform level, or ``-1`` for a plain tensor."""

    return int(torch._C._functorch.maybe_get_level(value))


def interpreter_stack() -> tuple[_Interpreter, ...]:
    """Return the active functorch interpreter stack in outer-to-inner order."""

    stack = torch._C._functorch.get_interpreter_stack()
    return () if stack is None else tuple(stack)


def is_jvp_transform(interpreter: _Interpreter) -> bool:
    """Return whether an interpreter stack entry is a JVP transform."""

    return bool(interpreter.key() == torch._C._functorch.TransformType.Jvp)


def unwrap_transform_tensor(value: torch.Tensor) -> torch.Tensor:
    """Remove one functorch wrapper without copying tensor storage."""

    return torch._C._functorch.get_unwrapped(value)


def disable_functorch() -> AbstractContextManager[object]:
    """Disable functorch dispatch inside native/custom-AD bridge code."""

    return cast(AbstractContextManager[object], torch._C._DisableFuncTorch())


def uses_cxx11_abi() -> bool:
    """Return the CXX11 ABI flag compiled into the active PyTorch runtime."""

    return bool(torch._C._GLIBCXX_USE_CXX11_ABI)


# --- Shared tensor validation contracts for native facades and typed rows


def validate_cuda_tensor(
    name: str, tensor: torch.Tensor, *, dtype: torch.dtype, ndim: int,
    trailing_shape: tuple[int, ...] = (), require_contiguous: bool = True,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if trailing_shape and tuple(tensor.shape[-len(trailing_shape) :]) != trailing_shape:
        raise ValueError(f"{name} must end with shape {trailing_shape}")
    if require_contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def require_tensor(
    name: str, value: object, *, dtype: torch.dtype, shape: tuple[int, ...] | None = None,
    ndim: int | None = None, device: torch.device | None = None, cuda: bool = False,
    contiguous: bool = False, dtype_error: type[Exception] = ValueError,
) -> torch.Tensor:
    """Validate one declared tensor field of a typed row or capacity contract.

    This is the single owner of that check. The row contracts, the capacity
    contracts, and the consumer contracts each carried their own copy; the only
    behaviour that ever differed between them is which exception a dtype
    mismatch raises, which ``dtype_error`` keeps caller-declared. The checks run
    in the order the copies used, so every rejected input still fails on the
    same clause with the same message.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if cuda and not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.dtype != dtype:
        raise dtype_error(f"{name} must use {dtype}, got {value.dtype}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {value.ndim}")
    if contiguous and not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    return value


# --- Load and validate the compiled Channel extension -------------------


CHANNEL_ABI_VERSION = 1

_PACKAGE_MODULE = "witwin.channel._channel"
_DEVELOPER_ENABLE_ENV = "WITWIN_CHANNEL_DEVELOPER_OVERRIDE"
_DEVELOPER_PATH_ENV = "WITWIN_CHANNEL_EXTENSION_PATH"
_DEVELOPER_FINGERPRINT_ENV = "WITWIN_CHANNEL_EXPECTED_FINGERPRINT"
_REQUIRED_SYMBOLS = ("build_info",)
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_DEFAULT_BUILD_INFO: dict[str, bool | str] = {
    "backend": "channel",
    "uses_dr_jit": False,
    "uses_rayd_native": False,
    "rayd_integration": "unavailable",
    "uses_path_native": False,
    "cuda_available": False,
    "optix_available": False,
}

_FINGERPRINT_FIELDS = (
    "build_type",
    "channel_abi_version",
    "channel_git_dirty",
    "channel_git_sha",
    "compiler",
    "cuda_architectures",
    "cuda_compiler_version",
    "cuda_version",
    "cxx_abi",
    "rayd_dirty",
    "rayd_commit",
    "rayd_integration_abi_sha256",
    "rayd_integration_abi_kind",
    "rayd_integration_abi_path",
    "rayd_repository_url",
    "rayd_source_kind",
    "rayd_source_manifest_sha256",
    "torch_version",
)


class ExtensionLoadError(ImportError):
    """The native extension could not be selected or loaded safely."""


class ExtensionSymbolError(ExtensionLoadError):
    """The native extension does not expose its required bootstrap API."""


class ExtensionABIError(ExtensionLoadError):
    """The native extension identity does not match this Python package."""


def _extension_origin(module: object) -> Path:
    raw_origin = getattr(module, "__file__", None)
    if not isinstance(raw_origin, str) or not raw_origin:
        spec = getattr(module, "__spec__", None)
        raw_origin = getattr(spec, "origin", None)
    if not isinstance(raw_origin, str) or not raw_origin:
        raise ExtensionLoadError("_channel does not report a file origin")
    try:
        return Path(raw_origin).resolve(strict=True)
    except OSError as exc:
        raise ExtensionLoadError(
            f"_channel reports an invalid file origin: {raw_origin!r}"
        ) from exc


def _installed_distribution_package_dir() -> Path | None:
    try:
        distribution = metadata.distribution("witwin-channel")
    except metadata.PackageNotFoundError:
        return None
    located = distribution.locate_file("witwin/channel")
    # A zip-backed distribution returns a zipfile.Path, which is not a filesystem
    # path and cannot hold the loaded extension, so it grants no extra origin.
    if not isinstance(located, os.PathLike):
        return None
    package_dir = Path(located)
    try:
        return package_dir.resolve(strict=True)
    except OSError:
        return None


def _assert_packaged_origin(module: object) -> None:
    origin = _extension_origin(module)
    package_dir = Path(__file__).resolve().parent
    installed_dir = _installed_distribution_package_dir()
    allowed_dirs = (package_dir,) if installed_dir is None else (package_dir, installed_dir)
    if not any(origin.is_relative_to(directory) for directory in allowed_dirs):
        raise ExtensionLoadError(
            f"refusing _channel resolved outside witwin.channel: {origin}"
        )


def _load_rayd_lock() -> dict[str, Any]:
    packaged = resources.files(__package__).joinpath("rayd.lock.json")
    if packaged.is_file():
        raw = packaged.read_text(encoding="utf-8")
    else:
        repository_lock = (
            Path(__file__).resolve().parents[2] / "dependencies" / "rayd.lock.json"
        )
        try:
            raw = repository_lock.read_text(encoding="utf-8")
        except OSError as exc:
            raise ExtensionABIError(
                "the packaged RayD identity lock is missing"
            ) from exc

    try:
        lock = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ExtensionABIError("the RayD identity lock is not valid JSON") from exc
    if not isinstance(lock, dict):
        raise ExtensionABIError("the RayD identity lock must contain an object")
    return lock


def _read_expected_fingerprint(raw: str, *, source: str) -> str:
    fingerprint = raw.strip()
    if _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise ExtensionABIError(f"{source} must contain one SHA-256 digest")
    return fingerprint


def _packaged_expected_fingerprint() -> str:
    sidecar = resources.files(__package__).joinpath("_channel.build-fingerprint")
    if not sidecar.is_file():
        installed_dir = _installed_distribution_package_dir()
        if installed_dir is not None:
            installed = installed_dir / "_channel.build-fingerprint"
            if installed.is_file():
                sidecar = installed
        if not sidecar.is_file():
            raise ExtensionABIError("the packaged build fingerprint is missing")
    try:
        raw = sidecar.read_text(encoding="ascii")
    except OSError as exc:
        raise ExtensionABIError("the packaged build fingerprint is unreadable") from exc
    return _read_expected_fingerprint(raw, source="the packaged build fingerprint")


def _require_value_type(
    info: Mapping[str, object], name: str, expected_type: type[object],
) -> object:
    if name not in info:
        raise ExtensionABIError(f"_channel.build_info() is missing {name!r}")
    value = info[name]
    if type(value) is not expected_type:
        raise ExtensionABIError(
            f"_channel.build_info()[{name!r}] must be {expected_type.__name__}"
        )
    return value


def _runtime_identity() -> tuple[str, str, str]:
    import torch

    torch_version = str(torch.__version__).split("+", maxsplit=1)[0]
    cuda_version = str(torch.version.cuda or "")
    if os.name == "nt":
        cxx_abi = "msvc"
    else:
        uses_cxx11 = uses_cxx11_abi()
        cxx_abi = "cxx11" if uses_cxx11 else "pre-cxx11"
    return torch_version, cuda_version, cxx_abi


def torch_install_command(torch_version: str, cuda_version: str) -> str:
    """Return the pip command that installs one CUDA build of PyTorch."""

    tag = "cu" + cuda_version.replace(".", "")
    return (
        f"pip install torch=={torch_version} "
        f"--index-url https://download.pytorch.org/whl/{tag}"
    )


def cpu_only_torch_message(torch_version: str, *, cuda_version: str = "12.8") -> str:
    """Explain that the active PyTorch has no CUDA support and how to replace it."""

    base = torch_version.split("+", maxsplit=1)[0]
    return (
        f"the installed PyTorch {torch_version} is a CPU-only build; Channel needs a "
        f"CUDA build of PyTorch. Install it with: {torch_install_command(base, cuda_version)}"
    )


def _runtime_mismatch_message(
    info: Mapping[str, object], running: Mapping[str, str], fields: list[str],
) -> str:
    built_torch = str(info["torch_version"])
    built_cuda = str(info["cuda_version"])
    if "cuda_version" in fields and not running["cuda_version"]:
        return "_channel cannot load: " + cpu_only_torch_message(
            running["torch_version"], cuda_version=built_cuda
        )
    details = ", ".join(
        f"{name} built for {info[name]!s}, running {running[name]}" for name in fields
    )
    return (
        "_channel does not match the active Torch runtime "
        f"({details}). This Channel build needs PyTorch {built_torch} with CUDA "
        f"{built_cuda}: {torch_install_command(built_torch, built_cuda)}"
    )


def _expected_fingerprint(info: Mapping[str, object]) -> str:
    payload = {name: info[name] for name in _FINGERPRINT_FIELDS}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _locked_rayd_source_manifest(info: Mapping[str, object], lock: Mapping[str, object]) -> object:
    source_manifest = str(info["rayd_source_manifest_sha256"])
    if _SHA256_PATTERN.fullmatch(source_manifest) is None:
        raise ExtensionABIError("RayD source manifest must be a SHA-256 digest")
    if info["rayd_source_kind"] not in {"git-checkout", "python-package"}:
        raise ExtensionABIError("RayD source kind is not recognized")
    source_bundle = lock.get("source_bundle")
    if not isinstance(source_bundle, Mapping):
        raise ExtensionABIError("the RayD identity lock has no source bundle object")
    return source_bundle.get("manifest_sha256")


def _validate_build_info(raw_info: object) -> dict[str, object]:
    if not isinstance(raw_info, Mapping):
        raise ExtensionABIError("_channel.build_info() must return a mapping")

    info = dict(raw_info)
    string_fields = (
        "backend",
        "rayd_integration",
        "channel_git_sha",
        "compiler",
        "cuda_compiler_version",
        "cuda_version",
        "cxx_abi",
        "rayd_commit",
        "rayd_integration_abi_sha256",
        "rayd_integration_abi_kind",
        "rayd_integration_abi_path",
        "rayd_repository_url",
        "rayd_source_kind",
        "rayd_source_manifest_sha256",
        "torch_version",
        "build_type",
        "build_fingerprint",
    )
    boolean_fields = (
        "uses_dr_jit",
        "uses_rayd_native",
        "uses_path_native",
        "cuda_available",
        "optix_available",
        "channel_git_dirty",
        "rayd_dirty",
    )
    for name in string_fields:
        _require_value_type(info, name, str)
    for name in boolean_fields:
        _require_value_type(info, name, bool)
    abi_version = _require_value_type(info, "channel_abi_version", int)
    architectures = cast(
        list[object], _require_value_type(info, "cuda_architectures", list)
    )

    if info["backend"] != "channel" or info["uses_dr_jit"] is not False:
        raise ExtensionABIError("_channel reports an unexpected backend")
    if abi_version != CHANNEL_ABI_VERSION:
        raise ExtensionABIError(
            "channel ABI mismatch: "
            f"expected {CHANNEL_ABI_VERSION}, got {abi_version}"
        )
    if not architectures or not all(
        isinstance(architecture, str) and architecture for architecture in architectures
    ):
        raise ExtensionABIError(
            "_channel.build_info()['cuda_architectures'] must be a "
            "non-empty list of strings"
        )

    channel_sha = str(info["channel_git_sha"])
    if channel_sha != "unknown" and _SHA_PATTERN.fullmatch(channel_sha) is None:
        raise ExtensionABIError(
            "channel Git SHA must be 40 lowercase hex digits"
        )
    rayd_sha = str(info["rayd_commit"])
    if _SHA_PATTERN.fullmatch(rayd_sha) is None:
        raise ExtensionABIError("RayD Git SHA must be 40 lowercase hex digits")
    rayd_abi = str(info["rayd_integration_abi_sha256"])
    if _SHA256_PATTERN.fullmatch(rayd_abi) is None:
        raise ExtensionABIError("RayD integration ABI must be a SHA-256 digest")
    lock = _load_rayd_lock()
    integration_abi = lock.get("integration_abi")
    if not isinstance(integration_abi, Mapping):
        raise ExtensionABIError("the RayD identity lock has no integration ABI object")
    locked_values = {
        "rayd_repository_url": lock.get("repository_url"),
        "rayd_commit": lock.get("commit"),
        "rayd_integration_abi_kind": integration_abi.get("kind"),
        "rayd_integration_abi_path": integration_abi.get("entrypoint"),
        "rayd_integration_abi_sha256": integration_abi.get("sha256"),
        "rayd_source_manifest_sha256": _locked_rayd_source_manifest(info, lock),
    }
    mismatched_lock_fields = [
        name for name, expected in locked_values.items() if info[name] != expected
    ]
    if mismatched_lock_fields:
        raise ExtensionABIError(
            "_channel does not match the RayD identity lock: "
            + ", ".join(mismatched_lock_fields)
        )

    torch_version, cuda_version, cxx_abi = _runtime_identity()
    runtime_values = {
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "cxx_abi": cxx_abi,
    }
    mismatched_runtime_fields = [
        name for name, expected in runtime_values.items() if info[name] != expected
    ]
    if mismatched_runtime_fields:
        raise ExtensionABIError(
            _runtime_mismatch_message(info, runtime_values, mismatched_runtime_fields)
        )

    fingerprint = str(info["build_fingerprint"])
    if _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise ExtensionABIError("build fingerprint must be a SHA-256 digest")
    if fingerprint != _expected_fingerprint(info):
        raise ExtensionABIError("_channel build fingerprint is invalid")
    return info


def _validate_extension(
    module: object, *, packaged: bool, expected_fingerprint: str | None = None,
) -> object:
    if packaged:
        _assert_packaged_origin(module)
    missing = [
        name for name in _REQUIRED_SYMBOLS if not callable(getattr(module, name, None))
    ]
    if missing:
        raise ExtensionSymbolError(
            "_channel is missing required symbols: " + ", ".join(missing)
        )
    info = _validate_build_info(cast(Any, module).build_info())
    if (
        expected_fingerprint is not None
        and info["build_fingerprint"] != expected_fingerprint
    ):
        raise ExtensionABIError(
            "developer extension does not match the expected build fingerprint"
        )
    return module


def _import_packaged_module() -> object:
    try:
        return importlib.import_module("._channel", package="witwin.channel")
    except ImportError as exc:
        if torch.version.cuda is None:
            raise ExtensionLoadError(
                "_channel cannot load: " + cpu_only_torch_message(str(torch.__version__))
            ) from exc
        raise


@lru_cache(maxsize=1)
def _load_packaged_extension() -> object:
    module = _import_packaged_module()
    return _validate_extension(
        module,
        packaged=True,
        expected_fingerprint=_packaged_expected_fingerprint(),
    )


def _developer_extension_config() -> tuple[Path, str] | None:
    enabled = os.environ.get(_DEVELOPER_ENABLE_ENV)
    raw_path = os.environ.get(_DEVELOPER_PATH_ENV)
    fingerprint = os.environ.get(_DEVELOPER_FINGERPRINT_ENV)
    if enabled is None and raw_path is None and fingerprint is None:
        return None
    if enabled != "1" or not raw_path or not fingerprint:
        raise ExtensionLoadError(
            f"developer extension loading requires {_DEVELOPER_ENABLE_ENV}=1 and "
            f"an absolute {_DEVELOPER_PATH_ENV} plus {_DEVELOPER_FINGERPRINT_ENV}"
        )
    if _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise ExtensionLoadError(
            f"{_DEVELOPER_FINGERPRINT_ENV} must be a SHA-256 digest"
        )
    path = Path(raw_path)
    if not path.is_absolute():
        raise ExtensionLoadError(f"{_DEVELOPER_PATH_ENV} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ExtensionLoadError(f"developer extension does not exist: {path}") from exc
    if not resolved.is_file() or not any(
        resolved.name.endswith(suffix) for suffix in machinery.EXTENSION_SUFFIXES
    ):
        raise ExtensionLoadError(
            f"developer extension path is not a Python extension module: {resolved}"
        )
    return resolved, fingerprint


def _load_extension_file(path: Path) -> ModuleType:
    loader = machinery.ExtensionFileLoader(_PACKAGE_MODULE, str(path))
    spec = util.spec_from_loader(_PACKAGE_MODULE, loader, origin=str(path))
    if spec is None:
        raise ExtensionLoadError(f"cannot create an import spec for {path}")
    module = util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_developer_extension(path: str, expected_fingerprint: str) -> object:
    module = _load_extension_file(Path(path))
    origin = _extension_origin(module)
    if origin != Path(path):
        raise ExtensionLoadError(
            f"developer extension origin mismatch: expected {path}, got {origin}"
        )
    return _validate_extension(
        module, packaged=False, expected_fingerprint=expected_fingerprint
    )


@lru_cache(maxsize=1)
def _load_native_extension() -> object:
    """Load the validated extension, refusing implicit global modules."""

    package_spec = util.find_spec(_PACKAGE_MODULE)
    if package_spec is not None:
        return _load_packaged_extension()

    developer_config = _developer_extension_config()
    if developer_config is not None:
        developer_path, expected_fingerprint = developer_config
        return _load_developer_extension(str(developer_path), expected_fingerprint)

    raise ExtensionLoadError(
        "witwin.channel._channel is not installed; a development "
        f"build requires {_DEVELOPER_ENABLE_ENV}=1, {_DEVELOPER_PATH_ENV}, and "
        f"{_DEVELOPER_FINGERPRINT_ENV}"
    )


@lru_cache(maxsize=1)
def _validated_native_build_info() -> dict[str, object]:
    return _validate_build_info(cast(Any, native_extension()).build_info())


def build_info() -> dict[str, object]:
    """Return validated native build and capability metadata."""

    info: dict[str, object] = dict(_DEFAULT_BUILD_INFO)
    info.update(_validated_native_build_info())
    return info


def _clear_loader_caches() -> None:
    """Reset process-local loader state for isolated contract tests."""

    _load_packaged_extension.cache_clear()
    _load_developer_extension.cache_clear()
    _load_native_extension.cache_clear()
    _validated_native_build_info.cache_clear()
    require_device_supported.cache_clear()


# --- GPU, driver and trace backend checks ------------------------------

TRACE_BACKEND_ENV = "WITWIN_CHANNEL_TRACE_BACKEND"
TRACE_BACKENDS = ("auto", "optix", "cuda")
SKIP_DEVICE_CHECK_ENV = "WITWIN_CHANNEL_SKIP_DEVICE_CHECK"

# Channel and RayD device code uses Volta warp intrinsics; nothing below sm_70 runs it.
MIN_COMPUTE_CAPABILITY = 70
# Jetson and embedded architectures run only their own cubin, and their cubins run
# nowhere else (the same rule RayD's device check applies).
EXACT_CUBIN_COMPUTE_CAPABILITIES = frozenset({72, 87, 101, 110})
# The first CUDA driver API version that loads cubins built by a CUDA 12 toolkit.
MIN_CUDA_DRIVER_VERSION = 12000

_SCENE_TRACE_BACKENDS: weakref.WeakSet[Any] = weakref.WeakSet()


class UnsupportedDeviceError(RuntimeError):
    """The active GPU or driver cannot run the installed Channel build."""


def requested_trace_backend() -> str:
    """Return the trace backend new scenes request: ``auto``, ``optix`` or ``cuda``."""

    raw = os.environ.get(TRACE_BACKEND_ENV)
    value = "auto" if raw is None or not raw.strip() else raw.strip().lower()
    if value not in TRACE_BACKENDS:
        raise ValueError(
            f"{TRACE_BACKEND_ENV} must be one of {', '.join(TRACE_BACKENDS)}; got {raw!r}"
        )
    return value


def record_scene_trace_backend(resource: object) -> None:
    """Remember a live native scene so diagnostics can report its trace backend."""

    _SCENE_TRACE_BACKENDS.add(resource)


def scene_trace_backends() -> list[dict[str, object]]:
    """Return the device and selected trace backend of every live compiled scene."""

    return [
        {"device_index": int(cast(Any, scene).device_index),
         "trace_backend": str(cast(Any, scene).trace_backend)}
        for scene in list(_SCENE_TRACE_BACKENDS)
    ]


def runtime_probe(device_index: int, *, probe_optix: bool = True) -> dict[str, object]:
    """Return the CUDA driver API version and, optionally, the OptiX probe for a device."""

    probe = cast(Any, required_symbol("runtime_probe"))(int(device_index), bool(probe_optix))
    return dict(probe)


def build_targets(architectures: list[object]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split ``build_info()["cuda_architectures"]`` into cubin and PTX SM numbers."""

    sass: set[int] = set()
    ptx: set[int] = set()
    for entry in architectures:
        match = re.fullmatch(r"(\d+)-(real|virtual)", str(entry))
        if match is None:
            raise ExtensionABIError(f"unrecognized CUDA architecture entry {entry!r}")
        (sass if match.group(2) == "real" else ptx).add(int(match.group(1)))
    return tuple(sorted(sass)), tuple(sorted(ptx))


def _cubin_runs(arch: int, sm: int) -> bool:
    if arch == sm:
        return True
    if arch in EXACT_CUBIN_COMPUTE_CAPABILITIES or sm in EXACT_CUBIN_COMPUTE_CAPABILITIES:
        return False
    return arch // 10 == sm // 10 and arch < sm


def device_code_mode(sm: int, sass: tuple[int, ...], ptx: tuple[int, ...]) -> str:
    """How a GPU runs this build: ``sass``, ``ptx-jit`` or ``unsupported``."""

    if sm < MIN_COMPUTE_CAPABILITY:
        return "unsupported"
    if any(_cubin_runs(arch, sm) for arch in sass):
        return "sass"
    if sm not in EXACT_CUBIN_COMPUTE_CAPABILITIES and any(arch <= sm for arch in ptx):
        return "ptx-jit"
    return "unsupported"


def _driver_version_text(version: int) -> str:
    return f"{version // 1000}.{(version % 1000) // 10}"


def device_support_problems(
    sm: int, architectures: list[object], driver_version: int, *, cuda_version: str,
) -> list[str]:
    """Return why a GPU and driver cannot run a build; empty when they can."""

    sass, ptx = build_targets(architectures)
    problems: list[str] = []
    mode = device_code_mode(sm, sass, ptx)
    if sm < MIN_COMPUTE_CAPABILITY:
        problems.append(
            f"Channel needs compute capability 7.0 (Volta) or newer; sm_{sm} cannot run "
            "it, from a wheel or from source"
        )
    elif mode == "unsupported":
        problems.append(
            f"the build has no CUDA code for sm_{sm} (cubins: "
            f"{', '.join(f'sm_{a}' for a in sass) or 'none'}; PTX: "
            f"{', '.join(f'compute_{a}' for a in ptx) or 'none'})"
        )
    if 0 < driver_version < MIN_CUDA_DRIVER_VERSION:
        problems.append(
            f"the NVIDIA driver supports CUDA {_driver_version_text(driver_version)}; "
            "Channel needs a driver for CUDA 12 or newer (R525+, R555+ for OptiX)"
        )
    elif mode == "ptx-jit" and cuda_version:
        major, _, minor = cuda_version.partition(".")
        needed = int(major) * 1000 + int(minor or 0) * 10
        if 0 < driver_version < needed:
            problems.append(
                f"sm_{sm} runs this build only by JIT-compiling its PTX, which needs a "
                f"driver for CUDA {cuda_version}; the driver supports CUDA "
                f"{_driver_version_text(driver_version)}"
            )
    return problems


def unsupported_device_message(
    *, device_index: int, name: str, sm: int, architectures: list[object], problems: list[str],
) -> str:
    """Format the error for a GPU or driver that cannot run the installed build."""

    lines = [
        f"Channel cannot run on CUDA device {device_index} '{name}' "
        f"(compute capability {sm // 10}.{sm % 10}, sm_{sm}):"
    ]
    lines += [f"  - {problem}" for problem in problems]
    lines.append(f"This build contains CUDA code for: {', '.join(map(str, architectures))}.")
    if sm >= MIN_COMPUTE_CAPABILITY:
        lines.append(
            "Fix: update the NVIDIA driver if it is named above; otherwise build Channel "
            f"from source for this GPU (CHANNEL_DEV_CUDA_ARCH={sm} scripts/dev_build.sh, or "
            f"CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES={sm}-real; see CONTRIBUTING.md). "
            "`python -m witwin.channel doctor` prints the full report."
        )
    lines.append(f"Set {SKIP_DEVICE_CHECK_ENV}=1 to skip this check.")
    return "\n".join(lines)


@lru_cache(maxsize=None)
def require_device_supported(device_index: int) -> None:
    """Fail before native scene work when a GPU or driver cannot run this build."""

    if os.environ.get(SKIP_DEVICE_CHECK_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        return
    if torch.version.cuda is None:
        raise UnsupportedDeviceError(cpu_only_torch_message(str(torch.__version__)))
    info = build_info()
    architectures = cast(list[object], info["cuda_architectures"])
    major, minor = torch.cuda.get_device_capability(device_index)
    sm = major * 10 + minor
    driver_version = int(cast(Any, runtime_probe(device_index, probe_optix=False)[
        "cuda_driver_version"
    ]))
    problems = device_support_problems(
        sm, architectures, driver_version, cuda_version=str(info["cuda_version"])
    )
    if problems:
        raise UnsupportedDeviceError(
            unsupported_device_message(
                device_index=device_index,
                name=torch.cuda.get_device_name(device_index),
                sm=sm,
                architectures=architectures,
                problems=problems,
            )
        )


# --- Validated access to Channel extension symbols ----------------------


class NativeSymbolError(RuntimeError):
    """A required Channel extension symbol is unavailable."""


def native_extension() -> object:
    """Return the process-cached, ABI-validated native extension."""

    return _load_native_extension()


def _required_symbol(extension: object, name: str) -> object:
    if extension is None or not hasattr(extension, name):
        raise NativeSymbolError(f"_channel.{name} CUDA kernel is required")
    return getattr(extension, name)


_native_symbols = sys.modules[__name__]


def _required_native_op(name: str) -> object:
    return _native_symbols._required_symbol(native_extension(), name)


def required_symbol(name: str) -> object:
    """Return a required symbol or raise :class:`NativeSymbolError`."""

    return _required_symbol(native_extension(), name)


def optional_symbol(name: str) -> object | None:
    """Return an optional symbol, or ``None`` when that symbol is absent."""

    extension = native_extension()
    if extension is None:
        return None
    return getattr(extension, name, None)


def has_symbol(name: str) -> bool:
    """Report whether the validated extension exposes ``name``."""

    extension = native_extension()
    return extension is not None and hasattr(extension, name)


# --- Typed native resource normalization at Python dispatch boundaries --


def _rayd_scene_resource(value: object) -> object:
    """Return a typed RayD scene resource; integer handles are forbidden."""

    if isinstance(value, int):
        raise TypeError("RayD scene operations require a typed scene resource")
    require = getattr(value, "require_resource", None)
    if callable(require):
        resource = require()
        if isinstance(resource, int):
            raise TypeError("RayD scene operations require a typed scene resource")
        return resource
    return value


# --- Native buffer constructors -----------------------------------------


def bdpt_zero_matrix(reference: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    out = cast(Callable[..., object], _required_native_op("bdpt_zero_matrix"))(reference, int(rows), int(cols))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.bdpt_zero_matrix must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != (int(rows), int(cols)):
        raise ValueError(
            "_channel.bdpt_zero_matrix returned an unexpected shape"
        )
    return out


def mc_pack_vec3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("x", x, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y", y, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z", z, dtype=torch.float32, ndim=1)
    if y.shape != x.shape or z.shape != x.shape:
        raise ValueError("x, y, and z must have the same shape")
    packed = cast(Callable[..., object], _required_native_op("mc_pack_vec3"))(x, y, z)
    if not isinstance(packed, torch.Tensor):
        raise TypeError("_channel.mc_pack_vec3 must return a tensor")
    validate_cuda_tensor(
        "packed", packed, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if packed.shape[0] != x.shape[0]:
        raise ValueError("_channel.mc_pack_vec3 returned an unexpected shape")
    return packed


def mc_receiver_grid_points(
    reference: torch.Tensor, *, origin: tuple[float, float, float],
    x_axis: tuple[float, float, float], y_axis: tuple[float, float, float], shape: tuple[int, int],
    spacing: tuple[float, float],
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    rows, cols = shape
    if rows < 0 or cols < 0:
        raise ValueError("shape entries must be non-negative")
    if spacing[0] <= 0.0 or spacing[1] <= 0.0:
        raise ValueError("spacing entries must be positive")
    points = cast(Callable[..., object], _required_native_op("mc_receiver_grid_points"))(
        reference,
        int(rows),
        int(cols),
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        float(x_axis[0]),
        float(x_axis[1]),
        float(x_axis[2]),
        float(y_axis[0]),
        float(y_axis[1]),
        float(y_axis[2]),
        float(spacing[0]),
        float(spacing[1]),
    )
    if not isinstance(points, torch.Tensor):
        raise TypeError("_channel.mc_receiver_grid_points must return a tensor")
    validate_cuda_tensor(
        "points", points, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if points.shape[0] != rows * cols:
        raise ValueError(
            "_channel.mc_receiver_grid_points returned an unexpected shape"
        )
    return points


# --- Shared autograd validation and transform contracts for native facades


def _ad_still_wrapped(value: torch.Tensor) -> bool:
    return is_transform_wrapped_tensor(value)


def _ad_raise_composed_transforms() -> None:
    # the AD contract: fail loudly instead of feeding the native
    # kernels an unwrapped tensor that has silently lost its transform
    # tracking (which would produce exact-zero tangents/gradients).
    raise NotImplementedError(
        "rayd_*_ad entry points support a single forward-mode transform"
        " level; composed functorch transforms (e.g. torch.func.grad over"
        " forward-mode jvp) are not supported by the native geometry kernels"
        " (first-order only)"
    )


def _ad_native_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    if transform_level(value) >= 0:
        # The tensor is functorch-wrapped. Unwrapping is only sound for a
        # single Jvp transform (torch.func.jvp); under nested transforms or
        # a Grad transform (e.g. torch.func.grad over forward-mode jvp, the
        # standard HVP recipe) unwrapping would silently sever the outer
        # transform and return exact zeros.
        stack = interpreter_stack()
        if len(stack) > 1 or any(
            not is_jvp_transform(entry) for entry in stack
        ):
            _ad_raise_composed_transforms()
    value = torch.autograd.forward_ad.unpack_dual(value).primal
    if _ad_still_wrapped(value):
        value = unwrap_transform_tensor(value)
    if _ad_still_wrapped(value):
        _ad_raise_composed_transforms()
    return value


def _ad_native_tangent_or_none(value: torch.Tensor | None) -> torch.Tensor | None:
    value = _ad_native_tensor(value)
    if value is None:
        return None
    try:
        # Efficient zero tangents (ZeroTensor) have no storage; treat them as
        # absent so the kernels take their tangent-free fast path.
        value.data_ptr()
    except RuntimeError:
        return None
    return value


def _ad_checked_tangent(
    name: str, tangent: torch.Tensor | None, primal_shape: tuple[int, ...],
) -> torch.Tensor | None:
    """Validate an unwrapped jvp tangent against its primal contract.

    Strided tangents are passed through unchanged: the native kernels consume
    explicit strides, so no Python-side layout copy or staging is needed.
    """

    if tangent is None:
        return None
    if tuple(tangent.shape) != tuple(primal_shape):
        raise ValueError(
            f"{name} must match its primal shape {tuple(primal_shape)};"
            f" got {tuple(tangent.shape)}"
        )
    if tangent.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if not tangent.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return tangent


def _ad_check_rows(name: str, tensor: torch.Tensor, rows: int) -> None:
    if tensor.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows to match the ray batch")


def _ad_check_active(active: torch.Tensor | None, rows: int) -> None:
    if active is None:
        return
    validate_cuda_tensor("active", active, dtype=torch.bool, ndim=1)
    if active.shape[0] not in (0, rows):
        raise ValueError("active must be empty or match the ray batch size")


def _ad_check_optional_grad(
    name: str, grad: torch.Tensor | None, allowed_shapes: tuple[tuple[int, ...], ...],
) -> None:
    # Cotangents from autograd may be strided views; the native kernels
    # consume explicit strides, so contiguity is deliberately not required.
    if grad is None:
        return
    if not isinstance(grad, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if grad.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if not grad.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tuple(grad.shape) not in allowed_shapes:
        raise ValueError(
            f"{name} must have shape in {allowed_shapes}; got {tuple(grad.shape)}"
        )


def _ad_check_tangent_vec3(name: str, tangent: torch.Tensor | None, rows: int | None) -> None:
    """Validate a facade-level jvp tangent.

    ``rows=None`` checks only the ``(V, 3)`` layout; the native entry point
    enforces that a vertex tangent matches the scene's global vertex table.
    Strided tangents are allowed: the native kernels consume explicit strides.
    """

    if tangent is None:
        return
    if not isinstance(tangent, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tangent.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if not tangent.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tangent.ndim != 2 or tangent.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if rows is not None and tangent.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows to match the ray batch")


def _ad_active_ctx(active: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if active is not None:
        return active
    return torch.empty((0,), device=like.device, dtype=torch.bool)


def _ad_frequency_value(frequency: torch.Tensor | float) -> float:
    """Read the scalar carrier frequency once per solve.

    A 0-d CUDA tensor frequency costs one device-to-host synchronization per
    read (documented material and frequency derivatives decision: one sync per solve, never per
    path); the native entry points keep a double scalar. The solve seams
    call this once and thread the float to every ``field_*_ad`` facade as
    ``frequency_value`` so no Function pays a second read; a
    facade called without it reads here exactly once.
    """

    if isinstance(frequency, torch.Tensor):
        if frequency.ndim != 0:
            raise ValueError("frequency must be a Python float or a 0-d tensor")
        return float(cast(torch.Tensor, _ad_native_tensor(frequency)).detach())
    return float(frequency)


def _ad_frequency_tangent(tangent: torch.Tensor | None) -> float:
    tangent = _ad_native_tangent_or_none(tangent)
    if tangent is None:
        return 0.0
    if tangent.ndim != 0:
        raise ValueError("frequency tangent must be a 0-d tensor")
    return float(tangent.detach())


def _ad_frequency_grad(
    grad_frequency: torch.Tensor, meta: tuple[torch.dtype, torch.device],
) -> torch.Tensor:
    dtype, device = meta
    return grad_frequency.to(dtype=dtype, device=device)[0]


def _ad_reject_fixed_inputs(
    op_name: str, needs_input_grad: tuple[bool, ...], fixed: tuple[tuple[int, str], ...],
) -> None:
    for index, name in fixed:
        if needs_input_grad[index]:
            raise NotImplementedError(
                f"{op_name} does not differentiate {name}: tx_power, the "
                "polarizations, mu_r, material ids and valid masks are constants "
                "of the fixed-topology derivative; detach this input"
            )


def _ad_reject_fixed_tangents(op_name: str, tangents: tuple[tuple[object, str], ...]) -> None:
    for tangent, name in tangents:
        if isinstance(tangent, torch.Tensor) and (
            _ad_native_tangent_or_none(tangent) is not None
        ):
            raise NotImplementedError(
                f"{op_name} does not differentiate {name}: tx_power, the "
                "polarizations, mu_r, material ids and valid masks are constants "
                "of the fixed-topology derivative; detach this input"
            )


def _ad_geometry_live(*values: object) -> bool:
    """True when any geometry input participates in AD (grad or tangent).

    Drives the AD need_grad_geometry plumbing and the conditional
    differentiability of path_length_m / delay_s: a materials-only graph
    keeps them detached exactly as in AD, so it never pays for geometry
    adjoints it did not request.
    """

    for value in values:
        if not isinstance(value, torch.Tensor):
            continue
        if value.requires_grad:
            return True
        if torch.autograd.forward_ad.unpack_dual(value).tangent is not None:
            return True
    return False


_participates_in_ad = _ad_geometry_live


def _frequency_participates_in_ad(frequency: float | torch.Tensor) -> bool:
    return _participates_in_ad(frequency)


def _ad_geometry_tangent(name: str, tangent: object, primal: torch.Tensor) -> torch.Tensor | None:
    """Unwrap and validate a geometry tangent against its primal tensor."""

    value = _ad_native_tangent_or_none(
        tangent if isinstance(tangent, torch.Tensor) else None
    )
    if value is None:
        return None
    if tuple(value.shape) != tuple(primal.shape):
        raise ValueError(
            f"{name} must match its primal shape {tuple(primal.shape)};"
            f" got {tuple(value.shape)}"
        )
    if value.dtype != primal.dtype:
        raise TypeError(f"{name} must match the primal dtype {primal.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


def _ad_first_order_only(backward: Callable[..., Any]) -> Callable[..., Any]:
    """The one backward decorator: first-order guard over ``once_differentiable``.

    first-order differentiation: Channel publishes first derivatives only. ``create_graph=True`` is
    precisely what leaves grad mode enabled while a backward runs, so the check
    below is an exact detector that fires before any native launch, names the
    owner, and produces no partial second-order result. Without it the first
    gradient comes back silently detached and the failure surfaces one step
    later as a generic Torch message that names Torch, not the owner that
    cannot answer.

    ``torch.autograd.function.once_differentiable`` is applied underneath, as
    defence in depth, from here rather than from every call site. It cannot
    replace the check: it runs the backward body inside ``torch.no_grad``, so
    the grad-mode signal is gone by the time the body executes, and it only
    fails when the detached gradient is later used.
    """

    once = torch.autograd.function.once_differentiable(backward)

    @functools.wraps(backward)
    def guarded(ctx: Any, *grad_outputs: Any) -> Any:
        if torch.is_grad_enabled():
            owner = f"{backward.__module__}.{backward.__qualname__}"
            raise NotImplementedError(
                f"{owner} is first-order only; Channel does not support "
                "higher-order AD (create_graph=True, grad-of-grad). "
                "capabilities().supports_higher_order_ad is False"
            )
        return once(ctx, *grad_outputs)

    return guarded


# --- Kernel metadata and AD launch accounting ---------------------------


@dataclass
class AdLaunchLedger:
    """Per-solve accounting of AD companion kernels.

    ``launches`` counts the native backward/jvp companion launches one full
    reverse pass (vjp) or forward-dual pass (jvp) performs for this solve:
    one per registered differentiable Function. ``tape_bytes`` sums the
    tensors the reverse pass retains via ``save_for_backward``; forward mode
    retains nothing past the solve, so jvp reports zero tape. One ledger
    shape for every solver (montecarlo.basic, deterministic, path).
    """

    launches: int = 0
    tape_bytes: int = 0

    def add(self, *saved: object) -> None:
        self.launches += 1
        for tensor in saved:
            if isinstance(tensor, torch.Tensor):
                self.tape_bytes += tensor.numel() * tensor.element_size()


ACCUMULATION_STRATEGIES = frozenset(
    {
        "none",
        "atomic_add",
        "cell_reduce",
        "compact_atomic_add",
        "sorted_segment_reduce",
        "shared_memory_private_reduce",
        "hybrid_tile_reduce",
    }
)

AD_STATUSES = frozenset({"none", "primal", "vjp", "jvp"})

REQUIRED_METADATA_FIELDS = (
    "primitive",
    "launch_count",
    "forward_launch_count",
    "backward_launch_count",
    "jvp_launch_count",
    "intermediate_bytes",
    "tape_bytes",
    "fused_stages",
    "accumulation_strategy",
    "scheduling_strategy",
    "registers_per_thread",
    "shared_memory_bytes",
    "occupancy_estimate",
    "spill_bytes",
    "rayd_native",
    "ad_status",
)


def make_metadata(
    *, primitive: str, forward_launch_count: int = 0, backward_launch_count: int = 0,
    jvp_launch_count: int = 0, intermediate_bytes: int = 0, tape_bytes: int = 0,
    fused_stages: int = 0, accumulation_strategy: str = "none", scheduling_strategy: str = "none",
    registers_per_thread: int = 0, shared_memory_bytes: int = 0, occupancy_estimate: float = 0.0,
    spill_bytes: int = 0, rayd_native: bool = False, ad_status: str = "none",
    forward_time_ms: float = 0.0, peak_memory_bytes: int = 0,
) -> dict[str, bool | float | int | str]:
    metadata: dict[str, bool | float | int | str] = {
        "primitive": primitive,
        "launch_count": forward_launch_count + backward_launch_count + jvp_launch_count,
        "forward_launch_count": forward_launch_count,
        "backward_launch_count": backward_launch_count,
        "jvp_launch_count": jvp_launch_count,
        "intermediate_bytes": intermediate_bytes,
        "tape_bytes": tape_bytes,
        "fused_stages": fused_stages,
        "accumulation_strategy": accumulation_strategy,
        "scheduling_strategy": scheduling_strategy,
        "registers_per_thread": registers_per_thread,
        "shared_memory_bytes": shared_memory_bytes,
        "occupancy_estimate": occupancy_estimate,
        "spill_bytes": spill_bytes,
        "rayd_native": rayd_native,
        "ad_status": ad_status,
        # Wall-clock (CUDA-synchronized) solve duration and the amount the
        # solve raised the process CUDA high-water mark. A jvp solve carries
        # its dual pass inside this forward time; a vjp solve cannot observe
        # its future backward, so reverse-pass time/memory budgets are pinned
        # by the tests/ad overhead gates instead of a metadata field.
        "forward_time_ms": float(forward_time_ms),
        "peak_memory_bytes": int(peak_memory_bytes),
    }
    validate_metadata(metadata)
    return metadata


def noop_metadata(*, accumulation_strategy: str = "none") -> dict[str, bool | float | int | str]:
    return make_metadata(
        primitive="noop_metadata",
        accumulation_strategy=accumulation_strategy,
        scheduling_strategy="none",
        ad_status="none",
    )


def validate_metadata(metadata: Mapping[str, object]) -> None:
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            raise ValueError(f"metadata missing required field: {field}")

    if not isinstance(metadata["primitive"], str) or not metadata["primitive"]:
        raise ValueError("metadata primitive must be a non-empty string")

    for field in (
        "launch_count",
        "forward_launch_count",
        "backward_launch_count",
        "jvp_launch_count",
        "intermediate_bytes",
        "tape_bytes",
        "fused_stages",
        "registers_per_thread",
        "shared_memory_bytes",
        "spill_bytes",
    ):
        value = metadata[field]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"metadata {field} must be a non-negative integer")

    expected_launch_count = (
        cast(int, metadata["forward_launch_count"])
        + cast(int, metadata["backward_launch_count"])
        + cast(int, metadata["jvp_launch_count"])
    )
    if metadata["launch_count"] != expected_launch_count:
        raise ValueError("metadata launch_count must equal forward+backward+jvp counts")

    if metadata["accumulation_strategy"] not in ACCUMULATION_STRATEGIES:
        raise ValueError("metadata accumulation_strategy is not recognized")

    if metadata["ad_status"] not in AD_STATUSES:
        raise ValueError("metadata ad_status is not recognized")

    if not isinstance(metadata["scheduling_strategy"], str):
        raise ValueError("metadata scheduling_strategy must be a string")

    occupancy = metadata["occupancy_estimate"]
    if not isinstance(occupancy, int | float) or occupancy < 0.0:
        raise ValueError("metadata occupancy_estimate must be non-negative")

    if not isinstance(metadata["rayd_native"], bool):
        raise ValueError("metadata rayd_native must be a boolean")

    for field in ("forward_time_ms", "peak_memory_bytes"):
        value = metadata.get(field, 0)
        if not isinstance(value, int | float) or value < 0:
            raise ValueError(f"metadata {field} must be non-negative")


# --- Pre-launch memory estimates and budgets ----------------------------


_MAX_SIGNED_BYTES = (1 << 63) - 1


class MemoryBudgetError(RuntimeError):
    """Raised before launch when a requested workload cannot fit its budget."""


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    persistent_bytes: int = 0
    temporary_bytes: int = 0
    output_bytes: int = 0
    tape_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "persistent_bytes",
            "temporary_bytes",
            "output_bytes",
            "tape_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def total_bytes(self) -> int:
        return _checked_sum(
            self.persistent_bytes,
            self.temporary_bytes,
            self.output_bytes,
            self.tape_bytes,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "persistent_bytes": self.persistent_bytes,
            "temporary_bytes": self.temporary_bytes,
            "output_bytes": self.output_bytes,
            "tape_bytes": self.tape_bytes,
            "total_bytes": self.total_bytes,
        }


def _checked_sum(*values: int) -> int:
    total = 0
    for value in values:
        total += int(value)
        if total > _MAX_SIGNED_BYTES:
            raise MemoryBudgetError(
                "memory estimate exceeds the supported signed 64-bit byte range"
            )
    return total


def checked_product(*values: int, label: str = "workload") -> int:
    product = 1
    for value in values:
        value = int(value)
        if value < 0:
            raise ValueError(f"{label} dimensions must be non-negative")
        if value and product > _MAX_SIGNED_BYTES // value:
            raise MemoryBudgetError(
                f"{label} memory estimate overflows the signed 64-bit byte range"
            )
        product *= value
    return product


def estimate_monte_carlo_memory(
    *, samples: int, transmitters: int, receivers: int, depth: int, bytes_per_path_state: int = 192,
    output_bytes_per_pair: int = 16, persistent_bytes: int = 0, tape_bytes: int = 0,
) -> MemoryEstimate:
    """Conservative, allocation-free estimate for MC/BDPT scale sweeps.

    The per-path default covers two complex3 states, geometry/PDF state, and
    compaction indices. Callers may override it with a measured solver value.
    """

    for name, value in {
        "samples": samples,
        "transmitters": transmitters,
        "receivers": receivers,
        "depth": depth,
        "bytes_per_path_state": bytes_per_path_state,
        "output_bytes_per_pair": output_bytes_per_pair,
        "persistent_bytes": persistent_bytes,
        "tape_bytes": tape_bytes,
    }.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative")

    active_depth = max(1, int(depth))
    temporary = checked_product(
        samples,
        transmitters,
        active_depth,
        bytes_per_path_state,
        label="Monte Carlo path state",
    )
    output = checked_product(
        transmitters,
        receivers,
        output_bytes_per_pair,
        label="Monte Carlo output",
    )
    return MemoryEstimate(
        persistent_bytes=int(persistent_bytes),
        temporary_bytes=temporary,
        output_bytes=output,
        tape_bytes=int(tape_bytes),
    )


def enforce_memory_budget(
    estimate: MemoryEstimate, *, budget_bytes: int, workload: str, headroom_bytes: int = 0,
) -> None:
    """Fail with an actionable error before any workload allocation occurs."""

    if budget_bytes < 0 or headroom_bytes < 0:
        raise ValueError("budget_bytes and headroom_bytes must be non-negative")
    required = _checked_sum(estimate.total_bytes, headroom_bytes)
    if required <= int(budget_bytes):
        return
    raise MemoryBudgetError(
        f"{workload} exceeds the GPU memory budget before launch: estimated "
        f"{estimate.total_bytes} bytes plus {headroom_bytes} bytes headroom "
        f"requires {required} bytes, budget is {int(budget_bytes)} bytes. "
        "Reduce samples, TX/RX count, depth, grid resolution, or exported paths."
    )


# --- Device-resident shared failure state and counts for capacity transactions


class CapacityFailureBit(enum.IntFlag):
    """Stable device failure bits recorded by shared solve transactions."""

    DIFFRACTION_STATE_OVERFLOW = 1 << 0
    DIFFRACTION_PATH_OVERFLOW = 1 << 1
    DIFFRACTION_PATH_CONTRACT_ERROR = 1 << 2
    PAIR_CAPACITY_OVERFLOW = 1 << 3
    PAIR_CONTRACT_ERROR = 1 << 4
    COUPLED_CANDIDATE_OVERFLOW = 1 << 5
    REFLECTION_CANDIDATE_OVERFLOW = 1 << 6
    SEGMENT_PENETRATION_FAILURE = 1 << 7


_CAPACITY_FAILURE_DESCRIPTIONS = {
    CapacityFailureBit.DIFFRACTION_STATE_OVERFLOW: "diffraction state capacity exceeded",
    CapacityFailureBit.DIFFRACTION_PATH_OVERFLOW: "diffraction path capacity exceeded",
    CapacityFailureBit.DIFFRACTION_PATH_CONTRACT_ERROR: "diffraction path rows violated their contract",
    CapacityFailureBit.PAIR_CAPACITY_OVERFLOW: "endpoint-pair capacity exceeded",
    CapacityFailureBit.PAIR_CONTRACT_ERROR: (
        "path rows referenced an endpoint, material or layer outside the compiled scene"
    ),
    CapacityFailureBit.COUPLED_CANDIDATE_OVERFLOW: "coupled candidate capacity exceeded",
    CapacityFailureBit.REFLECTION_CANDIDATE_OVERFLOW: "reflection candidate capacity exceeded",
    CapacityFailureBit.SEGMENT_PENETRATION_FAILURE: (
        "a transmission segment could not be traced (non-finite or out-of-range endpoint "
        "coordinates, or inconsistent penetration rows)"
    ),
}


def decode_capacity_failure_bits(bits: int) -> tuple[str, ...]:
    """Return the names of the set failure bits, low bit first; unknown bits are named by value."""

    if type(bits) is not int:
        raise TypeError("bits must be an int")
    names = [flag.name for flag in CapacityFailureBit if bits & int(flag) and flag.name]
    unknown = bits & ~int(sum(int(flag) for flag in CapacityFailureBit))
    if unknown:
        names.append(f"UNKNOWN_0x{unknown & 0xFFFFFFFF:08x}")
    return tuple(names)


class CapacityError(RuntimeError):
    """A solve's device work reported a failure; no partial result was returned.

    ``bits`` is the raw failure bitmask and ``bit_names`` its decoded names.
    """

    def __init__(self, bits: int) -> None:
        self.bits = int(bits)
        self.bit_names = decode_capacity_failure_bits(self.bits)
        details = "; ".join(
            f"{name}: {_CAPACITY_FAILURE_DESCRIPTIONS[CapacityFailureBit[name]]}"
            if name in CapacityFailureBit.__members__
            else name
            for name in self.bit_names
        )
        super().__init__(f"the solve failed and returned no result: {details}")


@dataclass(frozen=True, slots=True, eq=False)
class CapacityFailureState:
    """One solve-owned CUDA ``int32[2]`` record: failure bitmask, then dropped-row count.

    Construction is metadata-only. Native operations atomically update the
    record on the active CUDA stream; the host reads it once per solve.
    """

    record: torch.Tensor
    # The int32[1] failure bitmask view that native operations OR into.
    bits: torch.Tensor = dataclass_field(init=False)
    # The int32[1] count of rows dropped for exceeding their hit capacity.
    dropped_rows: torch.Tensor = dataclass_field(init=False)

    def __post_init__(self) -> None:
        validate_cuda_tensor("record", self.record, dtype=torch.int32, ndim=1)
        if self.record.shape != (2,):
            raise ValueError("record must have shape (2,)")
        object.__setattr__(self, "bits", self.record[0:1])
        object.__setattr__(self, "dropped_rows", self.record[1:2])

    @property
    def device(self) -> torch.device:
        return self.record.device


def require_host_count(name: str, value: object) -> int:
    """Validate a host-known non-negative count carried by a typed contract."""

    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class CapacityExecutionCounts:
    """Host capacity plus CUDA-resident actual diagnostic counts.

    Public metadata may expose ``candidate_capacity``. The actual candidate
    and guardrail counts remain device sidecars so result assembly never hides
    a device-to-host synchronization behind metadata construction.
    """

    candidate_capacity: int
    failure_state: CapacityFailureState
    device_candidate_count: torch.Tensor
    device_guardrail_count: torch.Tensor

    def __post_init__(self) -> None:
        require_host_count("candidate_capacity", self.candidate_capacity)
        candidate_count = require_tensor(
            "device_candidate_count",
            self.device_candidate_count,
            dtype=torch.int32,
            shape=(1,),
            cuda=True,
            contiguous=True,
        )
        require_capacity_failure_state(
            self.failure_state, device=candidate_count.device
        )
        require_tensor(
            "device_guardrail_count",
            self.device_guardrail_count,
            dtype=torch.int32,
            shape=(1,),
            device=candidate_count.device,
            cuda=True,
            contiguous=True,
        )

    @property
    def device(self) -> torch.device:
        return self.device_candidate_count.device


@dataclass(slots=True, eq=False)
class SolveCapacityTransaction:
    """Solve-scoped owner of one failure record and its single host observation.

    Solvers pass ``failure_state`` unchanged to every capacity intermediate.
    The record reaches the host exactly once: either inside an existing compact
    count copy (``observe``) or, where a solve has no compact boundary, through
    one read at result finalization (``terminal_check``). A non-zero bitmask
    raises :class:`CapacityError`, so no partial result is returned.
    """

    failure_state: CapacityFailureState
    _observed_dropped_rows: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.failure_state, CapacityFailureState):
            raise TypeError("failure_state must be a CapacityFailureState")

    @property
    def device(self) -> torch.device:
        return self.failure_state.device

    @property
    def observed(self) -> bool:
        return self._observed_dropped_rows is not None

    @property
    def dropped_rows(self) -> int:
        """Rows dropped for exceeding their hit capacity; available after observation."""

        if self._observed_dropped_rows is None:
            raise RuntimeError("capacity transaction has not been observed")
        return self._observed_dropped_rows

    def observe(self, bits: int, dropped_rows: int) -> None:
        """Record the host copy of the failure record and raise if any bit is set."""

        if self._observed_dropped_rows is not None:
            raise RuntimeError("capacity transaction already observed")
        if type(bits) is not int or type(dropped_rows) is not int:
            raise TypeError("observed capacity values must be ints")
        if dropped_rows < 0:
            raise ValueError("dropped_rows must be non-negative")
        self._observed_dropped_rows = dropped_rows
        if bits != 0:
            raise CapacityError(bits)

    def terminal_check(self) -> int:
        """Observe the record once if no compact boundary did, and return the dropped rows."""

        if self._observed_dropped_rows is None:
            bits, dropped_rows = capacity_failure_state_read(self.failure_state)
            self.observe(bits, dropped_rows)
        return self.dropped_rows


def create_capacity_failure_state(reference: torch.Tensor) -> CapacityFailureState:
    """Create a native-zeroed failure record on ``reference``'s CUDA device."""

    if not isinstance(reference, torch.Tensor):
        raise TypeError("reference must be a torch.Tensor")
    if not reference.is_cuda:
        raise ValueError("reference must be a CUDA tensor")
    record = cast(Any, required_symbol("capacity_failure_state_create"))(reference)
    if not isinstance(record, torch.Tensor):
        raise TypeError("native capacity failure state must be a tensor")
    return CapacityFailureState(record=record)


def create_solve_capacity_transaction(reference: torch.Tensor) -> SolveCapacityTransaction:
    """Create the one capacity transaction owned by a participating solve."""

    return SolveCapacityTransaction(
        failure_state=create_capacity_failure_state(reference)
    )


def require_capacity_failure_state(state: object, *, device: torch.device) -> CapacityFailureState:
    """Validate a required typed state without reading its device value."""

    if not isinstance(state, CapacityFailureState):
        raise TypeError("failure_state must be a CapacityFailureState")
    if state.device != device:
        raise ValueError("failure_state must share the input device")
    return state


def capacity_failure_state_read(failure_state: CapacityFailureState) -> tuple[int, int]:
    """Copy the failure record to the host once (one copy, one stream sync)."""

    if not isinstance(failure_state, CapacityFailureState):
        raise TypeError("failure_state must be a CapacityFailureState")
    bits, dropped_rows = cast(Any, required_symbol("capacity_failure_state_read"))(
        failure_state.record
    )
    return int(bits), int(dropped_rows)


# --- Stable CUDA profiler annotations for architecture evidence ---------


class CudaProfileRange(str, enum.Enum):
    """Closed set of semantic ranges consumed by performance evidence."""

    ENUMERATED_PENETRATION_DISCOVERY = (
        "witwin.channel:enumerated_penetration_discovery"
    )
    MONTECARLO_BASIC_PENETRATION_DISCOVERY = (
        "witwin.channel:montecarlo_basic_penetration_discovery"
    )
    DIFFRACTION_EXPORTER = "witwin.channel:diffraction_exporter"
    CAPACITY_STATUS = "witwin.channel:capacity_status"
    DIFFRACTION_PAIR_REDUCER = "witwin.channel:diffraction_pair_reducer"
    DIFFRACTION_TOPOLOGY_PACKING = (
        "witwin.channel:diffraction_topology_packing"
    )
    DIFFRACTION_TOTAL_STAGE = "witwin.channel:diffraction_total_stage"


class CudaProfileMark(str, enum.Enum):
    """Closed set of semantic point annotations consumed by the runner."""

    OPTIX_TRAVERSAL = "witwin.channel:optix_traversal"
    DIFFRACTION_EXPORTER_REQUEST = (
        "witwin.channel:diffraction_exporter_request"
    )


@contextmanager
def cuda_profile_range(name: CudaProfileRange) -> Iterator[None]:
    """Emit one balanced NVTX range without CUDA work or synchronization."""

    torch.cuda.nvtx.range_push(name.value)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


_P = ParamSpec("_P")
_R = TypeVar("_R")


def profiled_cuda_range(name: CudaProfileRange) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Wrap an operation owner in one balanced semantic NVTX range."""

    def decorate(operation: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(operation)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with cuda_profile_range(name):
                return operation(*args, **kwargs)

        return wrapped

    return decorate


def cuda_profile_mark(name: CudaProfileMark) -> None:
    """Emit one semantic NVTX point annotation without CUDA work."""

    torch.cuda.nvtx.mark(name.value)


__all__ = [
    "ACCUMULATION_STRATEGIES",
    "AD_STATUSES",
    "AdLaunchLedger",
    "CHANNEL_ABI_VERSION",
    "CapacityError",
    "CapacityExecutionCounts",
    "CapacityFailureBit",
    "CapacityFailureState",
    "CudaProfileMark",
    "CudaProfileRange",
    "ExtensionABIError",
    "ExtensionLoadError",
    "ExtensionSymbolError",
    "MemoryBudgetError",
    "MemoryEstimate",
    "NativeSymbolError",
    "REQUIRED_METADATA_FIELDS",
    "SKIP_DEVICE_CHECK_ENV",
    "SolveCapacityTransaction",
    "TRACE_BACKENDS",
    "TRACE_BACKEND_ENV",
    "UnsupportedDeviceError",
    "bdpt_zero_matrix",
    "build_info",
    "build_targets",
    "capacity_failure_state_read",
    "checked_product",
    "create_capacity_failure_state",
    "create_solve_capacity_transaction",
    "cuda_profile_mark",
    "cuda_profile_range",
    "cpu_only_torch_message",
    "decode_capacity_failure_bits",
    "device_code_mode",
    "device_support_problems",
    "disable_functorch",
    "enforce_memory_budget",
    "estimate_monte_carlo_memory",
    "has_symbol",
    "interpreter_stack",
    "is_jvp_transform",
    "is_transform_wrapped_tensor",
    "make_metadata",
    "mc_pack_vec3",
    "mc_receiver_grid_points",
    "native_extension",
    "noop_metadata",
    "optional_symbol",
    "profiled_cuda_range",
    "record_scene_trace_backend",
    "require_device_supported",
    "require_capacity_failure_state",
    "require_host_count",
    "require_tensor",
    "required_symbol",
    "requested_trace_backend",
    "runtime_probe",
    "scene_trace_backends",
    "torch_install_command",
    "transform_level",
    "unsupported_device_message",
    "unwrap_transform_tensor",
    "uses_cxx11_abi",
    "validate_cuda_tensor",
    "validate_metadata",
]
