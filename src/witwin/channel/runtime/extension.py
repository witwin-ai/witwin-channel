"""Load and validate the compiled Channel Native extension."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import importlib
from importlib import machinery, resources, util
import json
import os
from pathlib import Path
import re
from types import ModuleType
from typing import Any


CHANNEL_NATIVE_ABI_VERSION = 1

_PACKAGE_MODULE = "witwin.channel._channel_native"
_DEVELOPER_ENABLE_ENV = "WITWIN_CHANNEL_NATIVE_DEVELOPER_OVERRIDE"
_DEVELOPER_PATH_ENV = "WITWIN_CHANNEL_NATIVE_EXTENSION_PATH"
_DEVELOPER_FINGERPRINT_ENV = "WITWIN_CHANNEL_NATIVE_EXPECTED_FINGERPRINT"
_REQUIRED_SYMBOLS = ("build_info",)
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_DEFAULT_BUILD_INFO: dict[str, bool | str] = {
    "backend": "channel-native",
    "uses_dr_jit": False,
    "uses_rayd_native": False,
    "rayd_integration": "unavailable",
    "uses_path_native": False,
    "cuda_available": False,
    "optix_available": False,
}

_FINGERPRINT_FIELDS = (
    "build_type",
    "channel_native_abi_version",
    "channel_native_git_dirty",
    "channel_native_git_sha",
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
        raise ExtensionLoadError("_channel_native does not report a file origin")
    try:
        return Path(raw_origin).resolve(strict=True)
    except OSError as exc:
        raise ExtensionLoadError(
            f"_channel_native reports an invalid file origin: {raw_origin!r}"
        ) from exc


def _assert_packaged_origin(module: object) -> None:
    origin = _extension_origin(module)
    package_dir = Path(__file__).resolve().parents[1]
    if not origin.is_relative_to(package_dir):
        raise ExtensionLoadError(
            f"refusing _channel_native resolved outside witwin.channel: {origin}"
        )


def _load_rayd_lock() -> dict[str, Any]:
    packaged = resources.files(__package__).joinpath("rayd.lock.json")
    if packaged.is_file():
        raw = packaged.read_text(encoding="utf-8")
    else:
        repository_lock = (
            Path(__file__).resolve().parents[4] / "dependencies" / "rayd.lock.json"
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
    sidecar = resources.files(__package__).joinpath("_channel_native.build-fingerprint")
    if not sidecar.is_file():
        raise ExtensionABIError("the packaged build fingerprint is missing")
    try:
        raw = sidecar.read_text(encoding="ascii")
    except OSError as exc:
        raise ExtensionABIError("the packaged build fingerprint is unreadable") from exc
    return _read_expected_fingerprint(raw, source="the packaged build fingerprint")


def _require_value_type(
    info: Mapping[str, object], name: str, expected_type: type[object]
) -> object:
    if name not in info:
        raise ExtensionABIError(f"_channel_native.build_info() is missing {name!r}")
    value = info[name]
    if type(value) is not expected_type:
        raise ExtensionABIError(
            f"_channel_native.build_info()[{name!r}] must be {expected_type.__name__}"
        )
    return value


def _runtime_identity() -> tuple[str, str, str]:
    import torch

    torch_version = str(torch.__version__).split("+", maxsplit=1)[0]
    cuda_version = str(torch.version.cuda or "")
    if os.name == "nt":
        cxx_abi = "msvc"
    else:
        from .torch_compat import uses_cxx11_abi

        uses_cxx11 = uses_cxx11_abi()
        cxx_abi = "cxx11" if uses_cxx11 else "pre-cxx11"
    return torch_version, cuda_version, cxx_abi


def _expected_fingerprint(info: Mapping[str, object]) -> str:
    payload = {name: info[name] for name in _FINGERPRINT_FIELDS}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _locked_rayd_source_manifest(
    info: Mapping[str, object], lock: Mapping[str, object]
) -> object:
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
        raise ExtensionABIError("_channel_native.build_info() must return a mapping")

    info = dict(raw_info)
    string_fields = (
        "backend",
        "rayd_integration",
        "channel_native_git_sha",
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
        "channel_native_git_dirty",
        "rayd_dirty",
    )
    for name in string_fields:
        _require_value_type(info, name, str)
    for name in boolean_fields:
        _require_value_type(info, name, bool)
    abi_version = _require_value_type(info, "channel_native_abi_version", int)
    architectures = _require_value_type(info, "cuda_architectures", list)

    if info["backend"] != "channel-native" or info["uses_dr_jit"] is not False:
        raise ExtensionABIError("_channel_native reports an unexpected backend")
    if abi_version != CHANNEL_NATIVE_ABI_VERSION:
        raise ExtensionABIError(
            "channel-native ABI mismatch: "
            f"expected {CHANNEL_NATIVE_ABI_VERSION}, got {abi_version}"
        )
    if not architectures or not all(
        isinstance(architecture, str) and architecture for architecture in architectures
    ):
        raise ExtensionABIError(
            "_channel_native.build_info()['cuda_architectures'] must be a "
            "non-empty list of strings"
        )

    channel_sha = str(info["channel_native_git_sha"])
    if channel_sha != "unknown" and _SHA_PATTERN.fullmatch(channel_sha) is None:
        raise ExtensionABIError(
            "channel-native Git SHA must be 40 lowercase hex digits"
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
        "rayd_integration_abi_path": integration_abi.get("path"),
        "rayd_integration_abi_sha256": integration_abi.get("sha256"),
        "rayd_source_manifest_sha256": _locked_rayd_source_manifest(info, lock),
    }
    mismatched_lock_fields = [
        name for name, expected in locked_values.items() if info[name] != expected
    ]
    if mismatched_lock_fields:
        raise ExtensionABIError(
            "_channel_native does not match the RayD identity lock: "
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
            "_channel_native does not match the active Torch runtime: "
            + ", ".join(mismatched_runtime_fields)
        )

    fingerprint = str(info["build_fingerprint"])
    if _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise ExtensionABIError("build fingerprint must be a SHA-256 digest")
    if fingerprint != _expected_fingerprint(info):
        raise ExtensionABIError("_channel_native build fingerprint is invalid")
    return info


def _validate_extension(
    module: object, *, packaged: bool, expected_fingerprint: str | None = None
) -> object:
    if packaged:
        _assert_packaged_origin(module)
    missing = [
        name for name in _REQUIRED_SYMBOLS if not callable(getattr(module, name, None))
    ]
    if missing:
        raise ExtensionSymbolError(
            "_channel_native is missing required symbols: " + ", ".join(missing)
        )
    info = _validate_build_info(module.build_info())
    if (
        expected_fingerprint is not None
        and info["build_fingerprint"] != expected_fingerprint
    ):
        raise ExtensionABIError(
            "developer extension does not match the expected build fingerprint"
        )
    return module


@lru_cache(maxsize=1)
def _load_packaged_extension() -> object:
    module = importlib.import_module(
        "._channel_native", package="witwin.channel"
    )
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
        "witwin.channel._channel_native is not installed; a development "
        f"build requires {_DEVELOPER_ENABLE_ENV}=1, {_DEVELOPER_PATH_ENV}, and "
        f"{_DEVELOPER_FINGERPRINT_ENV}"
    )


@lru_cache(maxsize=1)
def _validated_native_build_info() -> dict[str, object]:
    return _validate_build_info(native_extension().build_info())


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


from .symbols import native_extension  # noqa: E402


__all__ = [
    "CHANNEL_NATIVE_ABI_VERSION",
    "ExtensionABIError",
    "ExtensionLoadError",
    "ExtensionSymbolError",
    "build_info",
    "native_extension",
]
