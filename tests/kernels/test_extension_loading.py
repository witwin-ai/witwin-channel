# Copyright Xingyu Chen.
# Tests extension loading.

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from witwin.channel import deployment
from witwin.channel import runtime


@pytest.fixture(autouse=True)
def _isolated_loader(monkeypatch: pytest.MonkeyPatch):
    runtime._clear_loader_caches()
    monkeypatch.delenv("WITWIN_CHANNEL_DEVELOPER_OVERRIDE", raising=False)
    monkeypatch.delenv("WITWIN_CHANNEL_EXTENSION_PATH", raising=False)
    monkeypatch.delenv("WITWIN_CHANNEL_EXPECTED_FINGERPRINT", raising=False)
    yield


def _valid_build_info() -> dict[str, object]:
    info: dict[str, object] = {
        "backend": "channel",
        "uses_dr_jit": False,
        "uses_rayd_native": True,
        "rayd_integration": "source-linked",
        "uses_path_native": True,
        "material_abi_version": 3,
        "cuda_available": True,
        "optix_available": True,
        "channel_abi_version": 1,
        "channel_git_dirty": False,
        "channel_git_sha": "1" * 40,
        "compiler": "MSVC 19.44",
        "cuda_architectures": ["89-real"],
        "cuda_compiler_version": "12.8",
        "cuda_version": "12.8",
        "cxx_abi": "msvc",
        "rayd_dirty": False,
        "rayd_commit": "2" * 40,
        "rayd_integration_abi_sha256": "3" * 64,
        "rayd_integration_abi_kind": "source-header-set-sha256",
        "rayd_integration_abi_path": "include/rayd/integration.h",
        "rayd_repository_url": "https://example.invalid/RayD.git",
        "rayd_source_kind": "git-checkout",
        "rayd_source_manifest_sha256": "4" * 64,
        "torch_version": "2.10.0",
        "build_type": "Release",
    }
    info["build_fingerprint"] = runtime._expected_fingerprint(info)
    return info


def _configure_identity_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime,
        "_load_rayd_lock",
        lambda: {
            "repository_url": "https://example.invalid/RayD.git",
            "commit": "2" * 40,
            "integration_abi": {
                "kind": "source-header-set-sha256",
                "entrypoint": "include/rayd/integration.h",
                "sha256": "3" * 64,
            },
            "source_bundle": {"manifest_sha256": "4" * 64},
        },
    )
    monkeypatch.setattr(runtime, "_runtime_identity", lambda: ("2.10.0", "12.8", "msvc"))


def test_no_compatibility_extension_facade_exists():
    """The dissolved ``core.kernels.extension`` shim must not come back."""

    assert importlib.util.find_spec("witwin.channel.core") is None
    # ``deployment`` publishes the runtime owner's function; it does not define
    # a second one, so the object still reports the runtime module.
    assert deployment.build_info is runtime.build_info
    assert deployment.build_info.__module__ == "witwin.channel.runtime"
    assert deployment._import_native_build_info() is runtime.build_info


def test_native_extension_prefers_packaged_module(monkeypatch: pytest.MonkeyPatch):
    packaged = object()
    calls: list[tuple[str, str | None]] = []

    def import_module(name: str, package: str | None = None):
        calls.append((name, package))
        return packaged

    monkeypatch.setattr(runtime.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    monkeypatch.setattr(
        runtime,
        "_validate_extension",
        lambda module, *, packaged, expected_fingerprint: (
            module if packaged and expected_fingerprint == "0" * 64 else None
        ),
    )
    monkeypatch.setattr(runtime, "_packaged_expected_fingerprint", lambda: "0" * 64)
    monkeypatch.setenv("WITWIN_CHANNEL_DEVELOPER_OVERRIDE", "1")
    monkeypatch.setenv("WITWIN_CHANNEL_EXTENSION_PATH", "ignored")
    monkeypatch.setenv("WITWIN_CHANNEL_EXPECTED_FINGERPRINT", "0" * 64)

    assert runtime.native_extension() is packaged
    assert runtime.native_extension() is packaged
    assert calls == [("._channel", "witwin.channel")]


def test_native_extension_never_imports_a_global_module(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str | None]] = []

    def import_module(name: str, package: str | None = None):
        calls.append((name, package))
        return object()

    monkeypatch.setattr(runtime.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(runtime.importlib, "import_module", import_module)

    with pytest.raises(runtime.ExtensionLoadError, match="is not installed"):
        runtime.native_extension()
    assert calls == []


def test_native_extension_preserves_packaged_dependency_import_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    error = ModuleNotFoundError("No module named 'missing_native_dependency'")
    error.name = "missing_native_dependency"

    def import_module(name: str, package: str | None = None):
        raise error

    monkeypatch.setattr(runtime.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(runtime.importlib, "import_module", import_module)

    with pytest.raises(ModuleNotFoundError, match="missing_native_dependency"):
        runtime.native_extension()


@pytest.mark.parametrize(
    ("enabled", "path"),
    (("1", None), (None, "C:/native/_channel.pyd"), ("yes", "C:/x.pyd")),
)
def test_developer_override_requires_switch_and_path(
    monkeypatch: pytest.MonkeyPatch, enabled: str | None, path: str | None,
):
    monkeypatch.setattr(runtime.util, "find_spec", lambda _name: None)
    if enabled is not None:
        monkeypatch.setenv("WITWIN_CHANNEL_DEVELOPER_OVERRIDE", enabled)
    if path is not None:
        monkeypatch.setenv("WITWIN_CHANNEL_EXTENSION_PATH", path)
    monkeypatch.setenv("WITWIN_CHANNEL_EXPECTED_FINGERPRINT", "0" * 64)

    with pytest.raises(runtime.ExtensionLoadError, match="requires"):
        runtime.native_extension()


def test_developer_override_loads_only_the_exact_absolute_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    suffix = runtime.machinery.EXTENSION_SUFFIXES[0]
    path = tmp_path / f"_channel{suffix}"
    path.touch()
    loaded = object()
    calls: list[tuple[str, str]] = []

    def load_developer_extension(raw_path: str, fingerprint: str) -> object:
        calls.append((raw_path, fingerprint))
        return loaded

    monkeypatch.setattr(runtime.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(runtime, "_load_developer_extension", load_developer_extension)
    monkeypatch.setenv("WITWIN_CHANNEL_DEVELOPER_OVERRIDE", "1")
    monkeypatch.setenv("WITWIN_CHANNEL_EXTENSION_PATH", str(path))
    monkeypatch.setenv("WITWIN_CHANNEL_EXPECTED_FINGERPRINT", "a" * 64)

    assert runtime.native_extension() is loaded
    assert calls == [(str(path.resolve()), "a" * 64)]


def test_packaged_module_origin_cannot_resolve_to_global_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    global_extension = tmp_path / "_channel.pyd"
    global_extension.touch()
    module = SimpleNamespace(__file__=str(global_extension), build_info=lambda: _valid_build_info())
    monkeypatch.setattr(runtime.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(runtime.importlib, "import_module", lambda *_a, **_k: module)
    monkeypatch.setattr(runtime, "_packaged_expected_fingerprint", lambda: "0" * 64)

    with pytest.raises(runtime.ExtensionLoadError, match="outside"):
        runtime.native_extension()


def test_extension_origin_accepts_spec_origin_when_file_is_missing(tmp_path: Path):
    extension_file = tmp_path / "_channel.pyd"
    extension_file.touch()
    module = SimpleNamespace(__spec__=SimpleNamespace(origin=str(extension_file)))

    assert runtime._extension_origin(module) == extension_file.resolve()


def test_missing_bootstrap_symbol_fails_before_any_computation():
    with pytest.raises(runtime.ExtensionSymbolError, match="build_info"):
        runtime._validate_extension(SimpleNamespace(), packaged=False)


def test_wrong_channel_abi_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _configure_identity_checks(monkeypatch)
    info = _valid_build_info()
    info["channel_abi_version"] = 99
    info["build_fingerprint"] = runtime._expected_fingerprint(info)

    with pytest.raises(runtime.ExtensionABIError, match="ABI mismatch"):
        runtime._validate_build_info(info)


def test_wrong_rayd_sha_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _configure_identity_checks(monkeypatch)
    info = _valid_build_info()
    info["rayd_commit"] = "4" * 40
    info["build_fingerprint"] = runtime._expected_fingerprint(info)

    with pytest.raises(runtime.ExtensionABIError, match="rayd_commit"):
        runtime._validate_build_info(info)


@pytest.mark.parametrize("field", ("torch_version", "cuda_version", "cxx_abi"))
def test_wrong_runtime_abi_is_rejected(monkeypatch: pytest.MonkeyPatch, field: str):
    _configure_identity_checks(monkeypatch)
    info = _valid_build_info()
    info[field] = "wrong"
    info["build_fingerprint"] = runtime._expected_fingerprint(info)

    with pytest.raises(runtime.ExtensionABIError, match=field):
        runtime._validate_build_info(info)


def test_tampered_build_fingerprint_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _configure_identity_checks(monkeypatch)
    info = _valid_build_info()
    info["compiler"] = "different compiler"

    with pytest.raises(runtime.ExtensionABIError, match="fingerprint"):
        runtime._validate_build_info(info)


def test_developer_expected_fingerprint_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _configure_identity_checks(monkeypatch)
    module = SimpleNamespace(build_info=lambda: _valid_build_info())

    with pytest.raises(runtime.ExtensionABIError, match="expected build fingerprint"):
        runtime._validate_extension(module, packaged=False, expected_fingerprint="f" * 64)


def test_valid_complete_identity_is_accepted(monkeypatch: pytest.MonkeyPatch):
    _configure_identity_checks(monkeypatch)
    assert runtime._validate_build_info(_valid_build_info()) == _valid_build_info()


def test_build_info_validates_once_and_returns_fresh_mappings(monkeypatch: pytest.MonkeyPatch):
    _configure_identity_checks(monkeypatch)
    calls = 0

    def native_build_info() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _valid_build_info()

    module = SimpleNamespace(build_info=native_build_info)
    monkeypatch.setattr(runtime, "native_extension", lambda: module)

    first = runtime.build_info()
    second = runtime.build_info()

    assert calls == 1
    assert first == second
    assert first is not second
    first["backend"] = "mutated"
    assert second["backend"] == "channel"


def test_real_rayd_lock_uses_the_canonical_schema():
    lock = runtime._load_rayd_lock()

    assert lock["schema_version"] == 2
    assert isinstance(lock["repository_url"], str)
    assert len(lock["commit"]) == 40
    assert set(lock["integration_abi"]) == {
        "api_version",
        "identity",
        "kind",
        "entrypoint",
        "headers",
        "sha256",
    }
    assert set(lock["source_bundle"]) == {
        "distribution",
        "distribution_version",
        "manifest_sha256",
        "metadata_path",
    }
