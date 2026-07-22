from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from witwin.channel.core.kernels import extension as compatibility_extension
from witwin.channel.runtime import extension


@pytest.fixture(autouse=True)
def _isolated_loader(monkeypatch: pytest.MonkeyPatch):
    extension._clear_loader_caches()
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
        "rayd_integration_abi_kind": "source-header-sha256",
        "rayd_integration_abi_path": "backends/torch/include/rayd/torch/integration.h",
        "rayd_repository_url": "https://example.invalid/RayD.git",
        "rayd_source_kind": "git-checkout",
        "rayd_source_manifest_sha256": "4" * 64,
        "torch_version": "2.10.0",
        "build_type": "Release",
    }
    info["build_fingerprint"] = extension._expected_fingerprint(info)
    return info


def _configure_identity_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        extension,
        "_load_rayd_lock",
        lambda: {
            "repository_url": "https://example.invalid/RayD.git",
            "commit": "2" * 40,
            "integration_abi": {
                "kind": "source-header-sha256",
                "path": "backends/torch/include/rayd/torch/integration.h",
                "sha256": "3" * 64,
            },
            "source_bundle": {"manifest_sha256": "4" * 64},
        },
    )
    monkeypatch.setattr(
        extension, "_runtime_identity", lambda: ("2.10.0", "12.8", "msvc")
    )


def test_core_extension_is_only_a_compatibility_facade():
    assert compatibility_extension.native_extension is extension.native_extension
    assert compatibility_extension.build_info is extension.build_info


def test_native_extension_prefers_packaged_module(monkeypatch: pytest.MonkeyPatch):
    packaged = object()
    calls: list[tuple[str, str | None]] = []

    def import_module(name: str, package: str | None = None):
        calls.append((name, package))
        return packaged

    monkeypatch.setattr(extension.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(extension.importlib, "import_module", import_module)
    monkeypatch.setattr(
        extension,
        "_validate_extension",
        lambda module, *, packaged, expected_fingerprint: (
            module if packaged and expected_fingerprint == "0" * 64 else None
        ),
    )
    monkeypatch.setattr(extension, "_packaged_expected_fingerprint", lambda: "0" * 64)
    monkeypatch.setenv("WITWIN_CHANNEL_DEVELOPER_OVERRIDE", "1")
    monkeypatch.setenv("WITWIN_CHANNEL_EXTENSION_PATH", "ignored")
    monkeypatch.setenv("WITWIN_CHANNEL_EXPECTED_FINGERPRINT", "0" * 64)

    assert extension.native_extension() is packaged
    assert extension.native_extension() is packaged
    assert calls == [("._channel", "witwin.channel")]


def test_native_extension_never_imports_a_global_module(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, str | None]] = []

    def import_module(name: str, package: str | None = None):
        calls.append((name, package))
        return object()

    monkeypatch.setattr(extension.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(extension.importlib, "import_module", import_module)

    with pytest.raises(extension.ExtensionLoadError, match="is not installed"):
        extension.native_extension()
    assert calls == []


def test_native_extension_preserves_packaged_dependency_import_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    error = ModuleNotFoundError("No module named 'missing_native_dependency'")
    error.name = "missing_native_dependency"

    def import_module(name: str, package: str | None = None):
        raise error

    monkeypatch.setattr(extension.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(extension.importlib, "import_module", import_module)

    with pytest.raises(ModuleNotFoundError, match="missing_native_dependency"):
        extension.native_extension()


@pytest.mark.parametrize(
    ("enabled", "path"),
    (("1", None), (None, "C:/native/_channel.pyd"), ("yes", "C:/x.pyd")),
)
def test_developer_override_requires_switch_and_path(
    monkeypatch: pytest.MonkeyPatch, enabled: str | None, path: str | None
):
    monkeypatch.setattr(extension.util, "find_spec", lambda _name: None)
    if enabled is not None:
        monkeypatch.setenv("WITWIN_CHANNEL_DEVELOPER_OVERRIDE", enabled)
    if path is not None:
        monkeypatch.setenv("WITWIN_CHANNEL_EXTENSION_PATH", path)
    monkeypatch.setenv("WITWIN_CHANNEL_EXPECTED_FINGERPRINT", "0" * 64)

    with pytest.raises(extension.ExtensionLoadError, match="requires"):
        extension.native_extension()


def test_developer_override_loads_only_the_exact_absolute_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    suffix = extension.machinery.EXTENSION_SUFFIXES[0]
    path = tmp_path / f"_channel{suffix}"
    path.touch()
    loaded = object()
    calls: list[tuple[str, str]] = []

    def load_developer_extension(raw_path: str, fingerprint: str) -> object:
        calls.append((raw_path, fingerprint))
        return loaded

    monkeypatch.setattr(extension.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(
        extension, "_load_developer_extension", load_developer_extension
    )
    monkeypatch.setenv("WITWIN_CHANNEL_DEVELOPER_OVERRIDE", "1")
    monkeypatch.setenv("WITWIN_CHANNEL_EXTENSION_PATH", str(path))
    monkeypatch.setenv("WITWIN_CHANNEL_EXPECTED_FINGERPRINT", "a" * 64)

    assert extension.native_extension() is loaded
    assert calls == [(str(path.resolve()), "a" * 64)]


def test_packaged_module_origin_cannot_resolve_to_global_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    global_extension = tmp_path / "_channel.pyd"
    global_extension.touch()
    module = SimpleNamespace(
        __file__=str(global_extension), build_info=lambda: _valid_build_info()
    )
    monkeypatch.setattr(extension.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(extension.importlib, "import_module", lambda *_a, **_k: module)
    monkeypatch.setattr(extension, "_packaged_expected_fingerprint", lambda: "0" * 64)

    with pytest.raises(extension.ExtensionLoadError, match="outside"):
        extension.native_extension()


def test_extension_origin_accepts_spec_origin_when_file_is_missing(tmp_path: Path):
    extension_file = tmp_path / "_channel.pyd"
    extension_file.touch()
    module = SimpleNamespace(__spec__=SimpleNamespace(origin=str(extension_file)))

    assert extension._extension_origin(module) == extension_file.resolve()


def test_missing_bootstrap_symbol_fails_before_any_computation():
    with pytest.raises(extension.ExtensionSymbolError, match="build_info"):
        extension._validate_extension(SimpleNamespace(), packaged=False)


def test_wrong_channel_abi_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _configure_identity_checks(monkeypatch)
    info = _valid_build_info()
    info["channel_abi_version"] = 99
    info["build_fingerprint"] = extension._expected_fingerprint(info)

    with pytest.raises(extension.ExtensionABIError, match="ABI mismatch"):
        extension._validate_build_info(info)


def test_wrong_rayd_sha_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _configure_identity_checks(monkeypatch)
    info = _valid_build_info()
    info["rayd_commit"] = "4" * 40
    info["build_fingerprint"] = extension._expected_fingerprint(info)

    with pytest.raises(extension.ExtensionABIError, match="rayd_commit"):
        extension._validate_build_info(info)


@pytest.mark.parametrize("field", ("torch_version", "cuda_version", "cxx_abi"))
def test_wrong_runtime_abi_is_rejected(monkeypatch: pytest.MonkeyPatch, field: str):
    _configure_identity_checks(monkeypatch)
    info = _valid_build_info()
    info[field] = "wrong"
    info["build_fingerprint"] = extension._expected_fingerprint(info)

    with pytest.raises(extension.ExtensionABIError, match=field):
        extension._validate_build_info(info)


def test_tampered_build_fingerprint_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _configure_identity_checks(monkeypatch)
    info = _valid_build_info()
    info["compiler"] = "different compiler"

    with pytest.raises(extension.ExtensionABIError, match="fingerprint"):
        extension._validate_build_info(info)


def test_developer_expected_fingerprint_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_identity_checks(monkeypatch)
    module = SimpleNamespace(build_info=lambda: _valid_build_info())

    with pytest.raises(extension.ExtensionABIError, match="expected build fingerprint"):
        extension._validate_extension(
            module, packaged=False, expected_fingerprint="f" * 64
        )


def test_valid_complete_identity_is_accepted(monkeypatch: pytest.MonkeyPatch):
    _configure_identity_checks(monkeypatch)
    assert extension._validate_build_info(_valid_build_info()) == _valid_build_info()


def test_build_info_validates_once_and_returns_fresh_mappings(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_identity_checks(monkeypatch)
    calls = 0

    def native_build_info() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _valid_build_info()

    module = SimpleNamespace(build_info=native_build_info)
    monkeypatch.setattr(extension, "native_extension", lambda: module)

    first = extension.build_info()
    second = extension.build_info()

    assert calls == 1
    assert first == second
    assert first is not second
    first["backend"] = "mutated"
    assert second["backend"] == "channel"


def test_real_rayd_lock_uses_the_canonical_schema():
    lock = extension._load_rayd_lock()

    assert lock["schema_version"] == 2
    assert isinstance(lock["repository_url"], str)
    assert len(lock["commit"]) == 40
    assert set(lock["integration_abi"]) == {
        "api_version",
        "identity",
        "kind",
        "path",
        "sha256",
    }
    assert set(lock["source_bundle"]) == {
        "distribution",
        "distribution_version",
        "manifest_sha256",
        "metadata_path",
    }
