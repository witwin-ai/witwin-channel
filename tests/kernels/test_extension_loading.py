from __future__ import annotations

import pytest

from witwin.channel_native.core.kernels import extension


def test_native_extension_prefers_packaged_module(monkeypatch: pytest.MonkeyPatch):
    packaged = object()
    calls = []

    def import_module(name: str, package: str | None = None):
        calls.append((name, package))
        return packaged

    monkeypatch.setattr(extension.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(extension.importlib, "import_module", import_module)

    assert extension.native_extension() is packaged
    assert calls == [("._channel_native", "witwin.channel_native")]


def test_native_extension_accepts_development_module(monkeypatch: pytest.MonkeyPatch):
    development = object()
    calls = []

    def import_module(name: str, package: str | None = None):
        calls.append((name, package))
        return development

    monkeypatch.setattr(extension.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(extension.importlib, "import_module", import_module)

    assert extension.native_extension() is development
    assert calls == [("_channel_native", None)]


def test_native_extension_preserves_dependency_import_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    error = ModuleNotFoundError("No module named 'missing_native_dependency'")
    error.name = "missing_native_dependency"

    def import_module(name: str, package: str | None = None):
        raise error

    monkeypatch.setattr(extension.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(extension.importlib, "import_module", import_module)

    with pytest.raises(ModuleNotFoundError, match="missing_native_dependency"):
        extension.native_extension()
