# Copyright Xingyu Chen.
# Tests that the test harness never selects a local extension build on its own.

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support import native_ext

_NAMES = (
    "WITWIN_CHANNEL_DEVELOPER_OVERRIDE",
    "WITWIN_CHANNEL_EXTENSION_PATH",
    "WITWIN_CHANNEL_EXPECTED_FINGERPRINT",
)


@pytest.fixture
def local_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(native_ext, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(native_ext, "_importable", lambda _name: False)
    build_dir = tmp_path / "artifacts" / "cmake-old"
    build_dir.mkdir(parents=True)
    extension = build_dir / f"_channel{native_ext.importlib.machinery.EXTENSION_SUFFIXES[0]}"
    extension.touch()
    (build_dir / "_channel.build-fingerprint").write_text("a" * 64, encoding="ascii")
    return extension


def test_local_build_is_listed_but_never_selected(local_build: Path) -> None:
    import os

    assert native_ext.inject_native_paths() is False
    assert all(os.environ.get(name) is None for name in _NAMES)
    guidance = native_ext.build_guidance()
    assert str(local_build.resolve()) in guidance
    assert "no developer override is set" in guidance


def test_explicit_override_selects_the_named_build(
    monkeypatch: pytest.MonkeyPatch, local_build: Path,
) -> None:
    monkeypatch.setenv("WITWIN_CHANNEL_DEVELOPER_OVERRIDE", "1")
    monkeypatch.setenv("WITWIN_CHANNEL_EXTENSION_PATH", str(local_build.resolve()))
    monkeypatch.setenv("WITWIN_CHANNEL_EXPECTED_FINGERPRINT", "a" * 64)

    assert native_ext.inject_native_paths() is True


def test_incomplete_override_reports_the_missing_variable(
    monkeypatch: pytest.MonkeyPatch, local_build: Path,
) -> None:
    monkeypatch.setenv("WITWIN_CHANNEL_DEVELOPER_OVERRIDE", "1")
    monkeypatch.setenv("WITWIN_CHANNEL_EXTENSION_PATH", str(local_build.resolve()))

    assert native_ext.inject_native_paths() is False
    assert "WITWIN_CHANNEL_EXPECTED_FINGERPRINT is not set" in native_ext.build_guidance()
