from __future__ import annotations

from email.message import Message
from pathlib import Path
import zipfile

import pytest

from ci import wheel_smoke


def _write_wheel(path: Path, *, name: str, version: str) -> bytes:
    metadata = Message()
    metadata["Metadata-Version"] = "2.1"
    metadata["Name"] = name
    metadata["Version"] = version
    payload = metadata.as_bytes()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("witwin_channel_native-0.1.0.dist-info/METADATA", payload)
        archive.writestr(
            "witwin/channel_native/_channel_native.cp311-win_amd64.pyd", b"native"
        )
        archive.writestr(
            "witwin/channel_native/runtime/_channel_native.build-fingerprint",
            b"a" * 64 + b"\n",
        )
        archive.writestr(
            "witwin/channel_native/runtime/rayd.lock.json", b'{"schema_version": 1}'
        )
    return path.read_bytes()


def test_wheel_identity_and_sha256_are_read_from_wheel(tmp_path: Path):
    wheel = tmp_path / "witwin_channel_native-0.1.0-py3-none-any.whl"
    payload = _write_wheel(wheel, name="witwin-channel-native", version="0.1.0")

    assert wheel_smoke._wheel_identity(wheel) == (
        "witwin-channel-native",
        "0.1.0",
    )
    assert (
        wheel_smoke._sha256(wheel) == __import__("hashlib").sha256(payload).hexdigest()
    )


def test_wheel_identity_rejects_unexpected_distribution(tmp_path: Path):
    wheel = tmp_path / "other-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="other", version="0.1.0")

    with pytest.raises(ValueError, match="wheel identity"):
        wheel_smoke._wheel_identity(wheel)


def test_wheel_content_audit_accepts_owned_runtime_files(tmp_path: Path):
    wheel = tmp_path / "witwin_channel_native-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel-native", version="0.1.0")

    wheel_smoke._audit_wheel_contents(wheel)


def test_wheel_content_audit_does_not_treat_https_url_as_drive_path(tmp_path: Path):
    wheel = tmp_path / "witwin_channel_native-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel-native", version="0.1.0")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(
            "witwin/channel_native/runtime/source.json",
            b'{"repository_url":"https://github.com/Asixa/RayD.git"}',
        )

    wheel_smoke._audit_wheel_contents(wheel)


@pytest.mark.parametrize(
    ("member", "payload"),
    [
        ("rayd/torch/_stable_ops.dll", b"leaked dependency"),
        ("witwin/channel_native/CMakeFiles/kernel.obj", b"build output"),
        ("witwin/channel_native/config.txt", b"E:\\private\\checkout"),
    ],
)
def test_wheel_content_audit_rejects_build_leaks(
    tmp_path: Path, member: str, payload: bytes
):
    wheel = tmp_path / "witwin_channel_native-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel-native", version="0.1.0")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(member, payload)

    with pytest.raises(ValueError, match="forbidden build content"):
        wheel_smoke._audit_wheel_contents(wheel)


def test_smoke_program_contains_strict_isolation_and_native_checks(tmp_path: Path):
    code = wheel_smoke._smoke_code(
        target=tmp_path,
        wheel=tmp_path / "package.whl",
        wheel_sha256="a" * 64,
        expected_name="witwin-channel-native",
        expected_version="0.1.0",
    )

    compile(code, "<wheel-smoke>", "exec")
    assert "is_relative_to(target)" in code
    assert 'find_spec("witwin.channel_native._channel_native")' in code
    assert 'build_info.get("uses_dr_jit") is not False' in code
    assert 'build_info.get("uses_rayd_native") is not True' in code
def test_resolve_wheel_accepts_one_file_or_one_wheel_directory(tmp_path: Path):
    wheel = tmp_path / "package.whl"
    wheel.write_bytes(b"wheel")

    assert wheel_smoke._resolve_wheel(wheel) == wheel.resolve()
    assert wheel_smoke._resolve_wheel(tmp_path) == wheel.resolve()


def test_resolve_wheel_rejects_empty_or_ambiguous_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        wheel_smoke._resolve_wheel(tmp_path)
    (tmp_path / "one.whl").write_bytes(b"one")
    (tmp_path / "two.whl").write_bytes(b"two")
    with pytest.raises(ValueError, match="found 2"):
        wheel_smoke._resolve_wheel(tmp_path)
