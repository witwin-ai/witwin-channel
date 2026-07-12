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
        archive.writestr(
            "witwin_channel_native-0.1.0.dist-info/METADATA", payload
        )
    return path.read_bytes()


def test_wheel_identity_and_sha256_are_read_from_wheel(tmp_path: Path):
    wheel = tmp_path / "witwin_channel_native-0.1.0-py3-none-any.whl"
    payload = _write_wheel(
        wheel, name="witwin-channel-native", version="0.1.0"
    )

    assert wheel_smoke._wheel_identity(wheel) == (
        "witwin-channel-native",
        "0.1.0",
    )
    assert wheel_smoke._sha256(wheel) == __import__("hashlib").sha256(
        payload
    ).hexdigest()


def test_wheel_identity_rejects_unexpected_distribution(tmp_path: Path):
    wheel = tmp_path / "other-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="other", version="0.1.0")

    with pytest.raises(ValueError, match="wheel identity"):
        wheel_smoke._wheel_identity(wheel)


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
    assert 'build_info.get("uses_raydn_native") is not True' in code
