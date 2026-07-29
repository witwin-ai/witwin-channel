# Copyright Xingyu Chen.
# Tests rayd native link.

from witwin.channel.deployment import build_info


def test_build_info_reports_rayd_native_capability_key():
    info = build_info()

    assert "uses_rayd_native" in info
    assert isinstance(info["uses_rayd_native"], bool)