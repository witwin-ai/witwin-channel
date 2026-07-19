from witwin.channel_native.core.kernels.extension import build_info


def test_build_info_reports_rayd_native_capability_key():
    info = build_info()

    assert "uses_rayd_native" in info
    assert isinstance(info["uses_rayd_native"], bool)
