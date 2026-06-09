from witwin.channel_native.core.kernels.extension import build_info


def test_build_info_reports_raydn_native_capability_key():
    info = build_info()

    assert "uses_raydn_native" in info
    assert isinstance(info["uses_raydn_native"], bool)
