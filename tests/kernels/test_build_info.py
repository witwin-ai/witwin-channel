from witwin.channel_native.core.kernels.extension import build_info


def test_build_info_contract():
    info = build_info()

    assert info["backend"] == "channel-native"
    assert info["uses_dr_jit"] is False
    assert isinstance(info["uses_raydn_native"], bool)
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["optix_available"], bool)
