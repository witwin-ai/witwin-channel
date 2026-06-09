import sys


def test_channel_native_import_does_not_import_drjit_or_raydn():
    sys.modules.pop("witwin.channel_native", None)
    sys.modules.pop("drjit", None)
    sys.modules.pop("raydn", None)

    import witwin.channel_native  # noqa: F401

    assert "drjit" not in sys.modules
    assert "raydn" not in sys.modules
