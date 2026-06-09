import sys


def test_raydn_scene_wrapper_does_not_import_python_raydn():
    sys.modules.pop("raydn", None)

    from witwin.channel_native.core.runtime.raydn import RayDNScene

    assert RayDNScene.__name__ == "RayDNScene"
    assert "raydn" not in sys.modules


def test_raydn_scene_exposes_opaque_handle():
    from witwin.channel_native.core.runtime.raydn import RayDNScene

    handle = object()
    scene = RayDNScene(handle)

    assert scene.handle is handle
