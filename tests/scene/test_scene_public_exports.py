from __future__ import annotations

from witwin.channel.core.scene import Mesh, Scene, SionnaAdaptor


def test_scene_module_exports_public_types() -> None:
    assert Scene.__name__ == "Scene"
    assert Mesh.__name__ == "Mesh"
    assert SionnaAdaptor.__name__ == "SionnaAdaptor"
