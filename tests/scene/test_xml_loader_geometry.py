# Copyright Xingyu Chen.
# Tests xml loader geometry.

import importlib.util

from witwin.core import Scene


def test_core_scene_has_no_channel_specific_mitsuba_loader_facade():
    assert not hasattr(Scene, "load_mitsuba")
    assert importlib.util.find_spec("witwin.channel.scene.loader") is None