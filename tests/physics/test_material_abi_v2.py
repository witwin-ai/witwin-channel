from __future__ import annotations

import importlib.util

from witwin.channel.materials.abi import MATERIAL_ABI_VERSION


def test_material_abi_v2_facades_are_deleted() -> None:
    assert MATERIAL_ABI_VERSION == 3
    assert importlib.util.find_spec("witwin.channel.core.materials") is None
    assert importlib.util.find_spec("witwin.channel.core.scene_loader") is None
    assert importlib.util.find_spec("witwin.channel.core.material_runtime") is None
