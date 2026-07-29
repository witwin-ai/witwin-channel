# Copyright Xingyu Chen.
# Tests material abi v2.

from __future__ import annotations

import importlib.util

from witwin.channel.materials import MATERIAL_ABI_VERSION


def test_material_abi_v2_facades_are_deleted() -> None:
    assert MATERIAL_ABI_VERSION == 3
    assert importlib.util.find_spec("witwin.channel.core") is None