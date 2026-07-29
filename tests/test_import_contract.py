# Copyright Xingyu Chen.
# Tests import contract.

import sys


def test_channel_import_does_not_import_drjit_or_rayd():
    sys.modules.pop("witwin.channel", None)
    sys.modules.pop("drjit", None)
    sys.modules.pop("rayd", None)

    import witwin.channel  # noqa: F401

    assert "drjit" not in sys.modules
    assert "rayd" not in sys.modules