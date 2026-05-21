from __future__ import annotations

from pathlib import Path


def test_diffraction_builders_do_not_create_dummy_transmitters():
    source = Path("witwin/channel/deterministic/diffraction/builders.py").read_text(
        encoding="utf-8",
    )

    assert "Tx((0.0, 0.0, 0.0))" not in source
