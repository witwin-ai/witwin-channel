from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OWNER_SECTIONS = {
    "## Ownership",
    "## Public entry points",
    "## Dependency rules",
    "## Numerical and AD contract",
    "## Forbidden fallback",
}
OWNER_DOCS = (
    ROOT / "src" / "witwin" / "channel_native" / "runtime" / "README.md",
    ROOT / "src" / "witwin" / "channel_native" / "scene" / "README.md",
    ROOT / "src" / "witwin" / "channel_native" / "propagation" / "README.md",
    ROOT / "src" / "witwin" / "channel_native" / "scattering" / "README.md",
    ROOT / "docs" / "dev" / "materials-owner.md",
)


@pytest.mark.parametrize("owner_doc", OWNER_DOCS, ids=lambda path: path.stem)
def test_domain_owner_docs_freeze_required_boundaries(owner_doc: Path):
    content = owner_doc.read_text(encoding="utf-8")

    assert OWNER_SECTIONS <= set(content.splitlines())


def test_materials_remains_a_module_until_its_dedicated_migration():
    package_root = ROOT / "src" / "witwin" / "channel_native"

    assert (package_root / "materials.py").is_file()
    assert not (package_root / "materials").exists()
