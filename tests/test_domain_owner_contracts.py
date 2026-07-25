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
    ROOT / "src" / "witwin" / "channel" / "runtime" / "README.md",
    ROOT / "src" / "witwin" / "channel" / "scene" / "README.md",
    ROOT / "src" / "witwin" / "channel" / "propagation" / "README.md",
    ROOT / "src" / "witwin" / "channel" / "scattering" / "README.md",
    ROOT / "docs" / "dev" / "materials-owner.md",
)


@pytest.mark.parametrize("owner_doc", OWNER_DOCS, ids=lambda path: path.stem)
def test_domain_owner_docs_freeze_required_boundaries(owner_doc: Path):
    content = owner_doc.read_text(encoding="utf-8")

    assert OWNER_SECTIONS <= set(content.splitlines())


def test_materials_package_preserves_public_identity_and_owns_kernel_contracts():
    import witwin.channel.materials as public_materials
    from witwin.channel.materials.kernels import validate_layer_csr

    package_root = ROOT / "src" / "witwin" / "channel"
    materials_root = package_root / "materials"

    assert not (package_root / "materials.py").exists()
    assert not (package_root / "core" / "materials.py").exists()
    assert (materials_root / "__init__.py").is_file()
    assert (materials_root / "kernels" / "contracts.py").is_file()
    assert public_materials.__all__ == ["validate_layer_csr"]
    assert public_materials.validate_layer_csr is validate_layer_csr
