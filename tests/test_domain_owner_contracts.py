# Copyright Xingyu Chen.
# Tests domain owner contracts.

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
# No owner document lives inside the package tree. Every domain owner document
# lives under `docs/dev/<owner>/README.md`, whether the owner is a single module
# with no directory to hold a README or a package.
OWNER_DOCS = (
    ROOT / "docs" / "dev" / "runtime" / "README.md",
    ROOT / "docs" / "dev" / "scene" / "README.md",
    ROOT / "docs" / "dev" / "propagation" / "README.md",
    ROOT / "docs" / "dev" / "scattering" / "README.md",
    ROOT / "docs" / "dev" / "materials" / "README.md",
)


# Owner document IDs use the parent directory because every file is README.md.
@pytest.mark.parametrize("owner_doc", OWNER_DOCS, ids=lambda path: path.parent.name)
def test_domain_owner_docs_freeze_required_boundaries(owner_doc: Path):
    content = owner_doc.read_text(encoding="utf-8")

    assert OWNER_SECTIONS <= set(content.splitlines())


def test_materials_package_preserves_public_identity_and_owns_kernel_contracts():
    import witwin.channel.materials as public_materials
    from witwin.channel.kernels.materials import validate_layer_csr

    package_root = ROOT / "witwin" / "channel"

    # `materials` is one module. A `materials/` package beside it would shadow
    # the module on import, so the directory must not come back.
    assert not (package_root / "materials").exists()
    assert not (package_root / "core" / "materials.py").exists()
    assert (package_root / "materials.py").is_file()
    assert (package_root / "kernels" / "materials.py").is_file()
    assert public_materials.__all__ == ["validate_layer_csr"]
    assert public_materials.validate_layer_csr is validate_layer_csr