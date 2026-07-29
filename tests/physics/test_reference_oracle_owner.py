# Copyright Xingyu Chen.
# The NumPy reference oracle has exactly one owner, and it lives under tests.

"""The NumPy reference oracle has exactly one owner, and it lives under tests."""

from __future__ import annotations

import ast
from pathlib import Path
import pickle

import numpy as np

from tests.reference import em_oracle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "witwin" / "channel"

_PUBLIC = (
    "Medium",
    "RTCoefficients",
    "coherent_attenuation",
    "complex_sqrt_passive",
    "fresnel_interface",
    "hemisphere_integral",
    "kirchhoff_diffuse_lobe_quadrature",
    "kirchhoff_diffuse_lobe_series",
    "layer_stack_rt",
    "medium_params",
    "phase_screen_patch_integral",
    "refraction_direction",
    "vacuum_medium",
)

_EXPECTED_ALL = ["C0", "EPS0", "ETA0", "MU0", *sorted(_PUBLIC)]


def test_reference_oracle_has_exactly_one_owner_module() -> None:
    assert em_oracle.__all__ == _EXPECTED_ALL
    for name in _PUBLIC:
        owner = getattr(em_oracle, name)
        assert owner.__module__ == "tests.reference.em_oracle", name


def test_no_facade_impersonates_the_oracle_module() -> None:
    """No object rewrites ``__module__`` to point at a re-export shim."""

    source = Path(em_oracle.__file__).read_text(encoding="utf-8")
    assert "__module__ =" not in source


def test_oracle_is_not_reachable_from_the_production_package() -> None:
    """The shipped package contains no path to the reference oracle."""

    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        assert "tests.reference" not in text, relative
        assert "witwin.channel.physics" not in text, relative
    assert not (PACKAGE_ROOT / "physics").exists()


def test_pickle_round_trips_against_the_owner() -> None:
    medium = em_oracle.vacuum_medium(3.5e9)
    restored = pickle.loads(pickle.dumps(medium))
    assert type(restored) is em_oracle.Medium
    assert restored == medium
    assert (
        pickle.loads(pickle.dumps(em_oracle.layer_stack_rt))
        is em_oracle.layer_stack_rt
    )


def test_reference_oracle_is_numpy_only_and_production_independent() -> None:
    tree = ast.parse(Path(em_oracle.__file__).read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "torch" not in imports
    assert not any(
        name.startswith("witwin.channel.materials")
        or name.startswith("witwin.channel.scene")
        or name.startswith("witwin.channel.interactions")
        or name.startswith("witwin.channel.propagation")
        for name in imports
    )
    # Physical constants are the one thing it shares with production, by design.
    assert em_oracle.np is np