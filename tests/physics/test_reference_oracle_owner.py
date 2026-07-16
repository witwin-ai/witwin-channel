from __future__ import annotations

import ast
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

import witwin.channel_native.physics as physics
from witwin.channel_native.physics import oracle as legacy
from witwin.channel_native.physics.reference import oracle as canonical


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

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

_EXPECTED_ALL = [
    "C0",
    "EPS0",
    "ETA0",
    "MU0",
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
]


def test_reference_oracle_has_one_canonical_owner_and_same_object_facades() -> None:
    assert physics.__all__ == _EXPECTED_ALL
    assert legacy.__all__ == _EXPECTED_ALL
    assert canonical.__all__ == _EXPECTED_ALL
    for name in _PUBLIC:
        owner = getattr(canonical, name)
        assert getattr(legacy, name) is owner
        assert getattr(physics, name) is owner
        assert owner.__module__ == "witwin.channel_native.physics.oracle"


@pytest.mark.parametrize(
    "order",
    (
        ("witwin.channel_native.physics", "witwin.channel_native.physics.oracle"),
        (
            "witwin.channel_native.physics.oracle",
            "witwin.channel_native.physics.reference.oracle",
        ),
        (
            "witwin.channel_native.physics.reference.oracle",
            "witwin.channel_native.physics",
        ),
    ),
)
def test_reference_oracle_identity_is_import_order_independent(
    order: tuple[str, str],
) -> None:
    code = f"""
import importlib
for name in {order!r}:
    importlib.import_module(name)
p = importlib.import_module('witwin.channel_native.physics')
l = importlib.import_module('witwin.channel_native.physics.oracle')
c = importlib.import_module('witwin.channel_native.physics.reference.oracle')
for name in {_PUBLIC!r}:
    assert getattr(p, name) is getattr(l, name) is getattr(c, name)
"""
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_legacy_pickle_paths_replay_to_canonical_objects() -> None:
    medium = canonical.vacuum_medium(3.5e9)
    restored = pickle.loads(pickle.dumps(medium))
    assert type(restored) is canonical.Medium
    assert restored == medium
    assert (
        pickle.loads(pickle.dumps(canonical.layer_stack_rt)) is canonical.layer_stack_rt
    )


def test_legacy_facade_preserves_pre_move_reachable_globals() -> None:
    for name in (
        "Callable",
        "Sequence",
        "dataclass",
        "np",
        "_admittances",
        "_interface_rt",
        "_power_coefficients",
        "_stack_rt_one_pol",
    ):
        assert getattr(legacy, name) is getattr(canonical, name)
    assert legacy.np is np


def test_reference_oracle_is_static_numpy_only_and_production_independent() -> None:
    tree = ast.parse(Path(canonical.__file__).read_text(encoding="utf-8"))
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
        name.startswith("witwin.channel_native.materials")
        or name.startswith("witwin.channel_native.scattering")
        or name.startswith("witwin.channel_native.propagation")
        for name in imports
    )


def test_legacy_and_package_facades_import_canonical_owner_directly() -> None:
    for module in (legacy, physics):
        source = Path(module.__file__).read_text(encoding="utf-8")
        imports = {
            node.module
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "witwin.channel_native.physics.reference.oracle" in imports
