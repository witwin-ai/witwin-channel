from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from witwin.channel.materials import (
    layer_stack_rt as production_layer_stack_rt,
)
from tests.reference.em_oracle import (
    kirchhoff_diffuse_lobe_series as reference_kirchhoff_series,
)
from tests.reference.em_oracle import (
    layer_stack_rt as reference_layer_stack_rt,
)
from witwin.channel.scene.resources import (
    _kirchhoff_diffuse_lobe_series as production_kirchhoff_series,
)


_RT_FIELDS = (
    "r_te",
    "r_tm",
    "t_te",
    "t_tm",
    "R_te",
    "R_tm",
    "T_te",
    "T_tm",
    "A_te",
    "A_tm",
)


@pytest.mark.parametrize(
    ("layers", "frequency_hz"),
    [
        ([], 1.0e9),
        ([(0.0, 8.0, 0.5, 1.0)], 28.0e9),
        (
            [
                (0.013, 4.7, 0.021, 1.0),
                (0.004, 2.2, 0.0, 1.1),
                (0.002, 8.0, 0.5, 1.0),
            ],
            28.0e9,
        ),
    ],
)
def test_production_layer_stack_is_bitwise_equal_to_reference(
    layers: list[tuple], frequency_hz: float
) -> None:
    cos_theta = np.array([0.015625, 0.125, 0.5, 0.984375], dtype=np.float64)
    production = production_layer_stack_rt(layers, cos_theta, frequency_hz)
    reference = reference_layer_stack_rt(layers, cos_theta, frequency_hz)

    for name in _RT_FIELDS:
        assert np.array_equal(getattr(production, name), getattr(reference, name)), name


def test_production_kirchhoff_series_is_bitwise_equal_to_reference() -> None:
    q = np.array([-32.0, 0.0, 11.0], dtype=np.float64)
    args = (
        q[:, None],
        q[None, :],
        np.array([[12.0], [40.0], [75.0]], dtype=np.float64),
        0.002,
        0.03,
        0.017,
    )

    production = production_kirchhoff_series(*args, n_terms=96)
    reference = reference_kirchhoff_series(*args, n_terms=96)
    assert np.array_equal(production, reference)
    assert production_kirchhoff_series(
        1.0, 2.0, 3.0, 0.0, 0.1, 0.2
    ) == reference_kirchhoff_series(1.0, 2.0, 3.0, 0.0, 0.1, 0.2)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def test_production_and_reference_precompute_have_static_zero_dependency() -> None:
    root = Path(__file__).parents[2] / "src" / "witwin" / "channel"
    production_paths = (
        root / "constants.py",
        root / "materials.py",
        # The compile-time Kirchhoff/phase-screen construction merged into the
        # scene resource owner; the per-solve evaluator stayed where it was.
        root / "scene" / "resources.py",
        root / "interactions" / "scattering.py",
    )
    for path in production_paths:
        assert "tests.reference.em_oracle" not in _imports(path), path

    # The oracle left the shipped package; it may only reach the constants owner.
    assert not (root / "physics").exists()
    oracle_imports = _imports(
        Path(__file__).parents[1] / "reference" / "em_oracle.py"
    )
    assert not any(
        name.startswith("witwin.channel.materials")
        or name.startswith("witwin.channel.scene")
        or name.startswith("witwin.channel.interactions")
        or name.startswith("witwin.channel.propagation")
        for name in oracle_imports
    )
