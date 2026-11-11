# Copyright Xingyu Chen.
# Tests that imported scenes, ITU presets and pattern factories compile and solve on CUDA.

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from witwin.core import AntennaState, Scene
from witwin.core.identity import new_antenna_id
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import solve as solve_deterministic
from witwin.channel.importers import load_mitsuba
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import solve as solve_path
from witwin.channel.presets import dipole_pattern, itu_properties, tr38901_pattern
from witwin.channel.scene import compile as compile_scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DATA = Path(__file__).resolve().parents[1] / "data" / "scenes"
FREQUENCY_HZ = 3.5e9


def _endpoints(*, tx_pattern="isotropic", rx_x: float = 10.0) -> list[AntennaState]:
    return [
        AntennaState(
            new_antenna_id(), "tx", torch.tensor([0.0, 0.0, 1.5]), power_w=1.0,
            pattern=tx_pattern,
        ),
        AntennaState(new_antenna_id(), "rx", torch.tensor([rx_x, 0.0, 1.5])),
    ]


def test_imported_itu_materials_compile_at_the_reference_frequency() -> None:
    imported = load_mitsuba(DATA / "canyon.xml", frequency_hz=FREQUENCY_HZ)
    compiled = compile_scene(imported.to_scene(), reference_frequency_hz=FREQUENCY_HZ)

    keys = compiled.materials.material_keys
    row = next(i for i, key in enumerate(keys) if key.endswith(":concrete"))
    eps_r, sigma = itu_properties("concrete", FREQUENCY_HZ)
    assert float(compiled.materials.eps_r[row]) == pytest.approx(eps_r, rel=1e-6)
    assert float(compiled.materials.sigma_e[row]) == pytest.approx(sigma, rel=1e-5)
    assert any(key.endswith(":concrete") for key in compiled.materials.frequency_dependent)


def test_compiling_an_itu_material_outside_its_range_fails_before_native_work() -> None:
    imported = load_mitsuba(DATA / "canyon.xml", frequency_hz=FREQUENCY_HZ)

    with pytest.raises(ValueError, match="'brick'|'concrete'|'glass'|'wood'"):
        compile_scene(imported.to_scene(), reference_frequency_hz=150e9)
    torch.zeros(1, device="cuda").add_(1.0)
    torch.cuda.synchronize()


def test_imported_ground_reflects_like_a_specular_plane() -> None:
    imported = load_mitsuba(DATA / "canyon.xml", frequency_hz=FREQUENCY_HZ)
    scene = imported.to_scene(endpoints=_endpoints())

    result = solve_deterministic(
        scene, DeterministicConfig(max_depth=1, components={"los", "reflection"}),
        reference_frequency_hz=FREQUENCY_HZ,
    )

    los = float(result.component_power["los"].sum())
    reflection = float(result.component_power["reflection"].sum())
    assert los > 0.0 and reflection > 0.0
    assert reflection < los


@pytest.mark.parametrize(
    ("pattern", "expected_db"),
    [(tr38901_pattern(), 8.0), (dipole_pattern("half_wave"), 2.150821)],
)
def test_path_solver_applies_preset_patterns(pattern, expected_db: float) -> None:
    config = PathConfig(max_depth=0, components={"los"})

    reference = solve_path(
        Scene(endpoints=_endpoints()), config, reference_frequency_hz=FREQUENCY_HZ
    )
    shaped = solve_path(
        Scene(endpoints=_endpoints(tx_pattern=pattern)), config,
        reference_frequency_hz=FREQUENCY_HZ,
    )

    ratio_db = 20 * torch.log10(shaped.a.abs().sum() / reference.a.abs().sum())
    assert float(ratio_db) == pytest.approx(expected_db, abs=1e-3)
