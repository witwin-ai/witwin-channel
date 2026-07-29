# Copyright Xingyu Chen.
# Tests ad contract.

import json
import math
from pathlib import Path

import pytest
import torch

from witwin.channel import capabilities
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import _metadata as deterministic_metadata
from witwin.channel.montecarlo.basic import Config as BasicConfig
from witwin.channel.montecarlo.basic import (
    make_solver_metadata as basic_metadata,
)
from witwin.channel.montecarlo.bdpt import Config as BdptConfig
from witwin.channel.montecarlo.bdpt import (
    make_solver_metadata as bdpt_metadata,
)
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import _metadata as path_metadata


_CONFIG_TYPES = (PathConfig, DeterministicConfig, BasicConfig, BdptConfig)


def test_public_ad_capability_advertises_fixed_topology_jvp_vjp():
    manifest = capabilities()

    assert manifest["supports_ad"] is True
    assert manifest["ad_contract"] == {
        "decision": "fixed_topology_jvp_vjp",
        "public_modes": ["none", "jvp", "vjp"],
        "fixed_topology_jvp": True,
        "fixed_topology_vjp": True,
        # No estimator for visibility/topology discontinuities: path
        # birth/death and shadow transitions stay out of contract.
        "visibility_discontinuity_estimator": False,
        "differentiable_solvers": ["path", "deterministic", "montecarlo_basic"],
        "differentiable_inputs": [
            "material_eps_r",
            "material_sigma_e",
            "material_gain",
            "material_thickness",
            "frequency",
            "tx_position",
            "rx_position",
            "mesh_vertices",
        ],
        "ad_excluded": {
            "path": ["scattering", "coupled_paths_mesh_vertex"],
            # coupled reflection and diffraction: the grid solver now carries coupled paths and shares the
            # path solver's coupled mesh-vertex AD refusal.
            "deterministic": ["scattering", "coupled_paths_mesh_vertex"],
            "montecarlo_basic": ["scattering"],
            "montecarlo_bdpt": ["all"],
        },
        # Monte Carlo Basic exposes native LoS AD as a supported primitive.
        "low_level_primitives": [
            "mc_los_path_gain_backward",
            "mc_los_path_gain_jvp",
        ],
    }
    for name in ("path", "deterministic", "montecarlo_basic"):
        solver = manifest["solvers"][name]
        assert solver["supports_ad"] is True
        assert solver["ad_modes"] == ["none", "jvp", "vjp"]
        assert "scattering" in solver["ad_excluded"]
    bdpt = manifest["solvers"]["montecarlo_bdpt"]
    assert bdpt["supports_ad"] is False
    assert bdpt["ad_modes"] == ["none"]


@pytest.mark.parametrize(
    "config_type", (PathConfig, DeterministicConfig, BasicConfig)
)
@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_fixed_topology_solvers_accept_ad_modes(config_type, ad_mode):
    assert config_type(ad_mode=ad_mode).ad_mode == ad_mode


@pytest.mark.parametrize("config_type", _CONFIG_TYPES)
@pytest.mark.parametrize("ad_mode", ["forward", "reverse", "grad"])
def test_every_solver_rejects_unknown_ad_modes(config_type, ad_mode):
    with pytest.raises((ValueError, RuntimeError), match="ad_mode"):
        config_type(ad_mode=ad_mode)


@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_bdpt_now_accepts_fixed_topology_ad_modes(ad_mode):
    # BDPT AD lifts the AD contract BDPT AD deferral: the solver wires native
    # fixed-topology jvp/vjp companions, so the config accepts these modes.
    # Unknown-mode rejection stays covered by
    # test_every_solver_rejects_unknown_ad_modes (BdptConfig is in
    # _CONFIG_TYPES).
    assert BdptConfig(ad_mode=ad_mode).ad_mode == ad_mode


def test_primal_metadata_reports_no_ad_for_every_solver():
    path = path_metadata(
        config=PathConfig(max_depth=0, components={"los"}),
        path_count=1,
        reflection_available=False,
        diffraction_available=False,
        path_native_available=True,
    )
    deterministic = deterministic_metadata(
        config=DeterministicConfig(max_depth=0, components={"los"}),
        native_info={
            "uses_rayd_native": False,
            "uses_path_native": True,
            "cuda_available": True,
            "optix_available": False,
        },
        path_count=1,
        component_counts={"los": 1, "reflection": 0, "diffraction": 0},
        launch_count=1,
    )
    basic = basic_metadata(
        config=BasicConfig(samples=1, max_depth=0, components={"los"}),
        path_count=1,
        contribution_capacity=1,
        reflection_available=False,
        diffraction_available=False,
    )
    bdpt = bdpt_metadata(
        config=BdptConfig(samples=1, max_depth=0, components={"los"}),
        selected_accumulation_strategy="atomic",
        path_counts_by_strategy={"endpoint": 1},
        valid_contribution_count=1,
        reflection_available=False,
        diffraction_available=False,
        cuda_available=True,
        optix_available=False,
        workspace_bytes=0,
        variance_enabled=False,
        launch_count=1,
        effective_max_depth=0,
    )

    assert path["kernel"]["ad_status"] == "none"
    assert deterministic["kernel"]["ad_status"] == "none"
    assert basic["kernel"]["ad_status"] == "none"
    assert basic["kernel"]["tape_bytes"] == 0
    assert bdpt["kernel"]["ad_status"] == "none"
    assert bdpt["ad_status"] == "none"


def test_free_space_derivative_oracle_matches_centered_finite_difference():
    frequency_hz = 3.0e9
    tx = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64)
    rx = torch.tensor([2.3, 1.1, -0.5], dtype=torch.float64)
    power = torch.tensor(1.7, dtype=torch.float64)
    tx_tangent = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    rx_tangent = torch.tensor([-0.1, 0.4, 0.2], dtype=torch.float64)
    power_tangent = torch.tensor(-0.25, dtype=torch.float64)

    scale = (299_792_458.0 / frequency_hz / (4.0 * math.pi)) ** 2

    def gain(tx_value, power_value, rx_value):
        distance_sq = (tx_value - rx_value).square().sum()
        return power_value * scale / distance_sq

    displacement = tx - rx
    distance_sq = displacement.square().sum()
    analytic = power_tangent * scale / distance_sq
    analytic -= (
        2.0
        * power
        * scale
        * displacement.dot(tx_tangent - rx_tangent)
        / distance_sq.square()
    )
    step = 1.0e-5
    finite_difference = (
        gain(
            tx + step * tx_tangent, power + step * power_tangent, rx + step * rx_tangent
        )
        - gain(
            tx - step * tx_tangent, power - step * power_tangent, rx - step * rx_tangent
        )
    ) / (2.0 * step)

    torch.testing.assert_close(analytic, finite_difference, rtol=1.0e-8, atol=1.0e-12)


def test_ad_inventory_records_solver_ad_contract():
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads(
        (
            root / "docs" / "dev" / "replacement" / "channel-api-inventory.json"
        ).read_text(encoding="utf-8")
    )
    audit = inventory["ad_audit"]

    assert audit["production_channel_ad_references"] == 0
    # material and frequency derivatives: the shared deterministic/path field seam calls the three
    # differentiable field entry points (LoS, reflection, transmission).
    # solver derivatives adds the six montecarlo.basic power-map entry points.
    # diffraction AD adds the wedge re-evaluation, receiver projection, coupled
    # R-D transport and coupled stationary re-solve entry points; AD adds
    # the montecarlo.basic diffraction radiomap entry point.
    assert audit["native_public_solver_ad_callers"] == 14
    assert len(audit["native_public_solver_ad_call_sites"]) == 14
    assert audit["legacy_reference_files_with_explicit_ad"] == 19
    assert audit["decision"] == "fixed_topology_material_frequency_ad_t1"
    # the AD capability gate flipped the capability manifest (AD).
    assert audit["supports_ad"] is True