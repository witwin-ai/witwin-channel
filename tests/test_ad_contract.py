import json
import math
from pathlib import Path

import pytest
import torch

from witwin.channel_native import capabilities
from witwin.channel_native.deterministic import Config as DeterministicConfig
from witwin.channel_native.deterministic.solver import (
    _metadata as deterministic_metadata,
)
from witwin.channel_native.montecarlo.basic import Config as BasicConfig
from witwin.channel_native.montecarlo.basic.metadata import (
    make_solver_metadata as basic_metadata,
)
from witwin.channel_native.montecarlo.bdpt import Config as BdptConfig
from witwin.channel_native.montecarlo.bdpt.metadata import (
    make_solver_metadata as bdpt_metadata,
)
from witwin.channel_native.path import Config as PathConfig
from witwin.channel_native.path.solver import _metadata as path_metadata


_CONFIG_TYPES = (PathConfig, DeterministicConfig, BasicConfig, BdptConfig)


def test_public_ad_capability_is_primal_only_for_every_solver():
    manifest = capabilities()

    assert manifest["supports_ad"] is False
    assert manifest["ad_contract"] == {
        "decision": "primal_only_first_replacement",
        "public_modes": ["none"],
        "fixed_topology_jvp": False,
        "fixed_topology_vjp": False,
        "visibility_discontinuity_estimator": False,
        "experimental_low_level_primitives": [
            "mc_los_path_gain_backward",
            "mc_los_path_gain_jvp",
        ],
    }
    for solver in manifest["solvers"].values():
        assert solver["supports_ad"] is False
        assert solver["ad_modes"] == ["none"]


@pytest.mark.parametrize("config_type", (PathConfig, DeterministicConfig))
@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_fixed_topology_solvers_accept_ad_modes(config_type, ad_mode):
    assert config_type(ad_mode=ad_mode).ad_mode == ad_mode


@pytest.mark.parametrize("config_type", _CONFIG_TYPES)
@pytest.mark.parametrize("ad_mode", ["forward", "reverse", "grad"])
def test_every_solver_rejects_unknown_ad_modes(config_type, ad_mode):
    with pytest.raises((ValueError, RuntimeError), match="ad_mode"):
        config_type(ad_mode=ad_mode)


@pytest.mark.parametrize("config_type", (BasicConfig, BdptConfig))
@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_montecarlo_solvers_still_reject_ad_modes(config_type, ad_mode):
    with pytest.raises((ValueError, RuntimeError), match="ad_mode"):
        config_type(ad_mode=ad_mode)


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
            "uses_raydn_native": False,
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
        valid_contribution_count=1,
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
    # Plan 07 AD-1: the shared deterministic/path field seam calls the three
    # differentiable field entry points (LoS, reflection, transmission).
    assert audit["native_public_solver_ad_callers"] == 3
    assert len(audit["native_public_solver_ad_call_sites"]) == 3
    assert audit["legacy_reference_files_with_explicit_ad"] == 19
    assert audit["decision"] == "fixed_topology_material_frequency_ad_t1"
    # The capability manifest flip happens at the plan 07 completion gate.
    assert audit["supports_ad"] is False
