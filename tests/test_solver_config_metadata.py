# Copyright Xingyu Chen.
# Tests solver config metadata.

import importlib
from dataclasses import fields

import pytest
import torch

from tests.support.core_world import make_receiver_grid
from witwin.core import Scene
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
from witwin.channel.montecarlo.bdpt import solve as bdpt_solve


def test_deterministic_metadata_reports_effective_component_depths():
    metadata = deterministic_metadata(
        config=DeterministicConfig(max_depth=3, components={"los", "reflection"}),
        native_info={
            "uses_rayd_native": True,
            "uses_path_native": True,
            "cuda_available": True,
            "optix_available": True,
        },
        path_count=2,
        component_counts={"los": 1, "reflection": 1, "diffraction": 0},
        launch_count=2,
    )

    assert metadata["requested_max_depth"] == 3
    assert metadata["effective_max_depth"] == 3
    assert metadata["component_max_depth"]["reflection"] == 3
    assert set(metadata["requested_config"]) == {
        field.name for field in fields(DeterministicConfig)
    }


def test_deterministic_rejects_depth_above_public_capability_at_config_time():
    with pytest.raises(RuntimeError, match="max_depth <= 5"):
        DeterministicConfig(max_depth=6, components={"reflection"})


def test_basic_metadata_reports_requested_and_effective_config():
    metadata = basic_metadata(
        config=BasicConfig(max_depth=2, components={"reflection"}),
        path_count=1,
        contribution_capacity=1,
        reflection_available=True,
        diffraction_available=True,
    )

    assert metadata["requested_config"] == metadata["effective_config"]
    assert metadata["component_max_depth"] == {
        "los": -1,
        "reflection": 2,
        "diffraction": -1,
        "transmission": -1,
        "scattering": -1,
    }
    assert set(metadata["requested_config"]) == {
        field.name for field in fields(BasicConfig)
    }


@pytest.mark.parametrize("config_type", [BasicConfig, BdptConfig])
@pytest.mark.parametrize(
    "component", ["reflection", "diffraction", "transmission", "scattering"]
)
def test_montecarlo_configs_reject_zero_depth_scattering_before_solve(
    config_type, component
):
    with pytest.raises(RuntimeError, match="max_depth >= 1"):
        config_type(max_depth=0, components={component})


def test_bdpt_rejects_disabled_diffraction_and_accepts_single_strategy_without_mis():
    with pytest.raises(RuntimeError, match="max_diffraction_order"):
        BdptConfig(components={"diffraction"}, max_diffraction_order=0)
    assert BdptConfig(components={"diffraction"}, mis="none", samples=2).mis == "none"


def test_bdpt_rejects_grid_receiver_strategy_before_scene_build(monkeypatch):
    grid = make_receiver_grid(
        origin=torch.zeros(3),
        x_axis=torch.tensor([1.0, 0.0, 0.0]),
        y_axis=torch.tensor([0.0, 1.0, 0.0]),
        shape=(1, 1),
        spacing=(1.0, 1.0),
    )

    def fail_scene_build():
        raise AssertionError("rayd_scene must not run for invalid receiver config")

    scene = Scene(endpoints=[grid])
    bdpt_solver_module = importlib.import_module(
        "witwin.channel.montecarlo.bdpt"
    )
    monkeypatch.setattr(
        bdpt_solver_module,
        "compile_scene",
        lambda *args, **kwargs: fail_scene_build(),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="requires point receivers"):
        bdpt_solve(
            scene,
            BdptConfig(components={"los"}, receiver_strategy="point_sphere"),
            reference_frequency_hz=3.0e9,
        )


def test_bdpt_metadata_exposes_depth_clamp():
    config = BdptConfig(
        max_depth=5,
        max_light_depth=2,
        components={"reflection"},
    )
    metadata = bdpt_metadata(
        config=config,
        selected_accumulation_strategy="atomic",
        path_counts_by_strategy={"light": 1, "sensor": 1},
        valid_contribution_count=1,
        reflection_available=True,
        diffraction_available=True,
        cuda_available=True,
        optix_available=True,
        workspace_bytes=0,
        variance_enabled=False,
        launch_count=1,
        effective_max_depth=2,
    )

    assert metadata["requested_max_depth"] == 5
    assert metadata["effective_max_depth"] == 2
    assert metadata["effective_config"]["max_depth"] == 2
    assert metadata["component_max_depth"]["reflection"] == 2
    assert set(metadata["requested_config"]) == {
        field.name for field in fields(BdptConfig)
    }