# Copyright Xingyu Chen.
# Tests diffraction single wedge.

import pytest
import torch

from witwin.core import PhysicalMaterial, ReceiverGrid, Scene, Structure
from tests.support.core_world import (
    make_mesh_structure,
    make_receiver_grid,
    make_transmitter,
)
from tests.support.scenes import wedge_diffraction_scene
from witwin.channel.deployment import build_info
from witwin.channel.montecarlo.bdpt import Config, solve
from witwin.channel.montecarlo import bdpt as bdpt_solver


def _grid() -> ReceiverGrid:
    return make_receiver_grid(
        origin=torch.tensor([3.0, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def _with_grid(scene: Scene) -> Scene:
    return scene.with_endpoints(
        (
            *tuple(endpoint for endpoint in scene.endpoints if endpoint.role == "tx"),
            _grid(),
        )
    )


def test_bdpt_single_wedge_diffraction_returns_finite_native_component_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    result = solve(
        _with_grid(wedge_diffraction_scene()),
        Config(samples=512, seed=7, components={"diffraction"}),
        reference_frequency_hz=3.0e9,
    )

    assert result.component_maps is not None
    diffraction = result.component_maps["diffraction"]
    assert diffraction.shape == (1, 4, 4)
    assert torch.isfinite(diffraction).all()
    assert torch.count_nonzero(diffraction > 0.0).item() > 0
    assert result.component_power["diffraction"].item() > 0.0
    torch.testing.assert_close(
        result.component_power["diffraction"], diffraction.sum(), rtol=1e-5, atol=1e-8
    )
    assert result.metadata["components"]["diffraction"] == "enabled"


def test_bdpt_single_wedge_diffraction_does_not_use_path_block_sampler():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    assert not hasattr(bdpt_solver, "bdpt_sample_path_block")

    result = solve(
        _with_grid(wedge_diffraction_scene()),
        Config(samples=512, seed=7, components={"diffraction"}),
        reference_frequency_hz=3.0e9,
    )

    assert result.component_power["diffraction"].item() > 0.0


def test_bdpt_single_wedge_diffraction_fixed_seed_is_stable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction reproducibility")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    scene = _with_grid(wedge_diffraction_scene())
    first = solve(
        scene,
        Config(samples=512, seed=7, components={"diffraction"}),
        reference_frequency_hz=3.0e9,
    )
    second = solve(
        scene,
        Config(samples=512, seed=7, components={"diffraction"}),
        reference_frequency_hz=3.0e9,
    )
    changed = solve(
        scene,
        Config(samples=512, seed=8, components={"diffraction"}),
        reference_frequency_hz=3.0e9,
    )

    # BDPT diffraction: standalone diffraction routes through the deterministic
    # enumerated engine, so the estimate is seed-invariant. Distinct seeds must
    # produce the identical map, unlike the retired stochastic Keller sampler.
    torch.testing.assert_close(
        first.component_maps["diffraction"], second.component_maps["diffraction"]
    )
    torch.testing.assert_close(
        first.component_maps["diffraction"], changed.component_maps["diffraction"]
    )


def test_bdpt_single_wedge_point_diffraction_matches_deterministic_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    from witwin.channel.deterministic import Config as DeterministicConfig
    from witwin.channel.deterministic import solve as deterministic_solve

    scene = wedge_diffraction_scene()
    observed = (
        solve(
            scene,
            Config(
                samples=512,
                seed=7,
                components={"diffraction"},
                receiver_strategy="point_sphere",
            ),
            reference_frequency_hz=3.0e9,
        )
        .component_power["diffraction"]
        .detach()
        .sum()
    )
    # BDPT diffraction: BDPT standalone diffraction now consumes the same first-order UTD
    # enumerated evaluation as the deterministic solver, so the point-receiver
    # estimate reproduces the deterministic reference. The retired crude power
    # heuristic over-counted this fixture by ~2175x (fossilized as 4.66e-05).
    reference = (
        deterministic_solve(
            scene,
            DeterministicConfig(
                components={"diffraction"}, max_depth=1, coherent=False
            ),
            reference_frequency_hz=3.0e9,
        )
        .component_power["diffraction"]
        .detach()
        .sum()
    )
    torch.testing.assert_close(observed, reference, rtol=1.0e-4, atol=1.0e-12)


def test_bdpt_grid_diffraction_power_is_additive_over_disjoint_wedges():
    """With the round-robin lane mapping the per-lane edge
 measure must scale by the state count, otherwise adding a second wedge
 halves each wedge's contribution (1/S underestimate)."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    def wedge_pair(offset_z: float, tag: str) -> list[Structure]:
        shift = torch.tensor([0.0, 0.0, offset_z])
        face_a = make_mesh_structure(
            vertices=torch.tensor([[2.0, 0.0, -1.0], [2.0, 0.0, 2.0], [2.0, 2.0, -1.0]])
            + shift,
            faces=torch.tensor([[0, 1, 2]]),
            material=PhysicalMaterial.perfect_conductor(),
            name=f"wedge-a-{tag}",
        )
        face_b = make_mesh_structure(
            vertices=torch.tensor([[2.0, 0.0, -1.0], [2.0, 0.0, 2.0], [4.0, 0.0, -1.0]])
            + shift,
            faces=torch.tensor([[0, 2, 1]]),
            material=PhysicalMaterial.perfect_conductor(),
            name=f"wedge-b-{tag}",
        )
        return [face_a, face_b]

    def scene_with(structures: list[Structure]) -> Scene:
        return Scene(
            structures=structures,
            endpoints=[
                make_transmitter(position=torch.tensor([0.0, -1.0, 0.5])),
                _grid(),
            ],
        )

    config = Config(samples=16384, seed=7, components={"diffraction"})
    near = solve(
        scene_with(wedge_pair(0.0, "near")), config, reference_frequency_hz=3.0e9
    )
    far = solve(
        scene_with(wedge_pair(30.0, "far")), config, reference_frequency_hz=3.0e9
    )
    both = solve(
        scene_with(wedge_pair(0.0, "near") + wedge_pair(30.0, "far")),
        config,
        reference_frequency_hz=3.0e9,
    )

    separate = (
        near.component_power["diffraction"].item()
        + far.component_power["diffraction"].item()
    )
    combined = both.component_power["diffraction"].item()
    assert separate > 0.0
    assert combined == pytest.approx(separate, rel=0.15)


def test_bdpt_grid_diffraction_is_seed_stable():
    """Without pdf compensation the Keller sampler had
 unbounded variance (20x swings across seeds at this sample count)."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    scene = _with_grid(wedge_diffraction_scene())
    values = [
        solve(
            scene,
            Config(samples=8192, seed=seed, components={"diffraction"}),
            reference_frequency_hz=3.0e9,
        )
        .component_power["diffraction"]
        .item()
        for seed in (1, 7, 21)
    ]
    mean = sum(values) / len(values)
    spread = (max(values) - min(values)) / mean
    assert mean > 0.0
    assert spread < 0.3


def test_bdpt_diffraction_mis_none_uses_one_unbiased_strategy():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")

    result = solve(
        wedge_diffraction_scene(),
        Config(
            samples=256,
            seed=7,
            components={"diffraction"},
            receiver_strategy="point_sphere",
            mis="none",
        ),
        reference_frequency_hz=3.0e9,
    )
    assert result.metadata["mis"] == "none"
    assert bool(torch.isfinite(result.path_gain).all())


def test_bdpt_diffraction_point_receiver_returns_native_component_without_path_block_fallback():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    assert not hasattr(bdpt_solver, "bdpt_sample_path_block")

    result = solve(
        wedge_diffraction_scene(),
        Config(
            samples=256,
            seed=7,
            components={"diffraction"},
            receiver_strategy="point_sphere",
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.component_maps is None
    assert result.path_gain.shape == (1, 1)
    assert torch.isfinite(result.path_gain).all()
    assert result.component_power["diffraction"].item() > 0.0
    torch.testing.assert_close(
        result.path_gain,
        result.component_power["diffraction"].reshape(1, 1),
        rtol=1e-5,
        atol=1e-8,
    )