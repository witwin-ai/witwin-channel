import pytest
import torch

from tests.support.scenes import wedge_diffraction_scene
from witwin.channel_native import ReceiverGrid
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.montecarlo.bdpt import Config, solve
from witwin.channel_native.montecarlo.bdpt import solver as bdpt_solver


def _grid() -> ReceiverGrid:
    return ReceiverGrid(
        origin=torch.tensor([3.0, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def test_bdpt_single_wedge_diffraction_returns_finite_native_component_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    result = solve(wedge_diffraction_scene().add(_grid()), Config(samples=512, seed=7, components={"diffraction"}))

    assert result.component_maps is not None
    diffraction = result.component_maps["diffraction"]
    assert diffraction.shape == (1, 4, 4)
    assert torch.isfinite(diffraction).all()
    assert torch.count_nonzero(diffraction > 0.0).item() > 0
    assert result.component_power["diffraction"].item() > 0.0
    torch.testing.assert_close(result.component_power["diffraction"], diffraction.sum(), rtol=1e-5, atol=1e-8)
    assert result.metadata["components"]["diffraction"] == "enabled"


def test_bdpt_single_wedge_diffraction_does_not_use_path_block_sampler():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    assert not hasattr(bdpt_solver, "bdpt_sample_path_block")

    result = solve(wedge_diffraction_scene().add(_grid()), Config(samples=512, seed=7, components={"diffraction"}))

    assert result.component_power["diffraction"].item() > 0.0


def test_bdpt_single_wedge_diffraction_fixed_seed_is_stable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction reproducibility")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    scene = wedge_diffraction_scene().add(_grid())
    first = solve(scene, Config(samples=512, seed=7, components={"diffraction"}))
    second = solve(scene, Config(samples=512, seed=7, components={"diffraction"}))
    changed = solve(scene, Config(samples=512, seed=8, components={"diffraction"}))

    torch.testing.assert_close(first.component_maps["diffraction"], second.component_maps["diffraction"])
    assert not torch.equal(first.component_maps["diffraction"], changed.component_maps["diffraction"])


def test_bdpt_single_wedge_diffraction_uses_original_direct_keller_split():
    assert bdpt_solver._diffraction_sample_split(16, mis="power_heuristic") == (6, 5, 0)
    assert bdpt_solver._diffraction_sample_split(17, mis="balance") == (6, 6, 0)
    assert bdpt_solver._diffraction_sample_split(512, mis="power_heuristic") == (171, 171, 0)
    assert bdpt_solver._diffraction_sample_split(16, mis="none") == (16, 0, 0)


@pytest.mark.parametrize(
    ("samples", "relative_tolerance"),
    [
        (4096, 0.35),
        (16384, 0.20),
    ],
)
def test_bdpt_single_wedge_point_diffraction_converges_to_maintained_reference(samples, relative_tolerance):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction convergence")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    result = solve(
        wedge_diffraction_scene(),
        Config(samples=samples, seed=7, components={"diffraction"}, receiver_strategy="point_sphere"),
    )

    observed = result.component_power["diffraction"].detach().cpu()
    # Maintained reference for the single-wedge point-receiver estimate. The
    # historical value 1.25e-04 fossilized a 2x double count of the identical
    # direct/keller strategies (strategy_count=1 gave each full MIS weight);
    # 6.25e-05 still carried the duplicate half-plane record for the shared
    # wedge edge (audit D-6), now merged into one 3*pi/2 wedge state.
    reference = torch.tensor(4.66e-05, dtype=observed.dtype)
    torch.testing.assert_close(observed, reference, rtol=relative_tolerance, atol=1.0e-8)


def test_bdpt_grid_diffraction_power_is_additive_over_disjoint_wedges():
    """Guards audit MC-2: with the round-robin lane mapping the per-lane edge
    measure must scale by the state count, otherwise adding a second wedge
    halves each wedge's contribution (1/S underestimate)."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    from witwin.channel_native import Scene, Structure, Transmitter
    from witwin.channel_native.core.materials import PerfectConductor

    def wedge_pair(offset_z: float, tag: str) -> list[Structure]:
        shift = torch.tensor([0.0, 0.0, offset_z])
        face_a = Structure(
            vertices=torch.tensor([[2.0, 0.0, -1.0], [2.0, 0.0, 2.0], [2.0, 2.0, -1.0]]) + shift,
            faces=torch.tensor([[0, 1, 2]]),
            material=PerfectConductor(),
            name=f"wedge-a-{tag}",
            surface_id=2,
        )
        face_b = Structure(
            vertices=torch.tensor([[2.0, 0.0, -1.0], [2.0, 0.0, 2.0], [4.0, 0.0, -1.0]]) + shift,
            faces=torch.tensor([[0, 2, 1]]),
            material=PerfectConductor(),
            name=f"wedge-b-{tag}",
            surface_id=3,
        )
        return [face_a, face_b]

    def scene_with(structures: list[Structure]) -> Scene:
        return Scene(
            structures=structures,
            transmitters=[Transmitter(position=torch.tensor([0.0, -1.0, 0.5]))],
            receivers=[],
            frequency=3.0e9,
        ).add(_grid())

    config = Config(samples=16384, seed=7, components={"diffraction"})
    near = solve(scene_with(wedge_pair(0.0, "near")), config)
    far = solve(scene_with(wedge_pair(30.0, "far")), config)
    both = solve(scene_with(wedge_pair(0.0, "near") + wedge_pair(30.0, "far")), config)

    separate = near.component_power["diffraction"].item() + far.component_power["diffraction"].item()
    combined = both.component_power["diffraction"].item()
    assert separate > 0.0
    assert combined == pytest.approx(separate, rel=0.15)


def test_bdpt_grid_diffraction_is_seed_stable():
    """Guards audit DF-6: without pdf compensation the Keller sampler had
    unbounded variance (20x swings across seeds at this sample count)."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    scene = wedge_diffraction_scene().add(_grid())
    values = [
        solve(scene, Config(samples=8192, seed=seed, components={"diffraction"}))
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
    )
    assert result.metadata["mis"] == "none"
    assert bool(torch.isfinite(result.path_gain).all())


def test_bdpt_diffraction_point_receiver_returns_native_component_without_path_block_fallback():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native diffraction is not built")

    assert not hasattr(bdpt_solver, "bdpt_sample_path_block")

    result = solve(
        wedge_diffraction_scene(),
        Config(samples=256, seed=7, components={"diffraction"}, receiver_strategy="point_sphere"),
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
