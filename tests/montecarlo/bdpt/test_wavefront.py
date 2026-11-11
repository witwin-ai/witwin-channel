# Copyright Xingyu Chen.
# Tests the fused BDPT wavefront sampler, counter-based draws, one-call oracle and receivers.

import pytest
import torch

import witwin.channel.montecarlo.bdpt as bdpt
from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_transmitter,
)
from witwin.channel.deployment import build_info
from witwin.channel.interactions.transmission import event_uniforms
from witwin.channel.kernels.montecarlo import bdpt_counter_uniforms
from witwin.core import MaterialLayer, PhysicalMaterial, Scene

_FREQUENCY = 3.0e9

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")


def _require_native() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native transmission is not built")


def _wall(material, *, x: float, surface_id: int):
    return make_mesh_structure(
        vertices=torch.tensor(
            [[x, -4.0, -4.0], [x, 4.0, -4.0], [x, -4.0, 4.0], [x, 4.0, 4.0]]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=material,
        name=f"wall-{surface_id}",
        surface_id=surface_id,
    )


def _lossy() -> PhysicalMaterial:
    return PhysicalMaterial(
        layers=(MaterialLayer(thickness_m=0.1, eps_r=4.0, sigma_e=0.05),), name="lossy-wall"
    )


def _receivers():
    return [
        make_receiver(position=torch.tensor([5.0, y, z]))
        for y, z in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, -1.0))
    ]


def _mixed_scene(transmitters=1) -> Scene:
    """A lossy front wall and a PEC back wall: transmit->reflect mixed chains."""

    sources = [
        make_transmitter(position=torch.tensor([0.0, 0.3 * index, 0.0]))
        for index in range(transmitters)
    ]
    return Scene(
        structures=[
            _wall(_lossy(), x=2.5, surface_id=1),
            _wall(PhysicalMaterial.perfect_conductor(), x=6.0, surface_id=2),
        ],
        endpoints=[*sources, *_receivers()],
    )


def _solve(scene, config, monkeypatch=None, *, composed=False):
    if composed:
        assert monkeypatch is not None
        monkeypatch.setattr(bdpt, "_wavefront_eligible", lambda config, runtimes: False)
    try:
        return bdpt.solve(scene, config, reference_frequency_hz=_FREQUENCY)
    finally:
        if composed:
            monkeypatch.undo()


@cuda
def test_counter_uniforms_depend_only_on_the_global_sample():
    reference = torch.empty((0,), device="cuda")
    wide = bdpt_counter_uniforms(reference, 64, seed=9, first=0, depth=2, stream="event_selection")
    window = bdpt_counter_uniforms(
        reference, 16, seed=9, first=20, depth=2, stream="event_selection"
    )
    assert torch.equal(wide[20:36], window)
    assert bool(((wide >= 0.0) & (wide < 1.0)).all())
    # Sample i of transmitter t is global sample t * count + i.
    assert torch.equal(
        event_uniforms(32, seed=9, tx_index=1, depth=2, device=torch.device("cuda")), wide[32:]
    )
    other_depth = bdpt_counter_uniforms(
        reference, 64, seed=9, first=0, depth=3, stream="event_selection"
    )
    other_seed = bdpt_counter_uniforms(
        reference, 64, seed=10, first=0, depth=2, stream="event_selection"
    )
    assert not torch.equal(wide, other_depth)
    assert not torch.equal(wide, other_seed)
    directions = bdpt_counter_uniforms(
        reference, 64, seed=9, first=0, depth=2, stream="scatter_direction"
    )
    assert directions.shape == (64, 2)
    assert not torch.equal(directions[:, 0], wide)


@cuda
def test_wavefront_matches_event_composed_sampler(monkeypatch):
    _require_native()
    scene = _mixed_scene(transmitters=2)
    config = bdpt.Config(samples=4096, seed=11, max_depth=3, components={"transmission"})
    wavefront = _solve(scene, config)
    composed = _solve(scene, config, monkeypatch, composed=True)

    assert wavefront.metadata["sampler"] == "wavefront"
    assert composed.metadata["sampler"] == "event_composed"
    assert float(wavefront.component_power["transmission"]) > 0.0
    torch.testing.assert_close(wavefront.path_gain, composed.path_gain, rtol=1.0e-6, atol=0.0)
    assert (
        wavefront.metadata["transmission"]["event_counts"]
        == composed.metadata["transmission"]["event_counts"]
    )
    assert (
        wavefront.metadata["valid_contribution_count"]
        == composed.metadata["valid_contribution_count"]
    )


@cuda
def test_wavefront_is_chunk_and_tile_invariant(monkeypatch):
    _require_native()
    scene = _mixed_scene(transmitters=2)
    config = bdpt.Config(samples=3000, seed=4, max_depth=2, components={"transmission"})
    reference = _solve(scene, config)
    # Chunks that split a transmitter's samples, and one receiver per tile.
    monkeypatch.setattr(bdpt, "_WAVEFRONT_CHUNK_SAMPLES", 1000)
    monkeypatch.setattr(bdpt, "_WAVEFRONT_PAIR_TILE", 1500)
    tiled = _solve(scene, config)
    torch.testing.assert_close(tiled.path_gain, reference.path_gain, rtol=1.0e-6, atol=0.0)
    assert tiled.metadata["transmission"] == reference.metadata["transmission"]


@cuda
def test_wavefront_seed_changes_the_estimate_but_not_its_reproducibility():
    _require_native()
    scene = _mixed_scene()
    config = bdpt.Config(samples=2048, seed=3, max_depth=2, components={"transmission"})
    first = _solve(scene, config)
    again = _solve(scene, config)
    other = _solve(
        scene, bdpt.Config(samples=2048, seed=4, max_depth=2, components={"transmission"})
    )
    assert torch.equal(first.path_gain, again.path_gain)
    assert not torch.equal(first.path_gain, other.path_gain)


def test_wavefront_workspace_estimate_fits_the_default_guard():
    config = bdpt.Config(samples=1 << 20, max_depth=2, components={"transmission"})
    estimate = bdpt._estimate_workspace_bytes(config, tx_count=1, grid_cells=0, rx_count=4)
    assert estimate < 100 * (1 << 20)
    default = bdpt.Config(
        samples=1 << 20,
        max_depth=2,
        components={"los", "reflection", "diffraction", "transmission"},
    )
    assert bdpt._estimate_workspace_bytes(
        default, tx_count=1, grid_cells=0, rx_count=4
    ) < 100 * (1 << 20)
    rows = bdpt.Config(
        samples=1 << 20, max_depth=2, components={"transmission"}, export_paths=True
    )
    assert bdpt._estimate_workspace_bytes(rows, tx_count=1, grid_cells=0, rx_count=4) > estimate


@cuda
def test_enumerated_oracle_is_called_once(monkeypatch):
    _require_native()
    calls = []
    original = bdpt.evaluate_enumerated_paths

    def spy(scene, options, **kwargs):
        calls.append(frozenset(options.components))
        return original(scene, options, **kwargs)

    monkeypatch.setattr(bdpt, "evaluate_enumerated_paths", spy)
    scene = Scene(
        structures=[_wall(_lossy(), x=2.5, surface_id=1)],
        endpoints=[make_transmitter(position=torch.tensor([0.0, 0.0, 0.0])), *_receivers()],
    )
    components = {"reflection", "diffraction", "transmission"}
    together = bdpt.solve(
        scene,
        bdpt.Config(samples=256, max_depth=1, components=components),
        reference_frequency_hz=_FREQUENCY,
    )
    assert calls == [frozenset(components)]
    for component in sorted(components):
        alone = bdpt.solve(
            scene,
            bdpt.Config(samples=256, max_depth=1, components={component}),
            reference_frequency_hz=_FREQUENCY,
        )
        torch.testing.assert_close(
            together.component_power[component],
            alone.component_power[component],
            rtol=1.0e-6,
            atol=0.0,
        )

