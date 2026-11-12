# Copyright Xingyu Chen.
# Tests the fused BDPT wavefront sampler, counter-based draws, one-call oracle and receivers.

import pytest
import torch

import witwin.channel.experimental.bdpt as bdpt
from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_transmitter,
)
from witwin.channel.deployment import build_info
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


def _solve(scene, config, *, frequency=_FREQUENCY):
    return bdpt.solve(scene, config, reference_frequency_hz=frequency)


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

    def power(subset: set[str], component: str) -> torch.Tensor:
        return bdpt.solve(
            scene, bdpt.Config(samples=256, max_depth=1, components=subset),
            reference_frequency_hz=_FREQUENCY,
        ).component_power[component]

    for component in sorted(components):
        # With transmission paths in the solve the diffraction's incident terms compensate only the
        # step those paths leave, so the joint solve's diffraction equals that of a solve that also
        # carries transmission.
        subset = {"diffraction", "transmission"} if component == "diffraction" else {component}
        torch.testing.assert_close(
            together.component_power[component], power(subset, component), rtol=1.0e-6, atol=0.0
        )
    # The weighting is active: diffraction alone, without transmission paths, differs.
    alone = power({"diffraction"}, "diffraction")
    assert not torch.allclose(together.component_power["diffraction"], alone, rtol=1.0e-3, atol=0.0)


