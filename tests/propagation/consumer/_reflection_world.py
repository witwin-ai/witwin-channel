# Copyright Xingyu Chen.
# Shared worlds for the fixed-topology reflection consumer tests.

"""Shared worlds for the fixed-topology reflection consumer tests."""

from __future__ import annotations

import torch

from witwin.channel.propagation.consumer import (
    EndpointBatch,
    PropagationRequest,
    evaluate,
)
from witwin.channel.scene import compile
from witwin.core import PhaseScreen, PhysicalMaterial, Scene

from tests.support.core_world import make_mesh_structure
from tests.support.scenes import rough_wall_structure


FREQUENCY_HZ = 1.0e9
WALL_X_M = 2.0
BACK_WALL_X_M = -2.0

# u lies in the incidence plane (the xy plane) after projection, v is the
# out-of-plane axis, so a single mirror maps them to the TM and TE responses.
REFERENCE_BASIS = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# Two different rotations of the reference frame about the propagation axis.
# The published operator is then sink-rotated on its rows and source-rotated
# on its columns, which makes it genuinely non-symmetric: an implementation
# that swapped the two indices publishes a different matrix.
SOURCE_BASIS_25DEG = ((0.0, 0.9063078, 0.4226183), (0.0, -0.4226183, 0.9063078))
SINK_BASIS_MINUS_40DEG = ((0.0, 0.7660444, -0.6427876), (0.0, 0.6427876, 0.7660444))


def _cuda(values) -> torch.Tensor:
    return torch.tensor(values, device="cuda", dtype=torch.float32)


def smooth_wall_scene(*, rms_height_m: float = 0.0, frequency_hz: float = FREQUENCY_HZ):
    scene = Scene(
        structures=(
            rough_wall_structure(
                WALL_X_M, rms_height_m=rms_height_m, corr_length_m=0.1
            ),
        )
    )
    return compile(scene, reference_frequency_hz=frequency_hz)


def flat_phase_screen_wall_scene(*, frequency_hz: float = FREQUENCY_HZ):
    """Geometrically smooth wall carrying a realization_coherent screen.

 ``rms_height_m == 0`` keeps ``scatter_model_id`` at 0, so this reaches the
 phase-screen branch of the scene gate rather than the roughness branch.
 """

    screen = PhaseScreen(
        height=torch.zeros(64, 64),
        height_scale_m=1.0e-9,
        realization_id=0,
        mode="realization_coherent",
    )
    scene = Scene(
        structures=(
            rough_wall_structure(
                WALL_X_M,
                rms_height_m=0.0,
                corr_length_m=0.1,
                phase_screen=screen,
                with_uv=True,
            ),
        )
    )
    return compile(scene, reference_frequency_hz=frequency_hz)


def two_wall_scene(*, frequency_hz: float = FREQUENCY_HZ):
    """Facing walls at x = +2 and x = -2, so depth-2 rows exist."""

    scene = Scene(
        structures=(
            rough_wall_structure(WALL_X_M, rms_height_m=0.0, corr_length_m=0.1),
            rough_wall_structure(
                BACK_WALL_X_M,
                rms_height_m=0.0,
                corr_length_m=0.1,
                name="back-wall",
                surface_id=2,
            ),
        )
    )
    return compile(scene, reference_frequency_hz=frequency_hz)


def occluder_scene(*, frequency_hz: float = FREQUENCY_HZ):
    """Reference wall plus a plate that blocks only a moved arrival leg.

 The plate spans ``y`` in ``[0.6, 1.4]`` at ``x = 1``. At the reference
 endpoints neither leg of the reflection nor the LoS segment reaches it; at
 the moved sink the arrival leg crosses it while the stationary point is
 still well inside the wall facet, which is the occlusion branch of row
 validity rather than the off-facet branch.
 """

    plate = make_mesh_structure(
        vertices=torch.tensor(
            [
                [1.0, 0.6, -0.5],
                [1.0, 1.4, -0.5],
                [1.0, 0.6, 0.5],
                [1.0, 1.4, 0.5],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32),
        material=PhysicalMaterial(eps_r=4.0, sigma_e=0.01),
        name="occluder",
        surface_id=3,
    )
    scene = Scene(
        structures=(
            rough_wall_structure(WALL_X_M, rms_height_m=0.0, corr_length_m=0.1),
            plate,
        )
    )
    return compile(scene, reference_frequency_hz=frequency_hz)


def los_blocker_scene(*, frequency_hz: float = FREQUENCY_HZ):
    """Reference wall plus a plate straddling the line of sight."""

    plate = make_mesh_structure(
        vertices=torch.tensor(
            [
                [-0.5, 0.0, -0.5],
                [0.5, 0.0, -0.5],
                [-0.5, 0.0, 0.5],
                [0.5, 0.0, 0.5],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32),
        material=PhysicalMaterial(eps_r=4.0, sigma_e=0.01),
        name="los-blocker",
        surface_id=5,
    )
    scene = Scene(
        structures=(
            rough_wall_structure(WALL_X_M, rms_height_m=0.0, corr_length_m=0.1),
            plate,
        )
    )
    return compile(scene, reference_frequency_hz=frequency_hz)


def _basis(values, count: int) -> torch.Tensor:
    return _cuda([values]).expand(count, 2, 3).contiguous()


def endpoints(
    *, source_positions: torch.Tensor | None = None, sink_positions: torch.Tensor | None = None,
    with_basis: bool = True, source_basis=REFERENCE_BASIS, sink_basis=REFERENCE_BASIS,
    source_ids: tuple[int, ...] = (101,), sink_ids: tuple[int, ...] = (707,),
) -> tuple[EndpointBatch, EndpointBatch]:
    source_count = len(source_ids)
    sink_count = len(sink_ids)
    sources = EndpointBatch(
        stable_ids=torch.tensor(source_ids, device="cuda", dtype=torch.int64),
        positions_m=(
            source_positions
            if source_positions is not None
            else _cuda([[0.0, -0.5, 0.0]])
        ),
        polarizations=_cuda([[0.0, 0.0, 1.0]]).expand(source_count, 3).contiguous(),
        polarization_basis=(
            _basis(source_basis, source_count) if with_basis else None
        ),
        powers_w=torch.ones((source_count,), device="cuda"),
    )
    sinks = EndpointBatch(
        stable_ids=torch.tensor(sink_ids, device="cuda", dtype=torch.int64),
        positions_m=(
            sink_positions
            if sink_positions is not None
            else _cuda([[0.0, 0.5, 0.0]])
        ),
        polarizations=_cuda([[0.0, 0.0, 1.0]]).expand(sink_count, 3).contiguous(),
        polarization_basis=_basis(sink_basis, sink_count) if with_basis else None,
    )
    return sources, sinks


def rotated_endpoints(
    *, source_positions: torch.Tensor | None = None, sink_positions: torch.Tensor | None = None,
) -> tuple[EndpointBatch, EndpointBatch]:
    """Reference geometry read out in two differently rotated frames."""

    return endpoints(
        source_positions=source_positions,
        sink_positions=sink_positions,
        source_basis=SOURCE_BASIS_25DEG,
        sink_basis=SINK_BASIS_MINUS_40DEG,
    )


def multi_endpoints() -> tuple[EndpointBatch, EndpointBatch]:
    """Two sources and three sinks, none of them interchangeable.

 Every position is off-axis in a different way, so a source/sink index
 swap, a lost bucket scatter-back, or a broken pair segmentation changes
 the answer instead of permuting equal rows.
 """

    return endpoints(
        source_positions=_cuda([[0.0, -0.5, 0.0], [0.1, -0.7, 0.2]]),
        sink_positions=_cuda(
            [[0.0, 0.5, 0.0], [-0.2, 0.6, -0.1], [0.15, 0.9, 0.3]]
        ),
        source_basis=SOURCE_BASIS_25DEG,
        sink_basis=SINK_BASIS_MINUS_40DEG,
        source_ids=(101, 102),
        sink_ids=(707, 708, 709),
    )


def discover(
    compiled, sources: EndpointBatch, sinks: EndpointBatch, *, response: str = "scalar_transport",
    components: frozenset[str] = frozenset({"los", "reflection"}), max_depth: int = 1,
    ad_mode: str = "none",
):
    return evaluate(
        compiled,
        PropagationRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            components=components,
            max_depth=max_depth,
            response=response,
            topology_mode="discover",
            ad_mode=ad_mode,
        ),
    )


class DeviceReadCounter:
    """Count host reads of CUDA tensors performed inside a block.

 The published validation budget is a self-reported constant. This measures
 the thing the constant claims: how many times the prepared route actually
 pulls a device value to the host. It is the only way a regression that
 adds a second count read can fail a test.
 """

    _METHODS = ("item", "tolist", "cpu", "numpy", "__bool__")

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def __enter__(self) -> DeviceReadCounter:
        self._originals = {}
        for name in self._METHODS:
            original = getattr(torch.Tensor, name)
            self._originals[name] = original

            def wrapper(tensor, *args, _name=name, _original=original, **kwargs):
                if tensor.is_cuda:
                    self.counts[_name] = self.counts.get(_name, 0) + 1
                return _original(tensor, *args, **kwargs)

            setattr(torch.Tensor, name, wrapper)
        return self

    def __exit__(self, *_exc: object) -> None:
        for name, original in self._originals.items():
            setattr(torch.Tensor, name, original)

    @property
    def total(self) -> int:
        return sum(self.counts.values())