"""The multi-endpoint fixture world shared by the Phase-7 slot-batching tests.

One concrete wall at ``x = 4`` spanning ``y in [-1.2, 1.2]`` and
``z in [-3, 3]``, two sources and two sinks, at 77 GHz. Discovery over
``{los, reflection}`` at depth 1 publishes exactly three rows with
``pair_offsets = [0, 2, 2, 3, 3]``: the second source publishes no row at all,
which is why an endpoint count can never be inferred from the largest index a
topology carries.
"""

from __future__ import annotations

import torch

from witwin.channel.propagation.consumer import (
    EndpointBatch,
    FixedTopologyRequest,
    PropagationRequest,
    evaluate,
    prepare_fixed_topology,
    reevaluate,
)
from witwin.channel.scene import compile as compile_scene
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure


FREQUENCY_HZ = 77.0e9
WALL_X_M = 4.0
WALL_VERTICES = (
    (WALL_X_M, -1.2, -3.0),
    (WALL_X_M, 1.2, -3.0),
    (WALL_X_M, 1.2, 3.0),
    (WALL_X_M, -1.2, 3.0),
)
WALL_FACES = ((0, 1, 2), (0, 2, 3))

SOURCE_POSITIONS = ((0.0, 0.0, 0.0), (6.0, -1.0, 0.0))
SOURCE_IDS = (10, 11)
SINK_POSITIONS = ((2.0, 0.6, 0.0), (2.0, 2.4, 0.0))
SINK_IDS = (20, 21)
SOURCE_POWER_W = 0.01

# Rows discovered on this world, in frozen order.
FROZEN_ROW_COUNT = 3
FROZEN_PAIR_OFFSETS = [0, 2, 2, 3, 3]
LOS_ROW = 0
REFLECTION_ROW = 1


def multi_endpoint_scene() -> Scene:
    # recenter=False: witwin.core.Mesh otherwise silently rewrites the authored
    # world coordinates and the wall stops being where this world says it is.
    mesh = Mesh(
        vertices=torch.tensor(WALL_VERTICES, dtype=torch.float32),
        faces=torch.tensor(WALL_FACES, dtype=torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )
    return Scene(
        structures=(
            Structure(
                geometry=mesh,
                material=PhysicalMaterial(
                    name="concrete", eps_r=5.24, sigma_e=0.0462
                ),
                structure_id=1,
                material_id=1,
                assignment_id=1,
                surface_id=1,
            ),
        )
    )


def compiled_world():
    return compile_scene(multi_endpoint_scene(), reference_frequency_hz=FREQUENCY_HZ)


def cuda_positions(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device="cuda")


# u lies in the incidence plane after projection and v is the out-of-plane
# axis, so one mirror maps them to the TM and TE responses.
REFERENCE_BASIS = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def source_batch(
    positions: torch.Tensor,
    *,
    power_w: float = SOURCE_POWER_W,
    with_basis: bool = False,
) -> EndpointBatch:
    return _batch(positions, SOURCE_IDS, power_w=power_w, with_basis=with_basis)


def sink_batch(
    positions: torch.Tensor, *, with_basis: bool = False
) -> EndpointBatch:
    return _batch(positions, SINK_IDS, power_w=None, with_basis=with_basis)


def _batch(
    positions: torch.Tensor,
    ids: tuple[int, ...],
    *,
    power_w: float | None,
    with_basis: bool = False,
) -> EndpointBatch:
    device = torch.device("cuda")
    rows = int(positions.shape[0])
    slots, remainder = divmod(rows, len(ids))
    assert remainder == 0, "a stacked batch repeats the same endpoints per slot"
    return EndpointBatch(
        stable_ids=torch.tensor(ids, dtype=torch.int64, device=device).repeat(slots),
        positions_m=positions,
        polarizations=(
            torch.tensor([(0.0, 0.0, 1.0)], dtype=torch.float32, device=device)
            .expand(rows, 3)
            .contiguous()
        ),
        polarization_basis=(
            torch.tensor([REFERENCE_BASIS], dtype=torch.float32, device=device)
            .expand(rows, 2, 3)
            .contiguous()
            if with_basis
            else None
        ),
        powers_w=(
            None
            if power_w is None
            else torch.full((rows,), power_w, dtype=torch.float32, device=device)
        ),
    )


def stack_over_slots(base: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Slot-major stack of ``base`` displaced by one offset per slot.

    ``offsets`` is ``[T, 3]``; the result is ``[T * base.shape[0], 3]`` with
    slot ``t`` holding ``base + offsets[t]``.
    """

    return (base.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1, 3).contiguous()


def slot_offsets(slot_count: int, *, step: float = 0.01) -> torch.Tensor:
    """Distinct per-slot displacements along ``+y``, never all equal."""

    steps = torch.arange(slot_count, dtype=torch.float32, device="cuda") * step
    return torch.stack(
        (torch.zeros_like(steps), steps, torch.zeros_like(steps)), dim=1
    )


def discover(compiled, sources: EndpointBatch, sinks: EndpointBatch):
    return evaluate(
        compiled,
        PropagationRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            components=frozenset({"los", "reflection"}),
            max_depth=1,
            response="scalar_transport",
            topology_mode="discover",
            ad_mode="none",
        ),
    )


def frozen_topology(compiled):
    """Discover once at the reference endpoints and freeze the rows."""

    sources = source_batch(cuda_positions(SOURCE_POSITIONS))
    sinks = sink_batch(cuda_positions(SINK_POSITIONS))
    return prepare_fixed_topology(discover(compiled, sources, sinks).paths.topology)


def replay(
    compiled,
    topology,
    sources: EndpointBatch,
    sinks: EndpointBatch,
    *,
    slot_count: int = 1,
    response: str = "scalar_transport",
    ad_mode: str = "none",
    frequency_offsets_hz: tuple[float, ...] | None = None,
):
    return reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            topology=topology,
            response=response,
            ad_mode=ad_mode,
            slot_count=slot_count,
            frequency_offsets_hz=frequency_offsets_hz,
        ),
    )
