"""Transmission: enumerated topology discovery and the shared event helpers.

One concept, one file. This module owns the enumerated straight-segment
transmission topology discovery and the specular-transmission event helpers the
Monte Carlo solvers share; the latter came from the former
``montecarlo/events/transmission.py``, which was never a Monte Carlo concept.

Each origin docstring is preserved verbatim.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from witwin.channel.scene import endpoints as scene_endpoints
from witwin.channel.kernels.geometry import (
    rayd_segment_penetration_ad,
    rayd_segment_penetration_forward,
)
from witwin.channel.propagation.penetration import (
    SegmentPenetrationPolicy,
    SegmentPenetrationResult,
)
from witwin.channel.propagation.topology.concatenate import _empty_path_block
from witwin.channel.propagation.topology.export import _ensure_topology_fields
from witwin.channel.kernels.topology import (
    enumerated_transmission_topology_pack,
)
from witwin.channel.runtime import (
    CapacityExecutionCounts,
    CapacityFailureState,
    CudaProfileMark,
    CudaProfileRange,
    cuda_profile_mark,
    profiled_cuda_range,
)

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene as Scene


def _pair_major_endpoints(
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Structurally expand endpoints as ``tx * rx_count + rx`` rows."""

    tx_count = int(tx_positions.shape[0])
    rx_count = int(rx_positions.shape[0])
    device = tx_positions.device
    tx_index = torch.arange(
        tx_count, device=device, dtype=torch.int64
    ).repeat_interleave(rx_count)
    rx_index = torch.arange(rx_count, device=device, dtype=torch.int64).repeat(tx_count)
    return (
        tx_positions.index_select(0, tx_index).contiguous(),
        rx_positions.index_select(0, rx_index).contiguous(),
    )


@profiled_cuda_range(CudaProfileRange.ENUMERATED_PENETRATION_DISCOVERY)
def _transmission_topology(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    max_depth: int,
    ad_mode: str,
    failure_state: CapacityFailureState | None,
) -> tuple[dict[str, torch.Tensor], int, CapacityExecutionCounts | None]:
    """Trace and pack one pair-major RayD penetration batch.

    Device-selected cardinality remains in the returned execution sidecar.
    Structural no-op scenes do not allocate a failure state or launch native
    work; every non-empty request shares the engine-owned failure state between
    RayD penetration and the component-5 topology pack.
    """

    device = tx_positions.device
    tx_count = int(tx_positions.shape[0])
    rx_count = int(rx_positions.shape[0])
    if not scene.structures or tx_count == 0 or rx_count == 0 or max_depth < 1:
        return _ensure_topology_fields(_empty_path_block(device)), 0, None
    if not isinstance(failure_state, CapacityFailureState):
        raise TypeError("non-empty transmission discovery requires failure_state")

    rayd = compiled.rayd
    if not rayd.available:
        raise RuntimeError(
            "deterministic transmission requires RayD native scene capability"
        )

    if ad_mode == "none":
        tx_geometry = tx_positions
        rx_geometry = rx_positions
        vertices = None
    else:
        vertices = scene_endpoints.scene_vertex_table(scene, compiled)
        tx_geometry = scene_endpoints.transmitter_positions_ad(
            scene, tx_positions, device=device
        )
        rx_geometry = scene_endpoints.receiver_positions_ad(
            scene, rx_positions, device=device
        )
    origins, targets = _pair_major_endpoints(tx_geometry, rx_geometry)

    penetration_args = {
        "input_active_any": True,
        "hit_capacity": int(max_depth),
        "policy": SegmentPenetrationPolicy.EnumeratedFullDistance,
        "scene_diagonal": compiled.enumerated_penetration_scene_diagonal_m,
        "failure_state": failure_state,
    }
    if ad_mode == "none":
        penetration = rayd_segment_penetration_forward(
            rayd,
            origins,
            targets,
            None,
            **penetration_args,
        )
    else:
        assert vertices is not None
        penetration = rayd_segment_penetration_ad(
            rayd,
            vertices,
            origins,
            targets,
            None,
            **penetration_args,
        )

    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    geometry_mode_id = compiled.materials.geometry_mode_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    topology = enumerated_transmission_topology_pack(
        penetration,
        face_material_id,
        geometry_mode_id,
        tx_count=tx_count,
        rx_count=rx_count,
    )
    capacity_block = topology.as_block()
    # Transitional ADR-027 activation: preserve the live compact row shape
    # until the ADR-029 atomic selector/gather switch. This device-selected
    # structural compaction is the remaining Phase D synchronization blocker;
    # no geometry or RF quantity is recomputed here.
    selected = torch.nonzero(topology.valid, as_tuple=False).reshape(-1)
    block = {
        name: value.index_select(0, selected).contiguous()
        for name, value in capacity_block.items()
    }
    return _ensure_topology_fields(block), 1, topology.execution


# -------------------------------------------------------------------------
# Shared Monte Carlo specular-transmission events (was
# montecarlo/events/transmission.py)
# -------------------------------------------------------------------------
#
# Shared specular-transmission helpers for the Monte Carlo solvers.
#
# Two evaluation contexts share the per-wall layer-stack algebra (implementation
# contract section 4):
#
# - Endpoint connection (``straight_transmission_chains``): one flattened RayD
#   target-inset batch exports ordered resident wall hits. MC Basic then applies
#   its native incident-polarized wall product; this shared event section owns no
#   second RF estimator.
# - Shooting (BDPT light-subpath continuation): handled by the native
#   ``bdpt_transmitted_light_subpath_state`` kernel with the exact lateral exit
#   offset; this section only supplies the seeded event-selection utilities.


_MIN_EPSILON_M = 1.0e-6
_RELATIVE_EPSILON = 1.0e-6
_EVENT_PROBABILITY_FLOOR = 0.05
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_MUL0 = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL1 = 0x94D049BB133111EB
_MASK64 = (1 << 64) - 1

_LAYER_CSR_FIELDS = (
    "layer_offset",
    "layer_count",
    "layer_thickness_m",
    "layer_eps_r",
    "layer_sigma_e",
    "layer_mu_r",
)


def layer_csr_view(material_bundle: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Material-level CSR layer tensors from a face_material_field_bundle."""

    return {name: material_bundle[name] for name in _LAYER_CSR_FIELDS}


def scene_diagonal_m(scene: Any) -> float:
    """Structure bounding-box diagonal for scale-aware ray epsilons."""

    minimum: torch.Tensor | None = None
    maximum: torch.Tensor | None = None
    for structure in scene.structures:
        vertices = structure.vertices
        low = vertices.amin(dim=0)
        high = vertices.amax(dim=0)
        minimum = low if minimum is None else torch.minimum(minimum, low)
        maximum = high if maximum is None else torch.maximum(maximum, high)
    if minimum is None or maximum is None:
        return 0.0
    return float((maximum - minimum).norm())


def scale_aware_epsilon(
    position: torch.Tensor, *, scene_diagonal: float
) -> torch.Tensor:
    """Per-row restart offset ``max(|p|*1e-6, diag*1e-6, 1e-6 m)`` (contract
    section 4)."""

    floor = max(_MIN_EPSILON_M, float(scene_diagonal) * _RELATIVE_EPSILON)
    return (position.abs().amax(dim=-1) * _RELATIVE_EPSILON).clamp_min(floor)


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_MUL0) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_MUL1) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def event_selection_seed(seed: int, tx_index: int, depth: int) -> int:
    """Deterministic per-(seed, tx, depth) generator seed via the native
    splitmix64 mixing pattern (matches bdpt_subpaths.cu)."""

    mixed = _splitmix64(int(seed))
    mixed = _splitmix64(mixed ^ (((int(tx_index) + 1) * 0xD1B54A32D192ED03) & _MASK64))
    mixed = _splitmix64(mixed ^ (((int(depth) + 1) * 0x8CB92BA72F3D8DD7) & _MASK64))
    return mixed & ((1 << 63) - 1)


def event_uniforms(
    count: int, *, seed: int, tx_index: int, depth: int, device: torch.device
) -> torch.Tensor:
    """Reproducible per-sample uniforms for reflect/transmit event selection."""

    generator = torch.Generator(device=device)
    generator.manual_seed(event_selection_seed(seed, tx_index, depth))
    return torch.rand((int(count),), device=device, generator=generator)


def unpolarized_power_budgets(
    stack: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """(R_eff, T_eff) as the unpolarized TE/TM mean of the smooth-stack power
    budgets. The mean is acceptable for EVENT PROBABILITIES only; the selected
    branch's kernel applies the exact polarized Jones coefficients."""

    r_eff = 0.5 * (stack["cap_R_te"] + stack["cap_R_tm"])
    t_eff = 0.5 * (stack["cap_T_te"] + stack["cap_T_tm"])
    return r_eff, t_eff


def transmission_event_probability(
    r_eff: torch.Tensor,
    t_eff: torch.Tensor,
    *,
    floor: float = _EVENT_PROBABILITY_FLOOR,
) -> torch.Tensor:
    """Event probability p_t = T/(R+T) with a minimum-probability floor when
    both budgets are nonzero (plan section 7.1). Absorption 1-R-T terminates
    implicitly through the field magnitudes; there is no absorption event."""

    total = (r_eff + t_eff).clamp_min(1.0e-12)
    p_t = (t_eff / total).clamp(floor, 1.0 - floor)
    p_t = torch.where(t_eff <= 0.0, torch.zeros_like(p_t), p_t)
    return torch.where((r_eff <= 0.0) & (t_eff > 0.0), torch.ones_like(p_t), p_t)


@profiled_cuda_range(CudaProfileRange.MONTECARLO_BASIC_PENETRATION_DISCOVERY)
def straight_transmission_chains(
    rayd: Any,
    origins: torch.Tensor,
    targets: torch.Tensor,
    *,
    vertices: torch.Tensor | None,
    max_depth: int,
    scene_diagonal: float,
    failure_state: CapacityFailureState,
    ad: bool = False,
) -> SegmentPenetrationResult:
    """Trace one flattened fixed-capacity target-inset penetration batch."""

    common = {
        "input_active_any": int(origins.shape[0]) > 0,
        "hit_capacity": int(max_depth),
        "policy": SegmentPenetrationPolicy.MonteCarloTargetInset,
        "scene_diagonal": float(scene_diagonal),
        "failure_state": failure_state,
    }
    cuda_profile_mark(CudaProfileMark.OPTIX_TRAVERSAL)
    if not ad:
        return rayd_segment_penetration_forward(rayd, origins, targets, None, **common)
    if vertices is None:
        raise TypeError("AD straight transmission requires the live scene vertices")
    return rayd_segment_penetration_ad(rayd, vertices, origins, targets, None, **common)
