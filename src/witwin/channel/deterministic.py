"""Native deterministic RF solver public API."""

from __future__ import annotations

from dataclasses import dataclass

from witwin.channel.components import (
    AD_MODES as _VALID_AD_MODES,
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
    DEPTH_CAPPED_COMPONENTS,
    validate_coupled_candidate_limit,
    validate_coupled_gate,
    validate_isb_boundary_taper,
    validate_max_depth,
    validate_scatter_chain,
    validated_components,
)

from typing import Any

import torch

from typing import TYPE_CHECKING
from witwin.channel.scene.compiler import compile as compile_scene
from witwin.channel.scene.endpoints import bind_solver_scene

from witwin.channel.propagation.enumerated.engine import (
    evaluate_enumerated_paths,
)
from witwin.channel.propagation.enumerated.scattering import (
    append_scattering_evaluated_paths,
)
from witwin.channel.propagation.geometry.endpoints import (
    apply_receiver_layout,
    receiver_positions_and_layout,
)
from witwin.channel.scene.compiler import _frequency_scalar

from time import perf_counter

from witwin.channel.scene.endpoints import validate_scalar_endpoint_features
from witwin.channel.capabilities import (
    capabilities,
    config_metadata,
    serialize_config,
)
from witwin.channel import build_info
from witwin.channel.runtime import make_metadata
from witwin.channel.constants import PHASE_CONVENTION
from witwin.channel.scene.endpoints import ReceiverGrid
from witwin.channel.kernels.topology import (
    deterministic_component_counts,
)
from witwin.channel.propagation.topology.export import EvaluatedPathSidecars
from witwin.channel.propagation.enumerated.capacity import (
    sanitize_enumerated_capacity_transaction,
)
from witwin.channel.components import (
    apply_exported_path_counts,
    component_availability_status,
    component_max_depth,
)

from collections.abc import Mapping

from witwin.channel.kernels import deterministic as accumulation_kernels
from witwin.channel.kernels import fields as field_kernels
from witwin.channel.propagation.geometry.endpoints import ReceiverLayout
from witwin.channel.propagation.rows import EvaluatedPaths

if TYPE_CHECKING:
    from witwin.core import Scene, SceneSnapshot
    from witwin.channel.scene.endpoints import SolverScene


# --- Configuration --------------------------------------------------------

# Public component set. transmission carries specular wall-penetration paths
# (wave 2); scattering carries single-bounce Kirchhoff rough-surface paths
# (wave 3). transmission depth is capped like reflection (chains count wall
# penetrations); scattering is single-bounce in v1.
# Default component set is unchanged: the new components are strictly opt-in.
_VALID_SORT_KEYS = frozenset({"receiver_transmitter_depth_component"})
_VALID_MAX_PATHS_SCOPES = frozenset({"global", "per_pair"})


def _validate_scattering_coherent(
    *, scattering_coherent: bool, components: frozenset[str]
) -> None:
    """Validate the ADR-021 D3 coherent-scattering combine precondition."""

    if scattering_coherent and "scattering" not in components:
        # ADR-021 D3: the coherent combine only applies to scattering rows.
        # The scene-level requirement (realization-coherent phase screens,
        # not ensemble surfaces) is enforced at solve time where the scene
        # is known; here we reject the config-level precondition loudly.
        raise RuntimeError(
            "scattering_coherent=True requires the 'scattering' component "
            "(ADR-021 D3 combines scattering rows coherently and has no "
            "effect on any other component)"
        )


@dataclass(frozen=True, slots=True)
class Config:
    max_depth: int = 1
    max_diffraction_order: int = 1
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = (
        _DEFAULT_COMPONENTS
    )
    coherent: bool = True
    return_field: bool = True
    export_paths: bool = False
    max_paths: int | None = None
    max_paths_scope: str = "global"
    sort_key: str = "receiver_transmitter_depth_component"
    diagnostics: bool = False
    ad_mode: str = "none"
    # Coupled reflection-diffraction paths (ADR-011). Opt-in; when set the
    # deterministic grid solver enumerates the R->D / D->R compensator rows
    # (component ids 3/4) and accumulates their coherent contribution into a
    # dedicated coupled field slot. coupled_candidate_limit is a per-receiver-
    # block work/safety budget: the deterministic engine streams coupled
    # discovery over receiver blocks sized so each block stays under it.
    coupled_paths: bool = False
    coupled_candidate_limit: int = 1_000_000
    # Rough-surface scattering quadrature (wave 3). The patch density is a
    # fixed per-area sample count with a documented per-face cap of 4096
    # samples; per (tx, rx) pair only the strongest samples up to
    # scattering_max_paths_per_pair survive (dropped power is reported in the
    # scattering metadata, never silently redistributed) and samples below
    # scattering_power_threshold (absolute path_gain floor) are discarded.
    scattering_samples_per_m2: float = 8.0
    scattering_max_paths_per_pair: int = 4096
    scattering_power_threshold: float = 0.0
    # Coherent scattering combine (ADR-021 D3). DEFAULT-OFF opt-in. OFF keeps
    # the scattering slot an incoherent POWER sum bit-identical to today; ON
    # sums the complex path_field of scattering rows per (tx, rx) and finalizes
    # |sum|^2 (the ADR-019 per-component phasor precedent). It is physical only
    # for realization-coherent phase-screen rows, which carry a true complex
    # field; ensemble rows are zero-phase power rows, so the pipeline refuses an
    # ensemble-only solve loudly. Requires the 'scattering' component.
    scattering_coherent: bool = False
    # Enumerated scatter-chain path class (ADR-021 D1). DEFAULT-OFF opt-in:
    # scattering_chain_max_depth = 0 disables chain discovery so the pipeline is
    # byte-identical to today. When >= 1 it is the cap on d1 + d2, the combined
    # specular reflection depth of the two legs around the single diffuse vertex
    # (TX --C1(d1)--> v_s --C2(d2)--> RX with 1 <= d1 + d2 <= cap). Each leg is
    # independently bounded by the native kMaxAdDepth = 8, so the public cap is
    # 2 * 8 = 16. The chain-sample vertices are drawn at a documented lower
    # density (scattering_chain_samples_per_m2) than the single-bounce sampler,
    # and only the strongest scattering_chain_max_rows joined rows per (tx, rx)
    # survive. Requires the 'scattering' component.
    scattering_chain_max_depth: int = 0
    scattering_chain_samples_per_m2: float = 2.0
    scattering_chain_max_rows: int = 256
    # ISB boundary taper (ADR-017). DEFAULT-OFF visual-continuity heuristic: the
    # hard LoS occlusion gate becomes a C1 membership taper tau(c / (width * w_F))
    # and the compensating order-1 diffraction odd step spreads over the same
    # congruent window. OFF (the default) is bit-identical to the hard gate and
    # the unchanged diffraction window for every existing caller (enforced by a
    # bitwise regression test); the switch must never default ON. The width
    # scales the Fresnel penumbra w_F of the grazed silhouette edge; the
    # projection-validated optimum is 0.5 (artifacts/isb-taper/report.json).
    isb_boundary_taper: bool = False
    isb_boundary_taper_width: float = 0.5
    def __post_init__(self) -> None:
        validate_max_depth(self.max_depth)
        validate_isb_boundary_taper(self.isb_boundary_taper_width)
        if self.scattering_samples_per_m2 <= 0.0:
            raise ValueError("scattering_samples_per_m2 must be positive")
        if self.scattering_max_paths_per_pair <= 0:
            raise ValueError("scattering_max_paths_per_pair must be positive")
        if self.scattering_power_threshold < 0.0:
            raise ValueError("scattering_power_threshold must be non-negative")
        if self.max_diffraction_order < 0:
            raise ValueError("max_diffraction_order must be 0 or 1")
        if self.max_diffraction_order > 1:
            raise RuntimeError("max_diffraction_order above 1 is not supported yet")

        components = validated_components(
            self.components,
            error_message="components must be a non-empty subset of {valid}",
        )
        _validate_scattering_coherent(
            scattering_coherent=self.scattering_coherent, components=components
        )
        validate_scatter_chain(
            max_depth=self.scattering_chain_max_depth,
            samples_per_m2=self.scattering_chain_samples_per_m2,
            max_rows=self.scattering_chain_max_rows,
            components=components,
        )
        if self.max_depth > 5 and components & DEPTH_CAPPED_COMPONENTS:
            raise RuntimeError(
                "deterministic reflection/transmission currently support max_depth <= 5"
            )
        validate_coupled_gate(
            coupled_paths=self.coupled_paths,
            max_depth=self.max_depth,
            components=components,
        )
        validate_coupled_candidate_limit(self.coupled_candidate_limit)
        if self.max_paths is not None and self.max_paths <= 0:
            raise ValueError("max_paths must be positive when set")
        if self.max_paths_scope not in _VALID_MAX_PATHS_SCOPES:
            raise ValueError("max_paths_scope must be 'global' or 'per_pair'")
        if self.sort_key not in _VALID_SORT_KEYS:
            raise ValueError(f"sort_key must be one of {sorted(_VALID_SORT_KEYS)}")
        if self.ad_mode not in _VALID_AD_MODES:
            raise RuntimeError(
                f"deterministic ad_mode must be one of {sorted(_VALID_AD_MODES)}"
            )

        object.__setattr__(self, "components", components)


# --- Results --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PathTable:
    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    depth: torch.Tensor
    component_id: torch.Tensor
    primitive_id: torch.Tensor
    edge_id: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    path_gain: torch.Tensor
    interaction_position: torch.Tensor
    interaction_normal: torch.Tensor
    material_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    field_real: torch.Tensor
    field_imag: torch.Tensor
    coefficient: torch.Tensor
    field_xyz: torch.Tensor
    field_direction: torch.Tensor
    phase_rad: torch.Tensor
    interaction_count: torch.Tensor


@dataclass(frozen=True, slots=True)
class Result:
    path_gain: torch.Tensor
    field: torch.Tensor
    component_power: dict[str, torch.Tensor]
    component_fields: dict[str, torch.Tensor]
    paths: PathTable | None
    metadata: dict[str, Any]
    diagnostics: dict[str, Any] | None = None


# --- Accumulation ---------------------------------------------------------

# Component slots materialized by the native accumulator
# (kAccumSlotCount=6 in kernels/deterministic_accum.cu); the path component
# ids map to them as 0/1/2 -> 0/1/2, 5 -> 3, 6 -> 4, and 3/4 -> 5 (ADR-011:
# reflection->diffraction and diffraction->reflection are the coupled classes,
# both summed coherently into the single coupled field slot). Under
# ad_mode != "none" the same native forward runs inside a dispatch-only
# autograd.Function whose backward/jvp are native CUDA companions
# (accumulation_kernels.deterministic_accumulate_flat_ad), so the accumulated
# result keeps the
# autograd graph with no torch mirror of the kernel math. transmission
# carries specular wall-penetration paths (wave 2) and joins the coherent
# field total like the first three slots; scattering carries Kirchhoff
# rough-surface patch paths (wave 3) and is an incoherent POWER slot (plan
# 05 sections 6.7.3 / 7.3): its rows always fold into the totals in the
# power domain, never as zero-phase amplitudes, and its complex cell field
# is a diagnostic only. coupled carries the reflection-diffraction
# compensator (ADR-011) and is an ordinary coherent field slot that joins
# the coherent field total like the first three slots.
_NATIVE_COMPONENT_SLOTS = {
    "los": 0,
    "reflection": 1,
    "diffraction": 2,
    "transmission": 3,
    "scattering": 4,
    "coupled": 5,
}
_BASE_COMPONENTS = ("los", "reflection", "diffraction")
_OPTIONAL_COMPONENTS = ("transmission", "scattering")


def empty_field_like_power(path_gain: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), device=path_gain.device, dtype=torch.complex64)


def accumulate_flat_components(
    *,
    valid: torch.Tensor,
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    component_id: torch.Tensor,
    path_gain: torch.Tensor,
    path_field: torch.Tensor,
    num_tx: int,
    num_rx: int,
    coherent: bool,
    extra_components: tuple[str, ...] = (),
    differentiable: bool = False,
    scattering_coherent: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    # ADR-021 D3 opt-in coherent scattering combine. Default OFF threads NO new
    # argument to the native accumulator, so the call stays byte-identical to
    # today; ON requests the native scattering_combine_domain=1 path where the
    # scattering slot's summed complex field squares into its power instead of
    # summing per-row powers (the ADR-019 per-component phasor precedent).
    combine_kwargs = {"scattering_combine_domain": 1} if scattering_coherent else {}
    if differentiable:
        # AD modes (plan 07): the same native accumulator kernels run inside
        # a dispatch-only autograd.Function with native backward/jvp
        # companions, so Result.path_gain / field / component_power carry
        # the complete graph of the per-path fields and powers.
        exported = accumulation_kernels.deterministic_accumulate_flat_ad(
            valid.contiguous(),
            tx_id.to(dtype=torch.int32).contiguous(),
            rx_id.to(dtype=torch.int32).contiguous(),
            component_id.to(dtype=torch.int32).contiguous(),
            path_gain.to(dtype=torch.float32).contiguous(),
            path_field.real.to(dtype=torch.float32).contiguous(),
            path_field.imag.to(dtype=torch.float32).contiguous(),
            num_tx=int(num_tx),
            num_rx=int(num_rx),
            coherent=bool(coherent),
            **combine_kwargs,
        )
        power_total = exported["power_total"]
        field_total = torch.complex(
            exported["field_total_real"], exported["field_total_imag"]
        )
        component_power_tensor = exported["component_power"]
        component_field_tensor = torch.complex(
            exported["component_field_real"], exported["component_field_imag"]
        )
    else:
        exported = accumulation_kernels.deterministic_accumulate_flat(
            valid.contiguous(),
            tx_id.to(dtype=torch.int32).contiguous(),
            rx_id.to(dtype=torch.int32).contiguous(),
            component_id.to(dtype=torch.int32).contiguous(),
            path_gain.to(dtype=torch.float32).contiguous(),
            path_field.real.to(dtype=torch.float32).contiguous(),
            path_field.imag.to(dtype=torch.float32).contiguous(),
            num_tx=int(num_tx),
            num_rx=int(num_rx),
            coherent=bool(coherent),
            **combine_kwargs,
        )
        power_total = exported["power_total"]
        field_total = field_kernels.deterministic_pack_complex(
            exported["field_total_real"].reshape(-1).contiguous(),
            exported["field_total_imag"].reshape(-1).contiguous(),
        ).reshape(exported["field_total_real"].shape)
        component_power_tensor = exported["component_power"]
        component_field_tensor = field_kernels.deterministic_pack_complex(
            exported["component_field_real"].reshape(-1).contiguous(),
            exported["component_field_imag"].reshape(-1).contiguous(),
        ).reshape(exported["component_field_real"].shape)
    # All five slots come out of the one native accumulator (forward and,
    # in the AD modes, its native backward/jvp companions); the result dicts
    # expose the base components plus the requested optional ones.
    exported_names = _BASE_COMPONENTS + tuple(extra_components)
    component_power = {
        name: component_power_tensor[_NATIVE_COMPONENT_SLOTS[name]].contiguous()
        for name in exported_names
    }
    component_fields = {
        name: component_field_tensor[_NATIVE_COMPONENT_SLOTS[name]].contiguous()
        for name in exported_names
    }
    return power_total, field_total, component_power, component_fields


def apply_layout_to_accumulation(
    *,
    path_gain: torch.Tensor,
    field: torch.Tensor,
    component_power: Mapping[str, torch.Tensor],
    component_fields: Mapping[str, torch.Tensor],
    layout: ReceiverLayout,
    return_field: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    laid_out_power = apply_receiver_layout(path_gain, layout)
    laid_out_field = (
        apply_receiver_layout(field, layout)
        if return_field
        else torch.empty((0,), device=field.device, dtype=torch.complex64)
    )
    laid_out_component_power = {
        name: apply_receiver_layout(value, layout)
        for name, value in component_power.items()
    }
    laid_out_component_fields = {
        name: apply_receiver_layout(value, layout)
        if return_field
        else torch.empty((0,), device=value.device, dtype=torch.complex64)
        for name, value in component_fields.items()
    }
    return (
        laid_out_power,
        laid_out_field,
        laid_out_component_power,
        laid_out_component_fields,
    )


def accumulate_path_result(
    paths: EvaluatedPaths,
    *,
    frequency_hz: float,
    num_tx: int,
    num_rx: int,
    layout: ReceiverLayout,
    coherent: bool,
    return_field: bool,
    extra_components: tuple[str, ...] = (),
    differentiable: bool = False,
    scattering_coherent: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    topology = paths.topology
    fields = paths.fields
    power, field, component_power, component_fields = accumulate_flat_components(
        valid=topology.valid,
        tx_id=topology.tx_id,
        rx_id=topology.rx_id,
        component_id=topology.component_id,
        path_gain=fields.path_gain,
        path_field=fields.path_field,
        num_tx=num_tx,
        num_rx=num_rx,
        coherent=coherent,
        extra_components=extra_components,
        differentiable=differentiable,
        scattering_coherent=scattering_coherent,
    )
    return apply_layout_to_accumulation(
        path_gain=power,
        field=field,
        component_power=component_power,
        component_fields=component_fields,
        layout=layout,
        return_field=return_field,
    )


def build_path_table(
    paths: EvaluatedPaths, *, frequency_hz: float, include_fields: bool = True
) -> PathTable:
    topology = paths.topology
    geometry = paths.geometry
    fields = paths.fields
    if include_fields:
        path_field = fields.path_field.to(dtype=torch.complex64).contiguous()
        phase = field_kernels.deterministic_phase_from_field(
            path_field.real.to(dtype=torch.float32).contiguous(),
            path_field.imag.to(dtype=torch.float32).contiguous(),
        )
    else:
        zero_field_phase = field_kernels.deterministic_zero_field_phase(
            fields.path_gain.to(dtype=torch.float32).contiguous()
        )
        path_field = zero_field_phase["path_field"]
        phase = zero_field_phase["phase_rad"]
    return PathTable(
        valid=topology.valid.contiguous(),
        tx_id=topology.tx_id.to(dtype=torch.int32).contiguous(),
        rx_id=topology.rx_id.to(dtype=torch.int32).contiguous(),
        depth=topology.depth.to(dtype=torch.int32).contiguous(),
        component_id=topology.component_id.to(dtype=torch.int32).contiguous(),
        primitive_id=topology.primitive_id.to(dtype=torch.int32).contiguous(),
        edge_id=topology.edge_id.to(dtype=torch.int32).contiguous(),
        path_length_m=geometry.path_length_m.to(dtype=torch.float32).contiguous(),
        delay_s=geometry.delay_s.to(dtype=torch.float32).contiguous(),
        path_gain=fields.path_gain.to(dtype=torch.float32).contiguous(),
        interaction_position=geometry.interaction_position.to(
            dtype=torch.float32
        ).contiguous(),
        interaction_normal=geometry.interaction_normal.to(
            dtype=torch.float32
        ).contiguous(),
        material_id=topology.material_id.to(dtype=torch.int32).contiguous(),
        primitive_sequence=topology.primitive_sequence.to(
            dtype=torch.int32
        ).contiguous(),
        material_sequence=topology.material_sequence.to(dtype=torch.int32).contiguous(),
        interaction_positions=geometry.interaction_positions.to(
            dtype=torch.float32
        ).contiguous(),
        interaction_normals=geometry.interaction_normals.to(
            dtype=torch.float32
        ).contiguous(),
        field_real=path_field.real.to(dtype=torch.float32).contiguous(),
        field_imag=path_field.imag.to(dtype=torch.float32).contiguous(),
        coefficient=fields.coefficient.to(dtype=torch.complex64).contiguous(),
        field_xyz=fields.field_xyz.to(dtype=torch.complex64).contiguous(),
        field_direction=geometry.field_direction.to(dtype=torch.float32).contiguous(),
        phase_rad=phase,
        interaction_count=topology.depth.to(dtype=torch.int32).contiguous(),
    )


# --- Pipeline -------------------------------------------------------------


def _validate_requested_components(config: Config) -> None:
    if config.isb_boundary_taper and config.ad_mode != "none":
        # ISB boundary taper (ADR-017) gate 3: the C1 clearance-factor AD
        # companion (d(tau)/d(endpoint)) is a documented follow-up. Until it
        # lands, taper + AD is rejected loudly rather than returning a silently
        # incomplete gradient. OFF-path AD stays bit-identical.
        raise RuntimeError(
            "isb_boundary_taper does not support ad_mode != 'none' yet "
            "(ADR-017 gate 3 C1 clearance companion is a follow-up)"
        )
    if "reflection" in config.components and config.max_depth < 1:
        raise RuntimeError("deterministic reflection requires max_depth >= 1")
    if "diffraction" in config.components:
        if config.max_depth < 1:
            raise RuntimeError("deterministic diffraction requires max_depth >= 1")
        if config.max_diffraction_order < 1:
            raise RuntimeError(
                "deterministic diffraction requires max_diffraction_order >= 1"
            )


def _validate_scattering_coherent_mode(scattering_info: dict[str, Any] | None) -> None:
    """ADR-021 D3 solve-time gate for the coherent scattering combine.

    The combine sums the complex ``path_field`` of scattering rows and
    finalizes ``|sum|^2``. Only realization-coherent phase-screen rows carry a
    physical complex field; ensemble rows are zero-phase power rows, so an
    ensemble-only (or empty-realization) solve would interfere meaningless
    phases. Both cases are refused loudly rather than returning a wrong number.
    """

    info = scattering_info or {}
    ensemble = int(info.get("ensemble_sample_count", 0))
    realization = int(info.get("realization_structure_count", 0))
    if ensemble > 0:
        raise RuntimeError(
            "scattering_coherent=True requires realization-coherent scattering "
            f"only, but the scene has ensemble scattering surfaces ({ensemble} "
            "ensemble samples). Ensemble rows are zero-phase power rows and "
            "cannot combine coherently; assign a realization_coherent "
            "PhaseScreen to every scattering surface or disable "
            "scattering_coherent (ADR-021 D3, contract 6.7.3)"
        )
    if realization == 0:
        raise RuntimeError(
            "scattering_coherent=True requires at least one "
            "realization_coherent phase-screen scattering surface, but the "
            "scene has none; the coherent combine has no physical complex "
            "field to interfere (ADR-021 D3)"
        )


def _scattering_metadata(
    scattering_info: dict[str, Any] | None, config: Config
) -> dict[str, Any] | None:
    """Scattering metadata sub-dict for the deterministic result (plan 05 wave 3).

    Returns ``None`` when no scattering rows were requested. Extracted as a
    sub-dict builder so ``_metadata`` stays within its complexity budget.
    Incoherent Kirchhoff patch quadrature means per-path phases are NOT physical
    for ensemble rows; ADR-021 D3 records how scattering rows combine together.
    """

    if scattering_info is None:
        return None
    block = dict(scattering_info)
    block["combine_domain"] = (
        "coherent" if config.scattering_coherent else "incoherent_power"
    )
    return block


def _coupled_paths_metadata(
    config: Config, component_counts: dict[str, int] | None = None
) -> dict[str, Any]:
    """Coupled higher-order compensator metadata block (ADR-011 + ADR-013).

    Mirrors the path solver's coupled_paths block. ``coupled_paths=True`` now
    enables the uniform order-2 compensator family {R->D, D->R, D->D}; the DD
    (component id 7) member is reported as its own row count for audits while it
    aggregates into the single coherent coupled slot.
    """

    if not config.coupled_paths:
        return {
            "requested": False,
            "geometry": "not_requested",
            "coefficient": "not_requested",
        }
    counts = component_counts or {}
    return {
        "requested": True,
        "geometry": "native_1r1d_reciprocal_plus_dd",
        "coefficient": "unified_complex3_jones",
        "double_diffraction": {
            "geometry": "native_dd_cascade",
            "component_id": 7,
            "row_count": int(counts.get("coupled_double_diffraction", 0)),
        },
    }


def _register_coupled_component(
    config: Config,
    topology: Any,
    component_counts: dict[str, int],
    extra_components: tuple[str, ...],
) -> tuple[str, ...]:
    """Record the coupled component count and export name (ADR-011 + ADR-013).

    Coupled rows carry component ids 3 (R->D), 4 (D->R) and 7 (D->D, ADR-013);
    all three accumulate into the single coupled slot. "coupled" is not a public
    component name, so it is enabled by the coupled_paths gate rather than the
    components set. The DD row count is recorded separately for audits.
    """

    if not config.coupled_paths:
        return extra_components
    component_id = topology.component_id
    component_counts["coupled"] = int(
        ((component_id == 3) | (component_id == 4) | (component_id == 7)).sum().item()
    )
    component_counts["coupled_double_diffraction"] = int(
        (component_id == 7).sum().item()
    )
    return extra_components + ("coupled",)


def _append_scattering(
    scene: SolverScene, config: Config, evaluated: Any, sidecars: Any
) -> tuple[Any, Any, dict[str, Any] | None]:
    """Append Kirchhoff scattering rows and gate the coherent combine.

    Single-purpose solve stage (plan 05 wave 3 + ADR-021 D3). When scattering is
    not requested it returns the inputs unchanged with ``scattering_info=None``.
    When ``scattering_coherent`` is set, the D3 solve-time gate refuses an
    ensemble-only or empty-realization solve before any accumulation runs.
    """

    if "scattering" not in config.components:
        return evaluated, sidecars, None
    evaluated, sidecars, scattering_info = append_scattering_evaluated_paths(
        scene,
        config,
        evaluated,
        sidecars,
    )
    if config.scattering_coherent:
        _validate_scattering_coherent_mode(scattering_info)
    return evaluated, sidecars, scattering_info


def _metadata(
    *,
    config: Config,
    native_info: dict[str, Any],
    path_count: int,
    component_counts: dict[str, int],
    launch_count: int,
    ad_companion_launches: int = 0,
    ad_tape_bytes: int = 0,
    forward_time_ms: float = 0.0,
    peak_memory_bytes: int = 0,
    scattering_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capability = {
        "rayd_native": bool(native_info["uses_rayd_native"]),
        "path_native": bool(native_info.get("uses_path_native", False)),
        "cuda_available": bool(native_info["cuda_available"]),
        "optix_available": bool(native_info["optix_available"]),
    }
    components = component_availability_status(
        config.components,
        reflection_available=capability["rayd_native"],
        diffraction_available=capability["rayd_native"],
        reflection_error="deterministic reflection requires RayD native capability",
        diffraction_error="deterministic diffraction requires RayD native capability",
    )
    # transmission carries specular wall-penetration paths since wave 2 and
    # scattering carries Kirchhoff rough-surface patch paths since wave 3.
    # Both keep the truthful requested-but-empty status when no paths were
    # found (e.g. every surface in the scene is smooth).
    apply_exported_path_counts(
        components,
        config.components,
        transmission_path_count=component_counts.get("transmission", 0),
        scattering_path_count=component_counts.get("scattering", 0),
    )
    if "transmission" in config.components:
        if not capability["rayd_native"]:
            raise RuntimeError(
                "deterministic transmission requires RayD native capability"
            )
        # Endpoint-connection thin_sheet contract (plan 05 section 4).
        metadata_transmission = {
            "thin_sheet_straight_path_approximation": True,
            "group_delay": "geometric",
        }
    else:
        metadata_transmission = None
    rayd_component_enabled = (
        components["reflection"] == "enabled" or components["diffraction"] == "enabled"
    )
    requested_config = serialize_config(config)
    effective_config = dict(requested_config)
    metadata = {
        "max_depth": config.max_depth,
        "max_diffraction_order": config.max_diffraction_order,
        "coherent": config.coherent,
        "return_field": config.return_field,
        "export_paths": config.export_paths,
        "max_paths": config.max_paths,
        "max_paths_scope": config.max_paths_scope,
        "sort_key": config.sort_key,
        "accumulation_strategy": "coherent" if config.coherent else "incoherent",
        "components": components,
        "counts": {
            "path_count": path_count,
            "valid_path_count": path_count,
            "components": component_counts,
        },
        "capability": capability,
        # Plan 07 AD-4: the real registered-companion accounting. vjp retains
        # tape and schedules its companions on the user's later backward; jvp
        # runs its dual companions inside this forward and retains no tape.
        "kernel": make_metadata(
            primitive="deterministic_solver",
            forward_launch_count=launch_count,
            backward_launch_count=(
                ad_companion_launches if config.ad_mode == "vjp" else 0
            ),
            jvp_launch_count=(ad_companion_launches if config.ad_mode == "jvp" else 0),
            tape_bytes=ad_tape_bytes if config.ad_mode == "vjp" else 0,
            accumulation_strategy="atomic_add",
            scheduling_strategy="native_fused"
            if rayd_component_enabled
            else "native_cuda",
            rayd_native=capability["rayd_native"],
            ad_status=config.ad_mode,
            forward_time_ms=forward_time_ms,
            peak_memory_bytes=peak_memory_bytes,
        ),
        "field_abi": "complex3_v1",
        "phase_convention": dict(PHASE_CONVENTION),
        "coefficient_semantics": "unit_excitation_dimensionless_receiver_projection",
        # Coupled higher-order compensator family (ADR-011 R->D/D->R + ADR-013
        # D->D); mirrors the path solver's coupled_paths metadata block.
        "coupled_paths": _coupled_paths_metadata(config, component_counts),
    }
    if metadata_transmission is not None:
        metadata["transmission"] = metadata_transmission
    scattering_metadata = _scattering_metadata(scattering_info, config)
    if scattering_metadata is not None:
        metadata["scattering"] = scattering_metadata
    metadata.update(
        config_metadata(
            requested=requested_config,
            effective=effective_config,
            component_max_depth=component_max_depth(
                config.components,
                chain_depth=config.max_depth,
                single_bounce_depth=1,
            ),
        )
    )
    metadata["semantic_capabilities"] = capabilities()["solvers"]["deterministic"]
    return metadata


def _terminal_check_capacity(sidecars: EvaluatedPathSidecars) -> None:
    transaction = sidecars.capacity_transaction
    if transaction is not None:
        transaction.terminal_check()


def _solve_pipeline(scene: SolverScene, config: Config) -> Result:
    validate_scalar_endpoint_features(
        scene.transmitters, scene.receivers, solver="deterministic"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel.deterministic requires CUDA")
    # Solve-level wall time and CUDA high-water-mark delta for the kernel
    # metadata (plan 07 AD-4). This is AD instrumentation only: the leading
    # synchronize would stall the host on the caller's queued work and the
    # trailing one drains the solve before returning, which an optimization
    # loop over ad_mode="none" must not pay. none-mode reports zeros and takes
    # no sync, preserving the byte-identical zero-overhead primal contract.
    ad_instrumented = config.ad_mode != "none"
    solve_start = 0.0
    peak_before = 0
    if ad_instrumented:
        torch.cuda.synchronize()
        solve_start = perf_counter()
        peak_before = torch.cuda.max_memory_allocated()

    native_info = build_info()
    _validate_requested_components(config)
    if "reflection" in config.components and not native_info["uses_rayd_native"]:
        raise RuntimeError("deterministic reflection requires RayD native capability")
    if "diffraction" in config.components and not native_info["uses_rayd_native"]:
        raise RuntimeError("deterministic diffraction requires RayD native capability")
    has_grid = any(
        receiver.__class__.__name__ == "ReceiverGrid" for receiver in scene.receivers
    )
    if has_grid and len(scene.receivers) > 1 and not config.export_paths:
        raise RuntimeError("mixed point/grid receivers require export_paths=True")

    device = torch.device("cuda")
    _, layout = receiver_positions_and_layout(scene, device=device)
    # One host read of a tensor frequency for the whole solve: topology
    # export, field evaluation, accumulation and path export share it
        # (audit M3). Channel scene.compile() keeps its own read: the material cache
    # token must see the live value to stay correct under in-place
    # frequency mutation.
    frequency_hz = _frequency_scalar(scene)
    evaluated, sidecars = evaluate_enumerated_paths(
        scene,
        config,
        frequency_value=frequency_hz,
        # Stream coupled discovery over receiver blocks so a full grid solve
        # stays under the per-block candidate budget (ADR-011).
        coupled_rx_streaming=config.coupled_paths,
        defer_capacity_terminal=True,
    )
    evaluated, sidecars, scattering_info = _append_scattering(
        scene, config, evaluated, sidecars
    )
    evaluated, sidecars = sanitize_enumerated_capacity_transaction(evaluated, sidecars)
    topology = evaluated.topology
    path_count = evaluated.row_count
    component_counts = deterministic_component_counts(topology.component_id)
    # The native counter materializes only los/reflection/diffraction slots.
    for name, cid in (("transmission", 5), ("scattering", 6)):
        if name in config.components:
            component_counts[name] = int((topology.component_id == cid).sum().item())
    extra_components = tuple(
        name for name in _OPTIONAL_COMPONENTS if name in config.components
    )
    extra_components = _register_coupled_component(
        config, topology, component_counts, extra_components
    )
    path_gain, field, component_power, component_fields = accumulate_path_result(
        evaluated,
        frequency_hz=frequency_hz,
        num_tx=len(scene.transmitters),
        num_rx=layout.receiver_count,
        layout=layout,
        coherent=config.coherent,
        return_field=config.return_field,
        extra_components=extra_components,
        # AD modes run the same native accumulator inside its dispatch-only
        # autograd.Function so Result.path_gain/field/component_power carry
        # the complete graph; none-mode keeps the bare zero-overhead kernel.
        differentiable=config.ad_mode != "none",
        # ADR-021 D3: opt-in coherent scattering combine (default OFF is
        # byte-identical). The solve-time gate above already rejected any
        # ensemble-only or empty-realization scene.
        scattering_coherent=config.scattering_coherent,
    )
    exact_diffraction = None
    if (
        sidecars.diffraction_vector_field is not None
        and len(scene.receivers) == 1
        and isinstance(scene.receivers[0], ReceiverGrid)
    ):
        vector_field = sidecars.diffraction_vector_field
        exact_diffraction_flat = vector_field.abs().square().sum(dim=-1)
        exact_diffraction = apply_receiver_layout(exact_diffraction_flat, layout)
        previous_diffraction = component_power["diffraction"]
        component_power["diffraction"] = exact_diffraction
        if not config.coherent:
            path_gain = path_gain - previous_diffraction + exact_diffraction
    if ad_instrumented:
        torch.cuda.synchronize()
        forward_time_ms = (perf_counter() - solve_start) * 1.0e3
        peak_memory_bytes = max(0, torch.cuda.max_memory_allocated() - peak_before)
    else:
        forward_time_ms = 0.0
        peak_memory_bytes = 0
    metadata = _metadata(
        config=config,
        native_info=native_info,
        path_count=path_count,
        component_counts=component_counts,
        launch_count=sidecars.execution.launch_count,
        ad_companion_launches=sidecars.execution.ad_companion_launches,
        ad_tape_bytes=sidecars.execution.ad_tape_bytes,
        forward_time_ms=forward_time_ms,
        peak_memory_bytes=peak_memory_bytes,
        scattering_info=scattering_info,
    )
    diagnostics = None
    if config.diagnostics:
        candidate_count = sidecars.execution.candidate_count
        diagnostics = {
            "path_gain_shape": tuple(path_gain.shape),
            "field_shape": tuple(field.shape),
            "path_count": path_count,
            "component_counts": component_counts,
            "coherent": config.coherent,
            "accumulation_mode": "coherent" if config.coherent else "incoherent",
            "native_launch_count": sidecars.execution.launch_count,
            "diffraction_accumulation": (
                "exact_vector_coherent_paths"
                if exact_diffraction is not None
                else "scalar_path_accumulation"
            ),
            "visibility_rejection_count": (
                sidecars.execution.visibility_rejection_count
            ),
            "selected_edge_count": sidecars.execution.selected_edge_count,
            "path_planning": {
                "max_paths": config.max_paths,
                "max_paths_scope": config.max_paths_scope,
                "candidate_count": candidate_count,
                "guardrail_count": sidecars.execution.guardrail_count,
                "truncated": config.max_paths is not None
                and path_count < candidate_count,
            },
        }
    result = Result(
        path_gain=path_gain,
        field=field,
        component_power=component_power,
        component_fields=component_fields,
        paths=(
            build_path_table(
                evaluated,
                frequency_hz=frequency_hz,
                include_fields=config.return_field,
            )
            if config.export_paths
            else None
        ),
        metadata=metadata,
        diagnostics=diagnostics,
    )
    _terminal_check_capacity(sidecars)
    return result


# --- Public entry point ---------------------------------------------------


# ``reference_frequency_hz`` is deliberately unannotated: it accepts a Python
# float or a Torch scalar, and its rendered text is part of the frozen public
# signature in ci/public-api-snapshot.json, so annotating it would break the
# recorded contract hash. mypy only sees this function now that the solver is
# one module, so the pre-existing shape is silenced here rather than changed.
def solve(  # type: ignore[no-untyped-def]
    scene: Scene | SceneSnapshot,
    config: Config,
    *,
    reference_frequency_hz,
) -> Result:
    """Run the deterministic propagation pipeline."""

    compiled = compile_scene(
        scene, reference_frequency_hz=reference_frequency_hz
    )
    return _solve_pipeline(bind_solver_scene(compiled), config)

__all__ = ["Config", "PathTable", "Result", "solve"]
