# Copyright Xingyu Chen.
# Coherent radiomap estimator: enumerated paths combined per receiver.

"""Coherent radiomap estimator: enumerated paths combined per receiver."""

from __future__ import annotations

from dataclasses import dataclass

from typing import ClassVar, Literal, cast

from witwin.channel.components import (
    DiscoveryMetadata,
    discovery_metadata,
    require_config_type,
    validate_vocabulary,
)

from typing import Any

import torch

from typing import TYPE_CHECKING
from witwin.channel.scene.compiler import compile as compile_scene, record_dropped_faces
from witwin.channel.scene.endpoints import bind_solver_scene, link_zero_gradient

from witwin.channel.propagation.enumerated import (
    EnumeratedOptions,
    evaluate_enumerated_paths,
)
from witwin.channel.interactions.scattering import (
    append_scattering_evaluated_paths,
)
from witwin.channel.propagation.geometry import (
    apply_receiver_layout,
    receiver_positions_and_layout,
)
from witwin.channel.scene.compiler import _frequency_scalar

from time import perf_counter

from witwin.channel.scene.endpoints import validate_scalar_endpoint_features
from witwin.channel.capabilities import (
    ConfigMetadata,
    ResultSemantics,
    component_quantities,
    config_metadata,
    estimator_metadata,
    serialize_config,
    solver_capabilities,
)
from witwin.channel import build_info
from witwin.channel.runtime import KernelMetadata, make_metadata
from witwin.channel.constants import PHASE_CONVENTION
from witwin.channel.kernels.topology import (
    deterministic_component_counts,
)
from witwin.channel.propagation.topology import EvaluatedPathSidecars
from witwin.channel.propagation.enumerated import (
    sanitize_enumerated_capacity_transaction,
    transmission_dropped_rows,
)
from witwin.channel.components import (
    apply_exported_path_counts,
    component_availability_status,
    component_max_depth,
)

from collections.abc import Mapping

from witwin.channel.kernels import deterministic as accumulation_kernels
from witwin.channel.kernels import fields as field_kernels
from witwin.channel.propagation.geometry import ReceiverLayout
from witwin.channel.propagation.rows import EvaluatedPaths

if TYPE_CHECKING:
    from witwin.core import Scene, SceneSnapshot
    from witwin.channel.scene.endpoints import SolverScene


# --- Configuration --------------------------------------------------------

# Public component set. transmission carries specular wall-penetration paths
# (transmission); scattering carries single-bounce Kirchhoff rough-surface paths
# (rough scattering). transmission depth is capped like reflection (chains count wall
# penetrations); scattering is single-bounce in v1.
# Default component set is unchanged: the new components are strictly opt-in.
_VALID_SORT_KEYS = frozenset({"receiver_transmitter_depth_component"})
_COMBINES = frozenset({"field", "power"})


def _validate_scattering_coherent(*, scattering_coherent: bool, components: frozenset[str]) -> None:
    """Validate the coherent scattering coherent-scattering combine precondition."""

    if scattering_coherent and "scattering" not in components:
        # coherent scattering: the coherent combine only applies to scattering rows.
        # The scene-level requirement (realization-coherent phase screens,
        # not ensemble surfaces) is enforced at solve time where the scene
        # is known; here we reject the config-level precondition loudly.
        raise ValueError(
            "scattering_coherent=True requires the 'scattering' component (it "
            "combines scattering rows coherently and affects no other component), "
            f"got components {sorted(components)}"
        )


@dataclass(frozen=True, slots=True)
class Coherent(EnumeratedOptions):
    """Coherent radiomap options: the shared enumerated options plus the terminal stage's."""

    # How rows reach a receiver value: "field" sums the complex path fields per (tx, rx) and
    # squares the sum; "power" sums the per-row powers.
    combine: Literal["field", "power"] = "field"
    return_field: bool = True
    export_paths: bool = False
    sort_key: str = "receiver_transmitter_depth_component"
    diagnostics: bool = False
    # Coherent scattering combine, off by default. Off keeps the scattering slot an
    # incoherent power sum; on sums the complex path_field of scattering rows per
    # (tx, rx) and finalizes |sum|^2. It is physical only for realization-coherent
    # phase-screen rows; an ensemble-only solve is refused. Requires 'scattering'.
    scattering_coherent: bool = False

    _option_owner: ClassVar[str] = "radiomap.Coherent"

    def __post_init__(self) -> None:
        EnumeratedOptions.__post_init__(self)
        _validate_scattering_coherent(
            scattering_coherent=self.scattering_coherent,
            components=frozenset(self.components),
        )
        validate_vocabulary("sort_key", self.sort_key, _VALID_SORT_KEYS)
        validate_vocabulary("combine", self.combine, _COMBINES)


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


class _MetadataKeys(ResultSemantics, ConfigMetadata):
    max_depth: int
    return_field: bool
    export_paths: bool
    max_paths: int | None
    max_paths_scope: str
    sort_key: str
    accumulation_strategy: str
    components: dict[str, str]
    counts: dict[str, Any]
    capability: dict[str, bool]
    kernel: KernelMetadata
    field_abi: str
    phase_convention: dict[str, Any]
    coefficient_semantics: str
    coupled_paths: dict[str, Any]
    discovery: DiscoveryMetadata
    semantic_capabilities: dict[str, Any]


class CoherentRadioMapMetadata(_MetadataKeys, total=False):
    """The stable keys of ``CoherentRadioMap.metadata``; a plain ``dict`` at run time."""

    # Present when the transmission component was requested.
    transmission: dict[str, Any]
    # Present when the scattering component was requested.
    scattering: dict[str, Any]
    # Present when compilation removed degenerate or duplicate faces.
    dropped_faces: dict[str, int]


@dataclass(frozen=True, slots=True)
class CoherentRadioMap:
    path_gain: torch.Tensor
    field: torch.Tensor
    component_power: dict[str, torch.Tensor]
    component_fields: dict[str, torch.Tensor]
    paths: PathTable | None
    metadata: CoherentRadioMapMetadata
    diagnostics: dict[str, Any] | None = None


# --- Accumulation ---------------------------------------------------------

# Component slots materialized by the native accumulator
# (kAccumSlotCount=6 in kernels/deterministic_accum.cu); the path component
# ids map to them as 0/1/2 -> 0/1/2, 5 -> 3, 6 -> 4, and the interaction chains and double
# diffraction (3, 4, 7, 9..15) -> 5, summed coherently into the single coupled field slot. Under
# ad_mode != "none" the same native forward runs inside a dispatch-only
# autograd.Function whose backward/jvp are native CUDA companions
# (accumulation_kernels.deterministic_accumulate_flat_ad), so the accumulated
# result keeps the
# autograd graph with no torch mirror of the kernel math. transmission
# carries specular wall-penetration paths (transmission) and joins the coherent
# field total like the first three slots; scattering carries Kirchhoff
# rough-surface patch paths (rough scattering) and is an incoherent POWER slot (plan
# the component layout): its rows always fold into the totals in the
# power domain, never as zero-phase amplitudes, and its complex cell field
# is a diagnostic only. coupled carries the reflection-diffraction
# compensator (coupled reflection and diffraction) and is an ordinary coherent field slot that joins
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
    # coherent scattering coherent scattering combine. Default OFF threads NO new
    # argument to the native accumulator, so the call stays byte-identical to
    # today; ON requests the native scattering_combine_domain=1 path where the
    # scattering slot's summed complex field squares into its power instead of
    # summing per-row powers.
    combine_kwargs = {"scattering_combine_domain": 1} if scattering_coherent else {}
    if differentiable:
        # AD modes (the AD contract): the same native accumulator kernels run inside
        # a dispatch-only autograd.Function with native backward/jvp
        # companions, so CoherentRadioMap.path_gain / field / component_power carry
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
    paths: EvaluatedPaths, *, frequency_hz: float, include_fields: bool = True,
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


def _link_zero_gradients(
    config: Coherent, scene: SolverScene, path_gain: torch.Tensor, field: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep reverse-mode outputs on the graph when no surviving path depends on a live leaf."""

    if config.ad_mode != "vjp":
        return path_gain, field
    return link_zero_gradient(path_gain, scene), link_zero_gradient(field, scene)


def _validate_requested_components(config: Coherent) -> None:
    if config.isb_boundary_taper and config.ad_mode != "none":
        # ISB boundary taper gate 3: the C1 clearance-factor AD
        # companion (d(tau)/d(endpoint)) is a documented follow-up. Until it
        # lands, taper + AD is rejected loudly rather than returning a silently
        # incomplete gradient. OFF-path AD stays bit-identical.
        raise RuntimeError(
            "isb_boundary_taper=True supports only ad_mode='none' (the taper has no "
            f"derivative companion), got ad_mode={config.ad_mode!r}"
        )
    if "reflection" in config.components and config.max_depth < 1:
        raise RuntimeError("deterministic reflection requires max_depth >= 1")
    if "diffraction" in config.components and config.max_depth < 1:
        raise RuntimeError("deterministic diffraction requires max_depth >= 1")


def _validate_scattering_coherent_mode(scattering_info: dict[str, Any] | None) -> None:
    """coherent scattering solve-time gate for the coherent scattering combine.

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
            "scattering_coherent"
        )
    if realization == 0:
        raise RuntimeError(
            "scattering_coherent=True requires at least one "
            "realization_coherent phase-screen scattering surface, but the "
            "scene has none; the coherent combine has no physical complex "
            "field to interfere"
        )


def _scattering_metadata(
    scattering_info: dict[str, Any] | None, config: Coherent,
) -> dict[str, Any] | None:
    """Scattering metadata sub-dict for the deterministic result.

    Returns ``None`` when no scattering rows were requested. Extracted as a
    sub-dict builder so ``_metadata`` stays within its complexity budget.
    Incoherent Kirchhoff patch quadrature means per-path phases are NOT physical
    for ensemble rows; coherent scattering how scattering rows combine together.
    """

    if scattering_info is None:
        return None
    block = dict(scattering_info)
    block["combine_domain"] = (
        "coherent" if config.scattering_coherent else "incoherent_power"
    )
    return block


def _coupled_paths_metadata(
    config: Coherent, component_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Coupled-paths metadata block: interaction chains plus double diffraction.

    Mirrors the path solver's coupled_paths block. ``coupled_paths=True`` enables the
    interaction chains (up to ``max_depth`` reflections with one diffraction anywhere) and
    double diffraction; the double-diffraction rows (component id 7) are
    reported as their own row count while they aggregate into the single coherent coupled slot.
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
        "geometry": "native_interaction_chains_plus_dd",
        "coefficient": "unified_complex3_jones",
        "double_diffraction": {
            "geometry": "native_dd_cascade",
            "component_id": 7,
            "row_count": int(counts.get("coupled_double_diffraction", 0)),
        },
    }


def _register_coupled_component(
    config: Coherent, topology: Any, component_counts: dict[str, int],
    extra_components: tuple[str, ...],
) -> tuple[str, ...]:
    """Record the coupled component count and export name.

    Coupled rows carry component ids 3 (diffraction last), 4 (diffraction first), 9..15
    (diffraction at an interior slot) and 7 (double diffraction); all accumulate into the single
    coupled slot. "coupled" is not a public component name, so it is enabled by the
    coupled_paths gate rather than the components set. The double-diffraction row count is
    recorded separately for audits.
    """

    if not config.coupled_paths:
        return extra_components
    component_id = topology.component_id
    component_counts["coupled"] = int(
        (
            (component_id == 3) | (component_id == 4) | (component_id == 7)
            | ((component_id > 8) & (component_id < 16))
        ).sum().item()
    )
    component_counts["coupled_double_diffraction"] = int(
        (component_id == 7).sum().item()
    )
    return extra_components + ("coupled",)


def _append_scattering(
    scene: SolverScene, config: Coherent, evaluated: Any, sidecars: Any,
) -> tuple[Any, Any, dict[str, Any] | None]:
    """Append Kirchhoff scattering rows and gate the coherent combine.

    Single-purpose solve stage (rough-surface and coherent scattering). When scattering is
    not requested it returns the inputs unchanged with ``scattering_info=None``.
    When ``scattering_coherent`` is set, the coherent-scattering gate refuses an
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
    *, config: Coherent, native_info: dict[str, Any], path_count: int,
    component_counts: dict[str, int], launch_count: int, ad_companion_launches: int = 0,
    ad_tape_bytes: int = 0, forward_time_ms: float = 0.0, peak_memory_bytes: int = 0,
    scattering_info: dict[str, Any] | None = None, transmission_dropped_rows: int = 0,
    reflection_discovery: dict[str, Any] | None = None,
) -> CoherentRadioMapMetadata:
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
    # transmission carries specular wall-penetration paths since transmission and
    # scattering carries Kirchhoff rough-surface patch paths since rough scattering.
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
        # Endpoint-connection thin_sheet contract.
        metadata_transmission = {
            "thin_sheet_straight_path_approximation": True,
            "group_delay": "geometric",
            # Endpoint pairs whose straight segment crosses more walls than max_depth.
            "dropped_rows_over_max_depth": int(transmission_dropped_rows),
        }
    else:
        metadata_transmission = None
    rayd_component_enabled = (
        components["reflection"] == "enabled" or components["diffraction"] == "enabled"
    )
    requested_config = serialize_config(config)
    effective_config = dict(requested_config)
    discovery = discovery_metadata(config.discovery, reflection_discovery)
    metadata: dict[str, Any] = {
        "max_depth": config.max_depth,
        "return_field": config.return_field,
        "export_paths": config.export_paths,
        "max_paths": config.max_paths,
        "max_paths_scope": config.max_paths_scope,
        "sort_key": config.sort_key,
        "accumulation_strategy": "coherent" if config.combine == "field" else "incoherent",
        **estimator_metadata(
            estimator="coherent",
            quantity=component_quantities(config.components),
            combine=config.combine,
            complete=bool(discovery["complete"]),
        ),
        "components": components,
        "counts": {
            "path_count": path_count,
            "valid_path_count": path_count,
            "components": component_counts,
        },
        "capability": capability,
        # diffraction AD: the real registered-companion accounting. vjp retains
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
        # Coupled higher-order compensator family (coupled reflection and diffraction>D/D->R + coupled double diffraction
        # D->D); mirrors the path solver's coupled_paths metadata block.
        "coupled_paths": _coupled_paths_metadata(config, component_counts),
        "discovery": discovery,
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
    metadata["semantic_capabilities"] = solver_capabilities("deterministic")
    return cast(CoherentRadioMapMetadata, metadata)


def _terminal_check_capacity(sidecars: EvaluatedPathSidecars) -> None:
    transaction = sidecars.capacity_transaction
    if transaction is not None:
        transaction.terminal_check()


def _solve_pipeline(scene: SolverScene, config: Coherent) -> CoherentRadioMap:
    validate_scalar_endpoint_features(
        scene.transmitters, scene.receivers, solver="deterministic"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel.radiomap requires CUDA")
    # Solve-level wall time and CUDA high-water-mark delta for the kernel
    # metadata (diffraction AD). This is AD instrumentation only: the leading
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
        #. Channel scene.compile keeps its own read: the material cache
    # token must see the live value to stay correct under in-place
    # frequency mutation.
    frequency_hz = _frequency_scalar(scene)
    evaluated, sidecars = evaluate_enumerated_paths(
        scene,
        config,
        frequency_value=frequency_hz,
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
        coherent=config.combine == "field",
        return_field=config.return_field,
        extra_components=extra_components,
        # AD modes run the same native accumulator inside its dispatch-only
        # autograd.Function so CoherentRadioMap.path_gain/field/component_power carry
        # the complete graph; none-mode keeps the bare zero-overhead kernel.
        differentiable=config.ad_mode != "none",
        # coherent scattering: opt-in coherent scattering combine (default OFF is
        # byte-identical). The solve-time gate above already rejected any
        # ensemble-only or empty-realization scene.
        scattering_coherent=config.scattering_coherent,
    )
    path_gain, field = _link_zero_gradients(config, scene, path_gain, field)
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
        transmission_dropped_rows=transmission_dropped_rows(sidecars),
        reflection_discovery=sidecars.execution.reflection_discovery,
    )
    diagnostics = None
    if config.diagnostics:
        candidate_count = sidecars.execution.candidate_count
        diagnostics = {
            "path_gain_shape": tuple(path_gain.shape),
            "field_shape": tuple(field.shape),
            "path_count": path_count,
            "component_counts": component_counts,
            "combine": config.combine,
            "accumulation_mode": "coherent" if config.combine == "field" else "incoherent",
            "native_launch_count": sidecars.execution.launch_count,
            "diffraction_accumulation": "scalar_path_accumulation",
            "visibility_rejection_count": (
                sidecars.execution.visibility_rejection_count
            ),
            "selected_edge_count": sidecars.execution.resolved_selected_edge_count(),
            "path_planning": {
                "max_paths": config.max_paths,
                "max_paths_scope": config.max_paths_scope,
                "candidate_count": candidate_count,
                "guardrail_count": sidecars.execution.guardrail_count,
                "truncated": config.max_paths is not None
                and path_count < candidate_count,
            },
        }
    result = CoherentRadioMap(
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
def solve(
    scene: Scene | SceneSnapshot, config: Coherent, *, reference_frequency_hz: float | torch.Tensor,
) -> CoherentRadioMap:
    """Run the Coherent radiomap estimator."""

    require_config_type(config, Coherent, "witwin.channel.radiomap")
    compiled = compile_scene(
        scene, reference_frequency_hz=reference_frequency_hz
    )
    result = _solve_pipeline(bind_solver_scene(compiled), config)
    record_dropped_faces(result, compiled)
    return result

# The public home of these types is ``witwin.channel.radiomap``.
Coherent.__module__ = "witwin.channel.radiomap"
CoherentRadioMap.__module__ = "witwin.channel.radiomap"
CoherentRadioMapMetadata.__module__ = "witwin.channel.radiomap"
PathTable.__module__ = "witwin.channel.radiomap"

__all__ = ["Coherent", "CoherentRadioMap", "CoherentRadioMapMetadata", "PathTable", "solve"]
