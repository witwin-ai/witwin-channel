"""Native CUDA/OptiX BDPT Monte Carlo solver with Torch tensor storage.

``docs/dev/montecarlo/README.md`` holds the ownership contract for
this solver, including the narrow ADR-008 exception that lets it consume the
public enumerated entry read-only. The sections below follow the former package
layout: configuration, public result contracts, accumulation helpers, the
ADR-022 native AD companion facades, the ADR-022 autograd wrappers,
connection-sample builders, endpoint packing, launch state, metadata, the
per-solve workspace, the shared solve pipeline, and the public entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from witwin.core import Scene, SceneSnapshot

from witwin.channel import build_info
from witwin.channel.capabilities import (
    capabilities,
    config_metadata,
    serialize_config,
)
from witwin.channel.components import (
    AD_MODES as _VALID_AD_MODES,
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
    component_availability_status,
    component_max_depth,
    validate_bounce_depth,
    validate_max_depth,
    validate_samples,
    validate_seed,
    validate_workspace_limit_bytes,
    validated_components,
)
from witwin.channel.scene.endpoints import (
    receiver_polarizations_f32,
    transmitter_polarizations_f32,
)
from witwin.channel.kernels import geometry as geometry_kernels
from witwin.channel.kernels.materials import em_layer_stack_eval
from witwin.channel.kernels.montecarlo import (
    _BDPT_COMPONENT_MATRIX_FIELDS,
    _BDPT_COMPONENT_MATRIX_ORDER,
    _BDPT_SUBPATH_SCHEMA,
    _bdpt_accumulate_bin_sum_args,
    _bdpt_mis_mode_id,
    _validate_bdpt_connection_samples,
    _validate_bdpt_subpath_state,
    bdpt_accumulate_connection_samples,
    bdpt_compact_connection_samples,
    # Re-exported for the canonical-owner contract test; this module has no
    # other caller for the two component-map buffer facades.
    bdpt_component_map_buffer as bdpt_component_map_buffer,
    bdpt_concat_connection_samples,
    bdpt_connection_variance,
    bdpt_count_valid_connection_samples,
    bdpt_endpoint_connection_samples,
    bdpt_endpoint_connection_visibility_inputs,
    bdpt_endpoint_subpath_state,
    bdpt_filter_connection_samples,
    bdpt_finalize_component_maps,
    bdpt_finalize_component_maps_backward,
    bdpt_finalize_component_maps_jvp,
    bdpt_finalize_point_components,
    bdpt_finalize_point_components_backward,
    bdpt_finalize_point_components_jvp,
    bdpt_host_vec3_tensor,
    bdpt_launch_state,
    bdpt_los_component_maps_from_matrix,
    bdpt_receiver_grid_points,
    bdpt_reflected_light_subpath_state,
    bdpt_reflection_launch_inputs,
    bdpt_sample_directions,
    bdpt_store_component_map as bdpt_store_component_map,
    bdpt_subpath_intersection_inputs,
    bdpt_transmitted_light_subpath_state,
    bdpt_transmitter_tensors,
    bdpt_zero_matrix,
)
from witwin.channel.materials import face_material_field_bundle
from witwin.channel.interactions.scattering import (
    MASK_SCATTERING,
    local_frames,
    rough_material_runtimes,
    scatter_carried_incident_power,
    scatter_direction_uniforms,
    scattered_subpath_state,
    scattering_nee_connection_samples,
    te_tm_incident_power,
    three_way_rough_probabilities,
    world_to_local,
)
from witwin.channel.interactions.transmission import (
    event_uniforms,
    layer_csr_view,
    scene_diagonal_m,
    transmission_event_probability,
    unpolarized_power_budgets,
)
from witwin.channel.propagation import EvaluatedPaths, evaluate_enumerated_paths
from witwin.channel.runtime import (
    AdLaunchLedger,
    MemoryEstimate,
    _ad_first_order_only,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_live,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
    disable_functorch,
    enforce_memory_budget,
    make_metadata,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)
from witwin.channel.scene.compiler import compile as compile_scene
from witwin.channel.scene.endpoints import (
    ReceiverGrid,
    ReceiverPoint,
    SolverScene,
    _endpoint_views,
    _validate_scalar_endpoint_boundary,
    bind_solver_scene,
    first_receiver_grid,
    require_compiled,
    validate_scalar_endpoint_features,
    vector3_tuple as _vector3_tuple,
)
from witwin.channel.scene.resources import resolve_scene_edge_policy


# --- Configuration ----------------------------------------------------------

# Public component set. transmission runs straight endpoint chains plus the
# event-selected shooting sampler for mixed chains; scattering is accepted
# plumbing that emits zero maps until its wave lands. Both are surface events
# that require at least one bounce (max_depth >= 1); transmission chains count
# wall penetrations, scattering is single-bounce in v1. component_mask bits:
# 1=los, 2=reflection, 4=diffraction, 8=transmission, 16=scattering.
# Default component set is unchanged: the new components are strictly opt-in.
_VALID_MIS = frozenset({"balance", "power_heuristic", "none"})
_VALID_RECEIVER_STRATEGIES = frozenset({"grid_area", "point_sphere"})
_VALID_ACCUMULATION_STRATEGIES = frozenset({"auto", "atomic", "staged", "compact"})


def _validate_coherent_combine(coherent: bool, components: frozenset[str]) -> None:
    """ADR-019: coherent combine is only defined for the enumerable delta/UTD
    family that carries a complex field. Refuse it loudly for the stochastic
    transmission/scattering samplers rather than silently combining Monte
    Carlo power samples as phasors.

    ADR-022 SUPERSEDES the former coherent+AD refusal: the coherent accumulate
    now carries native backward/jvp companions
    (``bdpt_accumulate_connection_samples_{backward,jvp}``, spec 6.4), so
    coherent solves are differentiable exactly like the power-domain solves."""
    if not coherent:
        return
    refused = components & {"transmission", "scattering"}
    if refused:
        raise RuntimeError(
            "coherent combine supports only {los, reflection, diffraction} "
            f"components; refused for {sorted(refused)}"
        )


def _validate_ad_readiness(ad_mode: str, components: frozenset[str]) -> None:
    """ADR-022 per-feature AD readiness gate.

    ``ad_mode='none'`` is the bitwise default and never builds a tape. Under
    ``jvp``/``vjp`` every BDPT estimator block is differentiable: the frozen
    material/EM/table/frequency/tx_power parameters ride the plan-07 field and
    ADR-015 scattering companions plus the ADR-022 subpath / endpoint /
    accumulate / finalize companions. ``max_scattering_order > 1`` is allowed:
    its extra diffuse factors ride ``scattering_table_eval_ad`` and the subpath
    ``_ad`` wrappers exactly as order 1 does. Any combination whose native
    companions are not registered fails loudly where they are dispatched (a
    missing-symbol error from ``runtime.required_symbol``); it is never silently
    detached. No component combination is refused here in v1 because every
    differentiable-parameter path has a registered companion; geometry
    gradients through the stochastic sampler are refused at the autograd
    boundary (``ad_geometry='enumerated_blocks_only'``), not here."""

    if ad_mode == "none":
        return
    # Reserved for future features whose companions are not yet registered.
    # None exist in v1, so every accepted ``ad_mode`` reaches its native
    # companions; the readiness contract is enforced loudly at dispatch.


def _validate_scattering_order(max_scattering_order: int) -> None:
    """ADR-021 D4: the diffuse multi-order cap must be a positive bounce count.

    Extracted to a module-level validator mirroring ``_validate_coherent_combine``
    so ``__post_init__`` stays within its maintenance-complexity budget.
    """
    if max_scattering_order < 1:
        raise ValueError("max_scattering_order must be >= 1")


@dataclass(frozen=True, slots=True)
class Config:
    samples: int = 4096
    seed: int = 0
    max_depth: int = 3
    max_light_depth: int | None = None
    max_diffraction_order: int = 1
    # ADR-021 D4: maximum number of diffuse-scatter events a single BDPT light
    # subpath may undergo. DEFAULT 1 is today's behavior BIT-IDENTICALLY: a
    # scattered subpath emits its NEE connection and terminates (single-bounce
    # terminal rule). >1 lifts the terminal rule so a scattered subpath
    # continues in its sampled direction and may reflect/transmit/scatter again
    # up to this cap, emitting an NEE row at every scatter vertex (power domain;
    # scattering stays excluded from the ADR-019 coherent combine).
    max_scattering_order: int = 1
    coupled_paths: bool = False
    coupled_candidate_limit: int = 1_000_000
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = (
        _DEFAULT_COMPONENTS
    )
    # Coherent combine (ADR-019). DEFAULT-OFF opt-in switch. OFF (the default)
    # keeps today's power-domain incoherent accumulation BIT-IDENTICAL (enforced
    # by a bitwise regression test). ON sums the complex projected field
    # coefficient of the enumerated delta/UTD discrete connections per
    # (tx, rx, component) and finalizes |sum|^2, so paths within a component
    # interfere coherently (matching the deterministic per-component coherent
    # power). Only the enumerable delta/UTD family carries a coherent field, so
    # coherent is scoped to components subset of {los, reflection, diffraction}
    # (coupled folds into diffraction); BDPT's stochastic transmission/scattering
    # samplers have no coherent field and are refused under coherent.
    coherent: bool = False
    mis: str = "power_heuristic"
    power_heuristic_beta: float = 2.0
    receiver_strategy: str = "grid_area"
    accumulation_strategy: str = "auto"
    sample_streams: int = 1
    diagnostics: bool = False
    export_paths: bool = False
    max_exported_paths: int | None = None
    ad_mode: str = "none"
    workspace_limit_bytes: int | None = 1 << 30

    def __post_init__(self) -> None:
        validate_samples(self.samples)
        validate_seed(self.seed)
        validate_max_depth(self.max_depth)
        max_light_depth = (
            self.max_depth if self.max_light_depth is None else self.max_light_depth
        )
        if max_light_depth < 0:
            raise ValueError("max_light_depth must be non-negative")
        if self.max_diffraction_order not in {0, 1}:
            raise ValueError("max_diffraction_order must be 0 or 1")
        _validate_scattering_order(self.max_scattering_order)
        components = validated_components(
            self.components,
            error_message="components must be a non-empty subset of {valid}",
        )
        validate_bounce_depth(
            self.max_depth,
            components,
            error_message="BDPT scattering requires max_depth >= 1",
        )
        if "diffraction" in components and self.max_diffraction_order == 0:
            raise RuntimeError("diffraction requires max_diffraction_order > 0")
        if self.coupled_paths:
            if self.max_depth < 2:
                raise RuntimeError("coupled paths require max_depth >= 2")
            if not {"reflection", "diffraction"}.issubset(components):
                raise RuntimeError(
                    "coupled paths require reflection and diffraction components"
                )
        if (
            self.coupled_candidate_limit <= 0
            or self.coupled_candidate_limit > 1_000_000
        ):
            raise ValueError("coupled_candidate_limit must be in [1, 1000000]")
        if self.mis not in _VALID_MIS:
            raise ValueError(f"mis must be one of {sorted(_VALID_MIS)}")
        if self.power_heuristic_beta <= 0.0:
            raise ValueError("power_heuristic_beta must be positive")
        if self.receiver_strategy not in _VALID_RECEIVER_STRATEGIES:
            raise ValueError(
                f"receiver_strategy must be one of {sorted(_VALID_RECEIVER_STRATEGIES)}"
            )
        if self.accumulation_strategy not in _VALID_ACCUMULATION_STRATEGIES:
            raise ValueError(
                f"accumulation_strategy must be one of {sorted(_VALID_ACCUMULATION_STRATEGIES)}"
            )
        if self.sample_streams <= 0:
            raise ValueError("sample_streams must be positive")
        if self.max_exported_paths is not None and self.max_exported_paths < 0:
            raise ValueError("max_exported_paths must be non-negative")
        if self.ad_mode not in _VALID_AD_MODES:
            raise ValueError(
                "ad_mode must be one of "
                f"{sorted(_VALID_AD_MODES)} (ADR-022 lifted the BDPT AD "
                "refusal to fixed-topology jvp/vjp)"
            )
        _validate_coherent_combine(self.coherent, components)
        _validate_ad_readiness(self.ad_mode, components)
        validate_workspace_limit_bytes(self.workspace_limit_bytes)

        object.__setattr__(self, "max_light_depth", max_light_depth)
        object.__setattr__(self, "components", components)
# --- Public result contracts ------------------------------------------------

@dataclass(frozen=True, slots=True)
class BDPTPathSamples:
    topology: torch.Tensor
    contribution: torch.Tensor
    pdf: torch.Tensor
    mis_weight: torch.Tensor
    component_id: torch.Tensor
    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    grid_linear_id: torch.Tensor
    light_depth: torch.Tensor
    sensor_depth: torch.Tensor
    path_length_m: torch.Tensor


@dataclass(frozen=True, slots=True)
class Result:
    path_gain: torch.Tensor
    component_power: dict[str, torch.Tensor]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any] | None = None
    component_maps: dict[str, torch.Tensor] | None = None
    variance: torch.Tensor | None = None
    path_samples: BDPTPathSamples | None = None
# --- Result accumulation and path-sample assembly ---------------------------

def _component_maps_from_matrices(
    component_matrices: dict[str, torch.Tensor],
    *,
    rows: int,
    cols: int,
) -> dict[str, torch.Tensor]:
    return {
        name: bdpt_los_component_maps_from_matrix(component, rows=rows, cols=cols)
        for name, component in component_matrices.items()
    }


def _path_samples_from_connection_export(
    exported: dict[str, torch.Tensor],
) -> BDPTPathSamples:
    return BDPTPathSamples(
        topology=exported["topology"],
        contribution=exported["contribution"],
        pdf=exported["pdf"],
        mis_weight=exported["mis_weight"],
        component_id=exported["component_id"],
        valid=exported["valid"],
        tx_id=exported["tx_id"],
        rx_id=exported["rx_id"],
        grid_linear_id=exported["grid_linear_id"],
        light_depth=exported["light_depth"],
        sensor_depth=exported["sensor_depth"],
        path_length_m=exported["path_length_m"],
    )


def _empty_path_samples(reference: torch.Tensor) -> BDPTPathSamples:
    device = reference.device
    empty_float = torch.empty((0,), device=device, dtype=torch.float32)
    empty_int = torch.empty((0,), device=device, dtype=torch.int32)
    return BDPTPathSamples(
        topology=torch.empty((0, 4), device=device, dtype=torch.int32),
        contribution=empty_float,
        pdf=empty_float.clone(),
        mis_weight=empty_float.clone(),
        component_id=empty_int,
        valid=torch.empty((0,), device=device, dtype=torch.bool),
        tx_id=empty_int.clone(),
        rx_id=empty_int.clone(),
        grid_linear_id=empty_int.clone(),
        light_depth=empty_int.clone(),
        sensor_depth=empty_int.clone(),
        path_length_m=empty_float.clone(),
    )
# --- ADR-022 native AD companion facades ------------------------------------

# Differentiable subpath output fields (the four the companions carry cotangents
# for). Every other subpath field is frozen structure.
_BDPT_SUBPATH_TANGENT_FIELDS = (
    "tangent_field_real",
    "tangent_field_imag",
    "tangent_throughput_real",
    "tangent_throughput_imag",
)


def _validate_subpath_field_cotangents(
    grad_field_real: torch.Tensor | None,
    grad_field_imag: torch.Tensor | None,
    grad_throughput_real: torch.Tensor | None,
    grad_throughput_imag: torch.Tensor | None,
    *,
    count: int,
) -> None:
    for name, tensor, trailing in (
        ("grad_field_real", grad_field_real, (3,)),
        ("grad_field_imag", grad_field_imag, (3,)),
        ("grad_throughput_real", grad_throughput_real, None),
        ("grad_throughput_imag", grad_throughput_imag, None),
    ):
        if tensor is None:
            continue
        ndim = 2 if trailing is not None else 1
        validate_cuda_tensor(
            name,
            tensor,
            dtype=torch.float32,
            ndim=ndim,
            trailing_shape=trailing,  # type: ignore[arg-type]
            require_contiguous=False,
        )
        if int(tensor.shape[0]) != int(count):
            raise ValueError(f"{name} must have {count} rows")


def bdpt_reflected_light_subpath_state_backward(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_thickness: torch.Tensor,
    *,
    frequency_hz: float,
    grad_field_real: torch.Tensor | None = None,
    grad_field_imag: torch.Tensor | None = None,
    grad_throughput_real: torch.Tensor | None = None,
    grad_throughput_imag: torch.Tensor | None = None,
    need_grad_material: bool = False,
    need_grad_field_in: bool = False,
    need_grad_frequency: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`bdpt_reflected_light_subpath_state` (spec 6.1).

    ``grad_field_in = O^H grad_field_out`` (``O`` = ReflectFrame rotation x
    Fresnel diag); material partials via ``field_transport_ad.cuh::stack_rt_dual``
    accumulate into the shared CSR/material grads by ``atomicAdd``; the frequency
    grad by ``atomicAdd``. Off-flag groups are ``None``."""

    _validate_bdpt_subpath_state("light", light, None)
    count = int(light["origin"].shape[0])
    _validate_subpath_field_cotangents(
        grad_field_real,
        grad_field_imag,
        grad_throughput_real,
        grad_throughput_imag,
        count=count,
    )
    exported = _required_native_op("bdpt_reflected_light_subpath_state_backward")(  # type: ignore[operator]
        light,
        intersection,
        material_gain,
        material_valid,
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        material_thickness,
        float(frequency_hz),
        grad_field_real,
        grad_field_imag,
        grad_throughput_real,
        grad_throughput_imag,
        bool(need_grad_material),
        bool(need_grad_field_in),
        bool(need_grad_frequency),
    )
    expected = {
        "grad_eps_r",
        "grad_sigma_e",
        "grad_gain",
        "grad_thickness",
        "grad_light_field_real",
        "grad_light_field_imag",
        "grad_light_throughput_real",
        "grad_light_throughput_imag",
        "grad_frequency",
    }
    if not isinstance(exported, dict) or set(exported) != expected:
        raise TypeError(
            "_channel.bdpt_reflected_light_subpath_state_backward "
            "returned unexpected fields"
        )
    return exported


def bdpt_reflected_light_subpath_state_jvp(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_thickness: torch.Tensor,
    *,
    frequency_hz: float,
    tangent_eps_r: torch.Tensor | None = None,
    tangent_sigma_e: torch.Tensor | None = None,
    tangent_gain: torch.Tensor | None = None,
    tangent_thickness: torch.Tensor | None = None,
    tangent_frequency: float = 0.0,
    tangent_light_field_real: torch.Tensor | None = None,
    tangent_light_field_imag: torch.Tensor | None = None,
    tangent_light_throughput_real: torch.Tensor | None = None,
    tangent_light_throughput_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`bdpt_reflected_light_subpath_state` (spec 6.1).

    The differentiable reflected material set is ``{eps_r, sigma_e, gain,
    thickness}`` (``mu_r`` is frozen); ``tangent_gain`` maps to the native
    kernel's gain tangent slot."""

    _validate_bdpt_subpath_state("light", light, None)
    exported = _required_native_op("bdpt_reflected_light_subpath_state_jvp")(  # type: ignore[operator]
        light,
        intersection,
        material_gain,
        material_valid,
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        material_thickness,
        float(frequency_hz),
        tangent_eps_r,
        tangent_sigma_e,
        tangent_gain,
        tangent_thickness,
        float(tangent_frequency),
        tangent_light_field_real,
        tangent_light_field_imag,
        tangent_light_throughput_real,
        tangent_light_throughput_imag,
    )
    if not isinstance(exported, dict) or set(exported) != set(
        _BDPT_SUBPATH_TANGENT_FIELDS
    ):
        raise TypeError(
            "_channel.bdpt_reflected_light_subpath_state_jvp "
            "returned unexpected fields"
        )
    return exported


def bdpt_transmitted_light_subpath_state_backward(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    face_material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency_hz: float,
    grad_field_real: torch.Tensor | None = None,
    grad_field_imag: torch.Tensor | None = None,
    grad_throughput_real: torch.Tensor | None = None,
    grad_throughput_imag: torch.Tensor | None = None,
    need_grad_layers: bool = False,
    need_grad_field_in: bool = False,
    need_grad_frequency: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`bdpt_transmitted_light_subpath_state` (spec 6.2).

    Layer grads via ``stack_rt_dual`` folded onto the CSR by ``atomicAdd``
    (identical to ``em_layer_stack_backward`` / the transmission-sequence
    backward). Off-flag groups are ``None``."""

    _validate_bdpt_subpath_state("light", light, None)
    count = int(light["origin"].shape[0])
    _validate_subpath_field_cotangents(
        grad_field_real,
        grad_field_imag,
        grad_throughput_real,
        grad_throughput_imag,
        count=count,
    )
    exported = _required_native_op("bdpt_transmitted_light_subpath_state_backward")(  # type: ignore[operator]
        light,
        intersection,
        face_material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
        grad_field_real,
        grad_field_imag,
        grad_throughput_real,
        grad_throughput_imag,
        bool(need_grad_layers),
        bool(need_grad_field_in),
        bool(need_grad_frequency),
    )
    expected = {
        "grad_layer_thickness",
        "grad_layer_eps_r",
        "grad_layer_sigma_e",
        "grad_light_field_real",
        "grad_light_field_imag",
        "grad_light_throughput_real",
        "grad_light_throughput_imag",
        "grad_frequency",
    }
    if not isinstance(exported, dict) or set(exported) != expected:
        raise TypeError(
            "_channel.bdpt_transmitted_light_subpath_state_backward "
            "returned unexpected fields"
        )
    return exported


def bdpt_transmitted_light_subpath_state_jvp(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    face_material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency_hz: float,
    tangent_layer_thickness: torch.Tensor | None = None,
    tangent_layer_eps_r: torch.Tensor | None = None,
    tangent_layer_sigma_e: torch.Tensor | None = None,
    tangent_frequency: float = 0.0,
    tangent_light_field_real: torch.Tensor | None = None,
    tangent_light_field_imag: torch.Tensor | None = None,
    tangent_light_throughput_real: torch.Tensor | None = None,
    tangent_light_throughput_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`bdpt_transmitted_light_subpath_state` (spec 6.2)."""

    _validate_bdpt_subpath_state("light", light, None)
    exported = _required_native_op("bdpt_transmitted_light_subpath_state_jvp")(  # type: ignore[operator]
        light,
        intersection,
        face_material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
        tangent_layer_thickness,
        tangent_layer_eps_r,
        tangent_layer_sigma_e,
        float(tangent_frequency),
        tangent_light_field_real,
        tangent_light_field_imag,
        tangent_light_throughput_real,
        tangent_light_throughput_imag,
    )
    if not isinstance(exported, dict) or set(exported) != set(
        _BDPT_SUBPATH_TANGENT_FIELDS
    ):
        raise TypeError(
            "_channel.bdpt_transmitted_light_subpath_state_jvp "
            "returned unexpected fields"
        )
    return exported


def bdpt_endpoint_connection_samples_backward(
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    frequency_hz: float,
    samples_per_tx: int,
    mis: str,
    beta: float,
    strategy_count: int,
    max_paths: int | None,
    grad_contribution: torch.Tensor | None = None,
    need_grad_field: bool = False,
    need_grad_frequency: bool = False,
    need_grad_tx_power: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`bdpt_endpoint_connection_samples` (spec 6.3).

    ``contribution = P_src |F|^2 (lambda/(4 pi L))^2 / N``: ``d/dF = 2 conj(F)
    rest`` folds onto the light/sensor field cotangents (direct stores),
    ``d/d lambda`` chains into ``grad_frequency`` (``atomicAdd``), ``d/d P_src``
    onto ``grad_tx_power`` (``atomicAdd``). ``L``, ``N``, visibility and MIS are
    frozen."""

    _validate_bdpt_subpath_state("light", light, None)
    _validate_bdpt_subpath_state("sensor", sensor, None)
    if grad_contribution is not None:
        validate_cuda_tensor(
            "grad_contribution",
            grad_contribution,
            dtype=torch.float32,
            ndim=1,
            require_contiguous=False,
        )
    max_paths_value = -1 if max_paths is None else int(max_paths)
    exported = _required_native_op("bdpt_endpoint_connection_samples_backward")(  # type: ignore[operator]
        light,
        sensor,
        float(frequency_hz),
        int(samples_per_tx),
        int(_bdpt_mis_mode_id(mis)),
        float(beta),
        int(strategy_count),
        int(max_paths_value),
        grad_contribution,
        bool(need_grad_field),
        bool(need_grad_frequency),
        bool(need_grad_tx_power),
    )
    expected = {
        "grad_light_field_real",
        "grad_light_field_imag",
        "grad_sensor_field_real",
        "grad_sensor_field_imag",
        "grad_frequency",
        "grad_tx_power",
    }
    if not isinstance(exported, dict) or set(exported) != expected:
        raise TypeError(
            "_channel.bdpt_endpoint_connection_samples_backward "
            "returned unexpected fields"
        )
    return exported


def bdpt_endpoint_connection_samples_jvp(
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    frequency_hz: float,
    samples_per_tx: int,
    mis: str,
    beta: float,
    strategy_count: int,
    max_paths: int | None,
    tangent_light_field_real: torch.Tensor | None = None,
    tangent_light_field_imag: torch.Tensor | None = None,
    tangent_sensor_field_real: torch.Tensor | None = None,
    tangent_sensor_field_imag: torch.Tensor | None = None,
    tangent_frequency: float = 0.0,
    tangent_tx_power: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`bdpt_endpoint_connection_samples` (spec 6.3)."""

    _validate_bdpt_subpath_state("light", light, None)
    _validate_bdpt_subpath_state("sensor", sensor, None)
    max_paths_value = -1 if max_paths is None else int(max_paths)
    exported = _required_native_op("bdpt_endpoint_connection_samples_jvp")(  # type: ignore[operator]
        light,
        sensor,
        float(frequency_hz),
        int(samples_per_tx),
        int(_bdpt_mis_mode_id(mis)),
        float(beta),
        int(strategy_count),
        int(max_paths_value),
        tangent_light_field_real,
        tangent_light_field_imag,
        tangent_sensor_field_real,
        tangent_sensor_field_imag,
        float(tangent_frequency),
        tangent_tx_power,
    )
    if not isinstance(exported, dict) or set(exported) != {"tangent_contribution"}:
        raise TypeError(
            "_channel.bdpt_endpoint_connection_samples_jvp "
            "returned unexpected fields"
        )
    return exported


def bdpt_accumulate_connection_samples_forward_ad(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    accumulation_strategy: str,
    combine_domain: str,
    coeff_real: torch.Tensor,
    coeff_imag: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Accumulate forward that also returns the coherent bin-sum buffers.

    ADR-022 spec 6.4 supervisor ruling: the coherent forward returns the
    per-component phasor bin sums (``S_b``) as non-differentiable outputs so the
    coherent backward can read them without a second atomic-double reduction.
    Returns ``(component_matrices, bin_sums)`` where ``bin_sums`` is an ordered
    tuple (native return order, empty for the power domain) forwarded
    positionally to the backward companion. Numerically the component matrices
    are bitwise the primal :func:`bdpt_accumulate_connection_samples` result."""

    strategy_ids = {"atomic": 0, "staged": 1, "compact": 2}
    combine_ids = {"power": 0, "coherent": 1}
    exported = _required_native_op("bdpt_accumulate_connection_samples")(  # type: ignore[operator]
        samples,
        int(tx_count),
        int(rx_count),
        int(strategy_ids[accumulation_strategy]),
        int(combine_ids[combine_domain]),
        coeff_real,
        coeff_imag,
    )
    if not isinstance(exported, dict) or not _BDPT_COMPONENT_MATRIX_FIELDS.issubset(
        exported
    ):
        raise TypeError(
            "_channel.bdpt_accumulate_connection_samples returned unexpected fields"
        )
    matrices = {name: exported[name] for name in _BDPT_COMPONENT_MATRIX_ORDER}
    bin_sums = tuple(
        tensor
        for name, tensor in exported.items()
        if name not in _BDPT_COMPONENT_MATRIX_FIELDS
    )
    return matrices, bin_sums


def bdpt_accumulate_connection_samples_backward(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    combine_domain: str,
    bin_sums: tuple[torch.Tensor, ...] = (),
    grad_path_gain: torch.Tensor | None = None,
    grad_los: torch.Tensor | None = None,
    grad_reflection: torch.Tensor | None = None,
    grad_diffraction: torch.Tensor | None = None,
    grad_transmission: torch.Tensor | None = None,
    grad_scattering: torch.Tensor | None = None,
    need_grad_contribution: bool = False,
    need_grad_coeff: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`bdpt_accumulate_connection_samples`, both domains (spec 6.4).

    Power: ``grad_contribution_r = mis_r grad_M[bin(r)]`` (gather, no atomics);
    this is also the concat-backward split view. Coherent:
    ``grad_c_r = 2 grad_P[b] S_b`` read from the forward-retained ``bin_sums``
    (supervisor ruling: no in-backward re-reduction). Both gathers are
    deterministic. ``mis``, the accumulation strategy, and the index structure
    are frozen: the VJP does not read the sample coefficients, only the six
    output-matrix cotangents plus (coherent) the ten forward phasor bin sums."""

    _validate_bdpt_connection_samples("samples", samples, None)
    combine_ids = {"power": 0, "coherent": 1}
    if combine_domain not in combine_ids:
        raise ValueError("combine_domain must be 'power' or 'coherent'")
    bin_args = _bdpt_accumulate_bin_sum_args(combine_domain, bin_sums)
    exported = _required_native_op("bdpt_accumulate_connection_samples_backward")(  # type: ignore[operator]
        samples,
        int(tx_count),
        int(rx_count),
        int(combine_ids[combine_domain]),
        grad_path_gain,
        grad_los,
        grad_reflection,
        grad_diffraction,
        grad_transmission,
        grad_scattering,
        *bin_args,
        bool(need_grad_contribution),
        bool(need_grad_coeff),
    )
    if not isinstance(exported, dict) or set(exported) != {
        "grad_contribution",
        "grad_coeff_real",
        "grad_coeff_imag",
    }:
        raise TypeError(
            "_channel.bdpt_accumulate_connection_samples_backward "
            "returned unexpected fields"
        )
    return exported


def bdpt_accumulate_connection_samples_jvp(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    combine_domain: str,
    bin_sums: tuple[torch.Tensor, ...] = (),
    tangent_contribution: torch.Tensor | None = None,
    tangent_coeff_real: torch.Tensor | None = None,
    tangent_coeff_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`bdpt_accumulate_connection_samples`, both domains (spec 6.4).

    Power: ``t_M[b] = SUM_r mis_r tangent_contribution_r``. Coherent:
    ``t_P = 2 Re(conj(S_b) t_S_b)``, ``t_S_b = SUM_r t_c_r``, with ``S_b`` read
    from the forward-retained ``bin_sums`` (supervisor ruling: no re-reduction).
    Both are fixed-order per-bin sums (deterministic, no float atomics on the
    JVP); the accumulation strategy and sample coefficients are frozen out."""

    _validate_bdpt_connection_samples("samples", samples, None)
    combine_ids = {"power": 0, "coherent": 1}
    if combine_domain not in combine_ids:
        raise ValueError("combine_domain must be 'power' or 'coherent'")
    bin_args = _bdpt_accumulate_bin_sum_args(combine_domain, bin_sums)
    exported = _required_native_op("bdpt_accumulate_connection_samples_jvp")(  # type: ignore[operator]
        samples,
        int(tx_count),
        int(rx_count),
        int(combine_ids[combine_domain]),
        tangent_contribution,
        tangent_coeff_real,
        tangent_coeff_imag,
        *bin_args,
    )
    expected = {
        "tangent_path_gain",
        "tangent_los",
        "tangent_reflection",
        "tangent_diffraction",
        "tangent_transmission",
        "tangent_scattering",
    }
    if not isinstance(exported, dict) or set(exported) != expected:
        raise TypeError(
            "_channel.bdpt_accumulate_connection_samples_jvp "
            "returned unexpected fields"
        )
    return exported
# --- ADR-022 subpath / endpoint autograd.Function wrappers ------------------

# Subpath field order and the four differentiable slots (spec 6.1/6.2).
_SUBPATH_FIELDS = tuple(_BDPT_SUBPATH_SCHEMA)
_SUBPATH_DIFF_FIELDS = (
    "field_real",
    "field_imag",
    "throughput_real",
    "throughput_imag",
)
_SUBPATH_DIFF_INDEX = {
    name: _SUBPATH_FIELDS.index(name) for name in _SUBPATH_DIFF_FIELDS
}


def _subpath_with_fields(
    base: dict[str, torch.Tensor],
    field_real: torch.Tensor,
    field_imag: torch.Tensor,
    throughput_real: torch.Tensor,
    throughput_imag: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Rebuild a subpath dict overriding only the four differentiable slots."""

    out = dict(base)
    out["field_real"] = field_real
    out["field_imag"] = field_imag
    out["throughput_real"] = throughput_real
    out["throughput_imag"] = throughput_imag
    return out


def _subpath_output_tuple(out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(out[name] for name in _SUBPATH_FIELDS)


def _mark_subpath_structural(ctx, output) -> None:  # type: ignore[no-untyped-def]
    """Mark every subpath output field non-differentiable except the four
    field/throughput slots that carry the advance's cotangents."""

    structural = [
        output[index]
        for index, name in enumerate(_SUBPATH_FIELDS)
        if name not in _SUBPATH_DIFF_FIELDS
    ]
    ctx.mark_non_differentiable(*structural)


def _subpath_backward_needs(  # type: ignore[no-untyped-def]
    needs_input_grad, material_indices: tuple[int, ...]
) -> tuple[bool, bool, bool]:
    """Derive the (field-in, material/layers, frequency) need flags for a subpath
    advance backward from ``ctx.needs_input_grad``.

    ``material_indices`` selects the differentiable material/layer slots (they
    differ between the reflected and transmitted advances); slots 0-3 are the
    upstream field/throughput inputs and slot 8 is the carrier frequency."""

    need_field_in = any(bool(needs_input_grad[i]) for i in range(4))
    need_material = any(bool(needs_input_grad[i]) for i in material_indices)
    need_frequency = bool(needs_input_grad[8])
    return need_field_in, need_material, need_frequency


# ---------------------------------------------------------------------------
# 6.1 reflected light subpath advance
# ---------------------------------------------------------------------------


class _BdptReflectedSubpathAdFunction(torch.autograd.Function):
    """Differentiable specular-reflection subpath advance (spec 6.1).

    Differentiable inputs: the upstream light field/throughput (4 tensors) and
    the per-face material eps_r / sigma_e / thickness plus the carrier
    frequency. Frozen (reject loudly): hit-point geometry (the ``intersection``
    dict), material_valid, material_gain, mu_r, and every structural subpath
    field. Hit geometry stays frozen in v1 (stochastic-sampler stance,
    ``ad_geometry='enumerated_blocks_only'``)."""

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        field_real,
        field_imag,
        throughput_real,
        throughput_imag,
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        material_thickness,
        frequency,
        frequency_value,
        base_light,
        intersection,
        material_gain,
        material_valid,
    ):
        light = _subpath_with_fields(
            base_light, field_real, field_imag, throughput_real, throughput_imag
        )
        out = bdpt_reflected_light_subpath_state(
            light,
            intersection,
            material_gain=material_gain,
            material_valid=material_valid,
            material_eps_r=material_eps_r,
            material_sigma_e=material_sigma_e,
            material_mu_r=material_mu_r,
            material_thickness=material_thickness,
            frequency_hz=frequency_value,
        )
        return _subpath_output_tuple(out)

    @staticmethod
    def setup_context(ctx, inputs, output):  # type: ignore[no-untyped-def]
        ctx.set_materialize_grads(False)
        (
            field_real,
            field_imag,
            throughput_real,
            throughput_imag,
            material_eps_r,
            material_sigma_e,
            material_mu_r,
            material_thickness,
            frequency,
            frequency_value,
            base_light,
            intersection,
            material_gain,
            material_valid,
        ) = inputs
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (
                field_real,
                field_imag,
                throughput_real,
                throughput_imag,
                material_eps_r,
                material_sigma_e,
                material_mu_r,
                material_thickness,
            )
        )
        ctx.frequency_value = frequency_value
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.base_light = {
            name: torch.autograd.forward_ad.unpack_dual(value).primal
            if isinstance(value, torch.Tensor)
            else value
            for name, value in base_light.items()
        }
        ctx.intersection = intersection
        ctx.material_gain = material_gain
        ctx.material_valid = material_valid
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        _mark_subpath_structural(ctx, output)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):  # type: ignore[no-untyped-def]
        none_grads = (None,) * 14
        _ad_reject_fixed_inputs(
            "bdpt_reflected_light_subpath_state_ad",
            ctx.needs_input_grad,
            ((6, "material_mu_r"),),
        )
        need_field_in, need_material, need_frequency = _subpath_backward_needs(
            ctx.needs_input_grad, (4, 5, 7)
        )
        grad_field_real = grad_outputs[_SUBPATH_DIFF_INDEX["field_real"]]
        grad_field_imag = grad_outputs[_SUBPATH_DIFF_INDEX["field_imag"]]
        grad_throughput_real = grad_outputs[_SUBPATH_DIFF_INDEX["throughput_real"]]
        grad_throughput_imag = grad_outputs[_SUBPATH_DIFF_INDEX["throughput_imag"]]
        grads = (
            grad_field_real,
            grad_field_imag,
            grad_throughput_real,
            grad_throughput_imag,
        )
        if not (need_field_in or need_material or need_frequency) or all(
            value is None for value in grads
        ):
            return none_grads
        (
            field_real,
            field_imag,
            throughput_real,
            throughput_imag,
            material_eps_r,
            material_sigma_e,
            material_mu_r,
            material_thickness,
        ) = ctx.saved_tensors
        light = _subpath_with_fields(
            ctx.base_light, field_real, field_imag, throughput_real, throughput_imag
        )
        out = bdpt_reflected_light_subpath_state_backward(
            light,
            ctx.intersection,
            ctx.material_gain,
            ctx.material_valid,
            material_eps_r,
            material_sigma_e,
            material_mu_r,
            material_thickness,
            frequency_hz=ctx.frequency_value,
            grad_field_real=grad_field_real,
            grad_field_imag=grad_field_imag,
            grad_throughput_real=grad_throughput_real,
            grad_throughput_imag=grad_throughput_imag,
            need_grad_material=need_material,
            need_grad_field_in=need_field_in,
            need_grad_frequency=need_frequency,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_light_field_real"] if ctx.needs_input_grad[0] else None,
            out["grad_light_field_imag"] if ctx.needs_input_grad[1] else None,
            out["grad_light_throughput_real"] if ctx.needs_input_grad[2] else None,
            out["grad_light_throughput_imag"] if ctx.needs_input_grad[3] else None,
            out["grad_eps_r"] if ctx.needs_input_grad[4] else None,
            out["grad_sigma_e"] if ctx.needs_input_grad[5] else None,
            None,
            out["grad_thickness"] if ctx.needs_input_grad[7] else None,
            grad_frequency,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(  # type: ignore[no-untyped-def]
        ctx,
        t_field_real,
        t_field_imag,
        t_throughput_real,
        t_throughput_imag,
        t_eps_r,
        t_sigma_e,
        t_mu_r,
        t_thickness,
        t_frequency,
        _t_frequency_value,
        _t_base_light,
        _t_intersection,
        _t_material_gain,
        _t_material_valid,
    ):
        _ad_reject_fixed_tangents(
            "bdpt_reflected_light_subpath_state_ad", ((t_mu_r, "material_mu_r"),)
        )
        saved = ctx.saved_tensors
        light = _subpath_with_fields(ctx.base_light, saved[0], saved[1], saved[2], saved[3])
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        tangents = {
            "tangent_light_field_real": _ad_native_tangent_or_none(t_field_real),
            "tangent_light_field_imag": _ad_native_tangent_or_none(t_field_imag),
            "tangent_light_throughput_real": _ad_native_tangent_or_none(
                t_throughput_real
            ),
            "tangent_light_throughput_imag": _ad_native_tangent_or_none(
                t_throughput_imag
            ),
            "tangent_eps_r": _ad_native_tangent_or_none(t_eps_r),
            "tangent_sigma_e": _ad_native_tangent_or_none(t_sigma_e),
            "tangent_thickness": _ad_native_tangent_or_none(t_thickness),
        }
        if tangent_frequency == 0.0 and all(v is None for v in tangents.values()):
            return (None,) * len(_SUBPATH_FIELDS)
        with disable_functorch():
            out = bdpt_reflected_light_subpath_state_jvp(
                light,
                ctx.intersection,
                ctx.material_gain,
                ctx.material_valid,
                _ad_native_tensor(saved[4]),
                _ad_native_tensor(saved[5]),
                _ad_native_tensor(saved[6]),
                _ad_native_tensor(saved[7]),
                frequency_hz=ctx.frequency_value,
                tangent_frequency=tangent_frequency,
                **tangents,
            )
        return _subpath_tangent_outputs(out)


def _subpath_tangent_outputs(out: dict[str, torch.Tensor]) -> tuple:
    """Map the four native tangent fields onto the full subpath output slots."""

    result: list[torch.Tensor | None] = [None] * len(_SUBPATH_FIELDS)
    result[_SUBPATH_DIFF_INDEX["field_real"]] = out["tangent_field_real"]
    result[_SUBPATH_DIFF_INDEX["field_imag"]] = out["tangent_field_imag"]
    result[_SUBPATH_DIFF_INDEX["throughput_real"]] = out["tangent_throughput_real"]
    result[_SUBPATH_DIFF_INDEX["throughput_imag"]] = out["tangent_throughput_imag"]
    return tuple(result)


def bdpt_reflected_light_subpath_state_ad(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    *,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_thickness: torch.Tensor,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_reflected_light_subpath_state` (spec 6.1)."""

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    base_light = {
        name: value
        for name, value in light.items()
        if name not in _SUBPATH_DIFF_FIELDS
    }
    values = _BdptReflectedSubpathAdFunction.apply(
        light["field_real"],
        light["field_imag"],
        light["throughput_real"],
        light["throughput_imag"],
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        material_thickness,
        frequency,
        float(frequency_value),
        base_light,
        intersection,
        material_gain,
        material_valid,
    )
    return dict(zip(_SUBPATH_FIELDS, values, strict=True))


# ---------------------------------------------------------------------------
# 6.2 transmitted light subpath advance
# ---------------------------------------------------------------------------


class _BdptTransmittedSubpathAdFunction(torch.autograd.Function):
    """Differentiable slab-transmission subpath advance (spec 6.2).

    Differentiable inputs: the upstream light field/throughput (4 tensors) and
    the CSR layer thickness / eps_r / sigma_e plus the carrier frequency. Frozen
    (reject loudly): hit-point geometry, face_material_id, the CSR index arrays
    (layer_offset / layer_count), layer_mu_r, and every structural subpath
    field."""

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        field_real,
        field_imag,
        throughput_real,
        throughput_imag,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        frequency_value,
        base_light,
        intersection,
        face_material_id,
        layer_offset,
        layer_count,
    ):
        light = _subpath_with_fields(
            base_light, field_real, field_imag, throughput_real, throughput_imag
        )
        out = bdpt_transmitted_light_subpath_state(
            light,
            intersection,
            face_material_id=face_material_id,
            layer_offset=layer_offset,
            layer_count=layer_count,
            layer_thickness_m=layer_thickness_m,
            layer_eps_r=layer_eps_r,
            layer_sigma_e=layer_sigma_e,
            layer_mu_r=layer_mu_r,
            frequency_hz=frequency_value,
        )
        return _subpath_output_tuple(out)

    @staticmethod
    def setup_context(ctx, inputs, output):  # type: ignore[no-untyped-def]
        ctx.set_materialize_grads(False)
        (
            field_real,
            field_imag,
            throughput_real,
            throughput_imag,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            frequency,
            frequency_value,
            base_light,
            intersection,
            face_material_id,
            layer_offset,
            layer_count,
        ) = inputs
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (
                field_real,
                field_imag,
                throughput_real,
                throughput_imag,
                layer_thickness_m,
                layer_eps_r,
                layer_sigma_e,
                layer_mu_r,
            )
        )
        ctx.frequency_value = frequency_value
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.base_light = {
            name: torch.autograd.forward_ad.unpack_dual(value).primal
            if isinstance(value, torch.Tensor)
            else value
            for name, value in base_light.items()
        }
        ctx.intersection = intersection
        ctx.face_material_id = face_material_id
        ctx.layer_offset = layer_offset
        ctx.layer_count = layer_count
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        _mark_subpath_structural(ctx, output)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):  # type: ignore[no-untyped-def]
        none_grads = (None,) * 15
        _ad_reject_fixed_inputs(
            "bdpt_transmitted_light_subpath_state_ad",
            ctx.needs_input_grad,
            ((7, "layer_mu_r"),),
        )
        need_field_in, need_layers, need_frequency = _subpath_backward_needs(
            ctx.needs_input_grad, (4, 5, 6)
        )
        grad_field_real = grad_outputs[_SUBPATH_DIFF_INDEX["field_real"]]
        grad_field_imag = grad_outputs[_SUBPATH_DIFF_INDEX["field_imag"]]
        grad_throughput_real = grad_outputs[_SUBPATH_DIFF_INDEX["throughput_real"]]
        grad_throughput_imag = grad_outputs[_SUBPATH_DIFF_INDEX["throughput_imag"]]
        grads = (
            grad_field_real,
            grad_field_imag,
            grad_throughput_real,
            grad_throughput_imag,
        )
        if not (need_field_in or need_layers or need_frequency) or all(
            value is None for value in grads
        ):
            return none_grads
        (
            field_real,
            field_imag,
            throughput_real,
            throughput_imag,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
        ) = ctx.saved_tensors
        light = _subpath_with_fields(
            ctx.base_light, field_real, field_imag, throughput_real, throughput_imag
        )
        out = bdpt_transmitted_light_subpath_state_backward(
            light,
            ctx.intersection,
            ctx.face_material_id,
            ctx.layer_offset,
            ctx.layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            frequency_hz=ctx.frequency_value,
            grad_field_real=grad_field_real,
            grad_field_imag=grad_field_imag,
            grad_throughput_real=grad_throughput_real,
            grad_throughput_imag=grad_throughput_imag,
            need_grad_layers=need_layers,
            need_grad_field_in=need_field_in,
            need_grad_frequency=need_frequency,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_light_field_real"] if ctx.needs_input_grad[0] else None,
            out["grad_light_field_imag"] if ctx.needs_input_grad[1] else None,
            out["grad_light_throughput_real"] if ctx.needs_input_grad[2] else None,
            out["grad_light_throughput_imag"] if ctx.needs_input_grad[3] else None,
            out["grad_layer_thickness"] if ctx.needs_input_grad[4] else None,
            out["grad_layer_eps_r"] if ctx.needs_input_grad[5] else None,
            out["grad_layer_sigma_e"] if ctx.needs_input_grad[6] else None,
            None,
            grad_frequency,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(  # type: ignore[no-untyped-def]
        ctx,
        t_field_real,
        t_field_imag,
        t_throughput_real,
        t_throughput_imag,
        t_thickness,
        t_eps_r,
        t_sigma_e,
        t_mu_r,
        t_frequency,
        _t_frequency_value,
        _t_base_light,
        _t_intersection,
        _t_face_material_id,
        _t_layer_offset,
        _t_layer_count,
    ):
        _ad_reject_fixed_tangents(
            "bdpt_transmitted_light_subpath_state_ad", ((t_mu_r, "layer_mu_r"),)
        )
        saved = ctx.saved_tensors
        light = _subpath_with_fields(ctx.base_light, saved[0], saved[1], saved[2], saved[3])
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        tangents = {
            "tangent_light_field_real": _ad_native_tangent_or_none(t_field_real),
            "tangent_light_field_imag": _ad_native_tangent_or_none(t_field_imag),
            "tangent_light_throughput_real": _ad_native_tangent_or_none(
                t_throughput_real
            ),
            "tangent_light_throughput_imag": _ad_native_tangent_or_none(
                t_throughput_imag
            ),
            "tangent_layer_thickness": _ad_native_tangent_or_none(t_thickness),
            "tangent_layer_eps_r": _ad_native_tangent_or_none(t_eps_r),
            "tangent_layer_sigma_e": _ad_native_tangent_or_none(t_sigma_e),
        }
        if tangent_frequency == 0.0 and all(v is None for v in tangents.values()):
            return (None,) * len(_SUBPATH_FIELDS)
        with disable_functorch():
            out = bdpt_transmitted_light_subpath_state_jvp(
                light,
                ctx.intersection,
                ctx.face_material_id,
                ctx.layer_offset,
                ctx.layer_count,
                _ad_native_tensor(saved[4]),
                _ad_native_tensor(saved[5]),
                _ad_native_tensor(saved[6]),
                _ad_native_tensor(saved[7]),
                frequency_hz=ctx.frequency_value,
                tangent_frequency=tangent_frequency,
                **tangents,
            )
        return _subpath_tangent_outputs(out)


def bdpt_transmitted_light_subpath_state_ad(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    *,
    face_material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_transmitted_light_subpath_state` (spec 6.2)."""

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    base_light = {
        name: value
        for name, value in light.items()
        if name not in _SUBPATH_DIFF_FIELDS
    }
    values = _BdptTransmittedSubpathAdFunction.apply(
        light["field_real"],
        light["field_imag"],
        light["throughput_real"],
        light["throughput_imag"],
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        float(frequency_value),
        base_light,
        intersection,
        face_material_id,
        layer_offset,
        layer_count,
    )
    return dict(zip(_SUBPATH_FIELDS, values, strict=True))


# ---------------------------------------------------------------------------
# 6.3 endpoint connection samples
# ---------------------------------------------------------------------------


_CONNECTION_FIELDS = (
    "topology",
    "contribution",
    "pdf",
    "mis_weight",
    "component_id",
    "valid",
    "tx_id",
    "rx_id",
    "grid_linear_id",
    "light_depth",
    "sensor_depth",
    "path_length_m",
)
_CONNECTION_CONTRIBUTION_INDEX = _CONNECTION_FIELDS.index("contribution")


class _BdptEndpointConnectionAdFunction(torch.autograd.Function):
    """Differentiable endpoint (LoS/NEE) connection contribution (spec 6.3).

    Differentiable inputs: the light and sensor subpath fields (8 tensors), the
    carrier frequency and ``tx_power`` (P_src). Frozen (reject loudly): the
    connection length L, samples_per_tx N, visibility, MIS mode, component_id,
    and every structural connection field. Only ``contribution`` is a
    differentiable output; the other 11 schema fields are frozen structure."""

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        light_field_real,
        light_field_imag,
        light_throughput_real,
        light_throughput_imag,
        sensor_field_real,
        sensor_field_imag,
        sensor_throughput_real,
        sensor_throughput_imag,
        frequency,
        frequency_value,
        tx_power,
        base_light,
        base_sensor,
        params,
    ):
        light = _subpath_with_fields(
            base_light,
            light_field_real,
            light_field_imag,
            light_throughput_real,
            light_throughput_imag,
        )
        sensor = _subpath_with_fields(
            base_sensor,
            sensor_field_real,
            sensor_field_imag,
            sensor_throughput_real,
            sensor_throughput_imag,
        )
        out = bdpt_endpoint_connection_samples(
            light,
            sensor,
            frequency_hz=frequency_value,
            samples_per_tx=params["samples_per_tx"],
            max_paths=params["max_paths"],
            mis=params["mis"],
            beta=params["beta"],
            strategy_count=params["strategy_count"],
        )
        return tuple(out[name] for name in _CONNECTION_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):  # type: ignore[no-untyped-def]
        ctx.set_materialize_grads(False)
        (
            light_field_real,
            light_field_imag,
            light_throughput_real,
            light_throughput_imag,
            sensor_field_real,
            sensor_field_imag,
            sensor_throughput_real,
            sensor_throughput_imag,
            frequency,
            frequency_value,
            tx_power,
            base_light,
            base_sensor,
            params,
        ) = inputs
        ctx.frequency_value = frequency_value
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.params = params

        def _detach_dict(source):  # type: ignore[no-untyped-def]
            return {
                name: torch.autograd.forward_ad.unpack_dual(value).primal
                if isinstance(value, torch.Tensor)
                else value
                for name, value in source.items()
            }

        ctx.base_light = _detach_dict(base_light)
        ctx.base_sensor = _detach_dict(base_sensor)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (
                light_field_real,
                light_field_imag,
                light_throughput_real,
                light_throughput_imag,
                sensor_field_real,
                sensor_field_imag,
                sensor_throughput_real,
                sensor_throughput_imag,
                tx_power,
            )
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        structural = [
            output[index]
            for index, name in enumerate(_CONNECTION_FIELDS)
            if name != "contribution"
        ]
        ctx.mark_non_differentiable(*structural)

    @staticmethod
    def _light_sensor(ctx):  # type: ignore[no-untyped-def]
        saved = ctx.saved_tensors
        light = _subpath_with_fields(
            ctx.base_light, saved[0], saved[1], saved[2], saved[3]
        )
        sensor = _subpath_with_fields(
            ctx.base_sensor, saved[4], saved[5], saved[6], saved[7]
        )
        return light, sensor

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):  # type: ignore[no-untyped-def]
        none_grads = (None,) * 14
        need_field = any(bool(ctx.needs_input_grad[i]) for i in range(8))
        need_frequency = bool(ctx.needs_input_grad[8])
        need_tx_power = bool(ctx.needs_input_grad[10])
        grad_contribution = grad_outputs[_CONNECTION_CONTRIBUTION_INDEX]
        if not (need_field or need_frequency or need_tx_power) or (
            grad_contribution is None
        ):
            return none_grads
        light, sensor = _BdptEndpointConnectionAdFunction._light_sensor(ctx)
        params = ctx.params
        out = bdpt_endpoint_connection_samples_backward(
            light,
            sensor,
            frequency_hz=ctx.frequency_value,
            samples_per_tx=params["samples_per_tx"],
            mis=params["mis"],
            beta=params["beta"],
            strategy_count=params["strategy_count"],
            max_paths=params["max_paths"],
            grad_contribution=grad_contribution,
            need_grad_field=need_field,
            need_grad_frequency=need_frequency,
            need_grad_tx_power=need_tx_power,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_light_field_real"] if ctx.needs_input_grad[0] else None,
            out["grad_light_field_imag"] if ctx.needs_input_grad[1] else None,
            None,
            None,
            out["grad_sensor_field_real"] if ctx.needs_input_grad[4] else None,
            out["grad_sensor_field_imag"] if ctx.needs_input_grad[5] else None,
            None,
            None,
            grad_frequency,
            None,
            out["grad_tx_power"] if need_tx_power else None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):  # type: ignore[no-untyped-def]
        light, sensor = _BdptEndpointConnectionAdFunction._light_sensor(ctx)
        params = ctx.params
        tangent_frequency = _ad_frequency_tangent(tangents[8])
        payload = {
            "tangent_light_field_real": _ad_native_tangent_or_none(tangents[0]),
            "tangent_light_field_imag": _ad_native_tangent_or_none(tangents[1]),
            "tangent_sensor_field_real": _ad_native_tangent_or_none(tangents[4]),
            "tangent_sensor_field_imag": _ad_native_tangent_or_none(tangents[5]),
            "tangent_tx_power": _ad_native_tangent_or_none(tangents[10]),
        }
        if tangent_frequency == 0.0 and all(v is None for v in payload.values()):
            return (None,) * len(_CONNECTION_FIELDS)
        with disable_functorch():
            out = bdpt_endpoint_connection_samples_jvp(
                light,
                sensor,
                frequency_hz=ctx.frequency_value,
                samples_per_tx=params["samples_per_tx"],
                mis=params["mis"],
                beta=params["beta"],
                strategy_count=params["strategy_count"],
                max_paths=params["max_paths"],
                tangent_frequency=tangent_frequency,
                **payload,
            )
        result: list[torch.Tensor | None] = [None] * len(_CONNECTION_FIELDS)
        result[_CONNECTION_CONTRIBUTION_INDEX] = out["tangent_contribution"]
        return tuple(result)


def bdpt_endpoint_connection_samples_ad(
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    tx_power: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
    samples_per_tx: int,
    max_paths: int | None = None,
    mis: str = "power_heuristic",
    beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_endpoint_connection_samples` (spec 6.3).

    ``tx_power`` (P_src) is carried so its gradient (``grad_tx_power``) is
    accumulated by the native backward; it is not consumed by the primal
    forward (source power rides the light subpath) so the primal is bitwise
    unchanged."""

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    base_light = {
        name: value for name, value in light.items() if name not in _SUBPATH_DIFF_FIELDS
    }
    base_sensor = {
        name: value for name, value in sensor.items() if name not in _SUBPATH_DIFF_FIELDS
    }
    params = {
        "samples_per_tx": int(samples_per_tx),
        "max_paths": max_paths,
        "mis": mis,
        "beta": float(beta),
        "strategy_count": int(strategy_count),
    }
    values = _BdptEndpointConnectionAdFunction.apply(
        light["field_real"],
        light["field_imag"],
        light["throughput_real"],
        light["throughput_imag"],
        sensor["field_real"],
        sensor["field_imag"],
        sensor["throughput_real"],
        sensor["throughput_imag"],
        frequency,
        float(frequency_value),
        tx_power,
        base_light,
        base_sensor,
        params,
    )
    return dict(zip(_CONNECTION_FIELDS, values, strict=True))
# --- ADR-022 accumulate / finalize autograd.Function wrappers ---------------

_FINALIZE_FIELDS = (
    "path_gain",
    "los_power",
    "reflection_power",
    "diffraction_power",
    "transmission_power",
    "scattering_power",
)


# ---------------------------------------------------------------------------
# 6.5 / 6.6 finalize (point components and component maps)
# ---------------------------------------------------------------------------


class _BdptFinalizeAdFunction(torch.autograd.Function):
    """Differentiable BDPT finalize (linear map; spec 6.5/6.6).

    The five component matrices/maps are all differentiable; the forward sums
    them into ``path_gain`` and reduces each into a 0-dim power. Backward is the
    native transpose companion, jvp the native forward map on the tangents;
    both deterministic, no atomics."""

    @staticmethod
    def forward(los, reflection, diffraction, transmission, scattering, kind):  # type: ignore[no-untyped-def]
        forward = (
            bdpt_finalize_point_components
            if kind == "point"
            else bdpt_finalize_component_maps
        )
        out = forward(los, reflection, diffraction, transmission, scattering)
        return tuple(out[name] for name in _FINALIZE_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):  # type: ignore[no-untyped-def]
        ctx.set_materialize_grads(False)
        los, reflection, diffraction, transmission, scattering, kind = inputs
        ctx.kind = kind
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (los, reflection, diffraction, transmission, scattering)
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, grad_path_gain, *grad_powers):  # type: ignore[no-untyped-def]
        if not any(bool(flag) for flag in ctx.needs_input_grad[:5]):
            return (None,) * 6
        backward = (
            bdpt_finalize_point_components_backward
            if ctx.kind == "point"
            else bdpt_finalize_component_maps_backward
        )
        out = backward(
            *ctx.saved_tensors,
            grad_path_gain=grad_path_gain,
            grad_los_power=grad_powers[0],
            grad_reflection_power=grad_powers[1],
            grad_diffraction_power=grad_powers[2],
            grad_transmission_power=grad_powers[3],
            grad_scattering_power=grad_powers[4],
            need_grad_components=True,
        )
        return (
            out["grad_los"],
            out["grad_reflection"],
            out["grad_diffraction"],
            out["grad_transmission"],
            out["grad_scattering"],
            None,
        )

    @staticmethod
    def jvp(ctx, t_los, t_reflection, t_diffraction, t_transmission, t_scattering, _t_kind):  # type: ignore[no-untyped-def]
        tangents = {
            "tangent_los": _ad_native_tangent_or_none(t_los),
            "tangent_reflection": _ad_native_tangent_or_none(t_reflection),
            "tangent_diffraction": _ad_native_tangent_or_none(t_diffraction),
            "tangent_transmission": _ad_native_tangent_or_none(t_transmission),
            "tangent_scattering": _ad_native_tangent_or_none(t_scattering),
        }
        if all(value is None for value in tangents.values()):
            return (None,) * len(_FINALIZE_FIELDS)
        jvp = (
            bdpt_finalize_point_components_jvp
            if ctx.kind == "point"
            else bdpt_finalize_component_maps_jvp
        )
        with disable_functorch():
            out = jvp(*(_ad_native_tensor(value) for value in ctx.saved_tensors), **tangents)
        return tuple(out[name] for name in _FINALIZE_TANGENT_FIELDS)


_FINALIZE_TANGENT_FIELDS = (
    "tangent_path_gain",
    "tangent_los_power",
    "tangent_reflection_power",
    "tangent_diffraction_power",
    "tangent_transmission_power",
    "tangent_scattering_power",
)


def bdpt_finalize_point_components_ad(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_finalize_point_components` (spec 6.5)."""

    values = _BdptFinalizeAdFunction.apply(
        los, reflection, diffraction, transmission, scattering, "point"
    )
    return dict(zip(_FINALIZE_FIELDS, values, strict=True))


def bdpt_finalize_component_maps_ad(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_finalize_component_maps` (spec 6.6)."""

    values = _BdptFinalizeAdFunction.apply(
        los, reflection, diffraction, transmission, scattering, "maps"
    )
    return dict(zip(_FINALIZE_FIELDS, values, strict=True))


# ---------------------------------------------------------------------------
# 6.4 accumulate (power AND coherent)
# ---------------------------------------------------------------------------


_ACCUMULATE_MATRIX_FIELDS = (
    "path_gain",
    "los",
    "reflection",
    "diffraction",
    "transmission",
    "scattering",
)


class _BdptAccumulateAdFunction(torch.autograd.Function):
    """Differentiable connection-sample accumulate, both domains (spec 6.4).

    Differentiable inputs: the row ``contribution`` (power domain) OR the
    complex ``coeff_real``/``coeff_imag`` (coherent domain). Frozen (reject
    loudly): mis_weight, tx_id/rx_id/component_id/valid, the whole index
    structure. The coherent forward retains its per-component phasor bin sums
    ``S_b`` so the coherent backward needs no re-reduction (supervisor ruling)."""

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        contribution,
        coeff_real,
        coeff_imag,
        base_samples,
        tx_count,
        rx_count,
        accumulation_strategy,
        combine_domain,
    ):
        samples = dict(base_samples)
        samples["contribution"] = contribution
        matrices, bin_sums = bdpt_accumulate_connection_samples_forward_ad(
            samples,
            tx_count=tx_count,
            rx_count=rx_count,
            accumulation_strategy=accumulation_strategy,
            combine_domain=combine_domain,
            coeff_real=coeff_real,
            coeff_imag=coeff_imag,
        )
        # Flat tensor output tuple: the six differentiable component matrices
        # followed by the coherent bin-sum buffers (empty for the power domain),
        # which are marked non-differentiable in setup_context (spec 6.4).
        return tuple(matrices[name] for name in _ACCUMULATE_MATRIX_FIELDS) + tuple(
            bin_sums
        )

    @staticmethod
    def setup_context(ctx, inputs, output):  # type: ignore[no-untyped-def]
        ctx.set_materialize_grads(False)
        (
            contribution,
            coeff_real,
            coeff_imag,
            base_samples,
            tx_count,
            rx_count,
            accumulation_strategy,
            combine_domain,
        ) = inputs
        ctx.tx_count = int(tx_count)
        ctx.rx_count = int(rx_count)
        ctx.accumulation_strategy = accumulation_strategy
        ctx.combine_domain = combine_domain
        ctx.base_samples = {
            name: torch.autograd.forward_ad.unpack_dual(value).primal
            if isinstance(value, torch.Tensor)
            else value
            for name, value in base_samples.items()
        }
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (contribution, coeff_real, coeff_imag)
        )
        # The trailing outputs past the six component matrices are the coherent
        # bin sums S_b, retained for the backward but carrying no gradient.
        ctx.bin_sums = tuple(output[len(_ACCUMULATE_MATRIX_FIELDS):])
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        if ctx.bin_sums:
            ctx.mark_non_differentiable(*ctx.bin_sums)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):  # type: ignore[no-untyped-def]
        grad_matrices = grad_outputs[: len(_ACCUMULATE_MATRIX_FIELDS)]
        none_grads = (None,) * 8
        # The coherent VJP/JVP read the forward-retained bin sums (ctx.bin_sums),
        # not the sample coefficients, so the saved coeff tensors are unused here.
        contribution, _coeff_real, _coeff_imag = ctx.saved_tensors
        samples = dict(ctx.base_samples)
        samples["contribution"] = contribution
        need_contribution = bool(ctx.needs_input_grad[0])
        need_coeff = bool(ctx.needs_input_grad[1]) or bool(ctx.needs_input_grad[2])
        if not (need_contribution or need_coeff) or all(
            value is None for value in grad_matrices
        ):
            return none_grads
        out = bdpt_accumulate_connection_samples_backward(
            samples,
            tx_count=ctx.tx_count,
            rx_count=ctx.rx_count,
            combine_domain=ctx.combine_domain,
            bin_sums=ctx.bin_sums,
            grad_path_gain=grad_matrices[0],
            grad_los=grad_matrices[1],
            grad_reflection=grad_matrices[2],
            grad_diffraction=grad_matrices[3],
            grad_transmission=grad_matrices[4],
            grad_scattering=grad_matrices[5],
            need_grad_contribution=need_contribution,
            need_grad_coeff=need_coeff,
        )
        return (
            out["grad_contribution"] if need_contribution else None,
            out["grad_coeff_real"] if bool(ctx.needs_input_grad[1]) else None,
            out["grad_coeff_imag"] if bool(ctx.needs_input_grad[2]) else None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(  # type: ignore[no-untyped-def]
        ctx,
        t_contribution,
        t_coeff_real,
        t_coeff_imag,
        _t_base_samples,
        _t_tx_count,
        _t_rx_count,
        _t_strategy,
        _t_combine,
    ):
        # The coherent VJP/JVP read the forward-retained bin sums (ctx.bin_sums),
        # not the sample coefficients, so the saved coeff tensors are unused here.
        contribution, _coeff_real, _coeff_imag = ctx.saved_tensors
        samples = dict(ctx.base_samples)
        samples["contribution"] = contribution
        tangent_contribution = _ad_native_tangent_or_none(t_contribution)
        tangent_coeff_real = _ad_native_tangent_or_none(t_coeff_real)
        tangent_coeff_imag = _ad_native_tangent_or_none(t_coeff_imag)
        n_out = len(_ACCUMULATE_MATRIX_FIELDS) + len(ctx.bin_sums)
        if (
            tangent_contribution is None
            and tangent_coeff_real is None
            and tangent_coeff_imag is None
        ):
            return (None,) * n_out
        with disable_functorch():
            out = bdpt_accumulate_connection_samples_jvp(
                samples,
                tx_count=ctx.tx_count,
                rx_count=ctx.rx_count,
                combine_domain=ctx.combine_domain,
                bin_sums=ctx.bin_sums,
                tangent_contribution=tangent_contribution,
                tangent_coeff_real=tangent_coeff_real,
                tangent_coeff_imag=tangent_coeff_imag,
            )
        tangent_matrices = (
            out["tangent_path_gain"],
            out["tangent_los"],
            out["tangent_reflection"],
            out["tangent_diffraction"],
            out["tangent_transmission"],
            out["tangent_scattering"],
        )
        # Bin-sum outputs are non-differentiable: their forward-mode tangent is
        # None.
        return tangent_matrices + (None,) * len(ctx.bin_sums)


def bdpt_accumulate_connection_samples_ad(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    accumulation_strategy: str = "atomic",
    combine_domain: str = "power",
    coeff_real: torch.Tensor | None = None,
    coeff_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_accumulate_connection_samples` (spec 6.4)."""

    device = samples["contribution"].device
    if combine_domain == "coherent":
        if coeff_real is None or coeff_imag is None:
            raise ValueError("coherent combine requires coeff_real and coeff_imag")
    else:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        coeff_real = empty if coeff_real is None else coeff_real
        coeff_imag = empty if coeff_imag is None else coeff_imag
    base_samples = {
        name: value for name, value in samples.items() if name != "contribution"
    }
    outputs = _BdptAccumulateAdFunction.apply(
        samples["contribution"],
        coeff_real,
        coeff_imag,
        base_samples,
        int(tx_count),
        int(rx_count),
        accumulation_strategy,
        combine_domain,
    )
    matrices = outputs[: len(_ACCUMULATE_MATRIX_FIELDS)]
    return dict(zip(_ACCUMULATE_MATRIX_FIELDS, matrices, strict=True))
# --- Connection-sample builders ---------------------------------------------

_MASK_REFLECTION = 2
_MASK_TRANSMISSION = 8


def _native_los_connection_samples(
    rayd: Any,
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    scene_has_structures: bool,
    frequency_hz: float | torch.Tensor,
    mis: str,
    beta: float,
    strategy_count: int,
    ad: bool = False,
    tx_power: torch.Tensor | None = None,
    frequency_value: float | None = None,
    ledger: object | None = None,
) -> dict[str, torch.Tensor]:
    if ad:
        # ADR-022: the LoS direct connection carries both a frequency gradient
        # (the lambda^2 radiometric factor) and a tx_power gradient (P_src),
        # dispatched natively through the endpoint-connection companion exactly
        # like the mixed-transmission path. The live frequency tensor and the
        # live tx_power leaf feed grad_frequency / grad_tx_power; the host scalar
        # (frequency_value) threads the frozen sampling/pdf path.
        samples = bdpt_endpoint_connection_samples_ad(
            light,
            sensor,
            tx_power,
            frequency=frequency_hz,
            frequency_value=frequency_value,
            samples_per_tx=1,
            max_paths=None,
            mis=mis,
            beta=beta,
            strategy_count=strategy_count,
        )
        if ledger is not None:
            ledger.add(  # type: ignore[attr-defined]
                light["field_real"],
                light["field_imag"],
                sensor["field_real"],
                sensor["field_imag"],
            )
    else:
        samples = bdpt_endpoint_connection_samples(
            light,
            sensor,
            frequency_hz=frequency_hz,
            samples_per_tx=1,
            max_paths=None,
            mis=mis,
            beta=beta,
            strategy_count=strategy_count,
        )
    if not scene_has_structures:
        return samples
    visibility_inputs = bdpt_endpoint_connection_visibility_inputs(
        light,
        sensor,
        sample_count=int(samples["valid"].shape[0]),
    )
    visible = geometry_kernels.rayd_visibility_forward(
        rayd.require_resource(),
        visibility_inputs["start"],
        visibility_inputs["end"],
        visibility_inputs["active"],
    )[0]
    return bdpt_filter_connection_samples(samples, visible)  # type: ignore[no-any-return]


def _merge_event_states(
    reflected: dict[str, torch.Tensor],
    transmitted: dict[str, torch.Tensor],
    choose_transmit: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Row-wise merge of the two event kernels' outputs.

    Both kernels are evaluated on the full batch and the per-row winner is
    selected, which preserves the tensor layout exactly (equivalent to
    partitioning the hit indices, running each kernel on its partition, and
    scattering back by original index).
    """

    wide = choose_transmit[:, None]
    merged: dict[str, torch.Tensor] = {}
    for key, reflected_value in reflected.items():
        condition = wide if reflected_value.dim() == 2 else choose_transmit
        merged[key] = torch.where(condition, transmitted[key], reflected_value)
    return merged


def _merge_scattered_state(
    merged: dict[str, torch.Tensor],
    scattered: dict[str, torch.Tensor],
    choose_scatter: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Row-wise overlay of the scattered branch onto the reflect/transmit
    merge (same evaluate-everywhere-select-per-row pattern as
    :func:`_merge_event_states`)."""

    wide = choose_scatter[:, None]
    out: dict[str, torch.Tensor] = {}
    for key, merged_value in merged.items():
        condition = wide if merged_value.dim() == 2 else choose_scatter
        out[key] = torch.where(condition, scattered[key], merged_value)
    return out


def _select_surface_events(
    *,
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    hit_ok: torch.Tensor,
    material_bundle: dict[str, torch.Tensor],
    layer_csr: dict[str, torch.Tensor],
    runtimes: dict[int, Any],
    frequency_value: float,
    samples: int,
    seed: int,
    tx_index: int,
    bounce: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Three-way (scatter / transmit / reflect) event selection at a surface hit.

    Pure lift of plan section 7.1's frozen event-probability stack: the smooth
    two-way split, the rough three-way budget overlay on rough rows, and the
    single seeded uniform that partitions scatter/transmit/reflect. Returns the
    per-row selection masks plus the probabilities the unbiased weighting reads.
    """

    stack = em_layer_stack_eval(
        cos_theta,
        material_id.clamp_min(0),
        layer_csr["layer_offset"],
        layer_csr["layer_count"],
        layer_csr["layer_thickness_m"],
        layer_csr["layer_eps_r"],
        layer_csr["layer_sigma_e"],
        layer_csr["layer_mu_r"],
        frequency_hz=frequency_value,
    )
    r_eff, t_eff = unpolarized_power_budgets(stack)
    p_transmit = transmission_event_probability(r_eff, t_eff)
    uniforms = event_uniforms(
        int(samples), seed=seed, tx_index=tx_index, depth=bounce, device=device
    )
    if runtimes:
        rough_probs = three_way_rough_probabilities(
            cos_theta,
            material_id,
            material_bundle,
            stack,
            frequency_hz=frequency_value,
        )
        rough = rough_probs["rough"] & hit_ok
        p_scatter = torch.where(
            rough, rough_probs["p_scatter"], torch.zeros_like(p_transmit)
        )
        # Smooth rows keep the exact wave-2 two-way probability;
        # rough rows switch to the three-way budget split.
        p_transmit = torch.where(rough, rough_probs["p_transmit"], p_transmit)
        coherent_amplitude = torch.where(
            rough,
            rough_probs["r_coh_amplitude"],
            torch.ones_like(p_transmit),
        )
    else:
        rough = torch.zeros_like(hit_ok)
        p_scatter = torch.zeros_like(p_transmit)
        coherent_amplitude = torch.ones_like(p_transmit)
    # One uniform partitions the three events: [0, p_s) scatter,
    # [p_s, p_s + p_t) transmit, else reflect. Smooth faces have
    # p_s = 0 exactly, so their transmit test u < p_t is unchanged.
    choose_scatter = hit_ok & (uniforms < p_scatter)
    choose_transmit = (
        hit_ok & ~choose_scatter & (uniforms < p_scatter + p_transmit)
    )
    return {
        "choose_scatter": choose_scatter,
        "choose_transmit": choose_transmit,
        "rough": rough,
        "p_scatter": p_scatter,
        "p_transmit": p_transmit,
        "coherent_amplitude": coherent_amplitude,
    }


def _emit_scatter_nee(
    *,
    rayd: Any,
    sensor: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    hit: dict[str, torch.Tensor],
    merged: dict[str, torch.Tensor],
    choose_scatter: torch.Tensor,
    p_scatter: torch.Tensor,
    material_id: torch.Tensor,
    material_axis_rad: torch.Tensor,
    runtimes: dict[int, Any],
    max_scattering_order: int,
    samples: int,
    seed: int,
    tx_index: int,
    bounce: int,
    device: torch.device,
    scene_diagonal: float,
    frequency_hz: float | torch.Tensor,
    frequency_value: float,
    tx_power: torch.Tensor,
    ad: bool,
    ledger: object | None,
    sample_blocks: list[dict[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Scatter branch: local frames, the scattered subpath overlay, and NEE rows.

    Pure lift of the scatter-selected emission block (plan section 7.1, ADR-021
    D4). Appends the NEE connection block (when any) to ``sample_blocks`` and
    returns the (possibly overlaid) merged state, the scattered-valid mask, and
    the count of emitted NEE rows."""

    scattered_valid = torch.zeros_like(choose_scatter)
    nee_rows = 0
    if runtimes and bool(choose_scatter.any()):
        # Local roughness frames from the shading normal flipped
        # toward the incident side (roughness applies to whichever
        # side is illuminated in v1; the store carries front-surface
        # statistics only).
        direction = state["direction"]
        hit_normal = hit["n"]
        normal_flipped = torch.where(
            ((direction * hit_normal).sum(dim=-1) > 0.0)[:, None],
            -hit_normal,
            hit_normal,
        )
        axis_rad = material_axis_rad.index_select(
            0, material_id.clamp_min(0).to(torch.int64)
        )
        frame_t1, frame_t2 = local_frames(normal_flipped, axis_rad)
        wi_world = -direction
        wi_local = world_to_local(wi_world, frame_t1, frame_t2, normal_flipped)
        p_te, p_tm = te_tm_incident_power(
            state["field_real"],
            state["field_imag"],
            direction,
            normal_flipped,
        )
        if int(max_scattering_order) > 1:
            # A subpath that has ALREADY scattered carries no Complex3
            # field (cleared at the previous scatter vertex); its
            # incident power lives in the scalar throughput. Route that
            # unpolarized power into the local TE/TM channels so both
            # the NEE row and the continuation weight see the correct
            # incident power at this vertex. Order 1 never reaches here
            # (no subpath is ever a continued scatter), so the default
            # stays bitwise the field-based decomposition above.
            already_scattered = (
                state["component_mask"] & MASK_SCATTERING
            ) != 0
            carried_te, carried_tm = scatter_carried_incident_power(
                state["throughput_real"], state["throughput_imag"]
            )
            p_te = torch.where(already_scattered, carried_te, p_te)
            p_tm = torch.where(already_scattered, carried_tm, p_tm)
        direction_uniforms = scatter_direction_uniforms(
            int(samples),
            seed=seed,
            tx_index=tx_index,
            depth=bounce,
            device=device,
        )
        scattered = scattered_subpath_state(
            state,
            hit,
            choose_scatter=choose_scatter,
            normal=normal_flipped,
            frame_t1=frame_t1,
            frame_t2=frame_t2,
            wi_local=wi_local,
            p_te=p_te,
            p_tm=p_tm,
            p_scatter=p_scatter,
            material_id=material_id,
            runtimes=runtimes,
            uniforms=direction_uniforms,
            scene_diagonal=scene_diagonal,
            ad=ad,
            ledger=ledger,
        )
        scattered_valid = scattered["valid"]
        merged = _merge_scattered_state(merged, scattered, choose_scatter)
        rows = torch.nonzero(scattered_valid, as_tuple=False).flatten()
        if int(rows.numel()):
            scatter_source_power = state["source_power"].index_select(0, rows)
            if ad:
                # ADR-022 tx_power threading: reattach the live per-tx
                # power's gradient onto the detached native source power
                # for the scatter-selected rows (values bitwise-identical,
                # so the scattering NEE primal is unchanged).
                scatter_tx_id = (
                    state["tx_id"].index_select(0, rows).to(torch.int64)
                )
                live_source_power = tx_power.index_select(0, scatter_tx_id)
                scatter_source_power = scatter_source_power + (
                    live_source_power - live_source_power.detach()
                )
            nee_block = scattering_nee_connection_samples(
                rayd,
                sensor,
                runtimes,
                position=hit["p"].index_select(0, rows),
                normal=normal_flipped.index_select(0, rows),
                frame_t1=frame_t1.index_select(0, rows),
                frame_t2=frame_t2.index_select(0, rows),
                wi_local=wi_local.index_select(0, rows),
                p_te=p_te.index_select(0, rows),
                p_tm=p_tm.index_select(0, rows),
                p_scatter=p_scatter.index_select(0, rows),
                material_id=material_id.index_select(0, rows),
                source_power=scatter_source_power,
                tx_id=state["tx_id"].index_select(0, rows),
                light_depth=scattered["depth"].index_select(0, rows),
                path_length_at_vertex=scattered["path_length"].index_select(
                    0, rows
                ),
                # ADR-015 Part A: hand the live frequency tensor to the
                # radiometric factor under ad; frequency_value stays the
                # host scalar for the sampling/pdf paths.
                frequency_hz=frequency_hz if ad else frequency_value,
                samples=int(samples),
                scene_diagonal=scene_diagonal,
                ad=ad,
                ledger=ledger,
            )
            if nee_block is not None:
                sample_blocks.append(nee_block)
                nee_rows += int(nee_block["valid"].sum())
    return merged, scattered_valid, nee_rows


def _emit_mixed_transmission(
    *,
    rayd: Any,
    sensor: dict[str, torch.Tensor],
    merged: dict[str, torch.Tensor],
    choose_scatter: torch.Tensor,
    emit_mixed_transmission: bool,
    sensor_count: int,
    samples: int,
    tx_power: torch.Tensor,
    frequency_hz: float | torch.Tensor,
    frequency_value: float,
    mis: str,
    beta: float,
    ad: bool,
    ledger: object | None,
    sample_blocks: list[dict[str, torch.Tensor]],
) -> None:
    """Emit the MIXED reflection+transmission endpoint connection (component 5).

    Pure lift of the mixed-transmission emission block (wave 2): connects only the
    reflect-and-transmit subpaths through the native endpoint kernel, filters by
    visibility, and appends the resulting block to ``sample_blocks``."""

    mask = merged["component_mask"]
    mixed = (
        merged["valid"]
        & ~choose_scatter
        & ((mask & _MASK_REFLECTION) != 0)
        & ((mask & _MASK_TRANSMISSION) != 0)
        # Post-scatter subpaths carry no Complex3 field (cleared at
        # the scatter vertex); their |F|^2 = 0 endpoint rows would
        # contribute nothing while contaminating the component-5
        # sample statistics. Their path class (S -> ... -> T) is
        # explicitly not covered in v1 (ADR-021 D4). At order 1 no
        # subpath survives with the scattering bit, so this term is
        # structurally inert for the default.
        & ((mask & MASK_SCATTERING) == 0)
    )
    if emit_mixed_transmission and bool(mixed.any()):
        if ad:
            samples_out = bdpt_endpoint_connection_samples_ad(
                merged,
                sensor,
                tx_power,
                frequency=frequency_hz,
                frequency_value=frequency_value,
                samples_per_tx=int(samples),
                max_paths=None,
                mis=mis,
                beta=beta,
                strategy_count=1,
            )
            if ledger is not None:
                ledger.add(  # type: ignore[attr-defined]
                    merged["field_real"],
                    merged["field_imag"],
                    sensor["field_real"],
                    sensor["field_imag"],
                )
        else:
            samples_out = bdpt_endpoint_connection_samples(
                merged,
                sensor,
                frequency_hz=frequency_value,
                samples_per_tx=int(samples),
                max_paths=None,
                mis=mis,
                beta=beta,
                strategy_count=1,
            )
        visibility_inputs = bdpt_endpoint_connection_visibility_inputs(
            merged,
            sensor,
            sample_count=int(samples_out["valid"].shape[0]),
        )
        visible = geometry_kernels.rayd_visibility_forward(
            rayd.require_resource(),
            visibility_inputs["start"],
            visibility_inputs["end"],
            visibility_inputs["active"],
        )[0]
        keep = visible & mixed.repeat_interleave(sensor_count)
        sample_blocks.append(bdpt_filter_connection_samples(samples_out, keep))


def _apply_scatter_continuation(
    *,
    merged: dict[str, torch.Tensor],
    scatter_count: torch.Tensor | None,
    choose_scatter: torch.Tensor,
    scattered_valid: torch.Tensor,
    max_scattering_order: int,
) -> torch.Tensor | None:
    """Terminate (order 1) or continue (order > 1) scattered subpaths.

    Pure lift of the continuation/kill logic (ADR-021 D4). Mutates
    ``merged['valid']`` in place and returns the updated scatter-event tally."""

    if scatter_count is None:
        # order 1 (default): scattered subpaths connected above and
        # terminate here; reflection/transmission never follow them.
        merged["valid"] = merged["valid"] & ~choose_scatter
        return None
    # order > 1 (ADR-021 D4): a successfully scattered subpath
    # CONTINUES (its new direction/origin/throughput are already
    # overlaid in ``merged`` by _merge_scattered_state) until it
    # reaches the scatter-event cap. NEE rows were emitted above at
    # this vertex exactly as at order 1. Count only successful
    # scatter events; a subpath that just hit the cap terminates
    # (its NEE is already recorded), and a scatter selection whose
    # direction sample failed is dropped.
    scatter_count = scatter_count + scattered_valid.to(scatter_count.dtype)
    reached_cap = scatter_count >= int(max_scattering_order)
    merged["valid"] = (
        merged["valid"]
        & ~(choose_scatter & ~scattered_valid)
        & ~reached_cap
    )
    return scatter_count


def _transmission_sampled_connection_samples(
    rayd: Any,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor,
    rx_polarization: torch.Tensor,
    sensor: dict[str, torch.Tensor],
    material_bundle: dict[str, torch.Tensor],
    *,
    frequency_hz: float,
    samples: int,
    max_depth: int,
    seed: int,
    mis: str,
    beta: float,
    scattering_runtimes: dict[int, Any] | None = None,
    emit_mixed_transmission: bool = True,
    scene_diagonal: float = 0.0,
    max_scattering_order: int = 1,
    ad: bool = False,
    ledger: object | None = None,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, int]]:
    """Shooting-context light subpaths with three-way event selection.

    Implements plan section 7.1. At every surface hit a seeded, reproducible
    uniform selects among the delta specular reflection, the continuous
    Kirchhoff scattering (rough faces only, when ``scattering_runtimes`` is
    provided) and the delta transmission events; the selected branch's field
    is divided by sqrt(p_event) so the power estimator stays unbiased (see
    the inline algebra note).

    Event probabilities: smooth faces keep the wave-2 two-way split
    p_t = T/(R+T) from the native stack budgets BIT-IDENTICALLY (their
    scatter probability is exactly zero, so the same uniform partitions the
    same way); rough faces use the native (R_coh, R_diff, T_bar) budgets
    with the same floor pattern. The rough reflect branch
    additionally multiplies the field by the coherent attenuation C_r so its
    amplitude represents sqrt(R_coh), matching the budget that selected it.

    Contribution routing (never double counts):
    - MIXED reflection+transmission chains connect through the native
      endpoint kernel (component 5), as in wave 2; emitted only when
      ``emit_mixed_transmission`` (the transmission component is requested).
    - Scatter-selected vertices emit torch-side NEE rows (component 6).
      Depth rule (ADR-021 D4, ``max_scattering_order``):
        * order 1 (default, BIT-IDENTICAL): the scattered subpath emits its
          NEE row and TERMINATES; reflection/transmission never follow.
        * order > 1: the scattered subpath CONTINUES in its lobe-sampled
          direction (power divided by ``p_scatter * pdf(wo)`` in
          :func:`scattered_subpath_state`) and may reflect/transmit/scatter
          again, emitting an NEE row at every scatter vertex, until it has
          undergone ``max_scattering_order`` scatter events. A post-scatter
          subpath carries power in the scalar throughput (its Complex3 Jones
          field is cleared at a scatter vertex), so its incident power at a
          further scatter vertex is the unpolarized throughput power.
    - Pure reflection stays with the discrete enumeration and pure
      transmission with the straight endpoint chains.
    """

    device = tx_positions.device
    # ADR-015 Part A: under AD the carrier crosses as a live 0-dim tensor
    # (``frequency_hz``) while the host scalar (``frequency_value``) is read once
    # and threaded to the frozen event-probability stack and every _ad facade.
    # The primal path keeps ``frequency_hz`` a float, so ``frequency_value`` is
    # exactly that float and every call is bitwise the pre-AD behaviour.
    frequency_value = (
        _ad_frequency_value(frequency_hz) if ad else float(frequency_hz)
    )
    layer_csr = layer_csr_view(material_bundle)
    face_material_id = material_bundle["material_id"]
    material_axis_rad = material_bundle["rough_axis_rad"]
    runtimes = scattering_runtimes or {}
    sensor_count = int(sensor["origin"].shape[0])
    sample_blocks: list[dict[str, torch.Tensor]] = []
    transmit_events = 0
    reflect_events = 0
    scatter_events = 0
    nee_rows = 0
    for tx_index in range(int(tx_positions.shape[0])):
        launch_inputs = bdpt_reflection_launch_inputs(
            tx_positions, tx_index=tx_index, sample_count=int(samples)
        )
        ray_d = bdpt_sample_directions(
            int(samples), tx_positions, seed=int(seed) + tx_index * 65537
        )
        state = bdpt_endpoint_subpath_state(
            tx_positions,
            tx_power,
            tx_polarization,
            rx_positions,
            rx_polarization,
            launch_inputs["tx_id"],
            launch_inputs["light_seed"],
        )["light"]
        state["direction"] = ray_d
        ray_inputs = {
            "ray_o": launch_inputs["ray_o"],
            "ray_d": ray_d,
            "ray_tmax": launch_inputs["ray_tmax"],
            "active": launch_inputs["active"],
        }
        # Per-subpath diffuse-scatter event tally for the order cap. Only
        # allocated when multi-order continuation is requested; order 1 keeps
        # the single-bounce terminal rule and never reads it.
        scatter_count = (
            torch.zeros((int(samples),), device=device, dtype=torch.int32)
            if int(max_scattering_order) > 1
            else None
        )
        for bounce in range(max(1, int(max_depth))):
            hit = geometry_kernels.rayd_intersect_forward(
                rayd.require_resource(),
                ray_inputs["ray_o"],
                ray_inputs["ray_d"],
                ray_inputs["ray_tmax"],
                ray_inputs["active"],
            )
            prim = hit["global_prim_id"]
            material_id = face_material_id.index_select(
                0, prim.clamp_min(0).to(torch.int64)
            )
            hit_ok = (
                state["valid"] & (prim >= 0) & (hit["t"] >= 0.0) & (material_id >= 0)
            )
            cos_theta = (
                (state["direction"] * hit["n"]).sum(dim=-1).abs().clamp(1.0e-6, 1.0)
            )
            # The event-probability stack is FROZEN (it drives sampling and MIS,
            # frozen under ADR-022): always the non-AD primal evaluation with the
            # host scalar. Material/layer gradients ride the subpath _ad kernels
            # below, not this selection stack.
            events = _select_surface_events(
                cos_theta=cos_theta,
                material_id=material_id,
                hit_ok=hit_ok,
                material_bundle=material_bundle,
                layer_csr=layer_csr,
                runtimes=runtimes,
                frequency_value=frequency_value,
                samples=samples,
                seed=seed,
                tx_index=tx_index,
                bounce=bounce,
                device=device,
            )
            choose_scatter = events["choose_scatter"]
            choose_transmit = events["choose_transmit"]
            rough = events["rough"]
            p_scatter = events["p_scatter"]
            p_transmit = events["p_transmit"]
            coherent_amplitude = events["coherent_amplitude"]
            if ad:
                if ledger is not None:
                    ledger.add(  # type: ignore[attr-defined]
                        state["field_real"],
                        state["field_imag"],
                        material_bundle["eps_r"],
                        material_bundle["sigma_e"],
                        material_bundle["thickness"],
                    )
                reflected = bdpt_reflected_light_subpath_state_ad(
                    state,
                    hit,
                    material_gain=material_bundle["gain"],
                    material_valid=material_bundle["valid"],
                    material_eps_r=material_bundle["eps_r"],
                    material_sigma_e=material_bundle["sigma_e"],
                    material_mu_r=material_bundle["mu_r"],
                    material_thickness=material_bundle["thickness"],
                    frequency=frequency_hz,
                    frequency_value=frequency_value,
                )
                if ledger is not None:
                    ledger.add(  # type: ignore[attr-defined]
                        state["field_real"],
                        state["field_imag"],
                        layer_csr["layer_thickness_m"],
                        layer_csr["layer_eps_r"],
                        layer_csr["layer_sigma_e"],
                    )
                transmitted = bdpt_transmitted_light_subpath_state_ad(
                    state,
                    hit,
                    face_material_id=face_material_id,
                    layer_offset=layer_csr["layer_offset"],
                    layer_count=layer_csr["layer_count"],
                    layer_thickness_m=layer_csr["layer_thickness_m"],
                    layer_eps_r=layer_csr["layer_eps_r"],
                    layer_sigma_e=layer_csr["layer_sigma_e"],
                    layer_mu_r=layer_csr["layer_mu_r"],
                    frequency=frequency_hz,
                    frequency_value=frequency_value,
                )
            else:
                reflected = bdpt_reflected_light_subpath_state(
                    state,
                    hit,
                    material_gain=material_bundle["gain"],
                    material_valid=material_bundle["valid"],
                    material_eps_r=material_bundle["eps_r"],
                    material_sigma_e=material_bundle["sigma_e"],
                    material_mu_r=material_bundle["mu_r"],
                    material_thickness=material_bundle["thickness"],
                    frequency_hz=frequency_value,
                )
                transmitted = bdpt_transmitted_light_subpath_state(
                    state,
                    hit,
                    face_material_id=face_material_id,
                    layer_offset=layer_csr["layer_offset"],
                    layer_count=layer_csr["layer_count"],
                    layer_thickness_m=layer_csr["layer_thickness_m"],
                    layer_eps_r=layer_csr["layer_eps_r"],
                    layer_sigma_e=layer_csr["layer_sigma_e"],
                    layer_mu_r=layer_csr["layer_mu_r"],
                    frequency_hz=frequency_value,
                )
            merged = _merge_event_states(reflected, transmitted, choose_transmit)
            # Unbiased event split: every contribution downstream has the form
            # source_power * |field|^2 * (geometry terms), and branch e was
            # selected with probability p_e, so the POWER must be divided by
            # p_e. Dividing the FIELD (and the real amplitude proxy) by
            # sqrt(p_e) achieves exactly that:
            #   E[|field_e / sqrt(p_e)|^2] = sum_e p_e * |field_e|^2 / p_e
            #                              = sum_e |field_e|^2.
            # source_power is deliberately untouched; scaling it too would
            # double count the correction. The reflect probability is
            # 1 - p_s - p_t (p_s = 0 on smooth faces, reproducing wave 2).
            p_event = torch.where(
                choose_transmit, p_transmit, 1.0 - p_scatter - p_transmit
            )
            inv_amplitude = torch.where(
                merged["valid"],
                torch.rsqrt(p_event.clamp_min(1.0e-4)),
                torch.ones_like(p_event),
            )
            # Rough reflect branch: the native kernel applied the SMOOTH
            # stack Jones (amplitude sqrt(R_bar)); multiplying by C_r turns
            # it into the coherent amplitude sqrt(R_coh) that matches the
            # budget driving its selection probability (contract 6.2).
            reflect_scale = torch.where(
                rough & ~choose_transmit & ~choose_scatter,
                coherent_amplitude,
                torch.ones_like(coherent_amplitude),
            )
            amplitude_scale = inv_amplitude * reflect_scale
            for key in ("throughput_real", "throughput_imag"):
                merged[key] = merged[key] * amplitude_scale
            for key in ("field_real", "field_imag"):
                merged[key] = merged[key] * amplitude_scale[:, None]
            # Reflection/transmission directions are delta events, but the
            # sampled event class still has a discrete probability mass. Keep
            # that mass in the proposal density; canonical enumerated delta
            # paths use a separate unit-mass block.
            for key in ("pdf_forward", "pdf_reverse"):
                merged[key] = torch.where(
                    merged["valid"], merged[key] * p_event, torch.zeros_like(p_event)
                )

            merged, scattered_valid, scatter_nee_rows = _emit_scatter_nee(
                rayd=rayd,
                sensor=sensor,
                state=state,
                hit=hit,
                merged=merged,
                choose_scatter=choose_scatter,
                p_scatter=p_scatter,
                material_id=material_id,
                material_axis_rad=material_axis_rad,
                runtimes=runtimes,
                max_scattering_order=max_scattering_order,
                samples=samples,
                seed=seed,
                tx_index=tx_index,
                bounce=bounce,
                device=device,
                scene_diagonal=scene_diagonal,
                frequency_hz=frequency_hz,
                frequency_value=frequency_value,
                tx_power=tx_power,
                ad=ad,
                ledger=ledger,
                sample_blocks=sample_blocks,
            )
            nee_rows += scatter_nee_rows
            transmit_events += int((choose_transmit & merged["valid"]).sum())
            scatter_events += int(scattered_valid.sum())
            reflect_events += int(
                (~choose_transmit & ~choose_scatter & merged["valid"]).sum()
            )
            _emit_mixed_transmission(
                rayd=rayd,
                sensor=sensor,
                merged=merged,
                choose_scatter=choose_scatter,
                emit_mixed_transmission=emit_mixed_transmission,
                sensor_count=sensor_count,
                samples=samples,
                tx_power=tx_power,
                frequency_hz=frequency_hz,
                frequency_value=frequency_value,
                mis=mis,
                beta=beta,
                ad=ad,
                ledger=ledger,
                sample_blocks=sample_blocks,
            )
            scatter_count = _apply_scatter_continuation(
                merged=merged,
                scatter_count=scatter_count,
                choose_scatter=choose_scatter,
                scattered_valid=scattered_valid,
                max_scattering_order=max_scattering_order,
            )
            if not bool(merged["valid"].any()):
                break
            state = merged
            ray_inputs = bdpt_subpath_intersection_inputs(merged)
    return sample_blocks, {
        "transmit": transmit_events,
        "reflect": reflect_events,
        "scatter": scatter_events,
        "scattering_nee_rows": nee_rows,
    }
# --- Transmitter/receiver endpoint tensor packing ---------------------------

def transmitter_tensors(scene: SolverScene) -> tuple[torch.Tensor, torch.Tensor]:
    flat_positions = tuple(
        component
        for transmitter in scene.transmitters
        for component in _vector3_tuple(transmitter.position)
    )
    # Read the host value for the native pack; a live power_w leaf is detached
    # here (its gradient is reattached under ad by the pipeline's _live_tx_power,
    # ADR-022 tx_power threading) so this stays a plain host read.
    powers = tuple(
        float(transmitter.power_w.detach())
        if isinstance(transmitter.power_w, torch.Tensor)
        else float(transmitter.power_w)
        for transmitter in scene.transmitters
    )
    exported = bdpt_transmitter_tensors(flat_positions, powers)
    return exported["positions"], exported["power"]


def receiver_positions(
    scene: SolverScene,
    *,
    reference: torch.Tensor,
    grid: ReceiverGrid | None = None,
) -> torch.Tensor:
    if grid is not None:
        return bdpt_receiver_grid_points(
            reference,
            origin=_vector3_tuple(grid.origin),
            x_axis=_vector3_tuple(grid.x_axis),
            y_axis=_vector3_tuple(grid.y_axis),
            shape=grid.shape,
            spacing=grid.spacing,
        )
    if len(scene.receivers) == 1 and isinstance(scene.receivers[0], ReceiverGrid):
        return receiver_positions(scene, reference=reference, grid=scene.receivers[0])

    flat_positions: list[float] = []
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            flat_positions.extend(_vector3_tuple(receiver.position))
        elif isinstance(receiver, ReceiverGrid):
            raise ValueError("BDPT supports either one ReceiverGrid or point receivers, not mixed receiver grids")
        else:
            raise TypeError(f"receiver type is not accepted: {type(receiver)!r}")
    return bdpt_host_vec3_tensor(tuple(flat_positions))
# --- Launch state -----------------------------------------------------------

def make_launch_state(reference: torch.Tensor, *, tx_count: int, config: Config) -> dict[str, torch.Tensor]:
    return bdpt_launch_state(  # type: ignore[no-any-return]
        reference,
        tx_count=tx_count,
        samples=config.samples,
        sample_streams=config.sample_streams,
        seed=config.seed,
    )
# --- Solver metadata --------------------------------------------------------

# ADR-022 differentiable-parameter inventory: the parameters BDPT AD carries
# gradients for, per estimator block. Reported in metadata so a caller can see
# exactly what is on the graph and what stays frozen. Geometry is differentiable
# only for the enumerated discrete blocks (fixed-winner endpoints / mesh
# vertices, inherited from the enumerated engine); the stochastic sampler keeps
# hit geometry frozen (ad_geometry='enumerated_blocks_only').
_AD_DIFFERENTIABLE_PARAMETERS = (
    "layer_eps_r",
    "layer_sigma_e",
    "layer_thickness",
    "roughness_sigma_h",
    "roughness_corr_x",
    "roughness_corr_y",
    "bsdf_table_values",
    "phase_screen_heights",
    "frequency",
    "tx_power",
)
_AD_GEOMETRY_SCOPE = "enumerated_blocks_only"


_KERNEL_ACCUMULATION = {
    "atomic": "atomic_add",
    "staged": "cell_reduce",
    "compact": "compact_atomic_add",
}

# BDPT per-path component_mask bit scheme (see subpaths.component_mask).
# Transmitted subpaths set bit 8 (delta specular transmission events);
# scattered subpaths set bit 16 (continuous Kirchhoff scattering events).
COMPONENT_MASK_LOS = 1
COMPONENT_MASK_REFLECTION = 2
COMPONENT_MASK_DIFFRACTION = 4
COMPONENT_MASK_TRANSMISSION = 8
COMPONENT_MASK_SCATTERING = 16
_COMPONENT_MASK_BITS = {
    "los": COMPONENT_MASK_LOS,
    "reflection": COMPONENT_MASK_REFLECTION,
    "diffraction": COMPONENT_MASK_DIFFRACTION,
    "transmission": COMPONENT_MASK_TRANSMISSION,
    "scattering": COMPONENT_MASK_SCATTERING,
}


def select_accumulation_strategy(
    config: Config, *, grid_cells: int, estimated_valid_ratio: float
) -> str:
    if config.accumulation_strategy != "auto":
        return config.accumulation_strategy
    if grid_cells <= 0:
        return "atomic"
    samples_per_cell = config.samples / float(grid_cells)
    if estimated_valid_ratio < 0.1:
        return "compact"
    if samples_per_cell >= 64:
        return "staged"
    return "atomic"


def _ad_launch_accounting(
    config: Config, ad_ledger: AdLaunchLedger | None
) -> tuple[int, int, int]:
    """ADR-022 companion accounting: backward/jvp launch counts and tape bytes.

    ad_mode='none' wires no companions and retains no tape (bitwise default).
    Under jvp/vjp report the companion launches this solve registered in the
    AdLaunchLedger, exactly as montecarlo.basic does."""

    ledger = ad_ledger if ad_ledger is not None else AdLaunchLedger()
    backward_launch_count = ledger.launches if config.ad_mode == "vjp" else 0
    jvp_launch_count = ledger.launches if config.ad_mode == "jvp" else 0
    tape_bytes = ledger.tape_bytes if config.ad_mode == "vjp" else 0
    return backward_launch_count, jvp_launch_count, tape_bytes


def make_solver_metadata(
    *,
    config: Config,
    selected_accumulation_strategy: str,
    path_counts_by_strategy: dict[str, int],
    valid_contribution_count: int,
    reflection_available: bool,
    diffraction_available: bool,
    cuda_available: bool,
    optix_available: bool,
    workspace_bytes: int,
    variance_enabled: bool,
    launch_count: int,
    effective_max_depth: int,
    ad_ledger: AdLaunchLedger | None = None,
) -> dict[str, Any]:
    rayd_component_enabled = (
        "reflection" in config.components and reflection_available
    ) or ("diffraction" in config.components and diffraction_available)
    # ADR-022: ad_mode='none' wires no companions and retains no tape (bitwise
    # default). Under jvp/vjp report the companion launches this solve
    # registered in the AdLaunchLedger, exactly as montecarlo.basic does.
    ad_active = config.ad_mode != "none"
    backward_launch_count, jvp_launch_count, tape_bytes = _ad_launch_accounting(
        config, ad_ledger
    )
    kernel_metadata = make_metadata(
        primitive="montecarlo_bdpt_primal",
        forward_launch_count=max(1, int(launch_count)),
        backward_launch_count=backward_launch_count,
        jvp_launch_count=jvp_launch_count,
        tape_bytes=tape_bytes,
        fused_stages=1 if rayd_component_enabled else 0,
        intermediate_bytes=int(workspace_bytes),
        accumulation_strategy=_KERNEL_ACCUMULATION[selected_accumulation_strategy],
        scheduling_strategy="native_fused"
        if rayd_component_enabled
        else "native_cuda",
        rayd_native=reflection_available or diffraction_available,
        ad_status=config.ad_mode if ad_active else "none",
    )
    requested_config = serialize_config(config)
    effective_config = dict(requested_config)
    effective_config["max_depth"] = int(effective_max_depth)
    metadata = {
        "samples": config.samples,
        "seed": config.seed,
        "stream_count": config.sample_streams,
        "sample_streams": config.sample_streams,
        "mis": config.mis,
        "power_heuristic_beta": config.power_heuristic_beta,
        # ADR-019: which combine domain produced the component powers. "power"
        # is the default incoherent per-path accumulation; "coherent" sums the
        # enumerated delta/UTD complex field per (tx, rx, component).
        "combine_domain": "coherent" if config.coherent else "power",
        "coherent": bool(config.coherent),
        "max_depth": config.max_depth,
        "max_light_depth": config.max_light_depth,
        "max_diffraction_order": config.max_diffraction_order,
        "path_counts_by_strategy": path_counts_by_strategy,
        "valid_contribution_count": valid_contribution_count,
        "components": component_availability_status(
            config.components,
            reflection_available=reflection_available,
            diffraction_available=diffraction_available,
            reflection_error="BDPT reflection requires RayD native capability",
            diffraction_error="BDPT diffraction requires RayD native capability",
        ),
        "native_capabilities": {
            "cuda": bool(cuda_available),
            "rayd": bool(reflection_available or diffraction_available),
            "reflection": bool(reflection_available),
            "diffraction": bool(diffraction_available),
            "optix": bool(optix_available),
        },
        "rayd": {
            "reflection": bool(reflection_available),
            "diffraction": bool(diffraction_available),
        },
        "launch_count": max(1, int(launch_count)),
        "accumulation_strategy": selected_accumulation_strategy,
        "workspace_bytes": int(workspace_bytes),
        "variance": bool(variance_enabled),
        "throughput_domain": "complex3_jones_coherent_events",
        # ADR-021 D4: BDPT multi-order diffuse scattering. Order 1 (default)
        # keeps the single-bounce terminal rule (a scattered subpath connects
        # via NEE and terminates); order > 1 lets a scattered subpath continue
        # and scatter again up to the cap, emitting an NEE row at every scatter
        # vertex (power domain, excluded from the coherent combine).
        "max_scattering_order": int(config.max_scattering_order),
        "scattering_depth_rule": (
            "single_bounce_terminal"
            if int(config.max_scattering_order) <= 1
            else "multi_order_continuation"
        ),
        "field_transport": {
            "authoritative_carrier": "complex3_jones",
            "scalar_throughput_role": "sampling_probability_proxy_only",
            "local_frame": "interaction_local_s_p_recomputed_per_event",
            "scattering": "incoherent_power_only_no_complex_field",
            "sensor_depth": "receiver_endpoint_only_always_zero",
        },
        "pdf_domain": "proposal_density_excludes_geometry_jacobian",
        "event_classification": {
            "endpoint": 0,
            "delta_specular_reflection": 1,
            "delta_specular_transmission": 2,
        },
        "component_mask_bits": dict(_COMPONENT_MASK_BITS),
        "delta_strategy": "canonical_enumeration_unit_bidirectional_mass",
        "sampled_delta_mass": "event_selection_probability_in_forward_reverse_pdf",
        "mis_capabilities": {
            "delta_specular_classification": True,
            "continuous_diffraction_strategies": True,
            "reflection_diffraction_coupled_bidirectional_pdf": True,
            "coupled_pdf_domain": "enumerated_bidirectional_discrete_mass",
        },
        "ad_status": config.ad_mode if ad_active else "none",
        # ADR-022: geometry gradients exist only for the enumerated discrete
        # blocks (fixed-winner endpoints / mesh vertices); the stochastic
        # sampler's hit geometry stays frozen in v1. Reported loudly so a caller
        # never mistakes a zero geometry grad through the sampler for a bug.
        "ad_geometry": _AD_GEOMETRY_SCOPE,
        "ad_differentiable_parameters": list(_AD_DIFFERENTIABLE_PARAMETERS),
        "kernel": kernel_metadata,
    }
    metadata.update(
        config_metadata(
            requested=requested_config,
            effective=effective_config,
            # BDPT's single-bounce components are additionally clamped by the
            # effective depth budget, which can be smaller than one bounce.
            component_max_depth=component_max_depth(
                config.components,
                chain_depth=int(effective_max_depth),
                single_bounce_depth=min(1, int(effective_max_depth)),
            ),
        )
    )
    metadata["semantic_capabilities"] = capabilities()["solvers"]["montecarlo_bdpt"]
    return metadata
# --- Endpoint subpath assembly and per-solve workspace contracts ------------

@dataclass(frozen=True, slots=True)
class _SolvePrep:
    """Workspace-sizing and native-capability results for one solve()."""

    native_samples: int
    native_max_depth: int
    selected_accumulation: str
    workspace_bytes: int
    info: dict[str, object]
    rayd: Any
    reflection_available: bool
    diffraction_available: bool
    transmission_available: bool


@dataclass(frozen=True, slots=True)
class _EndpointWorkspace:
    """Endpoint subpath tensors and derived counts shared across stages."""

    tx_reference: torch.Tensor
    tx_power: torch.Tensor
    rx_positions: torch.Tensor
    topology_scene: SolverScene
    tx_polarization: torch.Tensor
    rx_polarization: torch.Tensor
    launch_state: dict[str, torch.Tensor]
    endpoint_subpaths: dict[str, Any]
    los_light_state: dict[str, torch.Tensor] | None
    endpoint_connection_samples: dict[str, torch.Tensor] | None
    endpoint_accumulation: dict[str, torch.Tensor] | None
    launch_count: int
    tx_count: int
    rx_count: int
    endpoint_only: bool


def _accumulate_connection_samples(
    config: Config,
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    accumulation_strategy: str,
    combine_domain: str = "power",
    coeff_real: torch.Tensor | None = None,
    coeff_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Accumulate dispatcher: the differentiable twin under ad_mode != 'none',
    else the bitwise primal. Both domains (power/coherent) route through the
    ADR-022 accumulate companions when AD is active (spec 6.4)."""

    if config.ad_mode != "none":
        return bdpt_accumulate_connection_samples_ad(
            samples,
            tx_count=tx_count,
            rx_count=rx_count,
            accumulation_strategy=accumulation_strategy,
            combine_domain=combine_domain,
            coeff_real=coeff_real,
            coeff_imag=coeff_imag,
        )
    return bdpt_accumulate_connection_samples(  # type: ignore[no-any-return]
        samples,
        tx_count=tx_count,
        rx_count=rx_count,
        accumulation_strategy=accumulation_strategy,
        combine_domain=combine_domain,
        coeff_real=coeff_real,
        coeff_imag=coeff_imag,
    )


def _reduced_light_endpoint_state(
    tx_reference: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor,
    rx_polarization: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """One light endpoint per transmitter for the deterministic LoS term.

    All depth-0 light samples share the transmitter position, so N samples
    per tx are N identical rows weighted 1/N. Connecting the T unique
    endpoints with samples_per_tx=1 yields the identical estimate while the
    connection table shrinks from T*N*R rows to T*R (audit P-1/P-5).
    """

    device = tx_reference.device
    tx_count = int(tx_reference.shape[0])
    tx_ids = torch.arange(tx_count, device=device, dtype=torch.int32)
    seeds = torch.zeros((tx_count,), device=device, dtype=torch.int64)
    return bdpt_endpoint_subpath_state(  # type: ignore[no-any-return]
        tx_reference,
        tx_power,
        tx_polarization,
        rx_positions,
        rx_polarization,
        tx_ids,
        seeds,
    )["light"]


def _live_tx_power(scene: SolverScene, *, reference: torch.Tensor) -> torch.Tensor:
    """Reattach the live per-tx power leaves' gradient onto the native power pack.

    ADR-022 tx_power threading: ``endpoints.transmitter_tensors`` reads
    ``float(power_w)`` and detaches the leaf. Under ad we pack the same values
    from the ``Transmitter.power_w`` tensors and add them (minus their detached
    selves) to the native ``reference`` so the returned tensor is bitwise-equal
    to the detached pack while carrying the leaves' gradient. A float ``power_w``
    packs a plain constant, so a materials-only ad graph is unchanged. The
    native endpoint kernels read the data pointer and detach on output, so the
    live gradient reaches the differentiable inputs (endpoint-connection
    companion, scattering-NEE source power) rather than the frozen subpaths."""

    powers = []
    for transmitter in scene.transmitters:
        power = transmitter.power_w
        if isinstance(power, torch.Tensor):
            powers.append(power.to(device=reference.device, dtype=reference.dtype))
        else:
            powers.append(
                torch.tensor(
                    float(power), device=reference.device, dtype=reference.dtype
                )
            )
    packed = torch.stack(powers)
    return reference + (packed - packed.detach())


def _build_endpoint_subpaths(
    scene: SolverScene,
    config: Config,
    *,
    grid: ReceiverGrid | None,
    transmitter_tensors_fn: Callable[[SolverScene], tuple[torch.Tensor, torch.Tensor]],
    selected_accumulation: str,
    ledger: AdLaunchLedger | None = None,
) -> _EndpointWorkspace:
    tx_reference, tx_power = transmitter_tensors_fn(scene)
    if config.ad_mode != "none":
        tx_power = _live_tx_power(scene, reference=tx_power)
    rx_positions = receiver_positions(scene, reference=tx_reference, grid=grid)
    topology_scene = (
        scene
        if grid is None
        else SolverScene(
            compiled=scene.compiled,
            structures=scene.structures,
            transmitters=scene.transmitters,
            receivers=(grid,),
            frequency=scene.frequency,
            metadata=scene.metadata,
        )
    )
    tx_polarization = transmitter_polarizations_f32(scene, device=tx_reference.device)
    rx_polarization = receiver_polarizations_f32(
        scene, device=tx_reference.device, grid=grid
    )
    launch_state = make_launch_state(
        tx_reference, tx_count=len(scene.transmitters), config=config
    )
    endpoint_subpaths = bdpt_endpoint_subpath_state(
        tx_reference,
        tx_power,
        tx_polarization,
        rx_positions,
        rx_polarization,
        launch_state["tx_id"],
        launch_state["light_seed"],
    )
    los_light_state = (
        _reduced_light_endpoint_state(
            tx_reference,
            tx_power,
            tx_polarization,
            rx_positions,
            rx_polarization,
        )
        if "los" in config.components
        else None
    )
    endpoint_connection_samples = None
    endpoint_accumulation = None
    if los_light_state is not None and not scene.structures:
        if config.ad_mode != "none":
            # ADR-022: the endpoint-only (no-structures) LoS fast path threads the
            # frequency and tx_power gradients through the endpoint-connection
            # companion, exactly like _native_los_connection_samples; the
            # accumulate dispatcher below chains the differentiable contribution.
            endpoint_connection_samples = bdpt_endpoint_connection_samples_ad(
                los_light_state,
                endpoint_subpaths["sensor"],
                tx_power,
                frequency=scene.frequency,
                frequency_value=(
                    float(scene.frequency.detach())
                    if isinstance(scene.frequency, torch.Tensor)
                    else float(scene.frequency)
                ),
                samples_per_tx=1,
                max_paths=None,
                mis=config.mis,
                beta=config.power_heuristic_beta,
                strategy_count=1,
            )
            if ledger is not None:
                ledger.add(
                    los_light_state["field_real"],
                    los_light_state["field_imag"],
                    endpoint_subpaths["sensor"]["field_real"],
                    endpoint_subpaths["sensor"]["field_imag"],
                )
        else:
            endpoint_connection_samples = bdpt_endpoint_connection_samples(
                los_light_state,
                endpoint_subpaths["sensor"],
                frequency_hz=float(scene.frequency),
                samples_per_tx=1,
                max_paths=None,
                mis=config.mis,
                beta=config.power_heuristic_beta,
                strategy_count=1,
            )
        endpoint_accumulation = _accumulate_connection_samples(
            config,
            endpoint_connection_samples,
            tx_count=len(scene.transmitters),
            rx_count=int(rx_positions.shape[0]),
            accumulation_strategy=selected_accumulation,
        )
    launch_count = 1

    tx_count = len(scene.transmitters)
    rx_count = int(rx_positions.shape[0])
    endpoint_only = (
        endpoint_accumulation is not None and config.components == frozenset({"los"})
    )
    return _EndpointWorkspace(
        tx_reference=tx_reference,
        tx_power=tx_power,
        rx_positions=rx_positions,
        topology_scene=topology_scene,
        tx_polarization=tx_polarization,
        rx_polarization=rx_polarization,
        launch_state=launch_state,
        endpoint_subpaths=endpoint_subpaths,
        los_light_state=los_light_state,
        endpoint_connection_samples=endpoint_connection_samples,
        endpoint_accumulation=endpoint_accumulation,
        launch_count=launch_count,
        tx_count=tx_count,
        rx_count=rx_count,
        endpoint_only=endpoint_only,
    )
# --- The shared solve pipeline ----------------------------------------------

_EXPORT_BYTES_PER_PATH = 96
_CONNECTION_BYTES_PER_ROW = 57
_VISIBILITY_BYTES_PER_ROW = 25


def _host_frequency(scene: SolverScene) -> float:
    value = scene.frequency
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


@dataclass(frozen=True, slots=True)
class _BDPTTopologyOptions:
    max_depth: int
    components: frozenset[str]
    max_paths: int | None = None
    _detach_field_tx_power: bool = True
    max_paths_scope: str = "per_pair"
    ad_mode: str = "none"
    coupled_paths: bool = False
    coupled_candidate_limit: int = 1_000_000

    def __post_init__(self) -> None:
        if self.max_depth > 5 and self.components & {
            "reflection",
            "transmission",
        }:
            raise RuntimeError("path reflection/transmission support max_depth <= 5")


def _estimate_workspace_bytes(
    config: Config, *, tx_count: int, grid_cells: int, rx_count: int
) -> int:
    launch_entries = (
        max(0, int(tx_count)) * int(config.samples) * int(config.sample_streams)
    )
    bytes_estimate = launch_entries * 32
    if grid_cells > 0:
        bytes_estimate += tx_count * grid_cells * 3 * 4
    if config.export_paths:
        exported = (
            config.max_exported_paths
            if config.max_exported_paths is not None
            else launch_entries
        )
        bytes_estimate += int(exported) * _EXPORT_BYTES_PER_PATH

    # Dense endpoint-connection tables materialize light_count x rx_count rows
    # (contribution fields + visibility inputs). The LoS term connects one
    # deterministic endpoint per transmitter, so only the sampled reflection
    # and diffraction connection paths scale with the sample budget.
    row_bytes = _CONNECTION_BYTES_PER_ROW + _VISIBILITY_BYTES_PER_ROW
    rx_rows = max(0, int(rx_count))
    dense_rows = 0
    if "los" in config.components:
        dense_rows += max(0, int(tx_count)) * rx_rows
    point_receivers = grid_cells <= 0
    if "reflection" in config.components and (point_receivers or config.export_paths):
        dense_rows += launch_entries * rx_rows * max(1, int(config.max_depth))
    if "diffraction" in config.components and point_receivers:
        dense_rows += launch_entries * rx_rows
    if "transmission" in config.components:
        # Straight endpoint chains: one deterministic row per (tx, rx) pair.
        dense_rows += max(0, int(tx_count)) * rx_rows
        # Sampled mixed reflection+transmission chains retain a connection
        # block per light depth.
        dense_rows += launch_entries * rx_rows * max(1, int(config.max_depth))
    if "scattering" in config.components:
        # NEE rows from scatter-selected vertices: bounded by one block of
        # (scatter events x receivers) per light depth.
        dense_rows += launch_entries * rx_rows * max(1, int(config.max_depth))
    bytes_estimate += dense_rows * row_bytes
    return int(bytes_estimate)


def _check_workspace_guardrail(config: Config, workspace_bytes: int) -> None:
    if config.workspace_limit_bytes is None:
        return
    enforce_memory_budget(
        MemoryEstimate(temporary_bytes=workspace_bytes),
        budget_bytes=config.workspace_limit_bytes,
        workload="workspace for BDPT",
    )


def _path_counts(config: Config, *, tx_count: int) -> dict[str, int]:
    return {
        component: int(config.samples) * int(config.sample_streams) * int(tx_count)
        if component in config.components
        else 0
        for component in (
            "los",
            "reflection",
            "diffraction",
            "transmission",
            "scattering",
        )
    }


def _effective_native_samples(config: Config) -> int:
    return int(config.samples) * int(config.sample_streams)


def _effective_native_depth(config: Config) -> int:
    return min(int(config.max_depth), int(config.max_light_depth))  # type: ignore[arg-type]


def _evaluated_connection_samples(
    paths: EvaluatedPaths,
    selected: torch.Tensor,
    *,
    component_out: int,
    tx_power: torch.Tensor | None = None,
) -> dict[str, torch.Tensor] | None:
    if int(selected.numel()) == 0:
        return None
    topology = paths.topology
    geometry = paths.geometry
    fields = paths.fields
    tx_id = topology.tx_id.index_select(0, selected).to(torch.int32)
    rx_id = topology.rx_id.index_select(0, selected).to(torch.int32)
    depth = topology.depth.index_select(0, selected).to(torch.int32)
    contribution = fields.path_gain.index_select(0, selected)
    if tx_power is not None:
        # ADR-022 tx_power threading (ADR-014 coefficient precedent). The
        # enumerated engine applies per-tx power as a frozen host scale, so each
        # contribution row is EXACTLY LINEAR in P[tx_id]: path_gain = P * base,
        # with base (geometry x |field|^2 / P) independent of P. Reattach the
        # live power's gradient by the exact-primal ratio P_live / P_live.detach()
        # -- x / x.detach() is exactly 1.0 in IEEE for finite nonzero P, so the
        # primal is bitwise unchanged, and d(contribution)/dP = path_gain / P =
        # base is exact by linearity. Any material/frequency gradient already on
        # ``contribution`` (from the enumerated oracle) is preserved because the
        # factor is 1.0. Zero-power rows (path_gain == 0) pass through untouched
        # (denominator clamped, gradient there is 0 anyway).
        power_rows = tx_power.index_select(0, tx_id.to(torch.int64))
        power_detached = power_rows.detach()
        safe_denominator = torch.where(
            power_detached != 0.0, power_detached, torch.ones_like(power_detached)
        )
        contribution = torch.where(
            power_detached != 0.0,
            contribution * (power_rows / safe_denominator),
            contribution,
        )
    count = int(selected.numel())
    component_id = torch.full_like(tx_id, int(component_out))
    one = torch.ones((count,), device=tx_id.device, dtype=torch.float32)
    valid = torch.ones((count,), device=tx_id.device, dtype=torch.bool)
    zero_depth = torch.zeros_like(depth)
    return {
        "topology": torch.stack((tx_id, rx_id, component_id, depth), dim=1),
        "contribution": contribution,
        "pdf": one,
        "mis_weight": one,
        "component_id": component_id,
        "valid": valid,
        "tx_id": tx_id,
        "rx_id": rx_id,
        "grid_linear_id": rx_id.clone(),
        "light_depth": depth,
        "sensor_depth": zero_depth,
        "path_length_m": geometry.path_length_m.index_select(0, selected),
    }


def _single_class_discrete_connection_samples(
    scene: SolverScene,
    config: Config,
    *,
    component: str,
    component_id: int,
    tx_power: torch.Tensor | None = None,
) -> dict[str, torch.Tensor] | None:
    """Enumerate one delta-like path class as unit-mass discrete connections.

    Shared by the reflection and standalone-diffraction BDPT connection builders:
    both consume the public ``evaluate_enumerated_paths`` for a single delta/UTD
    component (ADR-008), select the matching enumerated rows, and pack them with
    unit forward/reverse discrete mass. The selected ``component_id`` also names
    the accumulation bucket (reflection -> 1, diffraction -> 2). The coupled
    builder differs (mixed-order discovery plus a >=3 class selection) and keeps
    its own body.
    """

    paths, _ = evaluate_enumerated_paths(
        scene,
        _BDPTTopologyOptions(
            max_depth=int(config.max_depth),
            components=frozenset({component}),
            # ADR-022: thread ad_mode read-only through the ADR-008 oracle so
            # the enumerated discrete block inherits the enumerated engine's
            # fixed-winner geometry/material AD. 'none' is a no-op.
            ad_mode=config.ad_mode,
        ),
    )
    selected = torch.nonzero(
        paths.topology.component_id == component_id, as_tuple=False
    ).flatten()
    return _evaluated_connection_samples(
        paths, selected, component_out=component_id, tx_power=tx_power
    )


def _reflection_discrete_connection_samples(
    scene: SolverScene, config: Config, *, tx_power: torch.Tensor | None = None
) -> dict[str, torch.Tensor] | None:
    """Enumerate delta-specular paths with unit forward/reverse discrete mass."""

    return _single_class_discrete_connection_samples(
        scene, config, component="reflection", component_id=1, tx_power=tx_power
    )


def _diffraction_discrete_connection_samples(
    scene: SolverScene, config: Config, *, tx_power: torch.Tensor | None = None
) -> dict[str, torch.Tensor] | None:
    """Enumerate first-order UTD diffraction paths with unit discrete mass.

    Standalone diffraction is a delta-like discrete path: first-order UTD emits
    one deterministic edge-diffraction connection per (tx, edge, rx) triple with
    the same field the deterministic solver evaluates. BDPT consumes it as an
    opaque discrete-path oracle (ADR-008/ADR-018) exactly as it does reflection,
    so the standalone diffraction component reproduces the deterministic
    reference instead of the retired crude power heuristic. The enumerated engine
    only implements order-1 diffraction, so max_diffraction_order stays at its
    default of 1.
    """

    return _single_class_discrete_connection_samples(
        scene, config, component="diffraction", component_id=2, tx_power=tx_power
    )


def _transmission_discrete_connection_samples(
    scene: SolverScene, config: Config, *, tx_power: torch.Tensor | None = None
) -> dict[str, torch.Tensor] | None:
    """Enumerate pure straight-segment transmission paths with unit discrete mass.

    ADR-020. Pure specular transmission is delta-like exactly as reflection is,
    so BDPT consumes it as an opaque enumerated discrete-path oracle (ADR-008),
    reproducing the deterministic full-Jones layer-stack field
    (``field_transmission_sequence``) instead of the retired straight-chain TE/TM
    mean. Mixed reflection+transmission chains are not enumerable and stay with
    the event-selected shooting sampler (native full-Jones field).
    """

    return _single_class_discrete_connection_samples(
        scene, config, component="transmission", component_id=5, tx_power=tx_power
    )


def _coupled_discrete_connection_samples(
    scene: SolverScene, config: Config, *, tx_power: torch.Tensor | None = None
) -> dict[str, torch.Tensor] | None:
    """Enumerate mixed delta/UTD paths with unit bidirectional discrete mass."""

    paths, _ = evaluate_enumerated_paths(
        scene,
        _BDPTTopologyOptions(
            max_depth=int(config.max_depth),
            components=frozenset({"reflection", "diffraction"}),
            coupled_paths=True,
            coupled_candidate_limit=int(config.coupled_candidate_limit),
            ad_mode=config.ad_mode,
        ),
    )
    selected = torch.nonzero(paths.topology.component_id >= 3, as_tuple=False).flatten()
    return _evaluated_connection_samples(
        paths, selected, component_out=2, tx_power=tx_power
    )


_COHERENT_ENUMERATED_COMPONENTS = (
    ("los", 0),
    ("reflection", 1),
    ("diffraction", 2),
)


def _enumerated_component_block_with_field(
    scene: SolverScene,
    config: Config,
    *,
    component: str,
    component_id: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None:
    """Enumerate one delta/UTD class as a discrete block plus its complex field.

    ADR-019 coherent path. Returns the same unit-mass discrete connection block
    the power-domain path uses, together with the per-row complex projected
    field coefficient (``path_field``) selected in row order so it aligns with
    the block. ``path_field`` is the natively evaluated complex field the
    deterministic coherent accumulator sums, so BDPT coherent reproduces the
    deterministic per-component coherent power.
    """

    paths, _ = evaluate_enumerated_paths(
        scene,
        _BDPTTopologyOptions(
            max_depth=int(config.max_depth),
            components=frozenset({component}),
            ad_mode=config.ad_mode,
        ),
    )
    selected = torch.nonzero(
        paths.topology.component_id == component_id, as_tuple=False
    ).flatten()
    block = _evaluated_connection_samples(paths, selected, component_out=component_id)
    if block is None:
        return None
    field = paths.fields.path_field.index_select(0, selected)
    return block, field.real, field.imag


def _coupled_component_block_with_field(
    scene: SolverScene, config: Config
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None:
    """Coupled reflection-diffraction discrete block plus its complex field."""

    paths, _ = evaluate_enumerated_paths(
        scene,
        _BDPTTopologyOptions(
            max_depth=int(config.max_depth),
            components=frozenset({"reflection", "diffraction"}),
            coupled_paths=True,
            coupled_candidate_limit=int(config.coupled_candidate_limit),
            ad_mode=config.ad_mode,
        ),
    )
    selected = torch.nonzero(paths.topology.component_id >= 3, as_tuple=False).flatten()
    block = _evaluated_connection_samples(paths, selected, component_out=2)
    if block is None:
        return None
    field = paths.fields.path_field.index_select(0, selected)
    return block, field.real, field.imag


def _collect_coherent_connection_samples(
    scene: SolverScene,
    config: Config,
    *,
    prep: _SolvePrep,
    workspace: _EndpointWorkspace,
    ledger: AdLaunchLedger | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
    int,
    dict[str, int],
    dict[int, Any],
    int,
]:
    """ADR-019 coherent collection over the enumerable delta/UTD family.

    Every coherent-eligible component (los / reflection / diffraction, plus the
    coupled compensator folded into diffraction) routes through the shared
    enumerated engine as a unit-mass discrete block carrying its complex field.
    The blocks concatenate through the native connection-sample concat (12-field
    schema, unchanged) while the per-row field coefficients concatenate in the
    same block order, then the coherent accumulate op sums the phasor per
    (tx, rx, component) and finalizes ``|sum|^2``. Config validation already
    guarantees ``components`` is a subset of {los, reflection, diffraction}, so
    transmission/scattering never reach here.
    """

    topology_scene = workspace.topology_scene
    tx_count = workspace.tx_count
    rx_count = workspace.rx_count
    launch_count = workspace.launch_count

    def zero_component_matrix() -> torch.Tensor:
        return bdpt_zero_matrix(
            workspace.tx_reference, rows=tx_count, cols=rx_count
        )

    sample_blocks: list[dict[str, torch.Tensor]] = []
    coeff_reals: list[torch.Tensor] = []
    coeff_imags: list[torch.Tensor] = []
    for component, component_id in _COHERENT_ENUMERATED_COMPONENTS:
        if component not in config.components:
            continue
        built = _enumerated_component_block_with_field(
            topology_scene, config, component=component, component_id=component_id
        )
        if built is not None:
            block, coeff_real, coeff_imag = built
            sample_blocks.append(block)
            coeff_reals.append(coeff_real)
            coeff_imags.append(coeff_imag)
            launch_count += 1
    if config.coupled_paths:
        built = _coupled_component_block_with_field(topology_scene, config)
        if built is not None:
            block, coeff_real, coeff_imag = built
            sample_blocks.append(block)
            coeff_reals.append(coeff_real)
            coeff_imags.append(coeff_imag)
            launch_count += 1

    empty_event_counts = {
        "transmit": 0,
        "reflect": 0,
        "scatter": 0,
        "scattering_nee_rows": 0,
    }
    if not sample_blocks:
        component_matrices = {
            component: zero_component_matrix()
            for component in (
                "los",
                "reflection",
                "diffraction",
                "transmission",
                "scattering",
            )
        }
        return component_matrices, None, 0, empty_event_counts, {}, launch_count

    estimate_samples = (
        sample_blocks[0]
        if len(sample_blocks) == 1
        else bdpt_concat_connection_samples(tuple(sample_blocks))
    )
    # Concatenate the field coefficients in the identical block order the
    # native concat uses so the coefficient rows align 1:1 with the samples.
    coeff_real = coeff_reals[0] if len(coeff_reals) == 1 else torch.cat(coeff_reals, dim=0)
    coeff_imag = coeff_imags[0] if len(coeff_imags) == 1 else torch.cat(coeff_imags, dim=0)
    accumulated = _accumulate_connection_samples(
        config,
        estimate_samples,
        tx_count=tx_count,
        rx_count=rx_count,
        accumulation_strategy=prep.selected_accumulation,
        combine_domain="coherent",
        coeff_real=coeff_real,
        coeff_imag=coeff_imag,
    )
    component_matrices = {
        component: accumulated[component]
        for component in (
            "los",
            "reflection",
            "diffraction",
            "transmission",
            "scattering",
        )
    }
    return (
        component_matrices,
        estimate_samples,
        0,
        empty_event_counts,
        {},
        launch_count,
    )


def _validate(scene: SolverScene, config: Config) -> ReceiverGrid | None:
    grid = first_receiver_grid(scene)
    if grid is not None and config.receiver_strategy != "grid_area":
        raise RuntimeError("receiver_strategy='point_sphere' requires point receivers")
    validate_scalar_endpoint_features(
        scene.transmitters, scene.receivers, solver="BDPT"
    )
    _reject_live_geometry_through_sampler(scene, config)
    return grid


def _reject_live_geometry_through_sampler(scene: SolverScene, config: Config) -> None:
    """ADR-022: mesh-vertex geometry is frozen for the stochastic sampler
    (``ad_geometry='enumerated_blocks_only'``).

    Reflection/diffraction/transmission (pure) and LoS route through the shared
    enumerated engine, which owns geometry adjoints; but the mixed-transmission
    shooting walk and the scattering NEE draw hit points from the stochastic
    sampler, whose geometry is not differentiable in v1. If a mesh-vertex leaf
    participates in AD and one of those stochastic blocks will run, refuse
    loudly instead of silently detaching. Purely-enumerated component sets keep
    their geometry gradients untouched."""

    if config.ad_mode == "none":
        return
    sampler_active = bool(scene.structures) and bool(
        {"transmission", "scattering"} & set(config.components)
    )
    if not sampler_active:
        return
    if _ad_geometry_live(*(structure.vertices for structure in scene.structures)):
        raise RuntimeError(
            "BDPT ad_geometry='enumerated_blocks_only' (ADR-022): a mesh-vertex "
            "gradient would reach the stochastic transmission/scattering sampler, "
            "whose hit-point geometry is frozen in v1; the geometry gradient is "
            "refused loudly rather than silently detached. Restrict AD to a "
            "purely-enumerated component set or to material/frequency/tx_power "
            "leaves."
        )


def _prepare_workspace_and_capabilities(
    scene: SolverScene,
    config: Config,
    *,
    grid: ReceiverGrid | None,
    build_info_fn: Callable[[], dict[str, object]],
) -> _SolvePrep:
    grid_cells = 0 if grid is None else int(grid.shape[0] * grid.shape[1])
    native_samples = _effective_native_samples(config)
    native_max_depth = _effective_native_depth(config)
    selected_accumulation = select_accumulation_strategy(
        config,
        grid_cells=grid_cells,
        estimated_valid_ratio=1.0 if "los" in config.components else 0.05,
    )
    rx_count_estimate = grid_cells if grid is not None else len(scene.receivers)
    workspace_bytes = _estimate_workspace_bytes(
        config,
        tx_count=len(scene.transmitters),
        grid_cells=grid_cells,
        rx_count=rx_count_estimate,
    )
    _check_workspace_guardrail(config, workspace_bytes)

    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel.montecarlo.bdpt requires CUDA")

    info = build_info_fn()
    rayd = require_compiled(scene).rayd
    rayd_available = bool(info["uses_rayd_native"]) and rayd.available
    reflection_available = rayd_available
    diffraction_available = rayd_available and config.max_diffraction_order > 0
    transmission_available = rayd_available
    if (
        "reflection" in config.components
        and scene.structures
        and not reflection_available
    ):
        raise RuntimeError("reflection requires RayD native capability")
    if (
        "diffraction" in config.components
        and scene.structures
        and not diffraction_available
    ):
        raise RuntimeError("diffraction requires RayD native capability")
    if (
        "transmission" in config.components
        and scene.structures
        and not transmission_available
    ):
        raise RuntimeError("transmission requires RayD native capability")
    return _SolvePrep(
        native_samples=native_samples,
        native_max_depth=native_max_depth,
        selected_accumulation=selected_accumulation,
        workspace_bytes=workspace_bytes,
        info=info,
        rayd=rayd,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        transmission_available=transmission_available,
    )


def _collect_connection_samples(
    scene: SolverScene,
    config: Config,
    *,
    prep: _SolvePrep,
    workspace: _EndpointWorkspace,
    ledger: AdLaunchLedger | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
    int,
    dict[str, int],
    dict[int, Any],
    int,
]:
    tx_reference = workspace.tx_reference
    tx_power = workspace.tx_power
    rx_positions = workspace.rx_positions
    topology_scene = workspace.topology_scene
    tx_polarization = workspace.tx_polarization
    rx_polarization = workspace.rx_polarization
    endpoint_subpaths = workspace.endpoint_subpaths
    los_light_state = workspace.los_light_state
    endpoint_connection_samples = workspace.endpoint_connection_samples
    endpoint_accumulation = workspace.endpoint_accumulation
    endpoint_only = workspace.endpoint_only
    tx_count = workspace.tx_count
    rx_count = workspace.rx_count
    launch_count = workspace.launch_count
    if config.coherent:
        # ADR-019 opt-in coherent combine. Config validation guarantees the
        # component set is coherent-eligible; route the whole solve through the
        # enumerated delta/UTD collector with phasor accumulation.
        return _collect_coherent_connection_samples(
            scene, config, prep=prep, workspace=workspace, ledger=ledger
        )
    rayd = prep.rayd
    native_samples = prep.native_samples
    native_max_depth = prep.native_max_depth
    selected_accumulation = prep.selected_accumulation
    reflection_available = prep.reflection_available
    diffraction_available = prep.diffraction_available
    transmission_available = prep.transmission_available
    def zero_component_matrix() -> torch.Tensor:
        return bdpt_zero_matrix(tx_reference, rows=tx_count, cols=rx_count)

    estimate_samples: dict[str, torch.Tensor] | None = None
    transmission_chain_count = 0
    event_counts = {"transmit": 0, "reflect": 0, "scatter": 0, "scattering_nee_rows": 0}
    scattering_runtimes: dict[int, Any] = {}
    if endpoint_only:
        component_matrices = {
            "los": endpoint_accumulation["los"],  # type: ignore[index]
            "reflection": zero_component_matrix(),
            "diffraction": zero_component_matrix(),
            "transmission": zero_component_matrix(),
            "scattering": zero_component_matrix(),
        }
        estimate_samples = endpoint_connection_samples
    else:
        ad = config.ad_mode != "none"
        # Live per-tx power under ad reattaches the tx_power gradient onto the
        # enumerated discrete blocks (linear coefficient) and the LoS companion.
        enumerated_tx_power = tx_power if ad else None
        sample_blocks: list[dict[str, torch.Tensor]] = []
        if los_light_state is not None:
            sample_blocks.append(
                _native_los_connection_samples(
                    rayd,
                    los_light_state,
                    endpoint_subpaths["sensor"],
                    scene_has_structures=bool(scene.structures),
                    frequency_hz=scene.frequency if ad else _host_frequency(scene),
                    mis=config.mis,
                    beta=config.power_heuristic_beta,
                    strategy_count=1,
                    ad=ad,
                    tx_power=enumerated_tx_power,
                    frequency_value=_host_frequency(scene),
                    ledger=ledger,
                )
            )
            launch_count += 2 if scene.structures else 1
        reflection_requested = (
            "reflection" in config.components
            and reflection_available
            and bool(scene.structures)
            and native_max_depth >= 1
        )
        diffraction_requested = (
            "diffraction" in config.components
            and diffraction_available
            and bool(scene.structures)
            and native_max_depth >= 1
        )
        if reflection_requested:
            reflection_samples = _reflection_discrete_connection_samples(
                topology_scene, config, tx_power=enumerated_tx_power
            )
            if reflection_samples is not None:
                sample_blocks.append(reflection_samples)
                launch_count += 1
        if diffraction_requested:
            # Standalone first-order diffraction routes through the shared
            # enumerated engine as a unit-mass discrete connection, exactly like
            # reflection above (ADR-008/ADR-018), replacing the retired crude
            # native power heuristic.
            diffraction_samples = _diffraction_discrete_connection_samples(
                topology_scene, config, tx_power=enumerated_tx_power
            )
            if diffraction_samples is not None:
                sample_blocks.append(diffraction_samples)
                launch_count += 1
        if config.coupled_paths:
            coupled_samples = _coupled_discrete_connection_samples(
                topology_scene, config, tx_power=enumerated_tx_power
            )
            if coupled_samples is not None:
                sample_blocks.append(coupled_samples)
                launch_count += 1
        transmission_requested = (
            "transmission" in config.components
            and scene.structures
            and transmission_available
            and native_max_depth >= 1
        )
        scattering_requested = (
            "scattering" in config.components
            and scene.structures
            and transmission_available
            and native_max_depth >= 1
        )
        if scattering_requested:
            # Kirchhoff tables per rough material (scatter_model_id == 1);
            # raises kirchhoff_domain_exceeded for out-of-domain roughness.
            scattering_runtimes = rough_material_runtimes(require_compiled(scene))
        sampler_requested = transmission_requested or bool(scattering_runtimes)
        if sampler_requested:
            material_bundle = face_material_field_bundle(
                scene, device=tx_reference.device
            )
            scene_diagonal = scene_diagonal_m(scene)
        if transmission_requested:
            # ADR-020: pure straight-segment transmission routes through the
            # shared enumerated engine as a unit-mass discrete connection, like
            # reflection/diffraction (ADR-008/ADR-018), replacing the
            # polarization-agnostic straight-chain TE/TM mean with the full-Jones
            # layer-stack field. Mixed reflection+transmission chains are handled
            # separately by the event-selected shooting sampler below.
            transmission_samples = _transmission_discrete_connection_samples(
                topology_scene, config, tx_power=enumerated_tx_power
            )
            if transmission_samples is not None:
                sample_blocks.append(transmission_samples)
                transmission_chain_count = int(transmission_samples["valid"].sum())
            launch_count += 1
        if sampler_requested:
            ad = config.ad_mode != "none"
            mixed_blocks, event_counts = _transmission_sampled_connection_samples(
                rayd,
                tx_reference,
                tx_power,
                tx_polarization,
                rx_positions,
                rx_polarization,
                endpoint_subpaths["sensor"],
                material_bundle,
                # ADR-015 Part A / ADR-022: under AD the carrier stays a live
                # tensor so lambda/frequency chains carry a gradient; the primal
                # reads the detached host scalar and is bitwise unchanged.
                frequency_hz=scene.frequency if ad else _host_frequency(scene),
                samples=native_samples,
                max_depth=native_max_depth,
                seed=int(config.seed),
                mis=config.mis,
                beta=config.power_heuristic_beta,
                scattering_runtimes=scattering_runtimes,
                emit_mixed_transmission=transmission_requested,
                scene_diagonal=scene_diagonal,
                ad=ad,
                ledger=ledger,
            )
            sample_blocks.extend(mixed_blocks)
            launch_count += 3
        if sample_blocks:
            estimate_samples = (
                sample_blocks[0]
                if len(sample_blocks) == 1
                else bdpt_concat_connection_samples(tuple(sample_blocks))
            )
            accumulated = _accumulate_connection_samples(
                config,
                estimate_samples,
                tx_count=tx_count,
                rx_count=rx_count,
                accumulation_strategy=selected_accumulation,
            )
            component_matrices = {
                component: accumulated[component]
                for component in (
                    "los",
                    "reflection",
                    "diffraction",
                    "transmission",
                    "scattering",
                )
            }
        else:
            component_matrices = {
                component: zero_component_matrix()
                for component in (
                    "los",
                    "reflection",
                    "diffraction",
                    "transmission",
                    "scattering",
                )
            }
    return (
        component_matrices,
        estimate_samples,
        transmission_chain_count,
        event_counts,
        scattering_runtimes,
        launch_count,
    )


def _accumulate_and_finalize(
    config: Config,
    *,
    grid: ReceiverGrid | None,
    prep: _SolvePrep,
    workspace: _EndpointWorkspace,
    component_matrices: dict[str, torch.Tensor],
    estimate_samples: dict[str, torch.Tensor] | None,
) -> tuple[
    dict[str, torch.Tensor] | None,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor | None,
]:
    tx_reference = workspace.tx_reference
    tx_count = workspace.tx_count
    rx_count = workspace.rx_count
    endpoint_only = workspace.endpoint_only
    native_samples = prep.native_samples
    # ADR-022: the finalize is a linear map with native backward/jvp companions;
    # dispatch the differentiable twin under ad_mode != 'none' so component
    # gradients reach path_gain and the per-component powers. 'none' calls the
    # identical primal symbol (bitwise).
    ad = config.ad_mode != "none"
    component_maps: dict[str, torch.Tensor] | None = None
    point_component_matrices: dict[str, torch.Tensor] | None = None
    if grid is not None:
        component_maps = _component_maps_from_matrices(
            component_matrices,
            rows=grid.shape[0],
            cols=grid.shape[1],
        )
        finalize_maps = (
            bdpt_finalize_component_maps_ad if ad else bdpt_finalize_component_maps
        )
        finalized = finalize_maps(
            component_maps["los"],
            component_maps["reflection"],
            component_maps["diffraction"],
            component_maps["transmission"],
            component_maps["scattering"],
        )
    else:
        point_component_matrices = component_matrices
        finalize_points = (
            bdpt_finalize_point_components_ad
            if ad
            else bdpt_finalize_point_components
        )
        finalized = finalize_points(
            point_component_matrices["los"],
            point_component_matrices["reflection"],
            point_component_matrices["diffraction"],
            point_component_matrices["transmission"],
            point_component_matrices["scattering"],
        )
    path_gain = finalized["path_gain"]
    component_power = {
        "los": finalized["los_power"],
        "reflection": finalized["reflection_power"],
        "diffraction": finalized["diffraction_power"],
    }
    # transmission and scattering maps are always finalized (zeros when the
    # component is off) but only exposed when requested.
    for optional in ("transmission", "scattering"):
        if optional in config.components:
            component_power[optional] = finalized[f"{optional}_power"]
        elif component_maps is not None:
            component_maps.pop(optional)

    variance = None
    if config.diagnostics:
        if estimate_samples is None:
            variance = bdpt_zero_matrix(tx_reference, rows=tx_count, cols=rx_count)
        else:
            variance = bdpt_connection_variance(
                estimate_samples,
                tx_count=tx_count,
                rx_count=rx_count,
                # LoS-only estimates are one deterministic row per (tx, rx)
                # connection, not native_samples Monte Carlo draws.
                samples_per_tx=1 if endpoint_only else native_samples,
            )
        if grid is not None:
            variance = bdpt_los_component_maps_from_matrix(
                variance,
                rows=grid.shape[0],
                cols=grid.shape[1],
            )
    return component_maps, path_gain, component_power, variance


def _export_paths(
    scene: SolverScene,
    config: Config,
    *,
    workspace: _EndpointWorkspace,
    estimate_samples: dict[str, torch.Tensor] | None,
) -> ReceiverGrid | None:
    endpoint_only = workspace.endpoint_only
    los_light_state = workspace.los_light_state
    endpoint_subpaths = workspace.endpoint_subpaths
    tx_reference = workspace.tx_reference
    path_samples = None
    if config.export_paths:
        if endpoint_only:
            exported_endpoint_samples = bdpt_endpoint_connection_samples(
                los_light_state,
                endpoint_subpaths["sensor"],
                frequency_hz=_host_frequency(scene),
                samples_per_tx=1,
                max_paths=config.max_exported_paths,
                mis=config.mis,
                beta=config.power_heuristic_beta,
                strategy_count=1,
            )
            path_samples = _path_samples_from_connection_export(
                exported_endpoint_samples
            )
        else:
            if estimate_samples is None:
                path_samples = _empty_path_samples(tx_reference)
            else:
                exported_path_samples = bdpt_compact_connection_samples(
                    estimate_samples,
                    max_paths=config.max_exported_paths,
                )
                path_samples = _path_samples_from_connection_export(
                    exported_path_samples
                )
    return path_samples


def _build_metadata(
    scene: SolverScene,
    config: Config,
    *,
    prep: _SolvePrep,
    workspace: _EndpointWorkspace,
    estimate_samples: dict[str, torch.Tensor] | None,
    path_gain: torch.Tensor,
    variance: torch.Tensor | None,
    launch_count: int,
    transmission_chain_count: int,
    event_counts: dict[str, int],
    scattering_runtimes: dict[int, Any],
    component_maps: dict[str, torch.Tensor] | None,
    ledger: AdLaunchLedger | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    launch_state = workspace.launch_state
    endpoint_subpaths = workspace.endpoint_subpaths
    selected_accumulation = prep.selected_accumulation
    reflection_available = prep.reflection_available
    diffraction_available = prep.diffraction_available
    info = prep.info
    workspace_bytes = prep.workspace_bytes
    native_max_depth = prep.native_max_depth
    valid_contribution_count = (
        bdpt_count_valid_connection_samples(estimate_samples)
        if estimate_samples is not None
        else (int(path_gain.numel()) if config.components else 0)
    )
    path_counts_by_strategy = _path_counts(config, tx_count=len(scene.transmitters))
    metadata = make_solver_metadata(
        config=config,
        selected_accumulation_strategy=selected_accumulation,
        path_counts_by_strategy=path_counts_by_strategy,
        valid_contribution_count=valid_contribution_count,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        cuda_available=bool(info.get("cuda_available", True)),
        optix_available=bool(info.get("optix_available", False)),
        workspace_bytes=workspace_bytes,
        variance_enabled=variance is not None,
        launch_count=launch_count,
        effective_max_depth=native_max_depth,
        ad_ledger=ledger,
    )
    if "transmission" in config.components:
        # component_mask bit 8 marks transmitted subpaths (contract section 1).
        metadata["transmission"] = {
            "straight_chain_paths": int(transmission_chain_count),
            "event_counts": {
                "transmit": int(event_counts["transmit"]),
                "reflect": int(event_counts["reflect"]),
            },
            "component_mask_bit": 8,
        }
    if "scattering" in config.components:
        # component_mask bit 16 marks scattered subpaths (contract section 1).
        metadata["scattering"] = {
            "event_counts": {"scatter": int(event_counts["scatter"])},
            "nee_connection_rows": int(event_counts["scattering_nee_rows"]),
            "rough_material_count": len(scattering_runtimes),
            "component_mask_bit": 16,
            # v1 depth rule: one Kirchhoff event per path; scattered
            # subpaths connect to the receivers and terminate.
            "max_scattering_order": 1,
        }
    edge_policy = resolve_scene_edge_policy(scene)
    metadata["edge_policy"] = {
        "edge_selection_mode": edge_policy.edge_selection_mode,
        "edge_diffraction": bool(edge_policy.edge_diffraction),
        "boundary_edge_policy": edge_policy.boundary_edge_policy,
    }
    diagnostics: dict[str, Any] | None = None
    if config.diagnostics:
        diagnostics = {
            "path_gain_shape": tuple(path_gain.shape),
            "device": str(path_gain.device),
            "launch_state_shape": tuple(launch_state["tx_id"].shape),
            "endpoint_subpath_shapes": {
                "light": tuple(endpoint_subpaths["light"]["origin"].shape),
                "sensor": tuple(endpoint_subpaths["sensor"]["origin"].shape),
            },
            "component_map_shapes": (
                None
                if component_maps is None
                else {key: tuple(value.shape) for key, value in component_maps.items()}
            ),
        }
    return metadata, diagnostics


def _solve_pipeline(
    scene: SolverScene,
    config: Config,
    *,
    build_info_fn: Callable[[], dict[str, object]],
    transmitter_tensors_fn: Callable[
        [SolverScene], tuple[torch.Tensor, torch.Tensor]
    ] = transmitter_tensors,
) -> Result:
    grid = _validate(scene, config)
    prep = _prepare_workspace_and_capabilities(
        scene, config, grid=grid, build_info_fn=build_info_fn
    )
    # ADR-022 per-solve companion accounting (mirrors montecarlo.basic). Under
    # ad_mode='none' no companion registers, so the ledger stays empty and the
    # metadata reports zero backward/jvp launches and zero tape.
    ledger = AdLaunchLedger()
    workspace = _build_endpoint_subpaths(
        scene,
        config,
        grid=grid,
        transmitter_tensors_fn=transmitter_tensors_fn,
        selected_accumulation=prep.selected_accumulation,
        ledger=ledger,
    )
    (
        component_matrices,
        estimate_samples,
        transmission_chain_count,
        event_counts,
        scattering_runtimes,
        launch_count,
    ) = _collect_connection_samples(
        scene, config, prep=prep, workspace=workspace, ledger=ledger
    )
    component_maps, path_gain, component_power, variance = _accumulate_and_finalize(
        config,
        grid=grid,
        prep=prep,
        workspace=workspace,
        component_matrices=component_matrices,
        estimate_samples=estimate_samples,
    )
    path_samples = _export_paths(
        scene, config, workspace=workspace, estimate_samples=estimate_samples
    )
    metadata, diagnostics = _build_metadata(
        scene,
        config,
        prep=prep,
        workspace=workspace,
        estimate_samples=estimate_samples,
        path_gain=path_gain,
        variance=variance,
        launch_count=launch_count,
        transmission_chain_count=transmission_chain_count,
        event_counts=event_counts,
        scattering_runtimes=scattering_runtimes,
        component_maps=component_maps,
        ledger=ledger,
    )
    return Result(
        path_gain=path_gain,
        component_power=component_power,
        metadata=metadata,
        diagnostics=diagnostics,
        component_maps=component_maps,
        variance=variance,
        path_samples=path_samples,
    )
# --- Public entry point -----------------------------------------------------

def solve(  # type: ignore[no-untyped-def]
    scene: Scene | SceneSnapshot,
    config: Config,
    *,
    reference_frequency_hz,
) -> Result:
    """Run the native CUDA/OptiX BDPT pipeline."""

    endpoint_views = _endpoint_views(scene)
    if (
        any(isinstance(view, ReceiverGrid) for view in endpoint_views)
        and config.receiver_strategy != "grid_area"
    ):
        raise RuntimeError(
            "receiver_strategy='point_sphere' requires point receivers"
        )
    _validate_scalar_endpoint_boundary(endpoint_views)
    validate_scalar_endpoint_features(
        tuple(view for view in endpoint_views if view.source.role == "tx"),
        tuple(view for view in endpoint_views if view.source.role == "rx"),
        solver="BDPT",
    )
    compiled = compile_scene(
        scene, reference_frequency_hz=reference_frequency_hz
    )
    return _solve_pipeline(
        bind_solver_scene(compiled),
        config,
        build_info_fn=build_info,
        transmitter_tensors_fn=transmitter_tensors,
    )


__all__ = ["BDPTPathSamples", "Config", "Result", "solve"]
