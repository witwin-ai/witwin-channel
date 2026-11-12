# Copyright Xingyu Chen.
# Native CUDA/OptiX BDPT Monte Carlo solver with Torch tensor storage.

"""Native CUDA/OptiX BDPT Monte Carlo solver with Torch tensor storage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from witwin.core import Scene, SceneSnapshot

from witwin.channel import build_info
from witwin.channel.capabilities import (
    _BDPT_AD_DIFFERENTIABLE_PARAMETERS, ConfigMetadata, ResultSemantics, capabilities,
    config_metadata, component_quantities, estimator_metadata,
    serialize_config,
)
from witwin.channel.components import (
    AD_MODES as _VALID_AD_MODES,
    DEFAULT_COMPONENTS as _DEFAULT_COMPONENTS,
    component_availability_status,
    component_max_depth,
    require_config_type, validate_bounce_depth, validate_chain_depth_cap, validate_config_fields,
    validate_coupled_candidate_limit, validate_coupled_gate, validate_max_depth, validate_samples,
    validate_seed,
    validate_vocabulary,
    validate_workspace_limit_bytes,
    validated_components,
)
from witwin.channel.scene.endpoints import (
    receiver_polarizations_f32,
    transmitter_polarizations_f32,
)
from witwin.channel.kernels import geometry as geometry_kernels
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
    bdpt_transmitter_tensors, bdpt_wavefront_transmission, bdpt_scatter_table_stack,
    BDPT_WAVEFRONT_GRADIENTS, bdpt_wavefront_backward, bdpt_wavefront_jvp,
)
from witwin.channel.materials import face_material_field_bundle, has_multilayer_material
from witwin.channel.interactions.scattering import (
    rough_material_runtimes,
)
from witwin.channel.interactions.transmission import (
    scene_diagonal_m,
)
from witwin.channel.propagation import EvaluatedPaths, evaluate_enumerated_paths
from witwin.channel.runtime import (
    AdLaunchLedger,
    bdpt_zero_matrix,
    KernelMetadata,
    MemoryEstimate,
    _ad_first_order_only,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_live,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    disable_functorch,
    enforce_memory_budget,
    make_metadata,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)
from witwin.channel.scene.compiler import (
    compile as compile_scene,
    receiver_positions as scene_receiver_positions,
    record_dropped_faces,
)
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
    """Require coherent combination to use enumerable components with native complex fields and AD companions."""
    if not coherent:
        return
    refused = components & {"transmission", "scattering"}
    if refused:
        raise ValueError(
            "coherent=True supports only the components {los, reflection, "
            f"diffraction}}; remove {sorted(refused)} or set coherent=False"
        )


def _validate_ad_readiness(ad_mode: str, components: frozenset[str]) -> None:
    """BDPT AD readiness gate.

    ``ad_mode='none'`` is the bitwise default and never builds a tape. Under
    ``jvp``/``vjp`` every BDPT estimator block is differentiable: the frozen
    material/EM/table/frequency/tx_power parameters ride the differentiable fields
    and scattering derivative companions, the fused wavefront's forward-mode and
    replay-reverse passes, and the BDPT endpoint / accumulate / finalize
    companions. ``max_scattering_order > 1`` is allowed: continued scatter
    events are part of the same wavefront derivative. Any combination whose native
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
    """coherent scattering: the diffuse multi-order cap must be a positive bounce count.

    Extracted to a module-level validator mirroring ``_validate_coherent_combine``
    so ``__post_init__`` stays within its maintenance-complexity budget.
    """
    if max_scattering_order < 1:
        raise ValueError(
            f"max_scattering_order must be a positive int, got {max_scattering_order}"
        )


@dataclass(frozen=True, slots=True)
class Config:
    samples: int = 4096
    seed: int = 0
    max_depth: int = 3
    max_light_depth: int | None = None
    max_diffraction_order: int = 1
    # coherent scattering: maximum number of diffuse-scatter events a single BDPT light
    # subpath may undergo. DEFAULT 1 is today's behavior BIT-IDENTICALLY: a
    # scattered subpath emits its NEE connection and terminates (single-bounce
    # terminal rule). >1 lifts the terminal rule so a scattered subpath
    # continues in its sampled direction and may reflect/transmit/scatter again
    # up to this cap, emitting an NEE row at every scatter vertex (power domain;
    # scattering stays excluded from the coherent combination).
    max_scattering_order: int = 1
    coupled_paths: bool = False
    coupled_candidate_limit: int = 1_000_000
    components: frozenset[str] | set[str] | tuple[str, ...] | list[str] = (
        _DEFAULT_COMPONENTS
    )
    # Coherent combine (coherent combination). DEFAULT-OFF opt-in switch. OFF (the default)
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
        validate_config_fields(self)
        validate_samples(self.samples)
        validate_seed(self.seed)
        validate_max_depth(self.max_depth)
        max_light_depth = (
            self.max_depth if self.max_light_depth is None else self.max_light_depth
        )
        if max_light_depth < 0:
            raise ValueError(
                f"max_light_depth must be None or a non-negative int, got {max_light_depth}"
            )
        if self.max_diffraction_order not in {0, 1}:
            raise ValueError(
                f"max_diffraction_order must be 0 or 1, got {self.max_diffraction_order}"
            )
        _validate_scattering_order(self.max_scattering_order)
        components = validated_components(
            self.components,
            error_message="components must be a non-empty subset of {valid}",
        )
        validate_bounce_depth(
            self.max_depth,
            components,
            error_message=(
                "BDPT requires max_depth >= 1 for components "
                f"{sorted(components - {'los'})}, got max_depth={self.max_depth}"
            ),
        )
        validate_chain_depth_cap("BDPT", self.max_depth, components)
        if "diffraction" in components and self.max_diffraction_order == 0:
            raise ValueError(
                "the 'diffraction' component requires max_diffraction_order=1, got 0"
            )
        validate_coupled_gate(
            coupled_paths=self.coupled_paths,
            max_depth=self.max_depth,
            components=components,
        )
        validate_coupled_candidate_limit(self.coupled_candidate_limit)
        validate_vocabulary("mis", self.mis, _VALID_MIS)
        if not self.power_heuristic_beta > 0.0:
            raise ValueError(
                f"power_heuristic_beta must be positive and finite, got {self.power_heuristic_beta!r}"
            )
        validate_vocabulary("receiver_strategy", self.receiver_strategy, _VALID_RECEIVER_STRATEGIES)
        validate_vocabulary(
            "accumulation_strategy", self.accumulation_strategy, _VALID_ACCUMULATION_STRATEGIES
        )
        if self.sample_streams <= 0:
            raise ValueError(f"sample_streams must be a positive int, got {self.sample_streams}")
        if self.max_exported_paths is not None and self.max_exported_paths < 0:
            raise ValueError(
                "max_exported_paths must be None or a non-negative int, got "
                f"{self.max_exported_paths}"
            )
        validate_vocabulary("BDPT ad_mode", self.ad_mode, _VALID_AD_MODES)
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


class _MetadataKeys(ResultSemantics, ConfigMetadata):
    samples: int
    seed: int
    stream_count: int
    sample_streams: int
    mis: str
    power_heuristic_beta: float
    combine_domain: str
    coherent: bool
    max_depth: int
    max_light_depth: int | None
    max_diffraction_order: int
    path_counts_by_strategy: dict[str, int]
    valid_contribution_count: int
    components: dict[str, str]
    native_capabilities: dict[str, bool]
    rayd: dict[str, bool]
    launch_count: int
    accumulation_strategy: str
    workspace_bytes: int
    variance: bool
    throughput_domain: str
    max_scattering_order: int
    scattering_depth_rule: str
    field_transport: dict[str, str]
    pdf_domain: str
    event_classification: dict[str, int]
    component_mask_bits: dict[str, int]
    delta_strategy: str
    sampled_delta_mass: str
    mis_capabilities: dict[str, Any]
    ad_status: str
    ad_geometry: str
    ad_differentiable_parameters: list[str]
    kernel: KernelMetadata
    semantic_capabilities: dict[str, Any]
    sampler: str
    edge_policy: dict[str, Any]


class Metadata(_MetadataKeys, total=False):
    """The stable keys of ``Result.metadata``; a plain ``dict`` at run time."""

    # Present when the transmission component was requested.
    transmission: dict[str, Any]
    # Present when the scattering component was requested.
    scattering: dict[str, Any]
    # Present when compilation removed degenerate or duplicate faces.
    dropped_faces: dict[str, int]


@dataclass(frozen=True, slots=True)
class Result:
    path_gain: torch.Tensor
    component_power: dict[str, torch.Tensor]
    metadata: Metadata
    diagnostics: dict[str, Any] | None = None
    component_maps: dict[str, torch.Tensor] | None = None
    variance: torch.Tensor | None = None
    path_samples: BDPTPathSamples | None = None
# --- Result accumulation and path-sample assembly ---------------------------

def _component_maps_from_matrices(
    component_matrices: dict[str, torch.Tensor], *, rows: int, cols: int,
) -> dict[str, torch.Tensor]:
    return {
        name: bdpt_los_component_maps_from_matrix(component, rows=rows, cols=cols)
        for name, component in component_matrices.items()
    }


def _path_samples_from_connection_export(exported: dict[str, torch.Tensor]) -> BDPTPathSamples:
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
# --- BDPT native AD facades ------------------------------------

def bdpt_endpoint_connection_samples_backward(
    light: dict[str, torch.Tensor], sensor: dict[str, torch.Tensor], *, frequency_hz: float,
    samples_per_tx: int, mis: str, beta: float, strategy_count: int, max_paths: int | None,
    grad_contribution: torch.Tensor | None = None, need_grad_field: bool = False,
    need_grad_frequency: bool = False, need_grad_tx_power: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`bdpt_endpoint_connection_samples` (endpoint-connection derivatives).

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
    exported = _required_native_op("bdpt_endpoint_connection_samples_backward")(
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
    light: dict[str, torch.Tensor], sensor: dict[str, torch.Tensor], *, frequency_hz: float,
    samples_per_tx: int, mis: str, beta: float, strategy_count: int, max_paths: int | None,
    tangent_light_field_real: torch.Tensor | None = None,
    tangent_light_field_imag: torch.Tensor | None = None,
    tangent_sensor_field_real: torch.Tensor | None = None,
    tangent_sensor_field_imag: torch.Tensor | None = None, tangent_frequency: float = 0.0,
    tangent_tx_power: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`bdpt_endpoint_connection_samples` (endpoint-connection derivatives)."""

    _validate_bdpt_subpath_state("light", light, None)
    _validate_bdpt_subpath_state("sensor", sensor, None)
    max_paths_value = -1 if max_paths is None else int(max_paths)
    exported = _required_native_op("bdpt_endpoint_connection_samples_jvp")(
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
    samples: dict[str, torch.Tensor], *, tx_count: int, rx_count: int, accumulation_strategy: str,
    combine_domain: str, coeff_real: torch.Tensor, coeff_imag: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Accumulate forward that also returns the coherent bin-sum buffers.

    Coherent BDPT accumulation: the coherent forward returns the
    per-component phasor bin sums (``S_b``) as non-differentiable outputs so the
    coherent backward can read them without a second atomic-double reduction.
    Returns ``(component_matrices, bin_sums)`` where ``bin_sums`` is an ordered
    tuple (native return order, empty for the power domain) forwarded
    positionally to the backward companion. Numerically the component matrices
    are bitwise the primal :func:`bdpt_accumulate_connection_samples` result."""

    strategy_ids = {"atomic": 0, "staged": 1, "compact": 2}
    combine_ids = {"power": 0, "coherent": 1}
    exported = _required_native_op("bdpt_accumulate_connection_samples")(
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
    samples: dict[str, torch.Tensor], *, tx_count: int, rx_count: int, combine_domain: str,
    bin_sums: tuple[torch.Tensor, ...] = (), grad_path_gain: torch.Tensor | None = None,
    grad_los: torch.Tensor | None = None, grad_reflection: torch.Tensor | None = None,
    grad_diffraction: torch.Tensor | None = None, grad_transmission: torch.Tensor | None = None,
    grad_scattering: torch.Tensor | None = None, need_grad_contribution: bool = False,
    need_grad_coeff: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`bdpt_accumulate_connection_samples`, both domains (connection accumulation).

    Power: ``grad_contribution_r = mis_r grad_M[bin(r)]`` (gather, no atomics);
    this is also the concat-backward split view. Coherent:
    ``grad_c_r = 2 grad_P[b] S_b`` read from the forward-retained ``bin_sums``
    (no backward re-reduction). Both gathers are
    deterministic. ``mis``, the accumulation strategy, and the index structure
    are frozen: the VJP does not read the sample coefficients, only the six
    output-matrix cotangents plus (coherent) the ten forward phasor bin sums."""

    _validate_bdpt_connection_samples("samples", samples, None)
    combine_ids = {"power": 0, "coherent": 1}
    if combine_domain not in combine_ids:
        raise ValueError("combine_domain must be 'power' or 'coherent'")
    bin_args = _bdpt_accumulate_bin_sum_args(combine_domain, bin_sums)
    exported = _required_native_op("bdpt_accumulate_connection_samples_backward")(
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
    samples: dict[str, torch.Tensor], *, tx_count: int, rx_count: int, combine_domain: str,
    bin_sums: tuple[torch.Tensor, ...] = (), tangent_contribution: torch.Tensor | None = None,
    tangent_coeff_real: torch.Tensor | None = None, tangent_coeff_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`bdpt_accumulate_connection_samples`, both domains (connection accumulation).

    Power: ``t_M[b] = SUM_r mis_r tangent_contribution_r``. Coherent:
    ``t_P = 2 Re(conj(S_b) t_S_b)``, ``t_S_b = SUM_r t_c_r``, with ``S_b`` read
    from the forward-retained ``bin_sums`` (no re-reduction).
    Both are fixed-order per-bin sums (deterministic, no float atomics on the
    JVP); the accumulation strategy and sample coefficients are frozen out."""

    _validate_bdpt_connection_samples("samples", samples, None)
    combine_ids = {"power": 0, "coherent": 1}
    if combine_domain not in combine_ids:
        raise ValueError("combine_domain must be 'power' or 'coherent'")
    bin_args = _bdpt_accumulate_bin_sum_args(combine_domain, bin_sums)
    exported = _required_native_op("bdpt_accumulate_connection_samples_jvp")(
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
# --- BDPT subpath and endpoint autograd wrappers ------------------

# Subpath field order and the four differentiable slots (reflected-subpath derivatives/6.2).
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




# ---------------------------------------------------------------------------
# 6.2 transmitted light subpath advance
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6.3 endpoint connection samples
# ---------------------------------------------------------------------------


def _subpath_with_fields(
    base: dict[str, torch.Tensor], field_real: torch.Tensor, field_imag: torch.Tensor,
    throughput_real: torch.Tensor, throughput_imag: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Rebuild a subpath dict overriding only the four differentiable slots."""

    out = dict(base)
    out["field_real"] = field_real
    out["field_imag"] = field_imag
    out["throughput_real"] = throughput_real
    out["throughput_imag"] = throughput_imag
    return out


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
    """Differentiable endpoint (LoS/NEE) connection contribution (endpoint-connection derivatives).

    Differentiable inputs: the light and sensor subpath fields (8 tensors), the
    carrier frequency and ``tx_power`` (P_src). Frozen (reject loudly): the
    connection length L, samples_per_tx N, visibility, MIS mode, component_id,
    and every structural connection field. Only ``contribution`` is a
    differentiable output; the other 11 schema fields are frozen structure."""

    @staticmethod
    def forward(
        light_field_real: Any, light_field_imag: Any, light_throughput_real: Any,
        light_throughput_imag: Any, sensor_field_real: Any, sensor_field_imag: Any,
        sensor_throughput_real: Any, sensor_throughput_imag: Any, frequency: Any,
        frequency_value: Any, tx_power: Any, base_light: Any, base_sensor: Any, params: Any,
    ) -> Any:
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
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
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

        def _detach_dict(source: Any) -> Any:
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
    def _light_sensor(ctx: Any) -> Any:
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
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
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
    def jvp(ctx: Any, *tangents: Any) -> Any:
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
    light: dict[str, torch.Tensor], sensor: dict[str, torch.Tensor], tx_power: torch.Tensor, *,
    frequency: torch.Tensor | float, frequency_value: float | None = None, samples_per_tx: int,
    max_paths: int | None = None, mis: str = "power_heuristic", beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_endpoint_connection_samples` (endpoint-connection derivatives).

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
# --- BDPT accumulation and finalization autograd wrappers ---------------

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
    """Differentiable BDPT finalize (linear map; point-result finalization/6.6).

    The five component matrices/maps are all differentiable; the forward sums
    them into ``path_gain`` and reduces each into a 0-dim power. Backward is the
    native transpose companion, jvp the native forward map on the tangents;
    both deterministic, no atomics."""

    @staticmethod
    def forward(
        los: Any, reflection: Any, diffraction: Any, transmission: Any, scattering: Any, kind: Any,
    ) -> Any:
        forward = (
            bdpt_finalize_point_components
            if kind == "point"
            else bdpt_finalize_component_maps
        )
        out = forward(los, reflection, diffraction, transmission, scattering)
        return tuple(out[name] for name in _FINALIZE_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
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
    def backward(ctx: Any, grad_path_gain: Any, *grad_powers: Any) -> Any:
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
    def jvp(
        ctx: Any, t_los: Any, t_reflection: Any, t_diffraction: Any, t_transmission: Any,
        t_scattering: Any, _t_kind: Any,
    ) -> Any:
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
    los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_finalize_point_components` (point-result finalization)."""

    values = _BdptFinalizeAdFunction.apply(
        los, reflection, diffraction, transmission, scattering, "point"
    )
    return dict(zip(_FINALIZE_FIELDS, values, strict=True))


def bdpt_finalize_component_maps_ad(
    los: torch.Tensor, reflection: torch.Tensor, diffraction: torch.Tensor,
    transmission: torch.Tensor, scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_finalize_component_maps` (map finalization)."""

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
    """Differentiable connection-sample accumulate, both domains (connection accumulation).

    Differentiable inputs: the row ``contribution`` (power domain) OR the
    complex ``coeff_real``/``coeff_imag`` (coherent domain). Frozen (reject
    loudly): mis_weight, tx_id/rx_id/component_id/valid, the whole index
    structure. The coherent forward retains its per-component phasor bin sums
    ``S_b`` so the coherent backward needs no re-reduction (retained forward bin sums)."""

    @staticmethod
    def forward(
        contribution: Any, coeff_real: Any, coeff_imag: Any, base_samples: Any, tx_count: Any,
        rx_count: Any, accumulation_strategy: Any, combine_domain: Any,
    ) -> Any:
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
        # which are marked non-differentiable in setup_context (connection accumulation).
        return tuple(matrices[name] for name in _ACCUMULATE_MATRIX_FIELDS) + tuple(
            bin_sums
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
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
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
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
    def jvp(
        ctx: Any, t_contribution: Any, t_coeff_real: Any, t_coeff_imag: Any, _t_base_samples: Any,
        _t_tx_count: Any, _t_rx_count: Any, _t_strategy: Any, _t_combine: Any,
    ) -> Any:
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
    samples: dict[str, torch.Tensor], *, tx_count: int, rx_count: int,
    accumulation_strategy: str = "atomic", combine_domain: str = "power",
    coeff_real: torch.Tensor | None = None, coeff_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`bdpt_accumulate_connection_samples` (connection accumulation)."""

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
    rayd: Any, light: dict[str, torch.Tensor], sensor: dict[str, torch.Tensor], *,
    scene_has_structures: bool, frequency_hz: float | torch.Tensor, mis: str, beta: float,
    strategy_count: int, ad: bool = False, tx_power: torch.Tensor | None = None,
    frequency_value: float | None = None, ledger: AdLaunchLedger | None = None,
) -> dict[str, torch.Tensor]:
    if ad:
        # BDPT AD: the LoS direct connection carries both a frequency gradient
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
            ledger.add(
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
    return bdpt_filter_connection_samples(samples, visible)


# --- Transmitter/receiver endpoint tensor packing ---------------------------

def transmitter_tensors(scene: SolverScene) -> tuple[torch.Tensor, torch.Tensor]:
    flat_positions = tuple(
        component
        for transmitter in scene.transmitters
        for component in _vector3_tuple(transmitter.position)
    )
    # Read the host value for the native pack; a live power_w leaf is detached
    # here (its gradient is reattached under ad by the pipeline's _live_tx_power,
    # BDPT transmitter-power gradients) so this stays a plain host read.
    powers = tuple(
        float(transmitter.power_w.detach())
        if isinstance(transmitter.power_w, torch.Tensor)
        else float(transmitter.power_w)
        for transmitter in scene.transmitters
    )
    exported = bdpt_transmitter_tensors(flat_positions, powers)
    return exported["positions"], exported["power"]


def receiver_positions(
    scene: SolverScene, *, reference: torch.Tensor, grid: ReceiverGrid | None = None,
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
    if any(isinstance(receiver, ReceiverGrid) for receiver in scene.receivers):
        # Several grids or grids mixed with points: one flat list in scene order,
        # the same receiver ids the enumerated oracle assigns.
        return scene_receiver_positions(scene, device=reference.device, reference=reference)

    flat_positions: list[float] = []
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverPoint):
            flat_positions.extend(_vector3_tuple(receiver.position))
        else:
            raise TypeError(f"receiver type is not accepted: {type(receiver)!r}")
    return bdpt_host_vec3_tensor(tuple(flat_positions))
# --- Launch state -----------------------------------------------------------

def make_launch_state(
    reference: torch.Tensor, *, tx_count: int, config: Config,
) -> dict[str, torch.Tensor]:
    return bdpt_launch_state(
        reference,
        tx_count=tx_count,
        samples=config.samples,
        sample_streams=config.sample_streams,
        seed=config.seed,
    )
# --- Solver metadata --------------------------------------------------------

# BDPT AD inventory: the parameters BDPT AD carries
# gradients for, per estimator block. Reported in metadata so a caller can see
# exactly what is on the graph and what stays frozen. Geometry is differentiable
# only for the enumerated discrete blocks (fixed-winner endpoints / mesh
# vertices, inherited from the enumerated engine); the stochastic sampler keeps
# hit geometry frozen (ad_geometry='enumerated_blocks_only').
_AD_DIFFERENTIABLE_PARAMETERS = _BDPT_AD_DIFFERENTIABLE_PARAMETERS
_AD_GEOMETRY_SCOPE = "enumerated_blocks_only"


_KERNEL_ACCUMULATION = {
    "atomic": "atomic_add",
    "staged": "sorted_segment_reduce",
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
    config: Config, *, grid_cells: int, estimated_valid_ratio: float,
) -> str:
    """Resolve ``accumulation_strategy='auto'``.

    ``auto`` picks the row-parallel atomic reduction, or the compacted atomic
    reduction for sparse grid estimates. It never picks ``staged``: the sorted
    segmented reduction costs a sort of the rows and is the explicit choice for a
    run-to-run reproducible sum.
    """

    if config.accumulation_strategy != "auto":
        return config.accumulation_strategy
    if grid_cells > 0 and estimated_valid_ratio < 0.1:
        return "compact"
    return "atomic"


def _ad_launch_accounting(config: Config, ad_ledger: AdLaunchLedger | None) -> tuple[int, int, int]:
    """BDPT AD companion accounting: backward/jvp launch counts and tape bytes.

    ad_mode='none' wires no companions and retains no tape (bitwise default).
    Under jvp/vjp report the companion launches this solve registered in the
    AdLaunchLedger, exactly as radiomap.Stochastic does."""

    ledger = ad_ledger if ad_ledger is not None else AdLaunchLedger()
    backward_launch_count = ledger.launches if config.ad_mode == "vjp" else 0
    jvp_launch_count = ledger.launches if config.ad_mode == "jvp" else 0
    tape_bytes = ledger.tape_bytes if config.ad_mode == "vjp" else 0
    return backward_launch_count, jvp_launch_count, tape_bytes


def make_solver_metadata(
    *, config: Config, selected_accumulation_strategy: str, path_counts_by_strategy: dict[str, int],
    valid_contribution_count: int, reflection_available: bool, diffraction_available: bool,
    cuda_available: bool, optix_available: bool, workspace_bytes: int, variance_enabled: bool,
    launch_count: int, effective_max_depth: int, ad_ledger: AdLaunchLedger | None = None,
) -> dict[str, Any]:
    rayd_component_enabled = (
        "reflection" in config.components and reflection_available
    ) or ("diffraction" in config.components and diffraction_available)
    # BDPT AD: ad_mode='none' wires no companions and retains no tape (bitwise
    # default). Under jvp/vjp report the companion launches this solve
    # registered in the AdLaunchLedger, exactly as radiomap.Stochastic does.
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
        # coherent combination: which combine domain produced the component powers. "power"
        # is the default incoherent per-path accumulation; "coherent" sums the
        # enumerated delta/UTD complex field per (tx, rx, component).
        "combine_domain": "coherent" if config.coherent else "power",
        **estimator_metadata(
            estimator="stochastic",
            quantity=component_quantities(config.components),
            combine="field" if config.coherent else "power",
            complete=False,
        ),
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
        # coherent scattering: BDPT multi-order diffuse scattering. Order 1 (default)
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
        # BDPT AD: geometry gradients exist only for the enumerated discrete
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
    """Workspace-sizing and native-capability results for one solve."""

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
    config: Config, samples: dict[str, torch.Tensor], *, tx_count: int, rx_count: int,
    accumulation_strategy: str, combine_domain: str = "power",
    coeff_real: torch.Tensor | None = None, coeff_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Accumulate dispatcher: the differentiable twin under ad_mode != 'none',
    else the bitwise primal. Both domains (power/coherent) route through the
    BDPT AD accumulate companions when AD is active (connection accumulation)."""

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
    return bdpt_accumulate_connection_samples(
        samples,
        tx_count=tx_count,
        rx_count=rx_count,
        accumulation_strategy=accumulation_strategy,
        combine_domain=combine_domain,
        coeff_real=coeff_real,
        coeff_imag=coeff_imag,
    )


def _reduced_light_endpoint_state(
    tx_reference: torch.Tensor, tx_power: torch.Tensor, tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor, rx_polarization: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """One light endpoint per transmitter for the deterministic LoS term.

    All depth-0 light samples share the transmitter position, so N samples
    per tx are N identical rows weighted 1/N. Connecting the T unique
    endpoints with samples_per_tx=1 yields the identical estimate while the
    connection table shrinks from T*N*R rows to T*R.
    """

    device = tx_reference.device
    tx_count = int(tx_reference.shape[0])
    tx_ids = torch.arange(tx_count, device=device, dtype=torch.int32)
    seeds = torch.zeros((tx_count,), device=device, dtype=torch.int64)
    return bdpt_endpoint_subpath_state(
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

    BDPT transmitter-power gradients: ``endpoints.transmitter_tensors`` reads
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
    scene: SolverScene, config: Config, *, grid: ReceiverGrid | None,
    transmitter_tensors_fn: Callable[[SolverScene], tuple[torch.Tensor, torch.Tensor]],
    selected_accumulation: str, ledger: AdLaunchLedger | None = None,
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
    # The per-sample launch state and light endpoints are only reported by the
    # diagnostics; every sampler builds its own light subpaths, so a solve
    # without diagnostics launches none of them.
    launch_state = make_launch_state(
        tx_reference,
        tx_count=len(scene.transmitters) if config.diagnostics else 0,
        config=config,
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
            # BDPT AD: the endpoint-only (no-structures) LoS fast path threads the
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
# Fused wavefront tiling: subpaths per chunk and connection pairs per
# visibility query. Per chunk sample the wavefront holds its state, one RayD hit
# record and its share of a pair tile; per pair a start/end/active/visibility row.
_WAVEFRONT_CHUNK_SAMPLES = 1 << 18
_WAVEFRONT_PAIR_TILE = 1 << 20
_WAVEFRONT_BYTES_PER_SAMPLE = 58 + 96
_WAVEFRONT_BYTES_PER_PAIR = 34


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
    config: Config, *, tx_count: int, grid_cells: int, rx_count: int,
) -> int:
    launch_entries = (
        max(0, int(tx_count)) * int(config.samples) * int(config.sample_streams)
    )
    # Diagnostics materialize the launch state and the light endpoints.
    bytes_estimate = launch_entries * 32 if config.diagnostics else 0
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
    # Reflection and diffraction are enumerated delta/UTD rows (no sampled
    # connection table): about one row per (tx, rx) and order.
    if "reflection" in config.components:
        dense_rows += max(0, int(tx_count)) * rx_rows * max(1, int(config.max_depth))
    if "diffraction" in config.components:
        dense_rows += max(0, int(tx_count)) * rx_rows
    if "transmission" in config.components:
        # Straight endpoint chains: one deterministic row per (tx, rx) pair.
        dense_rows += max(0, int(tx_count)) * rx_rows
        # The fused wavefront holds one chunk and one pair tile, never rows.
        chunk = min(launch_entries, _WAVEFRONT_CHUNK_SAMPLES)
        tile_receivers = max(1, min(rx_rows, _WAVEFRONT_PAIR_TILE // max(1, chunk)))
        bytes_estimate += chunk * _WAVEFRONT_BYTES_PER_SAMPLE
        bytes_estimate += chunk * tile_receivers * _WAVEFRONT_BYTES_PER_PAIR
        bytes_estimate += max(0, int(tx_count)) * rx_rows * 8
        if config.ad_mode == "vjp":
            # The reverse pass keeps a per-bounce replay tape of one chunk.
            bytes_estimate += (
                min(launch_entries, _WAVEFRONT_AD_CHUNK_SAMPLES)
                * max(1, int(config.max_depth))
                * _WAVEFRONT_TAPE_BYTES_PER_SLOT
            )
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
    # Config normalizes an unset max_light_depth to max_depth.
    return min(int(config.max_depth), int(cast(int, config.max_light_depth)))


# Enumerated component id -> BDPT accumulation bucket. LoS 0, reflection 1 and
# diffraction 2 keep their id; the interaction chains (3, 4, 9..15) and double
# diffraction (7) fold into diffraction; transmission keeps 5.
_ENUMERATED_BUCKET = (0, 1, 2, 2, 2, 5, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2)


def _evaluated_connection_samples(
    paths: EvaluatedPaths, *, tx_power: torch.Tensor | None = None,
) -> dict[str, torch.Tensor] | None:
    """Pack every evaluated row as a unit-mass discrete connection sample."""

    count = int(paths.row_count)
    if count == 0:
        return None
    topology = paths.topology
    tx_id = topology.tx_id.to(torch.int32)
    rx_id = topology.rx_id.to(torch.int32)
    depth = topology.depth.to(torch.int32)
    bucket = torch.tensor(_ENUMERATED_BUCKET, device=tx_id.device, dtype=torch.int32)
    component_id = bucket.index_select(0, topology.component_id.to(torch.int64))
    contribution = paths.fields.path_gain
    if tx_power is not None:
        # The enumerated engine applies per-tx power as a frozen host scale, so each
        # row is exactly linear in P[tx_id]. Multiplying by P_live / P_live.detach()
        # (exactly 1.0 for finite nonzero P) leaves the primal bitwise unchanged and
        # carries d(contribution)/dP = path_gain / P. Zero-power rows pass through.
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
    one = torch.ones((count,), device=tx_id.device, dtype=torch.float32)
    valid = torch.ones((count,), device=tx_id.device, dtype=torch.bool)
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
        "sensor_depth": torch.zeros_like(depth),
        "path_length_m": paths.geometry.path_length_m,
    }


@dataclass(frozen=True, slots=True)
class _EnumeratedBlock:
    """One enumerated-oracle call packed as unit-mass discrete connection rows."""

    samples: dict[str, torch.Tensor] | None
    dropped_rows: int
    coeff_real: torch.Tensor | None = None
    coeff_imag: torch.Tensor | None = None


def _enumerated_discrete_connection_samples(
    scene: SolverScene, config: Config, *, components: frozenset[str],
    tx_power: torch.Tensor | None = None, with_field: bool = False,
) -> _EnumeratedBlock:
    """Evaluate every requested delta/UTD class in one enumerated-oracle call.

    BDPT consumes the public ``evaluate_enumerated_paths`` read-only. Reflection,
    standalone first-order diffraction, straight-segment transmission, the coupled
    reflection-diffraction classes (when ``coupled_paths``) and, for the coherent
    combine, LoS are delta-like paths with unit forward/reverse discrete mass. One
    call returns all of them; the row's enumerated ``component_id`` names its
    accumulation bucket. ``with_field`` also returns the per-row complex projected
    field the coherent accumulator sums. The block also reports the endpoint pairs
    whose straight segment crosses more walls than ``max_depth``.
    """

    if not components:
        return _EnumeratedBlock(samples=None, dropped_rows=0)
    paths, sidecars = evaluate_enumerated_paths(
        scene,
        _BDPTTopologyOptions(
            max_depth=int(config.max_depth),
            components=components,
            coupled_paths=bool(config.coupled_paths),
            coupled_candidate_limit=int(config.coupled_candidate_limit),
            # Thread ad_mode read-only so the enumerated block inherits the
            # enumerated engine's fixed-winner geometry/material AD.
            ad_mode=config.ad_mode,
        ),
    )
    transaction = sidecars.capacity_transaction
    dropped_rows = (
        int(transaction.terminal_check())
        if "transmission" in components and transaction is not None
        else 0
    )
    samples = _evaluated_connection_samples(paths, tx_power=tx_power)
    if samples is None or not with_field:
        return _EnumeratedBlock(samples=samples, dropped_rows=dropped_rows)
    field = paths.fields.path_field
    return _EnumeratedBlock(
        samples=samples,
        dropped_rows=dropped_rows,
        coeff_real=field.real,
        coeff_imag=field.imag,
    )


def _collect_coherent_connection_samples(
    scene: SolverScene, config: Config, *, prep: _SolvePrep, workspace: _EndpointWorkspace,
    ledger: AdLaunchLedger | None = None,
) -> _CollectedSamples:
    """Coherent combination over the enumerable delta/UTD family.

    Every coherent-eligible component (los / reflection / diffraction, plus the
    coupled classes folded into diffraction) comes from one enumerated-oracle call
    as unit-mass discrete rows carrying their complex field; the coherent
    accumulate op sums the phasor per (tx, rx, component) and finalizes
    ``|sum|^2``. Config validation already guarantees ``components`` is a subset
    of {los, reflection, diffraction}.
    """

    tx_count = workspace.tx_count
    rx_count = workspace.rx_count
    components = frozenset(config.components) & frozenset(
        {"los", "reflection", "diffraction"}
    )
    block = _enumerated_discrete_connection_samples(
        workspace.topology_scene, config, components=components, with_field=True
    )
    launch_count = workspace.launch_count + (1 if block.samples is not None else 0)
    if block.samples is None:
        matrices = {
            component: bdpt_zero_matrix(workspace.tx_reference, rows=tx_count, cols=rx_count)
            for component in _COMPONENT_ORDER
        }
        return _CollectedSamples(matrices=matrices, launch_count=launch_count)
    accumulated = _accumulate_connection_samples(
        config,
        block.samples,
        tx_count=tx_count,
        rx_count=rx_count,
        accumulation_strategy=prep.selected_accumulation,
        combine_domain="coherent",
        coeff_real=block.coeff_real,
        coeff_imag=block.coeff_imag,
    )
    return _CollectedSamples(
        matrices={component: accumulated[component] for component in _COMPONENT_ORDER},
        estimate_samples=block.samples,
        launch_count=launch_count,
    )


def _single_receiver_grid(scene: SolverScene) -> ReceiverGrid | None:
    """The grid of a scene whose only receiver is one ``ReceiverGrid``, else None.

    One grid keeps the grid result layout (component maps); several grids, or
    grids mixed with point receivers, are solved as one flat receiver list in
    scene order (each grid contributes its cells), like the deterministic solver.
    """

    if len(scene.receivers) == 1 and isinstance(scene.receivers[0], ReceiverGrid):
        return scene.receivers[0]
    return None


def _receiver_count(scene: SolverScene) -> int:
    return sum(
        int(receiver.shape[0] * receiver.shape[1]) if isinstance(receiver, ReceiverGrid) else 1
        for receiver in scene.receivers
    )


def _validate(scene: SolverScene, config: Config) -> ReceiverGrid | None:
    grid = _single_receiver_grid(scene)
    if first_receiver_grid(scene) is not None and config.receiver_strategy != "grid_area":
        raise RuntimeError("receiver_strategy='point_sphere' requires point receivers")
    validate_scalar_endpoint_features(
        scene.transmitters, scene.receivers, solver="BDPT"
    )
    _reject_live_geometry_through_sampler(scene, config)
    return grid


def _reject_live_geometry_through_sampler(scene: SolverScene, config: Config) -> None:
    """Refuse endpoint-position and mesh-vertex AD leaves before any launch.

    BDPT differentiates material, frequency and transmitter-power leaves. Its
    sampled estimator does not carry endpoint-position or mesh-vertex
    derivatives through every block, so such a leaf would come back with no
    gradient or a partial one; it is refused instead.
    """

    if config.ad_mode == "none":
        return
    live: list[str] = []
    for index, transmitter in enumerate(scene.transmitters):
        if _ad_geometry_live(transmitter.position):
            live.append(f"transmitter[{index}].position")
    for index, receiver in enumerate(scene.receivers):
        if _ad_geometry_live(receiver.position):
            live.append(f"receiver[{index}].position")
    for index, structure in enumerate(scene.structures):
        if _ad_geometry_live(structure.vertices):
            live.append(f"structure[{index}].vertices")
    if live:
        raise NotImplementedError(
            f"witwin.channel.experimental.bdpt ad_mode={config.ad_mode!r} differentiates "
            "material, frequency and transmitter-power leaves only; "
            f"{', '.join(live)} carries a gradient or tangent. Detach it, or use "
            "witwin.channel.radiomap.Coherent or witwin.channel.path for geometry derivatives"
        )


def _prepare_workspace_and_capabilities(
    scene: SolverScene, config: Config, *, grid: ReceiverGrid | None,
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
    rx_count_estimate = _receiver_count(scene)
    workspace_bytes = _estimate_workspace_bytes(
        config,
        tx_count=len(scene.transmitters),
        grid_cells=grid_cells,
        rx_count=rx_count_estimate,
    )
    _check_workspace_guardrail(config, workspace_bytes)

    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel.experimental.bdpt requires CUDA")

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


_COMPONENT_ORDER = ("los", "reflection", "diffraction", "transmission", "scattering")


@dataclass(slots=True)
class _CollectedSamples:
    """Component matrices and the row blocks, counts and runtimes behind them.

    ``device_counts`` holds 0-dim int64 CUDA tensors that metadata reads in one
    host copy at the end of the solve; ``wavefront_valid_rows`` counts the
    connection rows the fused wavefront accumulated without materializing them.
    """

    matrices: dict[str, torch.Tensor]
    estimate_samples: dict[str, torch.Tensor] | None = None
    launch_count: int = 0
    dropped_rows: int = 0
    device_counts: dict[str, torch.Tensor] = field(default_factory=dict)
    scattering_runtimes: dict[int, Any] = field(default_factory=dict)
    sampler: str = "none"


@dataclass
class _Wavefront:
    """Arguments of one solve's fused wavefront calls."""

    scene: SolverScene
    config: Config
    prep: _SolvePrep
    workspace: _EndpointWorkspace
    materials: dict[str, torch.Tensor]
    runtimes: dict[int, Any]
    emit_mixed: bool
    scatter_stack: dict[str, torch.Tensor] | None = None
    counts: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.runtimes:
            self.scatter_stack = bdpt_scatter_table_stack(self.runtimes, self.materials)

    def positional(self, first: torch.Tensor, second: torch.Tensor) -> tuple[Any, ...]:
        workspace = self.workspace
        return (
            self.prep.rayd,
            workspace.tx_reference,
            workspace.tx_power,
            workspace.tx_polarization,
            workspace.rx_positions,
            workspace.rx_polarization,
            self.materials,
            first,
            second,
        )

    def arguments(
        self, f_te: torch.Tensor | None = None, f_tm: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        scatter = self.scatter_stack
        if scatter is not None and f_te is not None and f_tm is not None:
            scatter = {**scatter, "f_te": f_te, "f_tm": f_tm}
        return {
            "scatter": scatter,
            "frequency_hz": _host_frequency(self.scene),
            "samples": self.prep.native_samples,
            "max_depth": self.prep.native_max_depth,
            "seed": int(self.config.seed),
            "mis_mode": _bdpt_mis_mode_id(self.config.mis),
            "beta": self.config.power_heuristic_beta,
            "emit_mixed": self.emit_mixed,
            "max_scattering_order": int(self.config.max_scattering_order),
            "scene_diagonal": scene_diagonal_m(self.scene) if self.runtimes else 0.0,
            "chunk_samples": _WAVEFRONT_CHUNK_SAMPLES,
            "pair_tile": _WAVEFRONT_PAIR_TILE,
        }

    def run(
        self, base: dict[str, torch.Tensor], rows_capacity: int | None = None,
    ) -> dict[str, Any]:
        return bdpt_wavefront_transmission(
            *self.positional(base["transmission"], base["scattering"]),
            rows_capacity=rows_capacity, **self.arguments(),
        )


# Differentiable leaves of the wavefront, in BDPT_WAVEFRONT_GRADIENTS order:
# (gradient name, material bundle field or None for the table / power /
# frequency leaves).
_WAVEFRONT_MATERIAL_LEAVES = (
    ("face_eps_r", "eps_r"),
    ("face_sigma_e", "sigma_e"),
    ("face_gain", "gain"),
    ("face_thickness", "thickness"),
    ("layer_thickness_m", "layer_thickness_m"),
    ("layer_eps_r", "layer_eps_r"),
    ("layer_sigma_e", "layer_sigma_e"),
)
# The reverse pass replays the wavefront in chunks of this many samples (its
# per-bounce tape is held for one chunk at a time).
_WAVEFRONT_AD_CHUNK_SAMPLES = 1 << 16
# Replay tape bytes per (sample, bounce): event code, primitive, incident
# direction, normal, incoming field, field cotangent, carrier, scale, scatter
# powers cotangent and the scatter record.
_WAVEFRONT_TAPE_BYTES_PER_SLOT = 1 + 4 + 18 * 4 + 4 * 4 + 14 * 4


def _reject_frozen_wavefront_leaves(materials: dict[str, torch.Tensor]) -> None:
    for name in ("mu_r", "layer_mu_r"):
        value = materials[name]
        if value.requires_grad or torch.autograd.forward_ad.unpack_dual(value).tangent is not None:
            raise NotImplementedError(
                f"witwin.channel.experimental.bdpt does not differentiate {name}: the "
                "permeability is a constant of the fixed-topology derivative; detach it"
            )


class _BdptWavefrontAdFunction(torch.autograd.Function):
    """Fixed-topology derivative of the fused wavefront estimate.

    Inputs: the wavefront runner, the base transmission / scattering matrices,
    then the leaves of ``BDPT_WAVEFRONT_GRADIENTS`` (``None`` when absent).
    Outputs: base plus the sampled estimate (or the base alone when the estimate
    is already in the base as accumulated rows). Hits, event choices, visibility,
    lobe densities and MIS weights are frozen; the tangent comes from one native
    forward-mode pass, the gradient from one native replay-and-reverse pass.
    """

    @staticmethod
    def forward(runner: Any, base_transmission: Any, base_scattering: Any, *leaves: Any) -> Any:
        return runner.forward(base_transmission, base_scattering)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        ctx.runner = inputs[0]

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, grad_transmission: Any, grad_scattering: Any) -> Any:
        leaf_needs = ctx.needs_input_grad[3:]
        needs = tuple(
            name for name, need in zip(BDPT_WAVEFRONT_GRADIENTS, leaf_needs, strict=True) if need
        )
        grads: dict[str, torch.Tensor] = {}
        if needs and (grad_transmission is not None or grad_scattering is not None):
            grads = ctx.runner.backward(grad_transmission, grad_scattering, needs)
        leaf_grads = tuple(grads.get(name) for name in BDPT_WAVEFRONT_GRADIENTS)
        return (None, grad_transmission, grad_scattering, *ctx.runner.cast_grads(leaf_grads))

    @staticmethod
    def jvp(
        ctx: Any, _t_runner: Any, t_base_transmission: Any, t_base_scattering: Any, *_t_leaves: Any,
    ) -> Any:
        return ctx.runner.tangents(t_base_transmission, t_base_scattering)


@dataclass
class _WavefrontAdRunner:
    """Native passes behind :class:`_BdptWavefrontAdFunction` for one solve.

    Under forward AD the single native pass computes primal and tangent
    together; ``leaf_tangents`` holds the unpacked leaf tangents.
    """

    wavefront: _Wavefront
    leaves: tuple[torch.Tensor | None, ...]
    include_estimate: bool
    leaf_tangents: dict[str, object] | None = None
    tangent_out: tuple[torch.Tensor, torch.Tensor] | None = None

    def _arguments(self) -> dict[str, Any]:
        return self.wavefront.arguments(self.leaves[7], self.leaves[8])

    def forward(self, base_transmission: torch.Tensor, base_scattering: torch.Tensor) -> Any:
        wavefront = self.wavefront
        if self.leaf_tangents is not None:
            tangents = dict(self.leaf_tangents)
            tangents["base_transmission"] = torch.zeros_like(base_transmission)
            tangents["base_scattering"] = torch.zeros_like(base_scattering)
            out = bdpt_wavefront_jvp(
                *wavefront.positional(base_transmission, base_scattering),
                tangents=tangents, **self._arguments(),
            )
            self.tangent_out = (out["tangent_transmission"], out["tangent_scattering"])
        elif self.include_estimate:
            out = bdpt_wavefront_transmission(
                *wavefront.positional(base_transmission, base_scattering), **self._arguments()
            )
        else:
            return base_transmission.clone(), base_scattering.clone()
        wavefront.counts = out["counts"]
        if not self.include_estimate:
            return base_transmission.clone(), base_scattering.clone()
        return out["transmission"], out["scattering"]

    def tangents(self, t_base_transmission: Any, t_base_scattering: Any) -> Any:
        if self.tangent_out is None:
            raise RuntimeError("BDPT wavefront tangents need the forward-mode pass")
        sampled = self.tangent_out
        out: list[torch.Tensor] = []
        for base, value in zip((t_base_transmission, t_base_scattering), sampled, strict=True):
            base = _ad_native_tangent_or_none(base)
            out.append(value if base is None else base + value)
        return tuple(out)

    def backward(self, grad_transmission: Any, grad_scattering: Any, needs: Any) -> Any:
        wavefront = self.wavefront
        reference = wavefront.workspace.tx_reference
        shape = (wavefront.workspace.tx_count, wavefront.workspace.rx_count)
        zeros = torch.zeros(shape, device=reference.device, dtype=torch.float32)
        arguments = self._arguments()
        arguments["chunk_samples"] = _WAVEFRONT_AD_CHUNK_SAMPLES
        return bdpt_wavefront_backward(
            *wavefront.positional(
                zeros if grad_transmission is None else grad_transmission.to(torch.float32),
                zeros if grad_scattering is None else grad_scattering.to(torch.float32),
            ),
            needs=needs, **arguments,
        )

    def cast_grads(self, grads: Any) -> Any:
        out: list[torch.Tensor | None] = []
        for name, leaf, grad in zip(BDPT_WAVEFRONT_GRADIENTS, self.leaves, grads, strict=True):
            if grad is None or leaf is None:
                out.append(None)
            elif name == "frequency":
                out.append(_ad_frequency_grad(grad, (leaf.dtype, leaf.device)))
            else:
                out.append(grad.to(dtype=leaf.dtype))
        return tuple(out)


def _wavefront_leaf_tangents(leaves: tuple[torch.Tensor | None, ...]) -> dict[str, object]:
    """Forward-mode tangents of the wavefront leaves (absent tangents are omitted)."""

    leaf_tangents: dict[str, object] = {}
    for name, leaf in zip(BDPT_WAVEFRONT_GRADIENTS, leaves, strict=True):
        if leaf is None:
            continue
        tangent = _ad_native_tangent_or_none(torch.autograd.forward_ad.unpack_dual(leaf).tangent)
        if tangent is None:
            continue
        if name == "frequency":
            leaf_tangents[name] = _ad_frequency_tangent(tangent)
        else:
            leaf_tangents[name] = tangent.to(torch.float32)
    # The native pass takes both table tangents or neither.
    for name, other in (("f_te", 8), ("f_tm", 7)):
        partner = leaves[other]
        if name in leaf_tangents and BDPT_WAVEFRONT_GRADIENTS[other] not in leaf_tangents:
            assert partner is not None
            leaf_tangents[BDPT_WAVEFRONT_GRADIENTS[other]] = torch.zeros(
                partner.shape, device=partner.device, dtype=torch.float32
            )
    return leaf_tangents


def _wavefront_ad_estimate(
    wavefront: _Wavefront, base: dict[str, torch.Tensor], *, include_estimate: bool,
    ledger: AdLaunchLedger | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable wavefront estimate of the transmission and scattering matrices."""

    materials = wavefront.materials
    _reject_frozen_wavefront_leaves(materials)
    stack = wavefront.scatter_stack
    frequency = wavefront.scene.frequency
    leaves: tuple[torch.Tensor | None, ...] = (
        *(materials[field_name] for _, field_name in _WAVEFRONT_MATERIAL_LEAVES),
        None if stack is None else stack["f_te"],
        None if stack is None else stack["f_tm"],
        wavefront.workspace.tx_power,
        frequency if isinstance(frequency, torch.Tensor) else None,
    )
    leaf_tangents = (
        _wavefront_leaf_tangents(leaves) if wavefront.config.ad_mode == "jvp" else None
    )
    if ledger is not None:
        ledger.add(*(leaf for leaf in leaves if leaf is not None))
    runner = _WavefrontAdRunner(
        wavefront=wavefront, leaves=leaves, include_estimate=include_estimate,
        leaf_tangents=leaf_tangents,
    )
    transmission, scattering = _BdptWavefrontAdFunction.apply(
        runner, base["transmission"], base["scattering"], *leaves
    )
    return transmission, scattering


def _wavefront_counts(counts: torch.Tensor, *, rows_counted: bool) -> dict[str, torch.Tensor]:
    """Device metadata counts of a wavefront call; rows already in the estimate are not recounted."""

    out = {
        "transmit": counts[0],
        "reflect": counts[1],
        "scatter": counts[2],
        "scattering_nee_rows": counts[4],
    }
    if not rows_counted:
        out["wavefront_valid_rows"] = counts[3] + counts[4]
    return out


class _BdptConcatAdFunction(torch.autograd.Function):
    """Row concatenation of connection blocks that keeps the contribution graph.

    The native concat packs the rows; the contribution cotangent splits back to
    the blocks and the tangents concatenate in the same row order (structural
    only)."""

    @staticmethod
    def forward(blocks: Any, *contributions: Any) -> Any:
        packed = dict(bdpt_concat_connection_samples(tuple(blocks)))
        return packed.pop("contribution"), packed

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        ctx.sizes = [int(value.shape[0]) for value in inputs[1:]]
        ctx.device = inputs[1].device
        ctx.mark_non_differentiable(*(v for v in output[1].values() if isinstance(v, torch.Tensor)))

    @staticmethod
    def backward(ctx: Any, grad_contribution: Any, _grad_packed: Any) -> Any:
        if grad_contribution is None:
            return (None, *(None for _ in ctx.sizes))
        return (None, *torch.split(grad_contribution, ctx.sizes))

    @staticmethod
    def jvp(ctx: Any, _t_blocks: Any, *t_contributions: Any) -> Any:
        parts = []
        for size, tangent in zip(ctx.sizes, t_contributions, strict=True):
            tangent = _ad_native_tangent_or_none(tangent)
            parts.append(
                tangent
                if tangent is not None
                else torch.zeros((size,), device=ctx.device, dtype=torch.float32)
            )
        return torch.cat(parts), None


def _concat_connection_samples_ad(blocks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenated connection blocks whose contribution stays differentiable."""

    detached = [
        {name: value.detach() if name == "contribution" else value for name, value in block.items()}
        for block in blocks
    ]
    contribution, packed = _BdptConcatAdFunction.apply(
        detached, *(block["contribution"] for block in blocks)
    )
    return {**packed, "contribution": contribution}


def _accumulate_blocks(
    config: Config, sample_blocks: list[dict[str, torch.Tensor]], *, prep: _SolvePrep,
    workspace: _EndpointWorkspace,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor]]:
    """Concatenate the row blocks and accumulate them into the component matrices."""

    if not sample_blocks:
        return None, {
            component: bdpt_zero_matrix(
                workspace.tx_reference, rows=workspace.tx_count, cols=workspace.rx_count
            )
            for component in _COMPONENT_ORDER
        }
    if len(sample_blocks) == 1:
        estimate_samples = sample_blocks[0]
    elif config.ad_mode != "none":
        estimate_samples = _concat_connection_samples_ad(sample_blocks)
    else:
        estimate_samples = bdpt_concat_connection_samples(tuple(sample_blocks))
    accumulated = _accumulate_connection_samples(
        config,
        estimate_samples,
        tx_count=workspace.tx_count,
        rx_count=workspace.rx_count,
        accumulation_strategy=prep.selected_accumulation,
    )
    return estimate_samples, {component: accumulated[component] for component in _COMPONENT_ORDER}


def _endpoint_only_samples(workspace: _EndpointWorkspace) -> _CollectedSamples:
    """LoS-only solve without structures: the endpoint connection block is the estimate."""

    matrices = {
        component: bdpt_zero_matrix(
            workspace.tx_reference, rows=workspace.tx_count, cols=workspace.rx_count
        )
        for component in _COMPONENT_ORDER
    }
    # An endpoint-only workspace always carries the endpoint accumulation.
    matrices["los"] = cast("dict[str, torch.Tensor]", workspace.endpoint_accumulation)["los"]
    return _CollectedSamples(
        matrices=matrices,
        estimate_samples=workspace.endpoint_connection_samples,
        launch_count=workspace.launch_count,
    )


def _enumerated_components(config: Config, prep: _SolvePrep) -> frozenset[str]:
    """Requested delta/UTD classes whose native capability is available."""

    available = {
        "reflection": prep.reflection_available,
        "diffraction": prep.diffraction_available,
        "transmission": prep.transmission_available,
    }
    return frozenset(
        component for component, ok in available.items() if component in config.components and ok
    )


def _apply_wavefront_estimate(
    wavefront: _Wavefront, matrices: dict[str, torch.Tensor], rows_pass: dict[str, Any] | None, *,
    ad: bool, ledger: AdLaunchLedger | None = None,
) -> dict[str, Any]:
    """Add the wavefront estimate to the accumulated matrices (in place).

    Without a rows pass the wavefront adds its mixed-chain and scatter
    next-event estimates without materializing rows. After a rows pass the
    matrices already hold the estimate; under AD the wavefront companion then
    adds only its derivative. Returns the call whose counts the metadata reads.
    """

    if rows_pass is not None and not ad:
        return rows_pass
    if ad:
        matrices["transmission"], matrices["scattering"] = _wavefront_ad_estimate(
            wavefront, matrices, include_estimate=rows_pass is None, ledger=ledger
        )
        return rows_pass if rows_pass is not None else {"counts": wavefront.counts}
    sampled = wavefront.run(matrices)
    matrices["transmission"] = sampled["transmission"]
    matrices["scattering"] = sampled["scattering"]
    return sampled


def _collect_connection_samples(
    scene: SolverScene, config: Config, *, prep: _SolvePrep, workspace: _EndpointWorkspace,
    ledger: AdLaunchLedger | None = None,
) -> _CollectedSamples:
    if config.coherent:
        # Config validation guarantees the component set is coherent-eligible;
        # route the whole solve through the enumerated collector with phasor
        # accumulation.
        return _collect_coherent_connection_samples(
            scene, config, prep=prep, workspace=workspace, ledger=ledger
        )
    tx_reference = workspace.tx_reference

    def zero_component_matrix() -> torch.Tensor:
        return bdpt_zero_matrix(tx_reference, rows=workspace.tx_count, cols=workspace.rx_count)

    if workspace.endpoint_only:
        return _endpoint_only_samples(workspace)
    ad = config.ad_mode != "none"
    # Live per-tx power under ad reattaches the tx_power gradient onto the
    # enumerated discrete blocks (linear coefficient) and the LoS companion.
    enumerated_tx_power = workspace.tx_power if ad else None
    launch_count = workspace.launch_count
    sample_blocks: list[dict[str, torch.Tensor]] = []
    device_counts: dict[str, torch.Tensor] = {}
    if workspace.los_light_state is not None:
        sample_blocks.append(
            _native_los_connection_samples(
                prep.rayd,
                workspace.los_light_state,
                workspace.endpoint_subpaths["sensor"],
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
    structured = bool(scene.structures) and prep.native_max_depth >= 1
    enumerated = _enumerated_components(config, prep) if structured else frozenset()
    block = _enumerated_discrete_connection_samples(
        workspace.topology_scene, config, components=enumerated, tx_power=enumerated_tx_power
    )
    if block.samples is not None:
        sample_blocks.append(block.samples)
        launch_count += 1
    transmission_requested = "transmission" in enumerated
    if transmission_requested:
        device_counts["straight_chain_paths"] = (
            torch.zeros((), device=tx_reference.device, dtype=torch.int64)
            if block.samples is None
            else (block.samples["component_id"] == 5).sum()
        )
    scattering_requested = (
        "scattering" in config.components and structured and prep.transmission_available
    )
    # Kirchhoff tables per rough material (scatter_model_id == 1); raises
    # kirchhoff_domain_exceeded for out-of-domain roughness.
    runtimes = rough_material_runtimes(require_compiled(scene)) if scattering_requested else {}
    sampler = "none"
    wavefront_materials: dict[str, torch.Tensor] | None = None
    if transmission_requested or runtimes:
        material_bundle = face_material_field_bundle(scene, device=tx_reference.device)
        sampler = "wavefront"
        wavefront_materials = material_bundle
    wavefront = (
        None
        if wavefront_materials is None
        else _Wavefront(
            scene=scene, config=config, prep=prep, workspace=workspace,
            materials=wavefront_materials, runtimes=runtimes, emit_mixed=transmission_requested,
        )
    )
    sampled: dict[str, Any] | None = None
    rows_needed = config.export_paths or config.diagnostics
    if wavefront is not None and rows_needed:
        # Exported paths and variance need the rows: one counting pass, the
        # sanctioned compact-cardinality copy of the row count, and a second pass
        # (same random numbers, same rows) that writes exactly that many rows.
        zeros = {name: zero_component_matrix() for name in ("transmission", "scattering")}
        counted = wavefront.run(zeros)["counts"]
        row_count = int(counted[3:5].sum())
        sampled = wavefront.run(zeros, rows_capacity=row_count)
        if row_count:
            sample_blocks.append(sampled["rows"])
        launch_count += 3
    estimate_samples, matrices = _accumulate_blocks(
        config, sample_blocks, prep=prep, workspace=workspace
    )
    if wavefront is not None:
        launch_count += 3 if sampled is None else 0
        sampled = _apply_wavefront_estimate(wavefront, matrices, sampled, ad=ad, ledger=ledger)
    if sampled is not None:
        device_counts.update(
            _wavefront_counts(sampled["counts"], rows_counted=estimate_samples is not None and rows_needed)
        )
    return _CollectedSamples(
        matrices=matrices,
        estimate_samples=estimate_samples,
        launch_count=launch_count,
        dropped_rows=block.dropped_rows,
        device_counts=device_counts,
        scattering_runtimes=runtimes,
        sampler=sampler,
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
    # BDPT AD: the finalize is a linear map with native backward/jvp companions;
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
    scene: SolverScene, config: Config, *, workspace: _EndpointWorkspace,
    estimate_samples: dict[str, torch.Tensor] | None,
) -> BDPTPathSamples | None:
    endpoint_only = workspace.endpoint_only
    los_light_state = workspace.los_light_state
    endpoint_subpaths = workspace.endpoint_subpaths
    tx_reference = workspace.tx_reference
    path_samples = None
    if config.export_paths:
        if endpoint_only:
            # An endpoint-only solve requests exactly the LoS component, so the LoS light
            # endpoint state exists.
            exported_endpoint_samples = bdpt_endpoint_connection_samples(
                cast("dict[str, torch.Tensor]", los_light_state),
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


def _read_counts(
    collected: _CollectedSamples, *, path_gain: torch.Tensor, config: Config,
) -> dict[str, int]:
    """Read every metadata count of the solve in one device-to-host copy."""

    counts = dict(collected.device_counts)
    estimate_samples = collected.estimate_samples
    if estimate_samples is not None:
        counts["valid_rows"] = bdpt_count_valid_connection_samples(estimate_samples)
    names = sorted(counts)
    host = (
        [int(value) for value in torch.stack([counts[name] for name in names]).tolist()]
        if names
        else []
    )
    values = dict(zip(names, host, strict=True))
    if estimate_samples is None and "wavefront_valid_rows" not in values:
        valid = int(path_gain.numel()) if config.components else 0
    else:
        valid = values.get("valid_rows", 0) + values.get("wavefront_valid_rows", 0)
    values["valid_contribution_count"] = valid
    return values


def _build_metadata(
    scene: SolverScene, config: Config, *, prep: _SolvePrep, workspace: _EndpointWorkspace,
    collected: _CollectedSamples, path_gain: torch.Tensor, variance: torch.Tensor | None,
    component_maps: dict[str, torch.Tensor] | None, ledger: AdLaunchLedger | None = None,
) -> tuple[Metadata, dict[str, Any] | None]:
    launch_state = workspace.launch_state
    endpoint_subpaths = workspace.endpoint_subpaths
    selected_accumulation = prep.selected_accumulation
    # A scene without structures has nothing to reflect or diffract from; like the
    # other solvers it reports the component enabled and returns the LoS answer.
    reflection_available = prep.reflection_available or not scene.structures
    diffraction_available = prep.diffraction_available or (
        not scene.structures and config.max_diffraction_order > 0
    )
    info = prep.info
    counts = _read_counts(collected, path_gain=path_gain, config=config)
    metadata = make_solver_metadata(
        config=config,
        selected_accumulation_strategy=selected_accumulation,
        path_counts_by_strategy=_path_counts(config, tx_count=len(scene.transmitters)),
        valid_contribution_count=counts["valid_contribution_count"],
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        cuda_available=bool(info.get("cuda_available", True)),
        optix_available=bool(info.get("optix_available", False)),
        workspace_bytes=prep.workspace_bytes,
        variance_enabled=variance is not None,
        launch_count=collected.launch_count,
        effective_max_depth=prep.native_max_depth,
        ad_ledger=ledger,
    )
    metadata["sampler"] = collected.sampler
    if "transmission" in config.components:
        # component_mask bit 8 marks transmitted subpaths (component classification).
        metadata["transmission"] = {
            "straight_chain_paths": counts.get("straight_chain_paths", 0),
            # Endpoint pairs whose straight segment crosses more walls than max_depth.
            "dropped_rows_over_max_depth": int(collected.dropped_rows),
            "event_counts": {
                "transmit": counts.get("transmit", 0),
                "reflect": counts.get("reflect", 0),
            },
            "component_mask_bit": 8,
        }
    if "scattering" in config.components:
        # component_mask bit 16 marks scattered subpaths (component classification).
        metadata["scattering"] = {
            "event_counts": {"scatter": counts.get("scatter", 0)},
            "nee_connection_rows": counts.get("scattering_nee_rows", 0),
            "rough_material_count": len(collected.scattering_runtimes),
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
    return cast(Metadata, metadata), diagnostics


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
    # BDPT AD per-solve companion accounting (mirrors radiomap.Stochastic). Under
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
    collected = _collect_connection_samples(
        scene, config, prep=prep, workspace=workspace, ledger=ledger
    )
    component_maps, path_gain, component_power, variance = _accumulate_and_finalize(
        config,
        grid=grid,
        prep=prep,
        workspace=workspace,
        component_matrices=collected.matrices,
        estimate_samples=collected.estimate_samples,
    )
    path_samples = _export_paths(
        scene, config, workspace=workspace, estimate_samples=collected.estimate_samples
    )
    metadata, diagnostics = _build_metadata(
        scene,
        config,
        prep=prep,
        workspace=workspace,
        collected=collected,
        path_gain=path_gain,
        variance=variance,
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

def solve(
    scene: Scene | SceneSnapshot, config: Config, *, reference_frequency_hz: float | torch.Tensor,
) -> Result:
    """Run the native CUDA/OptiX BDPT pipeline."""

    require_config_type(config, Config, "witwin.channel.experimental.bdpt")
    endpoint_views = _endpoint_views(scene)
    if (
        any(isinstance(view, ReceiverGrid) for view in endpoint_views)
        and config.receiver_strategy != "grid_area"
    ):
        raise ValueError(
            f"receiver_strategy={config.receiver_strategy!r} requires point receivers "
            "only, but the scene has a ReceiverGrid; use receiver_strategy='grid_area'"
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
    if "reflection" in frozenset(config.components) and has_multilayer_material(compiled):
        # BDPT's sampled connections still use the single-slab reflection law; refuse rather than
        # mix it with the layered law of the enumerated paths.
        raise NotImplementedError(
            "BDPT does not yet evaluate reflection off multilayer materials; use the "
            "deterministic, path or Monte Carlo basic solver, or a single-layer material"
        )
    result = _solve_pipeline(
        bind_solver_scene(compiled),
        config,
        build_info_fn=build_info,
        transmitter_tensors_fn=transmitter_tensors,
    )
    record_dropped_faces(result, compiled)
    return result


__all__ = ["BDPTPathSamples", "Config", "Metadata", "Result", "solve"]
