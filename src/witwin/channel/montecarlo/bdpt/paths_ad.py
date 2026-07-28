"""ADR-022 BDPT fixed-topology AD companion facades (plan 10a section 6).

These validate the differentiable-input contracts, request the registered native
backward/jvp symbol through ``runtime``, and assert the returned dict. They never
reconstruct the RF physics in Torch: the numerical work runs entirely in the
native companion. The autograd.Function wrappers in ``montecarlo/bdpt/autograd.py``
and ``montecarlo/bdpt/autograd_accumulate.py`` own the tape, the frozen-input
rejection and the need-flag derivation; these facades are pure dispatch.

The primal forward facades and their shared validation helpers live in
``kernels/montecarlo.py``; this module imports the internal contract
validators from there so each contract keeps a single owner.
"""

from __future__ import annotations

import torch

from witwin.channel.runtime import (
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)

from witwin.channel.kernels.montecarlo import (
    _BDPT_COMPONENT_MATRIX_FIELDS,
    _BDPT_COMPONENT_MATRIX_ORDER,
    _bdpt_accumulate_bin_sum_args,
    _bdpt_mis_mode_id,
    _validate_bdpt_connection_samples,
    _validate_bdpt_subpath_state,
)


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
            trailing_shape=trailing,
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
    exported = _required_native_op("bdpt_reflected_light_subpath_state_backward")(
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
    exported = _required_native_op("bdpt_reflected_light_subpath_state_jvp")(
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
    exported = _required_native_op("bdpt_transmitted_light_subpath_state_backward")(
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
    exported = _required_native_op("bdpt_transmitted_light_subpath_state_jvp")(
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
