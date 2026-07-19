from __future__ import annotations

import torch

from witwin.channel_native.materials import validate_layer_csr as _validate_layer_csr
from witwin.channel_native.propagation.geometry import (
    BDPT_INTERSECTION_FIELDS as _BDPT_INTERSECTION_FIELDS,
)
from witwin.channel_native.runtime.symbols import (
    native_extension,
    required_symbol as _required_native_op,
)
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


def bdpt_launch_state(
    reference: torch.Tensor,
    *,
    tx_count: int,
    samples: int,
    sample_streams: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if tx_count < 0:
        raise ValueError("tx_count must be non-negative")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if sample_streams <= 0:
        raise ValueError("sample_streams must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    native = native_extension()
    if native is None or not hasattr(native, "bdpt_launch_state"):
        raise RuntimeError("_channel_native.bdpt_launch_state CUDA kernel is required")
    exported = native.bdpt_launch_state(
        reference,
        int(tx_count),
        int(samples),
        int(sample_streams),
        int(seed),
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_launch_state must return a dict")
    expected = int(tx_count) * int(samples) * int(sample_streams)
    for name in ("tx_id", "sample_id", "stream_id"):
        validate_cuda_tensor(name, exported[name], dtype=torch.int32, ndim=1)
        if exported[name].shape != (expected,):
            raise ValueError(
                f"_channel_native.bdpt_launch_state returned bad {name} shape"
            )
    for name in ("light_seed",):
        validate_cuda_tensor(name, exported[name], dtype=torch.int64, ndim=1)
        if exported[name].shape != (expected,):
            raise ValueError(
                f"_channel_native.bdpt_launch_state returned bad {name} shape"
            )
    return exported


_BDPT_SUBPATH_SCHEMA: dict[str, tuple[torch.dtype, tuple[int | None, ...]]] = {
    "origin": (torch.float32, (None, 3)),
    "direction": (torch.float32, (None, 3)),
    "throughput_real": (torch.float32, (None,)),
    "throughput_imag": (torch.float32, (None,)),
    "pdf_forward": (torch.float32, (None,)),
    "pdf_reverse": (torch.float32, (None,)),
    "depth": (torch.int32, (None,)),
    "component_mask": (torch.int32, (None,)),
    "primitive_id": (torch.int32, (None,)),
    "edge_id": (torch.int32, (None,)),
    "tx_id": (torch.int32, (None,)),
    "rx_id": (torch.int32, (None,)),
    "grid_linear_id": (torch.int32, (None,)),
    "valid": (torch.bool, (None,)),
    "path_length": (torch.float32, (None,)),
    "field_real": (torch.float32, (None, 3)),
    "field_imag": (torch.float32, (None, 3)),
    "source_power": (torch.float32, (None,)),
    "event_type": (torch.int32, (None,)),
}


def _validate_bdpt_subpath_state(
    name: str, exported: dict[str, torch.Tensor], expected_count: int | None
) -> None:
    if not isinstance(exported, dict):
        raise TypeError(f"{name} must be a dict")
    if set(exported) != set(_BDPT_SUBPATH_SCHEMA):
        raise ValueError(f"{name} returned unexpected fields")
    inferred_count: int | None = expected_count
    for field, (dtype, shape_spec) in _BDPT_SUBPATH_SCHEMA.items():
        tensor = exported[field]
        validate_cuda_tensor(field, tensor, dtype=dtype, ndim=len(shape_spec))
        if inferred_count is None:
            inferred_count = int(tensor.shape[0])
        expected_shape = tuple(
            inferred_count if dim is None else dim for dim in shape_spec
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} returned bad {field} shape")


_BDPT_CONNECTION_SCHEMA: dict[str, tuple[torch.dtype, tuple[int | None, ...]]] = {
    "topology": (torch.int32, (None, 4)),
    "contribution": (torch.float32, (None,)),
    "pdf": (torch.float32, (None,)),
    "mis_weight": (torch.float32, (None,)),
    "component_id": (torch.int32, (None,)),
    "valid": (torch.bool, (None,)),
    "tx_id": (torch.int32, (None,)),
    "rx_id": (torch.int32, (None,)),
    "grid_linear_id": (torch.int32, (None,)),
    "light_depth": (torch.int32, (None,)),
    "sensor_depth": (torch.int32, (None,)),
    "path_length_m": (torch.float32, (None,)),
}


def _validate_bdpt_connection_samples(
    name: str,
    exported: dict[str, torch.Tensor],
    expected_count: int | None,
) -> None:
    if not isinstance(exported, dict):
        raise TypeError(f"{name} must be a dict")
    if set(exported) != set(_BDPT_CONNECTION_SCHEMA):
        raise ValueError(f"{name} returned unexpected fields")
    inferred_count = expected_count
    for field, (dtype, shape_spec) in _BDPT_CONNECTION_SCHEMA.items():
        tensor = exported[field]
        validate_cuda_tensor(
            f"{name}.{field}", tensor, dtype=dtype, ndim=len(shape_spec)
        )
        if inferred_count is None:
            inferred_count = int(tensor.shape[0])
        expected_shape = tuple(
            inferred_count if dim is None else dim for dim in shape_spec
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} returned bad {field} shape")


# The six accumulate outputs, in the order every caller reads them. The set is
# used for the ADR-022 subset key-set check (the coherent forward may add
# bin-sum buffers as extra keys).
_BDPT_COMPONENT_MATRIX_ORDER = (
    "path_gain",
    "los",
    "reflection",
    "diffraction",
    "transmission",
    "scattering",
)
_BDPT_COMPONENT_MATRIX_FIELDS = frozenset(_BDPT_COMPONENT_MATRIX_ORDER)

# ADR-022 spec 6.4: the coherent forward returns the per-component phasor bin
# sums S_b as non-differentiable outputs; the coherent backward reads them as
# explicit args in this order (real/imag per accumulating component). Absent for
# the power domain.
_BDPT_ACCUMULATE_BIN_SUM_ORDER = (
    "los_re",
    "los_im",
    "reflection_re",
    "reflection_im",
    "diffraction_re",
    "diffraction_im",
    "transmission_re",
    "transmission_im",
    "scattering_re",
    "scattering_im",
)


def _bdpt_accumulate_bin_sum_args(
    combine_domain: str, bin_sums: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor | None, ...]:
    """Expand the coherent forward's phasor bin sums into the ten positional
    ``los_re..scattering_im`` args the native accumulate VJP/JVP consume.

    ADR-022 spec 6.4 (supervisor ruling): the coherent backward/jvp read the
    per-component bin sums ``S_b`` retained by the forward, so no in-backward
    re-reduction and no sample coefficients are needed. The power domain takes
    no bin sums; every slot is ``None``."""

    bins = tuple(bin_sums)
    if combine_domain == "coherent":
        if len(bins) != len(_BDPT_ACCUMULATE_BIN_SUM_ORDER):
            raise ValueError(
                "coherent accumulate backward/jvp requires the ten forward "
                "phasor bin sums"
            )
        return bins
    if bins:
        raise ValueError("power-domain accumulate takes no bin sums")
    return (None,) * len(_BDPT_ACCUMULATE_BIN_SUM_ORDER)


def _bdpt_mis_mode_id(mis: str) -> int:
    if mis == "none":
        return 0
    if mis == "balance":
        return 1
    if mis == "power_heuristic":
        return 2
    raise ValueError("mis is not supported")


def bdpt_empty_subpath_state(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    exported = _required_native_op("bdpt_empty_subpath_state")(reference)
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_empty_subpath_state", exported, 0
    )
    return exported


def bdpt_endpoint_subpath_state(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor,
    rx_polarization: torch.Tensor,
    launch_tx_id: torch.Tensor,
    light_seed: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "tx_polarization", tx_polarization, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_polarization", rx_polarization, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("launch_tx_id", launch_tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("light_seed", light_seed, dtype=torch.int64, ndim=1)
    if tx_power.shape != (tx_positions.shape[0],):
        raise ValueError("tx_power must match tx_positions")
    if tx_polarization.shape != tx_positions.shape:
        raise ValueError("tx_polarization must match tx_positions")
    if rx_polarization.shape != rx_positions.shape:
        raise ValueError("rx_polarization must match rx_positions")
    if light_seed.shape != launch_tx_id.shape:
        raise ValueError("light_seed must match launch_tx_id")
    device = tx_positions.get_device()
    if (
        tx_power.get_device() != device
        or tx_polarization.get_device() != device
        or rx_positions.get_device() != device
        or rx_polarization.get_device() != device
        or launch_tx_id.get_device() != device
        or light_seed.get_device() != device
    ):
        raise ValueError("BDPT endpoint tensors must share one CUDA device")
    exported = _required_native_op("bdpt_endpoint_subpath_state")(
        tx_positions,
        tx_power,
        tx_polarization,
        rx_positions,
        rx_polarization,
        launch_tx_id,
        light_seed,
    )
    if not isinstance(exported, dict) or set(exported) != {"light", "sensor"}:
        raise TypeError(
            "_channel_native.bdpt_endpoint_subpath_state must return light/sensor dicts"
        )
    light = exported["light"]
    sensor = exported["sensor"]
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_endpoint_subpath_state.light",
        light,
        int(launch_tx_id.shape[0]),
    )
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_endpoint_subpath_state.sensor",
        sensor,
        int(rx_positions.shape[0]),
    )
    return {"light": light, "sensor": sensor}


def bdpt_subpath_intersection_inputs(
    subpath: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("subpath", subpath, None)
    exported = _required_native_op("bdpt_subpath_intersection_inputs")(subpath)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_subpath_intersection_inputs must return a dict"
        )
    if set(exported) != {"ray_o", "ray_d", "ray_tmax", "active"}:
        raise ValueError(
            "_channel_native.bdpt_subpath_intersection_inputs returned unexpected fields"
        )
    validate_cuda_tensor(
        "ray_o", exported["ray_o"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "ray_d", exported["ray_d"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", exported["ray_tmax"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    if (
        exported["ray_o"].shape != subpath["origin"].shape
        or exported["ray_d"].shape != subpath["direction"].shape
    ):
        raise ValueError(
            "_channel_native.bdpt_subpath_intersection_inputs returned bad ray shape"
        )
    if exported["active"].shape != subpath["valid"].shape:
        raise ValueError(
            "_channel_native.bdpt_subpath_intersection_inputs returned bad active shape"
        )
    if exported["ray_tmax"].shape != (0,):
        raise ValueError(
            "_channel_native.bdpt_subpath_intersection_inputs returned bad ray_tmax shape"
        )
    return exported


def bdpt_reflected_light_subpath_state(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    *,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_thickness: torch.Tensor,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    if not isinstance(intersection, dict) or set(intersection) != set(
        _BDPT_INTERSECTION_FIELDS
    ):
        raise ValueError("intersection returned unexpected fields")
    count = int(light["origin"].shape[0])
    validate_cuda_tensor(
        "intersection.t", intersection["t"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "intersection.p",
        intersection["p"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.n",
        intersection["n"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.global_prim_id",
        intersection["global_prim_id"],
        dtype=torch.int32,
        ndim=1,
    )
    validate_cuda_tensor("material_gain", material_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_valid", material_valid, dtype=torch.bool, ndim=1)
    validate_cuda_tensor("material_eps_r", material_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_sigma_e", material_sigma_e, dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("material_mu_r", material_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_thickness", material_thickness, dtype=torch.float32, ndim=1
    )
    if int(material_gain.shape[0]) != int(material_valid.shape[0]):
        raise ValueError("material_gain and material_valid must have matching length")
    for name, tensor in (
        ("material_eps_r", material_eps_r),
        ("material_sigma_e", material_sigma_e),
        ("material_mu_r", material_mu_r),
        ("material_thickness", material_thickness),
    ):
        if int(tensor.shape[0]) != int(material_gain.shape[0]):
            raise ValueError(f"{name} must match material_gain length")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if (
        material_gain.get_device() != light["origin"].get_device()
        or material_valid.get_device() != light["origin"].get_device()
        or material_eps_r.get_device() != light["origin"].get_device()
        or material_sigma_e.get_device() != light["origin"].get_device()
        or material_mu_r.get_device() != light["origin"].get_device()
        or material_thickness.get_device() != light["origin"].get_device()
    ):
        raise ValueError("material tensors must share light device")
    for name in ("t", "p", "n", "global_prim_id"):
        if int(intersection[name].shape[0]) != count:
            raise ValueError("intersection must match light subpath count")
        if intersection[name].get_device() != light["origin"].get_device():
            raise ValueError("intersection tensors must share light device")
    exported = _required_native_op("bdpt_reflected_light_subpath_state")(
        light,
        intersection,
        material_gain,
        material_valid,
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        material_thickness,
        float(frequency_hz),
    )
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_reflected_light_subpath_state", exported, count
    )
    return exported


def bdpt_transmitted_light_subpath_state(
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
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    if not isinstance(intersection, dict) or set(intersection) != set(
        _BDPT_INTERSECTION_FIELDS
    ):
        raise ValueError("intersection returned unexpected fields")
    count = int(light["origin"].shape[0])
    validate_cuda_tensor(
        "intersection.t", intersection["t"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "intersection.p",
        intersection["p"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.n",
        intersection["n"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.global_prim_id",
        intersection["global_prim_id"],
        dtype=torch.int32,
        ndim=1,
    )
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    device = light["origin"].get_device()
    if face_material_id.get_device() != device:
        raise ValueError("face_material_id must share light device")
    _validate_layer_csr(
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        device,
    )
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    for name in ("t", "p", "n", "global_prim_id"):
        if int(intersection[name].shape[0]) != count:
            raise ValueError("intersection must match light subpath count")
        if intersection[name].get_device() != device:
            raise ValueError("intersection tensors must share light device")
    exported = _required_native_op("bdpt_transmitted_light_subpath_state")(
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
    )
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_transmitted_light_subpath_state", exported, count
    )
    return exported


def bdpt_endpoint_connection_samples(
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    frequency_hz: float,
    samples_per_tx: int,
    max_paths: int | None = None,
    mis: str = "power_heuristic",
    beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    _validate_bdpt_subpath_state("sensor", sensor, None)
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if samples_per_tx <= 0:
        raise ValueError("samples_per_tx must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if strategy_count != 1:
        raise ValueError("endpoint connections support exactly one strategy")
    max_paths_value = -1 if max_paths is None else int(max_paths)
    if max_paths_value < -1:
        raise ValueError("max_paths must be non-negative")
    mode_id = _bdpt_mis_mode_id(mis)
    expected_total = int(light["origin"].shape[0]) * int(sensor["origin"].shape[0])
    expected_count = (
        expected_total if max_paths is None else min(int(max_paths), expected_total)
    )
    exported = _required_native_op("bdpt_endpoint_connection_samples")(
        light,
        sensor,
        float(frequency_hz),
        int(samples_per_tx),
        int(mode_id),
        float(beta),
        int(strategy_count),
        int(max_paths_value),
    )
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_endpoint_connection_samples", exported, expected_count
    )
    return exported


def bdpt_endpoint_connection_visibility_inputs(
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    _validate_bdpt_subpath_state("sensor", sensor, None)
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    expected_total = int(light["origin"].shape[0]) * int(sensor["origin"].shape[0])
    if int(sample_count) > expected_total:
        raise ValueError("sample_count exceeds endpoint pair count")
    exported = _required_native_op("bdpt_endpoint_connection_visibility_inputs")(
        light,
        sensor,
        int(sample_count),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_endpoint_connection_visibility_inputs must return a dict"
        )
    if set(exported) != {"start", "end", "active"}:
        raise ValueError(
            "_channel_native.bdpt_endpoint_connection_visibility_inputs returned unexpected fields"
        )
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "end", exported["end"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    if tuple(exported["start"].shape) != (int(sample_count), 3):
        raise ValueError(
            "_channel_native.bdpt_endpoint_connection_visibility_inputs returned bad start shape"
        )
    if exported["end"].shape != exported["start"].shape or exported["active"].shape != (
        int(sample_count),
    ):
        raise ValueError(
            "_channel_native.bdpt_endpoint_connection_visibility_inputs returned bad visibility shape"
        )
    return exported


def bdpt_accumulate_connection_samples(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    accumulation_strategy: str = "atomic",
    combine_domain: str = "power",
    coeff_real: torch.Tensor | None = None,
    coeff_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Accumulate connection samples into per-component matrices.

    ``combine_domain='power'`` (default) is the incoherent per-path power
    accumulation, bit-identical to the pre-ADR-019 behaviour; the coefficient
    tensors are ignored. ``combine_domain='coherent'`` (ADR-019, opt-in) sums
    the complex projected field coefficient (``coeff_real``/``coeff_imag``,
    row-aligned to ``samples``) into per-(tx, rx, component) phasor bins and
    finalizes ``|sum|^2``; the ``accumulation_strategy`` perf axis stays
    orthogonal (the coherent phasor sum always uses the atomic-double
    reduction).
    """

    _validate_bdpt_connection_samples("samples", samples, None)
    if tx_count < 0 or rx_count < 0:
        raise ValueError("tx_count and rx_count must be non-negative")
    strategy_ids = {"atomic": 0, "staged": 1, "compact": 2}
    if accumulation_strategy not in strategy_ids:
        raise ValueError(
            "accumulation_strategy must be 'atomic', 'staged', or 'compact'"
        )
    combine_ids = {"power": 0, "coherent": 1}
    if combine_domain not in combine_ids:
        raise ValueError("combine_domain must be 'power' or 'coherent'")
    if combine_domain == "coherent":
        if coeff_real is None or coeff_imag is None:
            raise ValueError("coherent combine requires coeff_real and coeff_imag")
        for name, tensor in (("coeff_real", coeff_real), ("coeff_imag", coeff_imag)):
            # The coefficients arrive as .real/.imag strided views of the
            # natively-computed complex path field; the one-time layout copy
            # happens at the C++ ABI boundary (mc hot-path layout-copy rule),
            # so contiguity is not required here.
            validate_cuda_tensor(
                name, tensor, dtype=torch.float32, ndim=1, require_contiguous=False
            )
            if tensor.shape != samples["contribution"].shape:
                raise ValueError(f"{name} must match connection-sample rows")
            if tensor.get_device() != samples["contribution"].get_device():
                raise ValueError(f"{name} must share the connection-sample device")
    else:
        empty = torch.empty(
            (0,), device=samples["contribution"].device, dtype=torch.float32
        )
        coeff_real = empty
        coeff_imag = empty
    exported = _required_native_op("bdpt_accumulate_connection_samples")(
        samples,
        int(tx_count),
        int(rx_count),
        int(strategy_ids[accumulation_strategy]),
        int(combine_ids[combine_domain]),
        coeff_real,
        coeff_imag,
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_accumulate_connection_samples must return a dict"
        )
    # ADR-022 spec 6.4 supervisor ruling: under combine_domain='coherent' the
    # forward additionally returns its per-component phasor bin-sum buffers
    # (``S_b``) as non-differentiable outputs so the coherent backward can read
    # them without a second atomic-double reduction. The primal component
    # matrices are unchanged bitwise; a subset check accepts the extra keys and
    # keeps the public return the six component matrices only.
    if not _BDPT_COMPONENT_MATRIX_FIELDS.issubset(exported):
        raise ValueError(
            "_channel_native.bdpt_accumulate_connection_samples returned unexpected fields"
        )
    for name in _BDPT_COMPONENT_MATRIX_ORDER:
        tensor = exported[name]
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=2)
        if tuple(tensor.shape) != (int(tx_count), int(rx_count)):
            raise ValueError(
                f"_channel_native.bdpt_accumulate_connection_samples returned bad {name} shape"
            )
    return {name: exported[name] for name in _BDPT_COMPONENT_MATRIX_ORDER}


# ---------------------------------------------------------------------------
# ADR-022 BDPT fixed-topology AD companion facades (plan 10a section 6). These
# validate the differentiable-input contracts, request the registered native
# backward/jvp symbol through ``runtime``, and assert the returned dict. They
# never reconstruct the RF physics in Torch: the numerical work runs entirely
# in the native companion. The autograd.Function wrappers in
# ``montecarlo/bdpt/kernels/autograd.py`` own the tape, the frozen-input
# rejection and the need-flag derivation; these facades are pure dispatch.
# ---------------------------------------------------------------------------

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
            "_channel_native.bdpt_reflected_light_subpath_state_backward "
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
            "_channel_native.bdpt_reflected_light_subpath_state_jvp "
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
            "_channel_native.bdpt_transmitted_light_subpath_state_backward "
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
            "_channel_native.bdpt_transmitted_light_subpath_state_jvp "
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
            "_channel_native.bdpt_endpoint_connection_samples_backward "
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
            "_channel_native.bdpt_endpoint_connection_samples_jvp "
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
            "_channel_native.bdpt_accumulate_connection_samples returned unexpected fields"
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
            "_channel_native.bdpt_accumulate_connection_samples_backward "
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
            "_channel_native.bdpt_accumulate_connection_samples_jvp "
            "returned unexpected fields"
        )
    return exported


def bdpt_filter_connection_samples(
    samples: dict[str, torch.Tensor],
    visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_connection_samples("samples", samples, None)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if visible.shape != samples["valid"].shape:
        raise ValueError("visible must match samples")
    if visible.get_device() != samples["valid"].get_device():
        raise ValueError("visible must share samples device")
    exported = _required_native_op("bdpt_filter_connection_samples")(samples, visible)
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_filter_connection_samples", exported, None
    )
    return exported


def bdpt_count_valid_connection_samples(samples: dict[str, torch.Tensor]) -> int:
    _validate_bdpt_connection_samples("samples", samples, None)
    count = _required_native_op("bdpt_count_valid_connection_samples")(samples)
    if not isinstance(count, int):
        raise TypeError(
            "_channel_native.bdpt_count_valid_connection_samples must return an int"
        )
    if count < 0 or count > int(samples["valid"].shape[0]):
        raise ValueError(
            "_channel_native.bdpt_count_valid_connection_samples returned bad count"
        )
    return count


def bdpt_compact_connection_samples(
    samples: dict[str, torch.Tensor],
    *,
    max_paths: int | None = None,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_connection_samples("samples", samples, None)
    max_paths_value = -1 if max_paths is None else int(max_paths)
    if max_paths_value < -1:
        raise ValueError("max_paths must be non-negative")
    exported = _required_native_op("bdpt_compact_connection_samples")(
        samples, int(max_paths_value)
    )
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_compact_connection_samples", exported, None
    )
    if max_paths is not None and int(exported["valid"].shape[0]) > int(max_paths):
        raise ValueError(
            "_channel_native.bdpt_compact_connection_samples exceeded max_paths"
        )
    return exported


def bdpt_concat_connection_samples(
    samples: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("samples must not be empty")
    expected_count = 0
    for index, block in enumerate(samples):
        _validate_bdpt_connection_samples(f"samples[{index}]", block, None)
        expected_count += int(block["valid"].shape[0])
    exported = _required_native_op("bdpt_concat_connection_samples")(tuple(samples))
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_concat_connection_samples", exported, expected_count
    )
    return exported


def bdpt_connection_variance(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    samples_per_tx: int,
) -> torch.Tensor:
    _validate_bdpt_connection_samples("samples", samples, None)
    if tx_count < 0 or rx_count < 0:
        raise ValueError("tx_count and rx_count must be non-negative")
    if samples_per_tx <= 0:
        raise ValueError("samples_per_tx must be positive")
    variance = _required_native_op("bdpt_connection_variance")(
        samples,
        int(tx_count),
        int(rx_count),
        int(samples_per_tx),
    )
    if not isinstance(variance, torch.Tensor):
        raise TypeError("_channel_native.bdpt_connection_variance must return a tensor")
    validate_cuda_tensor("variance", variance, dtype=torch.float32, ndim=2)
    if tuple(variance.shape) != (int(tx_count), int(rx_count)):
        raise ValueError("_channel_native.bdpt_connection_variance returned bad shape")
    return variance


def bdpt_mis_weights(
    pdf: torch.Tensor,
    strategy_pdf_sum: torch.Tensor,
    *,
    mis: str,
    beta: float = 2.0,
) -> torch.Tensor:
    validate_cuda_tensor("pdf", pdf, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "strategy_pdf_sum", strategy_pdf_sum, dtype=torch.float32, ndim=0
    )
    mode_id = _bdpt_mis_mode_id(mis)
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "bdpt_mis_weights"):
        raise RuntimeError("_channel_native.bdpt_mis_weights CUDA kernel is required")
    weights = native.bdpt_mis_weights(pdf, strategy_pdf_sum, int(mode_id), float(beta))
    if not isinstance(weights, torch.Tensor):
        raise TypeError("_channel_native.bdpt_mis_weights must return a tensor")
    validate_cuda_tensor("weights", weights, dtype=torch.float32, ndim=1)
    if weights.shape != pdf.shape:
        raise ValueError(
            "_channel_native.bdpt_mis_weights returned an unexpected shape"
        )
    return weights


def bdpt_diffraction_connection_samples_from_tape(
    tape: dict[str, torch.Tensor],
    states: tuple[torch.Tensor, ...],
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    tx_index: int,
    state_count: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    grid_cell_area: float,
    wavelength: float,
    direct_samples: int,
    keller_samples: int,
    mis: str = "power_heuristic",
    beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor]:
    expected_tape = {
        "active": torch.bool,
        "state_idx": torch.int32,
        "cell": torch.int32,
        "material_idx": torch.int32,
        "edge_u": torch.float32,
    }
    if set(tape) != set(expected_tape):
        raise ValueError("diffraction tape returned unexpected fields")
    inferred_count: int | None = None
    for field, dtype in expected_tape.items():
        tensor = tape[field]
        validate_cuda_tensor(f"tape.{field}", tensor, dtype=dtype, ndim=1)
        if inferred_count is None:
            inferred_count = int(tensor.shape[0])
        if int(tensor.shape[0]) != inferred_count:
            raise ValueError("diffraction tape fields must share shape")
    if len(states) != 12:
        raise ValueError("states must contain 12 tensors")
    validate_cuda_tensor("material_gain", material_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_valid", material_valid, dtype=torch.bool, ndim=1)
    if material_gain.shape != material_valid.shape:
        raise ValueError("material_gain and material_valid must have matching shape")
    if state_count < 0:
        raise ValueError("state_count must be non-negative")
    if direct_samples < 0 or keller_samples < 0:
        raise ValueError("sample counts must be non-negative")
    if strategy_count <= 0:
        raise ValueError("strategy_count must be positive")
    actual_strategy_count = int(direct_samples > 0) + int(keller_samples > 0)
    if strategy_count != actual_strategy_count:
        raise ValueError("strategy_count must match enabled direct/Keller proposals")
    if mis == "none" and actual_strategy_count != 1:
        raise ValueError("mis='none' requires exactly one diffraction proposal")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    mode_id = _bdpt_mis_mode_id(mis)
    exported = _required_native_op("bdpt_diffraction_connection_samples_from_tape")(
        tape,
        tuple(states),
        material_gain,
        material_valid,
        int(tx_index),
        int(state_count),
        int(grid_axis),
        float(grid_position),
        float(grid_coord0_min),
        float(grid_coord0_max),
        float(grid_coord1_min),
        float(grid_coord1_max),
        int(grid_resolution0),
        int(grid_resolution1),
        float(grid_cell_area),
        float(wavelength),
        int(direct_samples),
        int(keller_samples),
        int(mode_id),
        float(beta),
        int(strategy_count),
    )
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_diffraction_connection_samples_from_tape",
        exported,
        inferred_count,
    )
    return exported


def bdpt_diffraction_point_connection_samples(
    rx_positions: torch.Tensor,
    states: tuple[torch.Tensor, ...],
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    tx_index: int,
    state_count: int,
    direct_samples: int,
    keller_samples: int,
    seed: int,
    wavelength: float,
    mis: str = "power_heuristic",
    beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if len(states) != 12:
        raise ValueError("states must contain 12 tensors")
    validate_cuda_tensor("material_gain", material_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_valid", material_valid, dtype=torch.bool, ndim=1)
    if material_gain.shape != material_valid.shape:
        raise ValueError("material_gain and material_valid must have matching shape")
    if state_count < 0:
        raise ValueError("state_count must be non-negative")
    if direct_samples < 0 or keller_samples < 0:
        raise ValueError("sample counts must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    if strategy_count <= 0:
        raise ValueError("strategy_count must be positive")
    actual_strategy_count = int(direct_samples > 0) + int(keller_samples > 0)
    if strategy_count != actual_strategy_count:
        raise ValueError("strategy_count must match enabled direct/Keller proposals")
    if mis == "none" and actual_strategy_count != 1:
        raise ValueError("mis='none' requires exactly one diffraction proposal")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    mode_id = _bdpt_mis_mode_id(mis)
    exported = _required_native_op("bdpt_diffraction_point_connection_samples")(
        rx_positions,
        tuple(states),
        material_gain,
        material_valid,
        int(tx_index),
        int(state_count),
        int(direct_samples),
        int(keller_samples),
        int(seed),
        float(wavelength),
        int(mode_id),
        float(beta),
        int(strategy_count),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_diffraction_point_connection_samples must return a dict"
        )
    expected = {
        "samples",
        "source_start",
        "source_end",
        "target_start",
        "target_end",
        "visibility_active",
    }
    if set(exported) != expected:
        raise ValueError(
            "_channel_native.bdpt_diffraction_point_connection_samples returned unexpected fields"
        )
    sample_count = int(rx_positions.shape[0]) * (
        int(direct_samples) + int(keller_samples)
    )
    if state_count == 0:
        sample_count = 0
    samples = exported["samples"]
    if not isinstance(samples, dict):
        raise TypeError(
            "_channel_native.bdpt_diffraction_point_connection_samples samples must be a dict"
        )
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_diffraction_point_connection_samples.samples",
        samples,
        sample_count,
    )
    for name in ("source_start", "source_end", "target_start", "target_end"):
        tensor = exported[name]
        validate_cuda_tensor(
            name, tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if tuple(tensor.shape) != (sample_count, 3):
            raise ValueError(
                f"_channel_native.bdpt_diffraction_point_connection_samples returned bad {name} shape"
            )
    active = exported["visibility_active"]
    validate_cuda_tensor("visibility_active", active, dtype=torch.bool, ndim=1)
    if tuple(active.shape) != (sample_count,):
        raise ValueError(
            "_channel_native.bdpt_diffraction_point_connection_samples returned bad visibility_active shape"
        )
    return exported

