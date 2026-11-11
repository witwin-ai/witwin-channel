# Copyright Xingyu Chen.
# Tests bdpt companion ad.

"""Tests bdpt companion ad."""

from __future__ import annotations

import pytest
import torch

from tests.ad._fd import relative_error
from tests.ad._tolerances import ABS_TOL
from tests.reference import bdpt_ad_oracles as O
from witwin.channel.kernels import montecarlo as M
from witwin.channel.montecarlo import bdpt as PA
from witwin.channel import runtime

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for BDPT companion AD"
)

_FREQ = 3.0e9
_REL_TOL_LOCK = 1.0e-2
_REL_TOL_FWD = 5.0e-3
_REL_TOL_FD = 5.0e-2

_BDPT_COMPANION_SYMBOLS = (
    "bdpt_endpoint_connection_samples_backward",
    "bdpt_endpoint_connection_samples_jvp",
    "bdpt_accumulate_connection_samples_backward",
    "bdpt_accumulate_connection_samples_jvp",
    "bdpt_finalize_point_components_backward",
    "bdpt_finalize_point_components_jvp",
    "bdpt_finalize_component_maps_backward",
    "bdpt_finalize_component_maps_jvp",
)


# ---------------------------------------------------------------------------
# Shared builders.
# ---------------------------------------------------------------------------


def _unit(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(v, dim=-1)


def _f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).contiguous()


def _subpath_state(
    origin: torch.Tensor, direction: torch.Tensor, field: torch.Tensor, throughput: torch.Tensor, *,
    component_mask: int = 3, depth: int = 1, tx_id: int = 0, rx_id: int = -1,
    source_power: torch.Tensor | None = None, path_length: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    n = origin.shape[0]
    dev = origin.device
    src_power = (
        torch.ones(n, device=dev, dtype=torch.float32)
        if source_power is None
        else _f32(source_power)
    )
    plen = (
        torch.zeros(n, device=dev, dtype=torch.float32)
        if path_length is None
        else _f32(path_length)
    )
    return {
        "origin": _f32(origin),
        "direction": _f32(direction),
        "throughput_real": _f32(throughput.real),
        "throughput_imag": _f32(throughput.imag),
        "pdf_forward": torch.full((n,), 0.25, device=dev, dtype=torch.float32),
        "pdf_reverse": torch.full((n,), 0.25, device=dev, dtype=torch.float32),
        "depth": torch.full((n,), depth, device=dev, dtype=torch.int32),
        "component_mask": torch.full((n,), component_mask, device=dev, dtype=torch.int32),
        "primitive_id": torch.full((n,), -1, device=dev, dtype=torch.int32),
        "edge_id": torch.full((n,), -1, device=dev, dtype=torch.int32),
        "tx_id": torch.full((n,), tx_id, device=dev, dtype=torch.int32),
        "rx_id": torch.full((n,), rx_id, device=dev, dtype=torch.int32),
        "grid_linear_id": torch.full((n,), -1, device=dev, dtype=torch.int32),
        "valid": torch.ones(n, device=dev, dtype=torch.bool),
        "path_length": plen,
        "field_real": _f32(field.real),
        "field_imag": _f32(field.imag),
        "source_power": src_power,
        "event_type": torch.zeros(n, device=dev, dtype=torch.int32),
    }


# ---------------------------------------------------------------------------
# 6.3 endpoint (LoS / NEE) connection contribution.
# ---------------------------------------------------------------------------


def _endpoint_fixture(seed: int, tx: int = 3, rx: int = 4):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)

    def randn(*s):
        return torch.randn(*s, generator=g, device=dev, dtype=torch.float64)

    def rand(*s):
        return torch.rand(*s, generator=g, device=dev, dtype=torch.float64)

    light_origin = randn(tx, 3)
    sensor_origin = randn(rx, 3) + torch.tensor([4.0, 0.0, 0.0], device=dev)
    light_field = randn(tx, 3) + 1.0j * randn(tx, 3)
    source_power = 0.5 + rand(tx)
    light_path_length = rand(tx)
    rx_pol = _unit(randn(rx, 3))
    return {
        "tx": tx, "rx": rx, "device": dev,
        "light_origin": light_origin, "sensor_origin": sensor_origin,
        "light_field": light_field, "source_power": source_power,
        "light_path_length": light_path_length, "rx_pol": rx_pol,
    }


def _endpoint_native(fx, samples_per_tx=4):
    dev = fx["device"]
    txn, rxn = fx["tx"], fx["rx"]
    # Sensor direction is one -z unit vector per sensor row (rxn), not txn.
    ez = torch.zeros(rxn, 3, device=dev, dtype=torch.float64)
    ez[:, 2] = -1.0
    light = _subpath_state(
        fx["light_origin"], _unit(torch.randn_like(fx["light_origin"])),
        fx["light_field"], torch.ones(txn, device=dev, dtype=torch.complex128),
        depth=0, tx_id=0,
        source_power=fx["source_power"], path_length=fx["light_path_length"],
    )
    # Distinct tx ids per light row and rx ids per sensor row (frozen topology).
    light["tx_id"] = torch.arange(txn, device=dev, dtype=torch.int32)
    sensor_field = torch.complex(
        fx["rx_pol"], torch.zeros_like(fx["rx_pol"])
    )
    sensor = _subpath_state(
        fx["sensor_origin"], ez, sensor_field,
        torch.ones(rxn, device=fx["device"], dtype=torch.complex128),
        depth=0, tx_id=-1, rx_id=0,
    )
    sensor["rx_id"] = torch.arange(rxn, device=fx["device"], dtype=torch.int32)
    sensor["grid_linear_id"] = torch.arange(rxn, device=fx["device"], dtype=torch.int32)
    return light, sensor


def _endpoint_pairwise_reference(fx, samples_per_tx, frequency):
    """Row-major (light, sensor) contribution stack matching the native kernel order."""

    txn, rxn = fx["tx"], fx["rx"]
    rows = []
    for li in range(txn):
        for si in range(rxn):
            rows.append(
                O.endpoint_connection_contribution_reference(
                    fx["light_field"][li : li + 1],
                    fx["source_power"][li : li + 1],
                    fx["light_origin"][li : li + 1],
                    fx["sensor_origin"][si : si + 1],
                    fx["rx_pol"][si : si + 1],
                    fx["light_path_length"][li : li + 1],
                    frequency,
                    samples_per_tx,
                )
            )
    return torch.cat(rows, dim=0)


def test_endpoint_forward_matches_oracle():
    fx = _endpoint_fixture(37)
    light, sensor = _endpoint_native(fx)
    spt = 4
    out = M.bdpt_endpoint_connection_samples(
        light, sensor, frequency_hz=_FREQ, samples_per_tx=spt, mis="none",
    )
    freq = torch.tensor(_FREQ, dtype=torch.float64, device=fx["device"])
    ref = _endpoint_pairwise_reference(fx, spt, freq)
    valid = out["valid"]
    assert relative_error(
        out["contribution"][valid], ref[valid], abs_floor=ABS_TOL
    ) <= _REL_TOL_FWD


def test_endpoint_backward_matches_oracle():
    fx = _endpoint_fixture(43)
    light, sensor = _endpoint_native(fx)
    spt = 4
    g = torch.Generator(device="cuda").manual_seed(94)
    grad_contribution = torch.randn(
        fx["tx"] * fx["rx"], generator=g, device="cuda", dtype=torch.float32
    )
    native = PA.bdpt_endpoint_connection_samples_backward(
        light, sensor, frequency_hz=_FREQ, samples_per_tx=spt, mis="none",
        beta=2.0, strategy_count=1, max_paths=None,
        grad_contribution=grad_contribution,
        need_grad_field=True, need_grad_frequency=True, need_grad_tx_power=True,
    )

    dev = fx["device"]
    field_real = fx["light_field"].real.clone().requires_grad_(True)
    field_imag = fx["light_field"].imag.clone().requires_grad_(True)
    source_power = fx["source_power"].clone().requires_grad_(True)
    frequency = torch.tensor(_FREQ, dtype=torch.float64, device=dev, requires_grad=True)
    field = torch.complex(field_real, field_imag)
    fx_leaf = dict(fx)
    fx_leaf["light_field"] = field
    fx_leaf["source_power"] = source_power
    ref = _endpoint_pairwise_reference(fx_leaf, spt, frequency)
    (grad_contribution.double() * ref).sum().backward()

    # Sensor fields (rx polarization) are frozen; only light field carries grad.
    assert relative_error(
        native["grad_light_field_real"], field_real.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    assert relative_error(
        native["grad_light_field_imag"], field_imag.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    # tx_power grad is scattered per tx (many rows share a tx).
    assert relative_error(
        native["grad_tx_power"], source_power.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    assert relative_error(
        native["grad_frequency"].reshape(()), frequency.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK


def test_endpoint_jvp_vjp_duality():
    fx = _endpoint_fixture(47)
    light, sensor = _endpoint_native(fx)
    spt = 4
    g = torch.Generator(device="cuda").manual_seed(95)

    def rc(*s):
        return torch.randn(*s, generator=g, device="cuda", dtype=torch.float32)

    tangents = {
        "tangent_light_field_real": rc(fx["tx"], 3),
        "tangent_light_field_imag": rc(fx["tx"], 3),
        "tangent_tx_power": rc(fx["tx"]),
    }
    jvp = PA.bdpt_endpoint_connection_samples_jvp(
        light, sensor, frequency_hz=_FREQ, samples_per_tx=spt, mis="none",
        beta=2.0, strategy_count=1, max_paths=None,
        tangent_frequency=0.0, **tangents,
    )
    cot = rc(*jvp["tangent_contribution"].shape)
    lhs = (cot.double() * jvp["tangent_contribution"].double()).sum()

    vjp = PA.bdpt_endpoint_connection_samples_backward(
        light, sensor, frequency_hz=_FREQ, samples_per_tx=spt, mis="none",
        beta=2.0, strategy_count=1, max_paths=None,
        grad_contribution=cot,
        need_grad_field=True, need_grad_frequency=False, need_grad_tx_power=True,
    )
    rhs = (
        (vjp["grad_light_field_real"].double() * tangents["tangent_light_field_real"].double()).sum()
        + (vjp["grad_light_field_imag"].double() * tangents["tangent_light_field_imag"].double()).sum()
        + (vjp["grad_tx_power"].double() * tangents["tangent_tx_power"].double()).sum()
    )
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_LOCK


def test_endpoint_backward_need_flag_gating():
    fx = _endpoint_fixture(59)
    light, sensor = _endpoint_native(fx)
    grad_contribution = torch.zeros(fx["tx"] * fx["rx"], device="cuda")
    out = PA.bdpt_endpoint_connection_samples_backward(
        light, sensor, frequency_hz=_FREQ, samples_per_tx=4, mis="none",
        beta=2.0, strategy_count=1, max_paths=None,
        grad_contribution=grad_contribution,
        need_grad_field=True, need_grad_frequency=False, need_grad_tx_power=False,
    )
    assert out["grad_light_field_real"] is not None
    assert out["grad_frequency"] is None
    assert out["grad_tx_power"] is None


# ---------------------------------------------------------------------------
# 6.4 accumulate connection samples (power + coherent).
# ---------------------------------------------------------------------------


def _accum_fixture(seed: int, rows: int = 24, tx: int = 3, rx: int = 4):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)

    def randn(*s):
        return torch.randn(*s, generator=g, device=dev, dtype=torch.float32)

    def randint(high, *s):
        return torch.randint(0, high, s, generator=g, device=dev, dtype=torch.int32)

    contribution = randn(rows).abs()
    mis = 0.5 + torch.rand(rows, generator=g, device=dev)
    coeff_real = randn(rows)
    coeff_imag = randn(rows)
    tx_id = randint(tx, rows)
    rx_id = randint(rx, rows)
    # Sample from the native component-id encoding {0,1,2,5,6} (los, reflection,
    # diffraction, transmission, scattering); ids 3 and 4 are unused, so a
    # dense randint(5) would fabricate non-existent components the accumulate
    # kernels drop to zero.
    _component_ids = torch.tensor(
        [O.COMPONENT_LOS, O.COMPONENT_REFLECTION, O.COMPONENT_DIFFRACTION,
         O.COMPONENT_TRANSMISSION, O.COMPONENT_SCATTERING],
        device=dev, dtype=torch.int32,
    )
    component_id = _component_ids[randint(_component_ids.numel(), rows).long()]
    valid = torch.rand(rows, generator=g, device=dev) > 0.15
    return {
        "rows": rows, "tx": tx, "rx": rx, "device": dev,
        "contribution": contribution, "mis": mis,
        "coeff_real": coeff_real, "coeff_imag": coeff_imag,
        "tx_id": tx_id, "rx_id": rx_id, "component_id": component_id, "valid": valid,
    }


def _accum_samples(fx):
    rows = fx["rows"]
    dev = fx["device"]
    return {
        "topology": torch.stack(
            [fx["tx_id"], fx["rx_id"], fx["component_id"],
             torch.zeros(rows, device=dev, dtype=torch.int32)], dim=1
        ).contiguous(),
        "contribution": fx["contribution"],
        "pdf": torch.ones(rows, device=dev, dtype=torch.float32),
        "mis_weight": fx["mis"].to(torch.float32),
        "component_id": fx["component_id"],
        "valid": fx["valid"],
        "tx_id": fx["tx_id"],
        "rx_id": fx["rx_id"],
        "grid_linear_id": fx["rx_id"],
        "light_depth": torch.ones(rows, device=dev, dtype=torch.int32),
        "sensor_depth": torch.zeros(rows, device=dev, dtype=torch.int32),
        "path_length_m": torch.ones(rows, device=dev, dtype=torch.float32),
    }


def test_accumulate_power_backward_matches_oracle():
    fx = _accum_fixture(71)
    samples = _accum_samples(fx)
    g = torch.Generator(device="cuda").manual_seed(96)
    cot = {
        name: torch.randn(fx["tx"], fx["rx"], generator=g, device="cuda")
        for name in ("grad_path_gain", "grad_los", "grad_reflection",
                     "grad_diffraction", "grad_transmission", "grad_scattering")
    }
    native = PA.bdpt_accumulate_connection_samples_backward(
        samples, tx_count=fx["tx"], rx_count=fx["rx"], combine_domain="power",
        **cot, need_grad_contribution=True, need_grad_coeff=False,
    )

    contribution = fx["contribution"].double().clone().requires_grad_(True)
    ref = O.accumulate_power_reference(
        contribution, fx["mis"], fx["tx_id"], fx["rx_id"], fx["component_id"],
        fx["valid"], fx["tx"], fx["rx"],
    )
    loss = sum(
        (cot[f"grad_{name}"].double() * ref[name]).sum()
        for name in ("path_gain", "los", "reflection", "diffraction",
                     "transmission", "scattering")
    )
    loss.backward()
    assert relative_error(
        native["grad_contribution"], contribution.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    assert native["grad_coeff_real"] is None and native["grad_coeff_imag"] is None


def test_accumulate_coherent_backward_matches_oracle():
    fx = _accum_fixture(73)
    samples = _accum_samples(fx)
    g = torch.Generator(device="cuda").manual_seed(97)
    cot = {
        name: torch.randn(fx["tx"], fx["rx"], generator=g, device="cuda")
        for name in ("grad_path_gain", "grad_los", "grad_reflection",
                     "grad_diffraction", "grad_transmission", "grad_scattering")
    }
    # BDPT AD: the coherent forward retains the ten phasor bin sums
    # S_b; the backward reads them (no in-backward re-reduction, no sample
    # coefficients) to form grad_c_r = 2 grad_P[b] S_b.
    _matrices, bin_sums = PA.bdpt_accumulate_connection_samples_forward_ad(
        samples, tx_count=fx["tx"], rx_count=fx["rx"],
        accumulation_strategy="atomic", combine_domain="coherent",
        coeff_real=fx["coeff_real"], coeff_imag=fx["coeff_imag"],
    )
    native = PA.bdpt_accumulate_connection_samples_backward(
        samples, tx_count=fx["tx"], rx_count=fx["rx"], combine_domain="coherent",
        bin_sums=bin_sums,
        **cot, need_grad_contribution=False, need_grad_coeff=True,
    )

    coeff_real = fx["coeff_real"].double().clone().requires_grad_(True)
    coeff_imag = fx["coeff_imag"].double().clone().requires_grad_(True)
    ref = O.accumulate_coherent_reference(
        coeff_real, coeff_imag, fx["tx_id"], fx["rx_id"], fx["component_id"],
        fx["valid"], fx["tx"], fx["rx"],
    )
    loss = sum(
        (cot[f"grad_{name}"].double() * ref[name]).sum()
        for name in ("path_gain", "los", "reflection", "diffraction",
                     "transmission", "scattering")
    )
    loss.backward()
    assert relative_error(
        native["grad_coeff_real"], coeff_real.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    assert relative_error(
        native["grad_coeff_imag"], coeff_imag.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    assert native["grad_contribution"] is None


def test_accumulate_power_jvp_vjp_duality():
    fx = _accum_fixture(75)
    samples = _accum_samples(fx)
    g = torch.Generator(device="cuda").manual_seed(98)
    tangent_contribution = torch.randn(fx["rows"], generator=g, device="cuda")
    jvp = PA.bdpt_accumulate_connection_samples_jvp(
        samples, tx_count=fx["tx"], rx_count=fx["rx"], combine_domain="power",
        tangent_contribution=tangent_contribution,
    )
    cot = {
        name: torch.randn(fx["tx"], fx["rx"], generator=g, device="cuda")
        for name in ("tangent_path_gain", "tangent_los", "tangent_reflection",
                     "tangent_diffraction", "tangent_transmission", "tangent_scattering")
    }
    lhs = sum((cot[k].double() * jvp[k].double()).sum() for k in cot)

    vjp = PA.bdpt_accumulate_connection_samples_backward(
        samples, tx_count=fx["tx"], rx_count=fx["rx"], combine_domain="power",
        grad_path_gain=cot["tangent_path_gain"], grad_los=cot["tangent_los"],
        grad_reflection=cot["tangent_reflection"], grad_diffraction=cot["tangent_diffraction"],
        grad_transmission=cot["tangent_transmission"], grad_scattering=cot["tangent_scattering"],
        need_grad_contribution=True, need_grad_coeff=False,
    )
    rhs = (vjp["grad_contribution"].double() * tangent_contribution.double()).sum()
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_LOCK


def test_accumulate_coherent_jvp_vjp_duality():
    fx = _accum_fixture(77)
    samples = _accum_samples(fx)
    g = torch.Generator(device="cuda").manual_seed(99)
    t_cr = torch.randn(fx["rows"], generator=g, device="cuda")
    t_ci = torch.randn(fx["rows"], generator=g, device="cuda")
    # Retain the coherent forward bin sums S_b; both the JVP and VJP read them
    # (BDPT AD), so neither companion re-derives them from coeff.
    _matrices, bin_sums = PA.bdpt_accumulate_connection_samples_forward_ad(
        samples, tx_count=fx["tx"], rx_count=fx["rx"],
        accumulation_strategy="atomic", combine_domain="coherent",
        coeff_real=fx["coeff_real"], coeff_imag=fx["coeff_imag"],
    )
    jvp = PA.bdpt_accumulate_connection_samples_jvp(
        samples, tx_count=fx["tx"], rx_count=fx["rx"], combine_domain="coherent",
        bin_sums=bin_sums,
        tangent_coeff_real=t_cr, tangent_coeff_imag=t_ci,
    )
    cot = {
        name: torch.randn(fx["tx"], fx["rx"], generator=g, device="cuda")
        for name in ("tangent_path_gain", "tangent_los", "tangent_reflection",
                     "tangent_diffraction", "tangent_transmission", "tangent_scattering")
    }
    lhs = sum((cot[k].double() * jvp[k].double()).sum() for k in cot)

    vjp = PA.bdpt_accumulate_connection_samples_backward(
        samples, tx_count=fx["tx"], rx_count=fx["rx"], combine_domain="coherent",
        bin_sums=bin_sums,
        grad_path_gain=cot["tangent_path_gain"], grad_los=cot["tangent_los"],
        grad_reflection=cot["tangent_reflection"], grad_diffraction=cot["tangent_diffraction"],
        grad_transmission=cot["tangent_transmission"], grad_scattering=cot["tangent_scattering"],
        need_grad_contribution=False, need_grad_coeff=True,
    )
    rhs = (
        (vjp["grad_coeff_real"].double() * t_cr.double()).sum()
        + (vjp["grad_coeff_imag"].double() * t_ci.double()).sum()
    )
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_LOCK


# ---------------------------------------------------------------------------
# 6.5 / 6.6 finalize point components / component maps.
# ---------------------------------------------------------------------------


def _finalize_fixture(seed: int, shape):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    return [
        torch.randn(*shape, generator=g, device=dev, dtype=torch.float32).abs()
        for _ in range(5)
    ]


_COMPONENT_ORDER = ("los", "reflection", "diffraction", "transmission", "scattering")


def _finalize_backward_lockstep(backward_fn, forward_fn, shape, seed):
    components = _finalize_fixture(seed, shape)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    grad_path_gain = torch.randn(*shape, generator=g, device="cuda")
    grad_powers = {
        f"grad_{name}_power": torch.randn((), generator=g, device="cuda")
        for name in _COMPONENT_ORDER
    }
    native = backward_fn(
        *components, grad_path_gain=grad_path_gain, **grad_powers,
        need_grad_components=True,
    )

    leaves = [c.double().clone().requires_grad_(True) for c in components]
    ref = forward_fn(*leaves)
    loss = (grad_path_gain.double() * ref["path_gain"]).sum()
    for name in _COMPONENT_ORDER:
        loss = loss + grad_powers[f"grad_{name}_power"].double() * ref[f"{name}_power"]
    loss.backward()
    for i, name in enumerate(_COMPONENT_ORDER):
        assert relative_error(
            native[f"grad_{name}"], leaves[i].grad, abs_floor=ABS_TOL
        ) <= _REL_TOL_LOCK, name


def test_finalize_point_components_backward_matches_oracle():
    _finalize_backward_lockstep(
        M.bdpt_finalize_point_components_backward,
        O.finalize_point_components_reference,
        (3, 4), 81,
    )


def test_finalize_component_maps_backward_matches_oracle():
    _finalize_backward_lockstep(
        M.bdpt_finalize_component_maps_backward,
        O.finalize_component_maps_reference,
        (2, 5, 6), 83,
    )


def test_finalize_point_components_jvp_vjp_duality():
    shape = (3, 4)
    components = _finalize_fixture(85, shape)
    g = torch.Generator(device="cuda").manual_seed(86)
    tangents = {
        f"tangent_{name}": torch.randn(*shape, generator=g, device="cuda")
        for name in _COMPONENT_ORDER
    }
    jvp = M.bdpt_finalize_point_components_jvp(*components, **tangents)
    cot_path = torch.randn(*shape, generator=g, device="cuda")
    cot_powers = {
        f"grad_{name}_power": torch.randn((), generator=g, device="cuda")
        for name in _COMPONENT_ORDER
    }
    lhs = (cot_path.double() * jvp["tangent_path_gain"].double()).sum()
    for name in _COMPONENT_ORDER:
        lhs = lhs + cot_powers[f"grad_{name}_power"].double() * jvp[f"tangent_{name}_power"].double()

    vjp = M.bdpt_finalize_point_components_backward(
        *components, grad_path_gain=cot_path, **cot_powers, need_grad_components=True,
    )
    rhs = sum(
        (vjp[f"grad_{name}"].double() * tangents[f"tangent_{name}"].double()).sum()
        for name in _COMPONENT_ORDER
    )
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_LOCK


# ---------------------------------------------------------------------------
# Fixed-input loud rejection + missing-symbol loud failure.
# ---------------------------------------------------------------------------


def test_companion_symbols_are_registered():
    for name in _BDPT_COMPANION_SYMBOLS:
        assert runtime.has_symbol(name), name


def test_missing_symbol_fails_loudly():
    with pytest.raises(runtime.NativeSymbolError):
        runtime.required_symbol("bdpt_endpoint_connection_samples_backward_absent")