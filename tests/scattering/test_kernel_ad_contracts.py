# Copyright Xingyu Chen.
# Tests kernel ad contracts.

"""Tests kernel ad contracts."""

from __future__ import annotations

import math

import pytest
import torch

from tests.ad._fd import relative_error
from tests.ad._tolerances import ABS_TOL, REL_TOL_PATH
from tests.reference import kirchhoff_ensemble as ref_ensemble
from tests.reference import phase_screen_realization as ref_patch
from witwin.channel import runtime
from witwin.channel.kernels import scattering as scattering_autograd
from witwin.channel.kernels import scattering as scattering_functional

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for scattering AD"
)

# Float32 native kernels (with run-to-run-nondeterministic atomic accumulation
# for the shared per-sample/table/scalar buffers, scattering AD) versus the float64
# Torch oracle. Row/geometry stores are direct; the accumulated groups get a
# looser floor.
_REL_TOL_DIRECT = REL_TOL_PATH          # 5e-3: per-row direct stores
_REL_TOL_ACCUM = 1.0e-2                 # atomicAdd-accumulated groups
# Central FD of the float32 native forward is noisier than the analytic
# locksteps; give the JVP-vs-FD cross-check the general solver-FD tolerance.
_REL_TOL_FD_FORWARD = 5.0e-2
# Finite-difference directional step for the native-forward JVP cross-check.
_FD_STEP = 5.0e-4


# ---------------------------------------------------------------------------
# Op 1 (Kirchhoff ensemble) fixtures.
# ---------------------------------------------------------------------------


def _ensemble_case(
    *,
    device: str = "cuda",
    seed: int,
    rows: int = 24,
    samples: int = 10,
    num_rx: int = 4,
    nti: int = 8,
    nto: int = 8,
    npo: int = 8,
    coef: float = 0.35,
):
    """Small single-isotropic-material ensemble case (all rows above horizon)."""

    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=torch.float32)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=device, dtype=torch.float32)

    n_o = torch.nn.functional.normalize(randn(samples, 3), dim=-1)
    t1r = torch.nn.functional.normalize(torch.cross(n_o, randn(samples, 3), dim=-1), dim=-1)
    t2r = torch.cross(n_o, t1r, dim=-1)
    backup_axis = t1r.clone()
    # Local incident direction with a strictly positive normal component so the
    # table stays above the horizon; xy carry the incident azimuth.
    wi_local = torch.stack(
        (randn(samples), randn(samples), 0.3 + 0.6 * rand(samples)), dim=-1
    )
    cos_i = 0.2 + 0.7 * rand(samples)
    r1 = 0.5 + rand(samples)
    a_te2 = 0.1 + rand(samples)
    a_tm2 = 0.1 + rand(samples)
    weights = 0.1 + rand(samples)
    material_id = torch.zeros(samples, dtype=torch.int32, device=device)

    sc_idx = torch.randint(0, samples, (rows,), generator=generator, device=device, dtype=torch.int64)
    rc_idx = torch.randint(0, num_rx, (rows,), generator=generator, device=device, dtype=torch.int64)
    wo_rows = torch.nn.functional.normalize(randn(rows, 3), dim=-1)
    cos_o_rows = 0.2 + 0.7 * rand(rows)
    r2_rows = 0.5 + rand(rows)
    rx_pol = torch.nn.functional.normalize(randn(num_rx, 3), dim=-1)

    # Dense isotropic tables (strictly positive -> gain > 0, amplitude smooth).
    f_te = 0.2 + rand(nti, 1, nto, npo)
    f_tm = 0.2 + rand(nti, 1, nto, npo)
    f_te_flat = f_te.reshape(-1).contiguous()
    f_tm_flat = f_tm.reshape(-1).contiguous()
    table_offset = torch.zeros(1, dtype=torch.int64, device=device)
    table_dims = torch.tensor([[nti, 1, nto, npo]], dtype=torch.int32, device=device)
    material_slot = torch.zeros(1, dtype=torch.int32, device=device)

    return {
        "valid": torch.ones(rows, dtype=torch.bool, device=device),
        "wo_rows": wo_rows,
        "r2_rows": r2_rows,
        "cos_o_rows": cos_o_rows,
        "n_o": n_o,
        "t1r": t1r,
        "t2r": t2r,
        "wi_local": wi_local,
        "cos_i": cos_i,
        "r1": r1,
        "a_te2": a_te2,
        "a_tm2": a_tm2,
        "weights": weights,
        "material_id": material_id,
        "backup_axis": backup_axis,
        "rx_pol": rx_pol,
        "rc_idx": rc_idx,
        "sc_idx": sc_idx,
        "f_te_flat": f_te_flat,
        "f_tm_flat": f_tm_flat,
        "table_offset": table_offset,
        "table_dims": table_dims,
        "material_slot": material_slot,
        "f_te": f_te,
        "f_tm": f_tm,
        "coef": coef,
        "nti": nti,
        "nto": nto,
        "npo": npo,
    }


_ENSEMBLE_FORWARD_ARGS = (
    "valid",
    "wo_rows",
    "r2_rows",
    "cos_o_rows",
    "n_o",
    "t1r",
    "t2r",
    "wi_local",
    "cos_i",
    "r1",
    "a_te2",
    "a_tm2",
    "weights",
    "material_id",
    "backup_axis",
    "rx_pol",
    "rc_idx",
    "sc_idx",
    "f_te_flat",
    "f_tm_flat",
    "table_offset",
    "table_dims",
    "material_slot",
)


def _ensemble_forward(case, threshold=-1.0):
    return scattering_functional.scattering_ensemble_eval(
        *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=case["coef"],
        threshold=threshold,
    )


def _ensemble_reference(case, *, dtype=torch.float64, make_leaves=False):
    """Build the float64 oracle inputs (optionally as autograd leaves)."""

    live = {
        "wo_rows": case["wo_rows"],
        "r2_rows": case["r2_rows"],
        "cos_o_rows": case["cos_o_rows"],
        "n_o": case["n_o"],
        "t1r": case["t1r"],
        "t2r": case["t2r"],
        "wi_local": case["wi_local"],
        "cos_i": case["cos_i"],
        "r1": case["r1"],
        "a_te2": case["a_te2"],
        "a_tm2": case["a_tm2"],
        "weights": case["weights"],
        "f_te": case["f_te"],
        "f_tm": case["f_tm"],
    }
    leaves = {name: value.to(dtype).clone() for name, value in live.items()}
    coef = torch.tensor(float(case["coef"]), dtype=dtype, device=case["wo_rows"].device)
    if make_leaves:
        for value in leaves.values():
            value.requires_grad_(True)
        coef.requires_grad_(True)
    out = ref_ensemble.kirchhoff_ensemble_gain_reference(
        leaves["wo_rows"],
        leaves["r2_rows"],
        leaves["cos_o_rows"],
        leaves["n_o"],
        leaves["t1r"],
        leaves["t2r"],
        leaves["wi_local"],
        leaves["cos_i"],
        leaves["r1"],
        leaves["a_te2"],
        leaves["a_tm2"],
        leaves["weights"],
        case["backup_axis"].to(dtype),
        case["rx_pol"].to(dtype),
        case["rc_idx"],
        case["sc_idx"],
        leaves["f_te"],
        leaves["f_tm"],
        coef,
    )
    return out, leaves, coef


# ---------------------------------------------------------------------------
# Op 1: forward parity + lockstep VJP.
# ---------------------------------------------------------------------------


def test_ensemble_forward_matches_reference():
    case = _ensemble_case(seed=11)
    native = _ensemble_forward(case)
    ref, _leaves, _coef = _ensemble_reference(case, dtype=torch.float64)
    for name in ("gain", "amplitude", "length"):
        assert (
            relative_error(native[name], ref[name], abs_floor=ABS_TOL) <= _REL_TOL_DIRECT
        ), name


def test_ensemble_invalid_row_short_circuits_poisoned_payload():
    case = _ensemble_case(seed=19)
    row = 3
    case["valid"][row] = False
    case["wo_rows"][row].fill_(float("nan"))
    case["r2_rows"][row] = float("nan")
    case["cos_o_rows"][row] = float("nan")
    case["rc_idx"][row] = torch.iinfo(torch.int64).max
    case["sc_idx"][row] = torch.iinfo(torch.int64).max

    out = _ensemble_forward(case)

    assert not bool(out["keep"][row])
    for name in ("gain", "amplitude", "length"):
        assert out[name][row].item() == 0.0, name


def test_ensemble_backward_matches_reference_autograd():
    case = _ensemble_case(seed=23)
    generator = torch.Generator(device="cuda").manual_seed(97)

    def randn(n):
        return torch.randn(n, generator=generator, device="cuda", dtype=torch.float32)

    rows = case["wo_rows"].shape[0]
    grad_gain = randn(rows)
    grad_amplitude = randn(rows)
    grad_length = randn(rows)

    native = scattering_functional.scattering_ensemble_eval_backward(
        *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=case["coef"],
        threshold=-1.0,
        grad_gain=grad_gain,
        grad_amplitude=grad_amplitude,
        grad_length=grad_length,
        need_grad_rows=True,
        need_grad_samples=True,
        need_grad_tables=True,
        need_grad_coef=True,
    )

    out, leaves, coef = _ensemble_reference(case, make_leaves=True)
    loss = (
        grad_gain.double() * out["gain"]
        + grad_amplitude.double() * out["amplitude"]
        + grad_length.double() * out["length"]
    ).sum()
    loss.backward()

    direct = {
        "grad_wo_rows": leaves["wo_rows"].grad,
        "grad_r2_rows": leaves["r2_rows"].grad,
        "grad_cos_o_rows": leaves["cos_o_rows"].grad,
    }
    for key, expected in direct.items():
        assert (
            relative_error(native[key], expected, abs_floor=ABS_TOL) <= _REL_TOL_DIRECT
        ), key

    accum = {
        "grad_n_o": leaves["n_o"].grad,
        "grad_t1r": leaves["t1r"].grad,
        "grad_t2r": leaves["t2r"].grad,
        "grad_wi_local": leaves["wi_local"].grad,
        "grad_cos_i": leaves["cos_i"].grad,
        "grad_r1": leaves["r1"].grad,
        "grad_a_te2": leaves["a_te2"].grad,
        "grad_a_tm2": leaves["a_tm2"].grad,
        "grad_weights": leaves["weights"].grad,
        "grad_f_te": leaves["f_te"].grad.reshape(-1),
        "grad_f_tm": leaves["f_tm"].grad.reshape(-1),
        "grad_coef": coef.grad.reshape(1),
    }
    for key, expected in accum.items():
        assert (
            relative_error(native[key], expected, abs_floor=ABS_TOL) <= _REL_TOL_ACCUM
        ), key


def test_ensemble_jvp_matches_reference_autograd():
    case = _ensemble_case(seed=31)
    generator = torch.Generator(device="cuda").manual_seed(53)

    def tangent_like(value):
        return torch.randn(*value.shape, generator=generator, device="cuda", dtype=torch.float32)

    tangents = {name: tangent_like(case[name]) for name in (
        "wo_rows", "r2_rows", "cos_o_rows", "n_o", "t1r", "t2r", "wi_local",
        "cos_i", "r1", "a_te2", "a_tm2", "weights",
    )}
    t_f_te = tangent_like(case["f_te_flat"])
    t_f_tm = tangent_like(case["f_tm_flat"])
    t_coef = 0.7

    native = scattering_functional.scattering_ensemble_eval_jvp(
        *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=case["coef"],
        threshold=-1.0,
        tangent_wo_rows=tangents["wo_rows"],
        tangent_r2_rows=tangents["r2_rows"],
        tangent_cos_o_rows=tangents["cos_o_rows"],
        tangent_n_o=tangents["n_o"],
        tangent_t1r=tangents["t1r"],
        tangent_t2r=tangents["t2r"],
        tangent_wi_local=tangents["wi_local"],
        tangent_cos_i=tangents["cos_i"],
        tangent_r1=tangents["r1"],
        tangent_a_te2=tangents["a_te2"],
        tangent_a_tm2=tangents["a_tm2"],
        tangent_weights=tangents["weights"],
        tangent_f_te_flat=t_f_te,
        tangent_f_tm_flat=t_f_tm,
        tangent_coef=t_coef,
    )

    # Forward-mode dual through the float64 reference.
    with torch.autograd.forward_ad.dual_level():
        duals = {}
        live = {
            "wo_rows": case["wo_rows"], "r2_rows": case["r2_rows"],
            "cos_o_rows": case["cos_o_rows"], "n_o": case["n_o"],
            "t1r": case["t1r"], "t2r": case["t2r"], "wi_local": case["wi_local"],
            "cos_i": case["cos_i"], "r1": case["r1"], "a_te2": case["a_te2"],
            "a_tm2": case["a_tm2"], "weights": case["weights"],
        }
        for name, value in live.items():
            duals[name] = torch.autograd.forward_ad.make_dual(
                value.double(), tangents[name].double()
            )
        f_te_dual = torch.autograd.forward_ad.make_dual(
            case["f_te"].double(), t_f_te.reshape(case["f_te"].shape).double()
        )
        f_tm_dual = torch.autograd.forward_ad.make_dual(
            case["f_tm"].double(), t_f_tm.reshape(case["f_tm"].shape).double()
        )
        coef_dual = torch.autograd.forward_ad.make_dual(
            torch.tensor(float(case["coef"]), dtype=torch.float64, device="cuda"),
            torch.tensor(t_coef, dtype=torch.float64, device="cuda"),
        )
        out = ref_ensemble.kirchhoff_ensemble_gain_reference(
            duals["wo_rows"], duals["r2_rows"], duals["cos_o_rows"], duals["n_o"],
            duals["t1r"], duals["t2r"], duals["wi_local"], duals["cos_i"],
            duals["r1"], duals["a_te2"], duals["a_tm2"], duals["weights"],
            case["backup_axis"].double(), case["rx_pol"].double(),
            case["rc_idx"], case["sc_idx"], f_te_dual, f_tm_dual, coef_dual,
        )
        expected = {
            "tangent_gain": torch.autograd.forward_ad.unpack_dual(out["gain"]).tangent,
            "tangent_amplitude": torch.autograd.forward_ad.unpack_dual(out["amplitude"]).tangent,
            "tangent_length": torch.autograd.forward_ad.unpack_dual(out["length"]).tangent,
        }
    for key in ("tangent_gain", "tangent_amplitude", "tangent_length"):
        assert (
            relative_error(native[key], expected[key], abs_floor=ABS_TOL) <= _REL_TOL_ACCUM
        ), key


def test_ensemble_jvp_vjp_inner_product_identity():
    # <g, J t> == <J^T g, t> for random cotangents g and tangents t.
    case = _ensemble_case(seed=42)
    generator = torch.Generator(device="cuda").manual_seed(1234)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)

    rows = case["wo_rows"].shape[0]
    g_gain, g_amp, g_len = randn(rows), randn(rows), randn(rows)
    live_names = (
        "wo_rows", "r2_rows", "cos_o_rows", "n_o", "t1r", "t2r", "wi_local",
        "cos_i", "r1", "a_te2", "a_tm2", "weights",
    )
    tangents = {name: randn(*case[name].shape) for name in live_names}
    t_f_te = randn(*case["f_te_flat"].shape)
    t_f_tm = randn(*case["f_tm_flat"].shape)
    t_coef = float(randn(1))

    jvp = scattering_functional.scattering_ensemble_eval_jvp(
        *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=case["coef"], threshold=-1.0,
        tangent_wo_rows=tangents["wo_rows"], tangent_r2_rows=tangents["r2_rows"],
        tangent_cos_o_rows=tangents["cos_o_rows"], tangent_n_o=tangents["n_o"],
        tangent_t1r=tangents["t1r"], tangent_t2r=tangents["t2r"],
        tangent_wi_local=tangents["wi_local"], tangent_cos_i=tangents["cos_i"],
        tangent_r1=tangents["r1"], tangent_a_te2=tangents["a_te2"],
        tangent_a_tm2=tangents["a_tm2"], tangent_weights=tangents["weights"],
        tangent_f_te_flat=t_f_te, tangent_f_tm_flat=t_f_tm, tangent_coef=t_coef,
    )
    lhs = (
        (g_gain * jvp["tangent_gain"]).sum()
        + (g_amp * jvp["tangent_amplitude"]).sum()
        + (g_len * jvp["tangent_length"]).sum()
    )

    vjp = scattering_functional.scattering_ensemble_eval_backward(
        *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=case["coef"], threshold=-1.0,
        grad_gain=g_gain, grad_amplitude=g_amp, grad_length=g_len,
        need_grad_rows=True, need_grad_samples=True,
        need_grad_tables=True, need_grad_coef=True,
    )
    rhs = torch.zeros((), device="cuda", dtype=torch.float64)
    pairs = {
        "grad_wo_rows": tangents["wo_rows"], "grad_r2_rows": tangents["r2_rows"],
        "grad_cos_o_rows": tangents["cos_o_rows"], "grad_n_o": tangents["n_o"],
        "grad_t1r": tangents["t1r"], "grad_t2r": tangents["t2r"],
        "grad_wi_local": tangents["wi_local"], "grad_cos_i": tangents["cos_i"],
        "grad_r1": tangents["r1"], "grad_a_te2": tangents["a_te2"],
        "grad_a_tm2": tangents["a_tm2"], "grad_weights": tangents["weights"],
        "grad_f_te": t_f_te, "grad_f_tm": t_f_tm,
    }
    for key, tangent in pairs.items():
        rhs = rhs + (vjp[key].double() * tangent.double()).sum()
    rhs = rhs + (vjp["grad_coef"].double().reshape(()) * t_coef)
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_ACCUM


def test_ensemble_jvp_matches_native_forward_fd():
    # Native JVP versus a central finite difference of the native forward along
    # a random direction (independent of the Torch oracle).
    case = _ensemble_case(seed=57)
    generator = torch.Generator(device="cuda").manual_seed(88)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)

    live_names = (
        "wo_rows", "r2_rows", "cos_o_rows", "n_o", "t1r", "t2r", "wi_local",
        "cos_i", "r1", "a_te2", "a_tm2", "weights",
    )
    tangents = {name: randn(*case[name].shape) for name in live_names}
    t_f_te = randn(*case["f_te_flat"].shape)
    t_f_tm = randn(*case["f_tm_flat"].shape)
    t_coef = float(randn(1))

    jvp = scattering_functional.scattering_ensemble_eval_jvp(
        *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=case["coef"], threshold=-1.0,
        tangent_wo_rows=tangents["wo_rows"], tangent_r2_rows=tangents["r2_rows"],
        tangent_cos_o_rows=tangents["cos_o_rows"], tangent_n_o=tangents["n_o"],
        tangent_t1r=tangents["t1r"], tangent_t2r=tangents["t2r"],
        tangent_wi_local=tangents["wi_local"], tangent_cos_i=tangents["cos_i"],
        tangent_r1=tangents["r1"], tangent_a_te2=tangents["a_te2"],
        tangent_a_tm2=tangents["a_tm2"], tangent_weights=tangents["weights"],
        tangent_f_te_flat=t_f_te, tangent_f_tm_flat=t_f_tm, tangent_coef=t_coef,
    )

    def forward_at(step):
        shifted = dict(case)
        for name in live_names:
            shifted[name] = case[name] + step * tangents[name]
        shifted["f_te_flat"] = case["f_te_flat"] + step * t_f_te
        shifted["f_tm_flat"] = case["f_tm_flat"] + step * t_f_tm
        shifted["coef"] = case["coef"] + step * t_coef
        return _ensemble_forward(shifted)

    plus = forward_at(_FD_STEP)
    minus = forward_at(-_FD_STEP)
    for out_name, t_name in (
        ("gain", "tangent_gain"),
        ("length", "tangent_length"),
    ):
        fd = (plus[out_name] - minus[out_name]) / (2.0 * _FD_STEP)
        assert relative_error(jvp[t_name], fd, abs_floor=ABS_TOL) <= _REL_TOL_FD_FORWARD, out_name


# ---------------------------------------------------------------------------
# Op 1: contract / negative / edge-case tests.
# ---------------------------------------------------------------------------


def test_ensemble_backward_need_flags_gate_outputs():
    case = _ensemble_case(seed=13)
    rows = case["wo_rows"].shape[0]
    grad_gain = torch.ones(rows, device="cuda")
    out = scattering_functional.scattering_ensemble_eval_backward(
        *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=case["coef"], threshold=-1.0,
        grad_gain=grad_gain, grad_amplitude=None, grad_length=None,
        need_grad_rows=True, need_grad_samples=False,
        need_grad_tables=False, need_grad_coef=False,
    )
    for key in ("grad_wo_rows", "grad_r2_rows", "grad_cos_o_rows"):
        assert out[key] is not None
    for key in (
        "grad_n_o", "grad_t1r", "grad_t2r", "grad_wi_local", "grad_cos_i",
        "grad_r1", "grad_a_te2", "grad_a_tm2", "grad_weights",
        "grad_f_te", "grad_f_tm", "grad_coef",
    ):
        assert out[key] is None, key


def test_ensemble_backward_rejects_wrong_dtype():
    case = _ensemble_case(seed=17)
    rows = case["wo_rows"].shape[0]
    bad = dict(case)
    bad["wo_rows"] = case["wo_rows"].double()  # wrong dtype
    with pytest.raises((TypeError, ValueError)):
        scattering_functional.scattering_ensemble_eval_backward(
            *(bad[name] for name in _ENSEMBLE_FORWARD_ARGS),
            coef=case["coef"], threshold=-1.0,
            grad_gain=torch.ones(rows, device="cuda"),
            need_grad_rows=True, need_grad_samples=False,
            need_grad_tables=False, need_grad_coef=False,
        )


def test_ensemble_backward_empty_rows():
    case = _ensemble_case(seed=19, rows=0)
    out = scattering_functional.scattering_ensemble_eval_backward(
        *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=case["coef"], threshold=-1.0,
        grad_gain=torch.zeros(0, device="cuda"),
        grad_amplitude=torch.zeros(0, device="cuda"),
        grad_length=torch.zeros(0, device="cuda"),
        need_grad_rows=True, need_grad_samples=True,
        need_grad_tables=True, need_grad_coef=True,
    )
    assert out["grad_wo_rows"].shape == (0, 3)
    # No row touches a sample/table entry -> accumulated grads are all zero.
    assert float(out["grad_f_te"].abs().sum()) == 0.0
    assert float(out["grad_coef"].abs().sum()) == 0.0


def test_ensemble_backward_requires_native_kernel(monkeypatch):
    case = _ensemble_case(seed=3)
    rows = case["wo_rows"].shape[0]
    monkeypatch.setattr(runtime, "native_extension", lambda: None)
    with pytest.raises(RuntimeError, match="CUDA kernel is required"):
        scattering_functional.scattering_ensemble_eval_backward(
            *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
            coef=case["coef"], threshold=-1.0,
            grad_gain=torch.ones(rows, device="cuda"),
            need_grad_rows=True, need_grad_samples=True,
            need_grad_tables=True, need_grad_coef=True,
        )


def test_ensemble_horizon_rows_are_gated():
    # Rows below the horizon (cos_o <= 0) contribute zero gain and no table
    # gradient; the native backward must match the gated reference.
    case = _ensemble_case(seed=61)
    case["cos_o_rows"][:4] = -0.1
    native = _ensemble_forward(case)
    assert torch.all(native["gain"][:4] == 0.0)

    rows = case["wo_rows"].shape[0]
    grad_gain = torch.ones(rows, device="cuda")
    out = scattering_functional.scattering_ensemble_eval_backward(
        *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=case["coef"], threshold=-1.0, grad_gain=grad_gain,
        need_grad_rows=True, need_grad_samples=False,
        need_grad_tables=True, need_grad_coef=True,
    )
    _out, leaves, _coef = _ensemble_reference(case, make_leaves=True)
    (grad_gain.double() * _out["gain"]).sum().backward()
    assert (
        relative_error(out["grad_f_te"], leaves["f_te"].grad.reshape(-1), abs_floor=ABS_TOL)
        <= _REL_TOL_ACCUM
    )


def test_ensemble_clamped_cos_axis_kills_row_gradient():
    # cos_o at 1.0 lands past the last table cell center -> the clamped table
    # axis contributes no gradient through cos_o_rows for those rows.
    case = _ensemble_case(seed=71)
    case["cos_o_rows"][:] = 1.0
    rows = case["wo_rows"].shape[0]
    out = scattering_functional.scattering_ensemble_eval_backward(
        *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=case["coef"], threshold=-1.0,
        grad_gain=torch.ones(rows, device="cuda"),
        need_grad_rows=True, need_grad_samples=False,
        need_grad_tables=True, need_grad_coef=False,
    )
    _out, leaves, _coef = _ensemble_reference(case, make_leaves=True)
    _out["gain"].sum().backward()
    # The reference cos_o gradient still carries the radiometric-cos term but no
    # table-axis term; the native store must match it exactly.
    assert (
        relative_error(out["grad_cos_o_rows"], leaves["cos_o_rows"].grad, abs_floor=ABS_TOL)
        <= _REL_TOL_DIRECT
    )


def test_ensemble_degenerate_backup_branch_matches_reference():
    # A row whose outgoing direction is parallel to the sample normal drives the
    # s/p frame into its constant backup branch; the lockstep must still hold.
    case = _ensemble_case(seed=83)
    case["wo_rows"][0] = case["n_o"][case["sc_idx"][0]]
    rows = case["wo_rows"].shape[0]
    grad_gain = torch.randn(rows, device="cuda")
    out = scattering_functional.scattering_ensemble_eval_backward(
        *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=case["coef"], threshold=-1.0, grad_gain=grad_gain,
        need_grad_rows=True, need_grad_samples=True,
        need_grad_tables=False, need_grad_coef=False,
    )
    _out, leaves, _coef = _ensemble_reference(case, make_leaves=True)
    (grad_gain.double() * _out["gain"]).sum().backward()
    assert torch.isfinite(out["grad_wo_rows"]).all()
    assert (
        relative_error(out["grad_n_o"], leaves["n_o"].grad, abs_floor=ABS_TOL)
        <= _REL_TOL_ACCUM
    )


def test_ensemble_isotropic_azimuth_coupling_reaches_wi_local():
    # npi == 1 relative-azimuth coupling: the incident azimuth (wi_local xy)
    # must receive a non-zero gradient through phi_o' = wrap(phi_o - phi_i).
    case = _ensemble_case(seed=91)
    rows = case["wo_rows"].shape[0]
    out = scattering_functional.scattering_ensemble_eval_backward(
        *(case[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=case["coef"], threshold=-1.0,
        grad_gain=torch.ones(rows, device="cuda"),
        need_grad_rows=False, need_grad_samples=True,
        need_grad_tables=False, need_grad_coef=False,
    )
    assert float(out["grad_wi_local"][:, :2].abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Op 1: public AD wrapper (torch.autograd.Function) parity + rejection.
# ---------------------------------------------------------------------------


def test_ensemble_ad_wrapper_matches_functional_backward():
    case = _ensemble_case(seed=101)
    device = "cuda"
    # Live leaves for the wrapper: the row/sample tensors, the flat tables and
    # the 0-dim coef scalar.
    leaf_names = (
        "wo_rows", "r2_rows", "cos_o_rows", "n_o", "t1r", "t2r", "wi_local",
        "cos_i", "r1", "a_te2", "a_tm2", "weights", "f_te_flat", "f_tm_flat",
    )
    leaves = {name: case[name].clone().requires_grad_(True) for name in leaf_names}
    coef = torch.tensor(float(case["coef"]), dtype=torch.float32, device=device, requires_grad=True)

    args = dict(case)
    args.update(leaves)
    out = scattering_autograd.scattering_ensemble_eval_ad(
        *(args[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=coef, threshold=-1.0,
    )
    generator = torch.Generator(device=device).manual_seed(202)
    rows = case["wo_rows"].shape[0]
    g_gain = torch.randn(rows, generator=generator, device=device)
    (g_gain * out["gain"]).sum().backward()
    assert leaves["wo_rows"].grad is not None
    assert leaves["f_te_flat"].grad is not None
    assert coef.grad is not None
    assert torch.isfinite(coef.grad).all()


def test_ensemble_ad_wrapper_rejects_fixed_input_grad():
    # rx_pol is a fixed input; requesting its gradient must fail loudly.
    case = _ensemble_case(seed=103)
    device = "cuda"
    rx_pol = case["rx_pol"].clone().requires_grad_(True)
    coef = torch.tensor(float(case["coef"]), dtype=torch.float32, device=device)
    args = dict(case)
    args["rx_pol"] = rx_pol
    out = scattering_autograd.scattering_ensemble_eval_ad(
        *(args[name] for name in _ENSEMBLE_FORWARD_ARGS),
        coef=coef, threshold=-1.0,
    )
    with pytest.raises(NotImplementedError):
        out["gain"].sum().backward()


def test_ensemble_ad_mode_none_has_no_autograd_function():
    case = _ensemble_case(seed=105)
    out = _ensemble_forward(case)
    assert not out["gain"].requires_grad


# ---------------------------------------------------------------------------
# Op 2 (realization phase-screen patch integral) fixtures.
# ---------------------------------------------------------------------------


def _patch_case(
    *,
    device: str = "cuda",
    seed: int,
    rows: int = 6,
    patches: int = 8,
    grid: int = 16,
    grazing: bool = False,
    zero_height: bool = False,
    frequency_hz: float = 3.0e9,
):
    """Small phase-screen patch case on the z = 0 plane (normal +z)."""

    generator = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=device, dtype=torch.float32)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=torch.float32)

    # Patch triangles covering a unit square, matching planar UVs.
    tris = []
    uvs = []
    for _ in range(patches):
        origin = rand(2)
        size = 0.1 + 0.2 * rand(2)
        p0 = torch.stack((origin[0], origin[1], torch.zeros((), device=device)))
        p1 = torch.stack((origin[0] + size[0], origin[1], torch.zeros((), device=device)))
        p2 = torch.stack((origin[0], origin[1] + size[1], torch.zeros((), device=device)))
        tris.append(torch.stack((p0, p1, p2)))
        uvs.append(
            torch.stack(
                (
                    torch.stack((origin[0], origin[1])),
                    torch.stack((origin[0] + size[0], origin[1])),
                    torch.stack((origin[0], origin[1] + size[1])),
                )
            )
        )
    patch_tris = torch.stack(tris).contiguous()
    patch_uvs = torch.stack(uvs).contiguous()

    row_index = torch.arange(rows, device=device, dtype=torch.int64) % patches
    centroids = patch_tris[row_index].mean(dim=1).contiguous()
    n_rows = torch.zeros(rows, 3, device=device)
    n_rows[:, 2] = 1.0

    # Incident direction from above (into the surface): d_i.z < 0. Outgoing up:
    # d_o.z > 0, so q_n = k0*(d_o.z - d_i.z) > 0. Grazing pushes q_n -> ~0.
    di_z = -0.05 if grazing else -(0.3 + 0.5 * rand(rows))
    do_z = 0.05 if grazing else (0.3 + 0.5 * rand(rows))
    d_i = torch.nn.functional.normalize(
        torch.stack((0.4 * randn(rows), 0.4 * randn(rows), di_z * torch.ones(rows, device=device)), dim=-1),
        dim=-1,
    ).contiguous()
    d_o = torch.nn.functional.normalize(
        torch.stack((0.4 * randn(rows), 0.4 * randn(rows), do_z * torch.ones(rows, device=device)), dim=-1),
        dim=-1,
    ).contiguous()

    r_te = torch.complex(0.3 + 0.4 * rand(rows), 0.2 * randn(rows)).contiguous()
    r_tm = torch.complex(0.3 + 0.4 * rand(rows), 0.2 * randn(rows)).contiguous()
    pol_t = torch.tensor([0.0, 1.0, 0.0], device=device)
    pol_r = torch.tensor([0.0, 1.0, 0.0], device=device)
    r1_rows = (1.0 + rand(rows)).contiguous()
    r2_rows = (1.0 + rand(rows)).contiguous()

    if zero_height:
        heights = torch.zeros(grid, grid, device=device)
    else:
        heights = (1.0e-3 * randn(grid, grid)).contiguous()
    k0 = 2.0 * math.pi * frequency_hz / 299792458.0

    return {
        "valid": torch.ones(rows, dtype=torch.bool, device=device),
        "patch_tris": patch_tris,
        "patch_uvs": patch_uvs,
        "rows": row_index,
        "d_i": d_i,
        "d_o": d_o,
        "n_rows": n_rows,
        "r_te": r_te,
        "r_tm": r_tm,
        "pol_t": pol_t,
        "pol_r": pol_r,
        "r1_rows": r1_rows,
        "r2_rows": r2_rows,
        "centroids": centroids,
        "heights": heights,
        "k0": k0,
    }


_PATCH_FORWARD_ARGS = (
    "valid",
    "patch_tris",
    "patch_uvs",
    "rows",
    "d_i",
    "d_o",
    "n_rows",
    "r_te",
    "r_tm",
    "pol_t",
    "pol_r",
    "r1_rows",
    "r2_rows",
    "centroids",
    "heights",
)


def _patch_forward(case):
    return scattering_functional.scattering_patch_integral_eval(
        *(case[name] for name in _PATCH_FORWARD_ARGS), k0=case["k0"]
    )


def _patch_reference(case, *, make_leaves=False):
    device = case["heights"].device
    quad_a, quad_b, quad_w = scattering_functional._duffy_nodes(device)
    live_names = (
        "heights", "d_i", "d_o", "r1_rows", "r2_rows", "centroids", "r_te", "r_tm",
    )
    leaves = {}
    for name in live_names:
        value = case[name]
        value = value.to(torch.complex128) if value.is_complex() else value.double()
        leaves[name] = value.clone()
    k0 = torch.tensor(float(case["k0"]), dtype=torch.float64, device=device)
    if make_leaves:
        for value in leaves.values():
            value.requires_grad_(True)
        k0.requires_grad_(True)
    out = ref_patch.realization_patch_eval_reference(
        leaves["heights"],
        case["patch_tris"].double(),
        case["patch_uvs"].double(),
        case["rows"],
        leaves["d_i"],
        leaves["d_o"],
        case["n_rows"].double(),
        leaves["r_te"],
        leaves["r_tm"],
        case["pol_t"].double(),
        case["pol_r"].double(),
        leaves["r1_rows"],
        leaves["r2_rows"],
        leaves["centroids"],
        quad_a,
        quad_b,
        quad_w,
        k0,
    )
    return out, leaves, k0


# ---------------------------------------------------------------------------
# Op 2: forward parity + lockstep VJP/JVP.
# ---------------------------------------------------------------------------


def test_patch_forward_matches_reference():
    case = _patch_case(seed=201)
    native = _patch_forward(case)
    ref, _leaves, _k0 = _patch_reference(case)
    assert relative_error(native["total"], ref["total"], abs_floor=ABS_TOL) <= _REL_TOL_DIRECT


def test_patch_invalid_row_short_circuits_poisoned_payload():
    case = _patch_case(seed=207)
    row = 2
    case["valid"][row] = False
    case["rows"][row] = torch.iinfo(torch.int64).max
    case["d_i"][row].fill_(float("nan"))
    case["d_o"][row].fill_(float("nan"))

    out = _patch_forward(case)

    assert out["integral"][row].item() == 0.0j
    assert out["row_value"][row].item() == 0.0j
    assert torch.isfinite(out["total"].real)
    assert torch.isfinite(out["total"].imag)


def test_patch_backward_matches_reference_autograd():
    case = _patch_case(seed=211)
    generator = torch.Generator(device="cuda").manual_seed(311)
    grad_total = torch.complex(
        torch.randn((), generator=generator, device="cuda"),
        torch.randn((), generator=generator, device="cuda"),
    ).to(torch.complex64)

    native = scattering_functional.scattering_patch_integral_eval_backward(
        *(case[name] for name in _PATCH_FORWARD_ARGS),
        k0=case["k0"], grad_total=grad_total,
        need_grad_heights=True, need_grad_jones=True,
        need_grad_geometry=True, need_grad_k0=True,
    )

    out, leaves, k0 = _patch_reference(case, make_leaves=True)
    g = grad_total.to(torch.complex128)
    loss = (g.real * out["total"].real + g.imag * out["total"].imag)
    loss.backward()

    assert (
        relative_error(native["grad_heights"], leaves["heights"].grad, abs_floor=ABS_TOL)
        <= _REL_TOL_ACCUM
    )
    for key, leaf in (
        ("grad_d_i", "d_i"),
        ("grad_d_o", "d_o"),
        ("grad_r1_rows", "r1_rows"),
        ("grad_r2_rows", "r2_rows"),
        ("grad_centroids", "centroids"),
    ):
        assert (
            relative_error(native[key], leaves[leaf].grad, abs_floor=ABS_TOL)
            <= _REL_TOL_DIRECT
        ), key
    for key, leaf in (("grad_r_te", "r_te"), ("grad_r_tm", "r_tm")):
        assert (
            relative_error(native[key], leaves[leaf].grad, abs_floor=ABS_TOL)
            <= _REL_TOL_DIRECT
        ), key
    assert (
        relative_error(native["grad_k0"].reshape(()), k0.grad, abs_floor=ABS_TOL)
        <= _REL_TOL_ACCUM
    )


def test_patch_jvp_matches_reference_autograd():
    case = _patch_case(seed=221)
    generator = torch.Generator(device="cuda").manual_seed(321)

    def real_tangent(value):
        return torch.randn(*value.shape, generator=generator, device="cuda", dtype=torch.float32)

    def complex_tangent(value):
        return torch.complex(real_tangent(value), real_tangent(value)).to(torch.complex64)

    tangents = {
        "heights": real_tangent(case["heights"]),
        "d_i": real_tangent(case["d_i"]),
        "d_o": real_tangent(case["d_o"]),
        "r1_rows": real_tangent(case["r1_rows"]),
        "r2_rows": real_tangent(case["r2_rows"]),
        "centroids": real_tangent(case["centroids"]),
        "r_te": complex_tangent(case["r_te"]),
        "r_tm": complex_tangent(case["r_tm"]),
    }
    t_k0 = 0.5

    native = scattering_functional.scattering_patch_integral_eval_jvp(
        *(case[name] for name in _PATCH_FORWARD_ARGS), k0=case["k0"],
        tangent_heights=tangents["heights"], tangent_d_i=tangents["d_i"],
        tangent_d_o=tangents["d_o"], tangent_r1_rows=tangents["r1_rows"],
        tangent_r2_rows=tangents["r2_rows"], tangent_centroids=tangents["centroids"],
        tangent_r_te=tangents["r_te"], tangent_r_tm=tangents["r_tm"],
        tangent_k0=t_k0,
    )

    device = "cuda"
    quad_a, quad_b, quad_w = scattering_functional._duffy_nodes(device)
    with torch.autograd.forward_ad.dual_level():
        def dual(name, dtype):
            base = case[name].to(dtype)
            tan = tangents[name]
            tan = tan.to(torch.complex128) if base.is_complex() else tan.double()
            return torch.autograd.forward_ad.make_dual(base, tan)

        heights_d = dual("heights", torch.float64)
        d_i_d = dual("d_i", torch.float64)
        d_o_d = dual("d_o", torch.float64)
        r1_d = dual("r1_rows", torch.float64)
        r2_d = dual("r2_rows", torch.float64)
        cen_d = dual("centroids", torch.float64)
        rte_d = dual("r_te", torch.complex128)
        rtm_d = dual("r_tm", torch.complex128)
        k0_d = torch.autograd.forward_ad.make_dual(
            torch.tensor(float(case["k0"]), dtype=torch.float64, device=device),
            torch.tensor(t_k0, dtype=torch.float64, device=device),
        )
        out = ref_patch.realization_patch_eval_reference(
            heights_d, case["patch_tris"].double(), case["patch_uvs"].double(),
            case["rows"], d_i_d, d_o_d, case["n_rows"].double(), rte_d, rtm_d,
            case["pol_t"].double(), case["pol_r"].double(), r1_d, r2_d, cen_d,
            quad_a, quad_b, quad_w, k0_d,
        )
        expected = torch.autograd.forward_ad.unpack_dual(out["total"]).tangent
    assert relative_error(native["tangent_total"], expected, abs_floor=ABS_TOL) <= _REL_TOL_ACCUM


def test_patch_jvp_vjp_inner_product_identity():
    # Re(conj(g) * (J t)) == <J^T g, t> with the real Wirtinger pairing.
    case = _patch_case(seed=231)
    generator = torch.Generator(device="cuda").manual_seed(331)

    def real_tangent(value):
        return torch.randn(*value.shape, generator=generator, device="cuda", dtype=torch.float32)

    def complex_tangent(value):
        return torch.complex(real_tangent(value), real_tangent(value)).to(torch.complex64)

    tangents = {
        "heights": real_tangent(case["heights"]),
        "d_i": real_tangent(case["d_i"]),
        "d_o": real_tangent(case["d_o"]),
        "r1_rows": real_tangent(case["r1_rows"]),
        "r2_rows": real_tangent(case["r2_rows"]),
        "centroids": real_tangent(case["centroids"]),
        "r_te": complex_tangent(case["r_te"]),
        "r_tm": complex_tangent(case["r_tm"]),
    }
    t_k0 = float(torch.randn((), generator=generator, device="cuda"))
    g = torch.complex(
        torch.randn((), generator=generator, device="cuda"),
        torch.randn((), generator=generator, device="cuda"),
    ).to(torch.complex64)

    jvp = scattering_functional.scattering_patch_integral_eval_jvp(
        *(case[name] for name in _PATCH_FORWARD_ARGS), k0=case["k0"],
        tangent_heights=tangents["heights"], tangent_d_i=tangents["d_i"],
        tangent_d_o=tangents["d_o"], tangent_r1_rows=tangents["r1_rows"],
        tangent_r2_rows=tangents["r2_rows"], tangent_centroids=tangents["centroids"],
        tangent_r_te=tangents["r_te"], tangent_r_tm=tangents["r_tm"],
        tangent_k0=t_k0,
    )
    lhs = (g.conj() * jvp["tangent_total"]).real.to(torch.float64)

    vjp = scattering_functional.scattering_patch_integral_eval_backward(
        *(case[name] for name in _PATCH_FORWARD_ARGS), k0=case["k0"], grad_total=g,
        need_grad_heights=True, need_grad_jones=True,
        need_grad_geometry=True, need_grad_k0=True,
    )
    rhs = torch.zeros((), device="cuda", dtype=torch.float64)
    for key, name in (
        ("grad_heights", "heights"), ("grad_d_i", "d_i"), ("grad_d_o", "d_o"),
        ("grad_r1_rows", "r1_rows"), ("grad_r2_rows", "r2_rows"),
        ("grad_centroids", "centroids"),
    ):
        rhs = rhs + (vjp[key].double() * tangents[name].double()).sum()
    # Complex pairing uses the real inner product Re(conj(grad) * tangent).
    for key, name in (("grad_r_te", "r_te"), ("grad_r_tm", "r_tm")):
        rhs = rhs + (vjp[key].conj() * tangents[name].to(torch.complex128)).real.sum()
    rhs = rhs + vjp["grad_k0"].double().reshape(()) * t_k0
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_ACCUM


# ---------------------------------------------------------------------------
# Op 2: contract / negative / edge-case tests.
# ---------------------------------------------------------------------------


def test_patch_backward_need_flags_gate_outputs():
    case = _patch_case(seed=241)
    g = torch.ones((), device="cuda", dtype=torch.complex64)
    out = scattering_functional.scattering_patch_integral_eval_backward(
        *(case[name] for name in _PATCH_FORWARD_ARGS), k0=case["k0"], grad_total=g,
        need_grad_heights=True, need_grad_jones=False,
        need_grad_geometry=False, need_grad_k0=False,
    )
    assert out["grad_heights"] is not None
    for key in (
        "grad_r_te", "grad_r_tm", "grad_d_i", "grad_d_o", "grad_r1_rows",
        "grad_r2_rows", "grad_centroids", "grad_k0",
    ):
        assert out[key] is None, key


def test_patch_backward_rejects_wrong_dtype():
    case = _patch_case(seed=243)
    bad = dict(case)
    bad["heights"] = case["heights"].double()  # wrong dtype
    with pytest.raises((TypeError, ValueError)):
        scattering_functional.scattering_patch_integral_eval_backward(
            *(bad[name] for name in _PATCH_FORWARD_ARGS), k0=case["k0"],
            grad_total=torch.ones((), device="cuda", dtype=torch.complex64),
            need_grad_heights=True, need_grad_jones=False,
            need_grad_geometry=False, need_grad_k0=False,
        )


def test_patch_backward_requires_native_kernel(monkeypatch):
    case = _patch_case(seed=245)
    monkeypatch.setattr(runtime, "native_extension", lambda: None)
    with pytest.raises(RuntimeError, match="CUDA kernel is required"):
        scattering_functional.scattering_patch_integral_eval_backward(
            *(case[name] for name in _PATCH_FORWARD_ARGS), k0=case["k0"],
            grad_total=torch.ones((), device="cuda", dtype=torch.complex64),
            need_grad_heights=True, need_grad_jones=True,
            need_grad_geometry=True, need_grad_k0=True,
        )


def test_patch_grazing_q_n_clamp_is_finite():
    # q_n -> 0 hits the max(q.n, 1e-9) clamp; gradients must stay finite and
    # track the clamped reference.
    case = _patch_case(seed=251, grazing=True)
    g = torch.ones((), device="cuda", dtype=torch.complex64)
    out = scattering_functional.scattering_patch_integral_eval_backward(
        *(case[name] for name in _PATCH_FORWARD_ARGS), k0=case["k0"], grad_total=g,
        need_grad_heights=True, need_grad_jones=True,
        need_grad_geometry=True, need_grad_k0=True,
    )
    assert torch.isfinite(out["grad_heights"]).all()
    assert torch.isfinite(out["grad_k0"]).all()
    ref, leaves, _k0 = _patch_reference(case, make_leaves=True)
    (g.to(torch.complex128).real * ref["total"].real
     + g.to(torch.complex128).imag * ref["total"].imag).backward()
    assert (
        relative_error(out["grad_heights"], leaves["heights"].grad, abs_floor=ABS_TOL)
        <= _REL_TOL_ACCUM
    )


def test_patch_zero_height_screen_has_finite_gradients():
    # A flat (h == 0) screen is the smooth-plate limit; height gradients are
    # still defined (they carry the -j*q_int_n phasor derivative) and finite.
    case = _patch_case(seed=261, zero_height=True)
    g = torch.complex(
        torch.tensor(0.7, device="cuda"), torch.tensor(-0.3, device="cuda")
    ).to(torch.complex64)
    out = scattering_functional.scattering_patch_integral_eval_backward(
        *(case[name] for name in _PATCH_FORWARD_ARGS), k0=case["k0"], grad_total=g,
        need_grad_heights=True, need_grad_jones=True,
        need_grad_geometry=True, need_grad_k0=True,
    )
    assert torch.isfinite(out["grad_heights"]).all()
    assert float(out["grad_heights"].abs().sum()) > 0.0


def test_patch_ad_wrapper_matches_functional_and_rejects_fixed():
    case = _patch_case(seed=271)
    device = "cuda"
    heights = case["heights"].clone().requires_grad_(True)
    k0 = torch.tensor(float(case["k0"]), dtype=torch.float32, device=device, requires_grad=True)
    args = dict(case)
    args["heights"] = heights
    out = scattering_autograd.scattering_patch_integral_eval_ad(
        *(args[name] for name in _PATCH_FORWARD_ARGS), k0=k0,
    )
    out["total"].real.backward()
    assert heights.grad is not None
    assert torch.isfinite(heights.grad).all()
    assert k0.grad is not None

    # patch_tris is a fixed input: requesting its gradient fails loudly.
    tris = case["patch_tris"].clone().requires_grad_(True)
    args2 = dict(case)
    args2["patch_tris"] = tris
    out2 = scattering_autograd.scattering_patch_integral_eval_ad(
        *(args2[name] for name in _PATCH_FORWARD_ARGS), k0=case["k0"],
    )
    with pytest.raises(NotImplementedError):
        out2["total"].real.backward()


def test_patch_ad_mode_none_has_no_autograd_function():
    case = _patch_case(seed=273)
    out = _patch_forward(case)
    assert not out["total"].requires_grad


# ---------------------------------------------------------------------------
# scattering AD: resident Kirchhoff table lookup (scattering_table_eval)
# backward / jvp companions. Lockstep against the float64 bsdf_table_interp
# oracle in tests.reference.kirchhoff_ensemble (the same helper the ensemble
# op-1 lockstep uses). The explicit valid mask is fixed; all four continuous
# inputs (wi, wo, f_te, f_tm) are live.
# ---------------------------------------------------------------------------


def _table_case(*, seed, rows=24, nti=8, npi=1, nto=8, npo=8, device="cuda"):
    """Rows above the horizon with interior cos axes (no clamp saturation)."""

    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=torch.float32)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=device, dtype=torch.float32)

    def dirs(n):
        # cos component (z) in (0.3, 0.8) sits interior to the clamped cos axis;
        # xy carry a strictly nonzero azimuth so the atan2 chain is defined.
        return torch.stack((randn(n), randn(n), 0.3 + 0.5 * rand(n)), dim=-1).contiguous()

    return {
        "valid": torch.ones(rows, dtype=torch.bool, device=device),
        "wi": dirs(rows),
        "wo": dirs(rows),
        "f_te": (0.2 + rand(nti, npi, nto, npo)).contiguous(),
        "f_tm": (0.2 + rand(nti, npi, nto, npo)).contiguous(),
    }


_TABLE_ARGS = ("valid", "wi", "wo", "f_te", "f_tm")
_TABLE_LIVE_ARGS = ("wi", "wo", "f_te", "f_tm")


def _table_reference(case, *, make_leaves=False):
    leaves = {name: case[name].double().clone() for name in _TABLE_LIVE_ARGS}
    if make_leaves:
        for value in leaves.values():
            value.requires_grad_(True)
    te, tm = ref_ensemble.bsdf_table_interp(
        leaves["wi"], leaves["wo"], leaves["f_te"], leaves["f_tm"]
    )
    return te, tm, leaves


def test_table_eval_forward_matches_reference():
    case = _table_case(seed=311)
    te, tm = scattering_functional.scattering_table_eval(*(case[n] for n in _TABLE_ARGS))
    ref_te, ref_tm, _ = _table_reference(case)
    assert relative_error(te, ref_te, abs_floor=ABS_TOL) <= _REL_TOL_DIRECT
    assert relative_error(tm, ref_tm, abs_floor=ABS_TOL) <= _REL_TOL_DIRECT


def test_table_eval_invalid_row_short_circuits_poisoned_payload():
    case = _table_case(seed=312)
    row = 5
    case["valid"][row] = False
    case["wi"][row].fill_(float("nan"))
    case["wo"][row].fill_(float("nan"))

    te, tm = scattering_functional.scattering_table_eval(
        *(case[name] for name in _TABLE_ARGS)
    )

    assert te[row].item() == 0.0
    assert tm[row].item() == 0.0


@pytest.mark.parametrize("npi", (1, 6))
def test_table_eval_backward_matches_reference_autograd(npi):
    case = _table_case(seed=313, npi=npi)
    rows = case["wi"].shape[0]
    generator = torch.Generator(device="cuda").manual_seed(929)

    def randn(n):
        return torch.randn(n, generator=generator, device="cuda", dtype=torch.float32)

    grad_te = randn(rows)
    grad_tm = randn(rows)
    native = scattering_functional.scattering_table_eval_backward(
        *(case[n] for n in _TABLE_ARGS),
        grad_out_f_te=grad_te,
        grad_out_f_tm=grad_tm,
        need_grad_dirs=True,
        need_grad_tables=True,
    )

    te, tm, leaves = _table_reference(case, make_leaves=True)
    (grad_te.double() * te + grad_tm.double() * tm).sum().backward()
    for key, leaf in (("grad_wi", "wi"), ("grad_wo", "wo")):
        assert (
            relative_error(native[key], leaves[leaf].grad, abs_floor=ABS_TOL)
            <= _REL_TOL_DIRECT
        ), key
    for key, leaf in (("grad_f_te", "f_te"), ("grad_f_tm", "f_tm")):
        assert (
            relative_error(native[key], leaves[leaf].grad, abs_floor=ABS_TOL)
            <= _REL_TOL_ACCUM
        ), key


def test_table_eval_jvp_matches_reference_autograd():
    case = _table_case(seed=317)
    generator = torch.Generator(device="cuda").manual_seed(731)

    def tangent_like(value):
        return torch.randn(*value.shape, generator=generator, device="cuda", dtype=torch.float32)

    tangents = {name: tangent_like(case[name]) for name in _TABLE_LIVE_ARGS}
    native = scattering_functional.scattering_table_eval_jvp(
        *(case[n] for n in _TABLE_ARGS),
        tangent_wi=tangents["wi"],
        tangent_wo=tangents["wo"],
        tangent_f_te=tangents["f_te"],
        tangent_f_tm=tangents["f_tm"],
    )
    with torch.autograd.forward_ad.dual_level():
        duals = {
            name: torch.autograd.forward_ad.make_dual(
                case[name].double(), tangents[name].double()
            )
            for name in _TABLE_LIVE_ARGS
        }
        te, tm = ref_ensemble.bsdf_table_interp(
            duals["wi"], duals["wo"], duals["f_te"], duals["f_tm"]
        )
        expected_te = torch.autograd.forward_ad.unpack_dual(te).tangent
        expected_tm = torch.autograd.forward_ad.unpack_dual(tm).tangent
    assert (
        relative_error(native["tangent_f_te"], expected_te, abs_floor=ABS_TOL)
        <= _REL_TOL_ACCUM
    )
    assert (
        relative_error(native["tangent_f_tm"], expected_tm, abs_floor=ABS_TOL)
        <= _REL_TOL_ACCUM
    )


def test_table_eval_jvp_vjp_inner_product_identity():
    case = _table_case(seed=319)
    rows = case["wi"].shape[0]
    generator = torch.Generator(device="cuda").manual_seed(1441)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)

    tangents = {name: randn(*case[name].shape) for name in _TABLE_LIVE_ARGS}
    g_te, g_tm = randn(rows), randn(rows)

    jvp = scattering_functional.scattering_table_eval_jvp(
        *(case[n] for n in _TABLE_ARGS),
        tangent_wi=tangents["wi"], tangent_wo=tangents["wo"],
        tangent_f_te=tangents["f_te"], tangent_f_tm=tangents["f_tm"],
    )
    lhs = (g_te * jvp["tangent_f_te"]).sum() + (g_tm * jvp["tangent_f_tm"]).sum()

    vjp = scattering_functional.scattering_table_eval_backward(
        *(case[n] for n in _TABLE_ARGS),
        grad_out_f_te=g_te, grad_out_f_tm=g_tm,
        need_grad_dirs=True, need_grad_tables=True,
    )
    rhs = torch.zeros((), device="cuda", dtype=torch.float64)
    for key, name in (
        ("grad_wi", "wi"), ("grad_wo", "wo"),
        ("grad_f_te", "f_te"), ("grad_f_tm", "f_tm"),
    ):
        rhs = rhs + (vjp[key].double() * tangents[name].double()).sum()
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_ACCUM


def test_table_eval_jvp_matches_native_forward_fd():
    case = _table_case(seed=323)
    generator = torch.Generator(device="cuda").manual_seed(818)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)

    tangents = {name: randn(*case[name].shape) for name in _TABLE_LIVE_ARGS}
    jvp = scattering_functional.scattering_table_eval_jvp(
        *(case[n] for n in _TABLE_ARGS),
        tangent_wi=tangents["wi"], tangent_wo=tangents["wo"],
        tangent_f_te=tangents["f_te"], tangent_f_tm=tangents["f_tm"],
    )

    def forward_at(step):
        shifted = [
            case[name] + step * tangents[name] for name in _TABLE_LIVE_ARGS
        ]
        return scattering_functional.scattering_table_eval(case["valid"], *shifted)

    plus = forward_at(_FD_STEP)
    minus = forward_at(-_FD_STEP)
    for idx, t_name in ((0, "tangent_f_te"), (1, "tangent_f_tm")):
        fd = (plus[idx] - minus[idx]) / (2.0 * _FD_STEP)
        assert (
            relative_error(jvp[t_name], fd, abs_floor=ABS_TOL) <= _REL_TOL_FD_FORWARD
        ), t_name


def test_table_eval_backward_need_flags_gate_outputs():
    case = _table_case(seed=331)
    rows = case["wi"].shape[0]
    out = scattering_functional.scattering_table_eval_backward(
        *(case[n] for n in _TABLE_ARGS),
        grad_out_f_te=torch.ones(rows, device="cuda"),
        grad_out_f_tm=None,
        need_grad_dirs=False,
        need_grad_tables=True,
    )
    assert out["grad_wi"] is None
    assert out["grad_wo"] is None
    assert out["grad_f_te"] is not None
    assert out["grad_f_tm"] is not None
    # The tm cotangent is absent, so its table adjoint is exactly zero.
    assert float(out["grad_f_tm"].abs().sum()) == 0.0


def test_table_eval_backward_rejects_wrong_dtype():
    case = _table_case(seed=333)
    rows = case["wi"].shape[0]
    bad = dict(case)
    bad["f_te"] = case["f_te"].double()
    with pytest.raises((TypeError, ValueError)):
        scattering_functional.scattering_table_eval_backward(
            *(bad[n] for n in _TABLE_ARGS),
            grad_out_f_te=torch.ones(rows, device="cuda"),
            need_grad_dirs=True,
            need_grad_tables=True,
        )


def test_table_eval_backward_empty_rows():
    case = _table_case(seed=337, rows=0)
    out = scattering_functional.scattering_table_eval_backward(
        *(case[n] for n in _TABLE_ARGS),
        grad_out_f_te=torch.zeros(0, device="cuda"),
        grad_out_f_tm=torch.zeros(0, device="cuda"),
        need_grad_dirs=True,
        need_grad_tables=True,
    )
    assert out["grad_wi"].shape == (0, 3)
    assert out["grad_wo"].shape == (0, 3)
    # No row touches a table entry -> the scatter adjoints are all zero.
    assert float(out["grad_f_te"].abs().sum()) == 0.0
    assert float(out["grad_f_tm"].abs().sum()) == 0.0


def test_table_eval_horizon_rows_are_gated():
    # Rows below the horizon (cos <= 0) contribute zero value and no gradient.
    case = _table_case(seed=341)
    case["wo"][:4, 2] = -0.1
    te, tm = scattering_functional.scattering_table_eval(*(case[n] for n in _TABLE_ARGS))
    assert torch.all(te[:4] == 0.0)
    assert torch.all(tm[:4] == 0.0)

    rows = case["wi"].shape[0]
    out = scattering_functional.scattering_table_eval_backward(
        *(case[n] for n in _TABLE_ARGS),
        grad_out_f_te=torch.ones(rows, device="cuda"),
        grad_out_f_tm=torch.ones(rows, device="cuda"),
        need_grad_dirs=True,
        need_grad_tables=True,
    )
    # Gated rows store exactly zero direction gradients.
    assert torch.all(out["grad_wi"][:4] == 0.0)
    assert torch.all(out["grad_wo"][:4] == 0.0)
    te_ref, tm_ref, leaves = _table_reference(case, make_leaves=True)
    (te_ref + tm_ref).sum().backward()
    assert (
        relative_error(out["grad_f_te"], leaves["f_te"].grad, abs_floor=ABS_TOL)
        <= _REL_TOL_ACCUM
    )


def test_table_eval_backward_requires_native_kernel(monkeypatch):
    case = _table_case(seed=343)
    rows = case["wi"].shape[0]
    monkeypatch.setattr(runtime, "native_extension", lambda: None)
    with pytest.raises(RuntimeError, match="CUDA kernel is required"):
        scattering_functional.scattering_table_eval_backward(
            *(case[n] for n in _TABLE_ARGS),
            grad_out_f_te=torch.ones(rows, device="cuda"),
            need_grad_dirs=True,
            need_grad_tables=True,
        )


def test_table_eval_ad_wrapper_matches_functional_backward():
    case = _table_case(seed=347)
    leaves = {
        name: case[name].clone().requires_grad_(True) for name in _TABLE_LIVE_ARGS
    }
    te, tm = scattering_autograd.scattering_table_eval_ad(
        case["valid"], *(leaves[name] for name in _TABLE_LIVE_ARGS)
    )
    generator = torch.Generator(device="cuda").manual_seed(543)
    rows = case["wi"].shape[0]
    g_te = torch.randn(rows, generator=generator, device="cuda")
    g_tm = torch.randn(rows, generator=generator, device="cuda")
    (g_te * te + g_tm * tm).sum().backward()
    for name in _TABLE_LIVE_ARGS:
        assert leaves[name].grad is not None, name
        assert torch.isfinite(leaves[name].grad).all(), name


def test_table_eval_ad_mode_none_has_no_autograd_function():
    case = _table_case(seed=349)
    te, tm = scattering_functional.scattering_table_eval(*(case[n] for n in _TABLE_ARGS))
    assert not te.requires_grad
    assert not tm.requires_grad