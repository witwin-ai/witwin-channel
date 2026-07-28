"""AD lockstep for the ADR-021 chain scattering ops (plan 10a s3/s4).

Native float32 companions of Op A (``scattering_chain_ensemble_eval``, power)
and Op B (``scattering_chain_realization_eval``, coherent) versus the committed
float64 Torch oracles ``tests.reference.chain_ensemble`` /
``tests.reference.chain_realization``, plus native-only self-consistency
(JVP-vs-VJP duality, JVP-vs-forward finite difference) and the plan-07 AD
wrapper contract (loud rejection of fixed inputs, no tape under ``ad="none"``).
These run after the supervisor rebuilds the extension with the ADR-021 kernels.

Convention bridge (frozen plan 10a s3/s4 vs the oracle parametrization):

* The oracle DERIVES the vertex directions ``d_i``/``d_o``, ``cos_i``/``cos_o``
  and ``wi_local`` from the chain endpoints, while the native op takes them as
  explicit fixed-winner inputs. The fixture builds the native inputs from the
  same geometry so the forwards agree, but the gradients w.r.t. those derived
  directions and w.r.t. ``L1``/``L2``/``sp1``/``sp2``/positions do NOT
  correspond one-to-one between the two parametrizations; those are covered by
  the native-forward finite-difference cross-check, not the oracle lockstep.
* The oracle chain legs carry a per-bounce rough ``C_r`` (``sigma_b``/``rough``)
  that the frozen native leg block omits; the fixture sets ``rough=False`` so
  the two agree (open issue: the native Op A leg block has no ``sigma_b`` slot).
* Op A carries no ``weights``/``A_patch`` input; the fixture sets the oracle
  ``weights=1`` and ``sp = 1/L`` (planar image theory). The per-bounce material
  lockstep uses the prefactor-cancelling ratio ``grad/gain`` so it is robust to
  the exact spreading exponent the kernel applies.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.ad._fd import relative_error
from tests.ad._tolerances import ABS_TOL
from tests.reference import chain_ensemble as ref_a
from tests.reference import chain_realization as ref_b
from witwin.channel.kernels import materials as materials_functional
from witwin.channel.kernels import scattering as chain_autograd
from witwin.channel.kernels import scattering as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for chain scattering AD"
)

_DMAX = 8
_C0 = 299792458.0
_FREQ = 3.0e9
_REL_TOL_DIRECT = 5.0e-3
_REL_TOL_ACCUM = 1.0e-2
_REL_TOL_FD = 5.0e-2
_FD_STEP = 5.0e-4


def _unit(v):
    return torch.nn.functional.normalize(v, dim=-1)


def _pad_leg(oracle_leg, dmax, device):
    """Pad an oracle ``[N, d, ...]`` leg into the native ``[N, Dmax, ...]`` block."""

    depth = oracle_leg["positions"].shape[1]
    rows = oracle_leg["positions"].shape[0]

    def pad2(name, fill):
        block = torch.full((rows, dmax), fill, dtype=torch.float32, device=device)
        if depth:
            block[:, :depth] = oracle_leg[name].to(torch.float32)
        return block.contiguous()

    positions = torch.zeros(rows, dmax, 3, device=device)
    normals = torch.zeros(rows, dmax, 3, device=device)
    if depth:
        positions[:, :depth] = oracle_leg["positions"].to(torch.float32)
        normals[:, :depth] = oracle_leg["normals"].to(torch.float32)
    return {
        "positions": positions.contiguous(),
        "normals": normals.contiguous(),
        "eps_r": pad2("eps_r", 1.0),
        "sigma_e": pad2("sigma_e", 0.0),
        "mu_r": pad2("mu_r", 1.0),
        "gain": pad2("gain", 1.0),
        "thickness": pad2("thickness", 0.0),
        "depth": torch.full((rows,), depth, dtype=torch.int32, device=device),
    }


def _oracle_leg(generator, rows, depth, device, dtype):
    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=device, dtype=dtype)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=dtype)

    return {
        "positions": (randn(rows, depth, 3) * 0.4).contiguous(),
        "normals": _unit(randn(rows, depth, 3)).contiguous(),
        "eps_r": (1.5 + 3.0 * rand(rows, depth)).contiguous(),
        "sigma_e": (0.01 + 0.05 * rand(rows, depth)).contiguous(),
        "mu_r": torch.ones(rows, depth, device=device, dtype=dtype),
        "gain": torch.ones(rows, depth, device=device, dtype=dtype),
        "thickness": (0.05 + 0.1 * rand(rows, depth)).contiguous(),
        "sigma_b": torch.zeros(rows, depth, device=device, dtype=dtype),
        "rough": torch.zeros(rows, depth, device=device, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Op A geometry / native-input construction.
# ---------------------------------------------------------------------------


def _ensemble_geo(*, seed, rows=10, d1=1, d2=1, device="cuda"):
    """Build a shared geometry: oracle leaves + the derived native inputs."""

    dtype = torch.float64
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=dtype)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=device, dtype=dtype)

    n_o = _unit(randn(rows, 3) * 0.15 + torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype))
    t1r = _unit(torch.cross(n_o, randn(rows, 3), dim=-1))
    t2r = torch.cross(n_o, t1r, dim=-1)
    backup = t1r.clone()
    vertex = torch.zeros(rows, 3, device=device, dtype=dtype)
    vertex[:, :2] = randn(rows, 2) * 0.3
    source = vertex + randn(rows, 3) * 0.3 + torch.tensor([0.0, 0.0, 2.5], device=device, dtype=dtype)
    target = vertex + randn(rows, 3) * 0.3 + torch.tensor([0.0, 0.0, 3.0], device=device, dtype=dtype)
    tx_pol = _unit(randn(rows, 3))
    rx_pol = _unit(randn(rows, 3))

    c1 = _oracle_leg(generator, rows, d1, device, dtype)
    c2 = _oracle_leg(generator, rows, d2, device, dtype)
    # Keep the reflection chain broadly forward-scattering so every row lands
    # above the horizon (positive cosines) for a clean lockstep.
    if d1:
        c1["positions"] = vertex[:, None, :] + torch.tensor([0.0, 0.0, 1.2], device=device, dtype=dtype) + randn(rows, d1, 3) * 0.2
    if d2:
        c2["positions"] = vertex[:, None, :] + torch.tensor([0.0, 0.0, 1.2], device=device, dtype=dtype) + randn(rows, d2, 3) * 0.2

    l1 = torch.linalg.vector_norm(vertex - source, dim=-1)
    l2 = torch.linalg.vector_norm(target - vertex, dim=-1)

    # Derived native vertex inputs (mirror the oracle's internal derivation).
    prev1 = c1["positions"][:, -1] if d1 else source
    d_i = _unit(vertex - prev1)
    next2 = c2["positions"][:, 0] if d2 else target
    d_o = _unit(next2 - vertex)
    wi_hat = -d_i
    cos_i = (wi_hat * n_o).sum(-1)
    cos_o = (d_o * n_o).sum(-1)
    wi_local = torch.stack(
        ((wi_hat * t1r).sum(-1), (wi_hat * t2r).sum(-1), cos_i), dim=-1
    )
    return {
        "n_o": n_o, "t1r": t1r, "t2r": t2r, "backup": backup, "vertex": vertex,
        "source": source, "target": target, "tx_pol": tx_pol, "rx_pol": rx_pol,
        "c1": c1, "c2": c2, "l1": l1, "l2": l2, "d_i": d_i, "d_o": d_o,
        "cos_i": cos_i, "cos_o": cos_o, "wi_local": wi_local, "rows": rows,
        "d1": d1, "d2": d2, "device": device,
    }


def _ensemble_table(device, nti=6, npi=1, nto=6, npo=8, seed=7):
    generator = torch.Generator(device=device).manual_seed(seed)
    f_te = 0.2 + torch.rand(nti, npi, nto, npo, generator=generator, device=device, dtype=torch.float64)
    f_tm = 0.2 + torch.rand(nti, npi, nto, npo, generator=generator, device=device, dtype=torch.float64)
    return f_te, f_tm, (nti, npi, nto, npo)


def _ensemble_native_args(geo, f_te_flat, f_tm_flat, table_dims, *, dtype=torch.float32):
    device = geo["device"]
    c1 = _pad_leg(geo["c1"], _DMAX, device)
    c2 = _pad_leg(geo["c2"], _DMAX, device)
    rows = geo["rows"]

    def f32(x):
        return x.to(dtype).contiguous()

    # weights = 1 matches the oracle's A_patch = 1 convention (the 1/(L1^2 L2^2)
    # spreading is applied in-kernel).
    args = {
        "valid": torch.ones(rows, dtype=torch.bool, device=device),
        "tx_pol": f32(geo["tx_pol"]), "rx_pol": f32(geo["rx_pol"]),
        "source": f32(geo["source"]), "vertex": f32(geo["vertex"]),
        "target": f32(geo["target"]),
        "c1": c1, "c2": c2,
        "d_i": f32(geo["d_i"]), "d_o": f32(geo["d_o"]),
        "n_o": f32(geo["n_o"]), "t1r": f32(geo["t1r"]),
        "t2r": f32(geo["t2r"]), "backup_axis": f32(geo["backup"]),
        "cos_i": f32(geo["cos_i"]), "cos_o": f32(geo["cos_o"]),
        "L1": f32(geo["l1"]), "L2": f32(geo["l2"]),
        "weights": torch.ones(rows, device=device, dtype=dtype).to(dtype).contiguous(),
        "wi_local": f32(geo["wi_local"]),
        "material_id": torch.zeros(rows, dtype=torch.int32, device=device),
        "f_te_flat": f_te_flat.to(dtype).contiguous(),
        "f_tm_flat": f_tm_flat.to(dtype).contiguous(),
        "table_offset": torch.zeros(1, dtype=torch.int64, device=device),
        "table_dims": torch.tensor([list(table_dims)], dtype=torch.int32, device=device),
        "material_slot": torch.zeros(1, dtype=torch.int32, device=device),
    }
    return args


def _ensemble_positional(args):
    c1, c2 = args["c1"], args["c2"]
    return (
        args["valid"],
        args["tx_pol"], args["rx_pol"],
        args["source"], args["vertex"], args["target"],
        c1["positions"], c1["normals"], c1["eps_r"], c1["sigma_e"], c1["mu_r"],
        c1["gain"], c1["thickness"], c1["depth"],
        c2["positions"], c2["normals"], c2["eps_r"], c2["sigma_e"], c2["mu_r"],
        c2["gain"], c2["thickness"], c2["depth"],
        args["n_o"], args["t1r"], args["t2r"], args["backup_axis"],
        args["wi_local"], args["cos_i"], args["cos_o"], args["d_i"], args["d_o"],
        args["L1"], args["L2"], args["weights"],
        args["material_id"], args["f_te_flat"], args["f_tm_flat"],
        args["table_offset"], args["table_dims"], args["material_slot"],
    )


def _oracle_ensemble(geo, f_te, f_tm, *, threshold=-1.0):
    return ref_a.chain_ensemble_gain_reference(
        geo["source"], geo["tx_pol"], geo["c1"], geo["vertex"], geo["n_o"],
        geo["t1r"], geo["t2r"], geo["backup"], f_te, f_tm, geo["c2"],
        geo["target"], geo["rx_pol"], geo["l1"], geo["l2"],
        torch.ones(geo["rows"], device=geo["device"], dtype=torch.float64),
        torch.tensor(1.0, device=geo["device"], dtype=torch.float64),
        torch.tensor(_FREQ, device=geo["device"], dtype=torch.float64),
        threshold,
    )


# ---------------------------------------------------------------------------
# Op A tests.
# ---------------------------------------------------------------------------


def test_chain_ensemble_forward_matches_oracle():
    geo = _ensemble_geo(seed=11)
    f_te, f_tm, dims = _ensemble_table(geo["device"])
    args = _ensemble_native_args(geo, f_te.reshape(-1), f_tm.reshape(-1), dims)
    native = F.scattering_chain_ensemble_eval(
        *_ensemble_positional(args), coef=1.0, threshold=-1.0, frequency_hz=_FREQ
    )
    ref = _oracle_ensemble(geo, f_te, f_tm)
    # gain matches up to the (documented) spreading/weights convention; length
    # is exact.
    torch.testing.assert_close(
        native["length"].double(), (geo["l1"] + geo["l2"]), rtol=1e-4, atol=1e-6
    )
    assert relative_error(native["gain"], ref["gain"], abs_floor=ABS_TOL) <= _REL_TOL_ACCUM


def test_chain_ensemble_backward_material_ratio_lockstep():
    # Prefactor-robust lockstep: grad(gain)/gain w.r.t. each per-bounce Fresnel
    # parameter matches the float64 oracle (cancels the unknown spreading scale).
    geo = _ensemble_geo(seed=23)
    f_te, f_tm, dims = _ensemble_table(geo["device"])
    args = _ensemble_native_args(geo, f_te.reshape(-1), f_tm.reshape(-1), dims)

    native_fwd = F.scattering_chain_ensemble_eval(
        *_ensemble_positional(args), coef=1.0, threshold=-1.0, frequency_hz=_FREQ
    )
    rows = geo["rows"]
    native = F.scattering_chain_ensemble_eval_backward(
        *_ensemble_positional(args), coef=1.0, threshold=-1.0, frequency_hz=_FREQ,
        grad_gain=torch.ones(rows, device="cuda"),
        need_grad_chain1=True, need_grad_chain2=True,
        need_grad_tables=False, need_grad_geometry=False,
        need_grad_coef=False, need_grad_frequency=False,
    )

    leaves = {}
    for leg in ("c1", "c2"):
        for name in ("eps_r", "sigma_e", "gain", "thickness"):
            key = f"{leg}_{name}"
            leaf = geo[leg][name].clone().requires_grad_(True)
            leaves[key] = leaf
            geo[leg][name] = leaf
    ref = _oracle_ensemble(geo, f_te, f_tm)
    ref["gain"].sum().backward()

    gain_native = native_fwd["gain"].double().clamp_min(1e-30)
    gain_ref = ref["gain"].detach().clamp_min(1e-30)
    for leg, d in (("c1", geo["d1"]), ("c2", geo["d2"])):
        if d == 0:
            continue
        for name in ("eps_r", "sigma_e", "gain", "thickness"):
            native_ratio = native[f"grad_{leg}_{name}"][:, :d].double() / gain_native[:, None]
            ref_ratio = leaves[f"{leg}_{name}"].grad / gain_ref[:, None]
            assert relative_error(native_ratio, ref_ratio, abs_floor=ABS_TOL) <= _REL_TOL_ACCUM, (leg, name)


def test_chain_ensemble_jvp_matches_forward_fd():
    geo = _ensemble_geo(seed=31)
    f_te, f_tm, dims = _ensemble_table(geo["device"])
    args = _ensemble_native_args(geo, f_te.reshape(-1), f_tm.reshape(-1), dims)
    generator = torch.Generator(device="cuda").manual_seed(88)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)

    # Tangents on the geometry leaves the oracle lockstep cannot reach.
    t_di = randn(geo["rows"], 3)
    t_do = randn(geo["rows"], 3)
    t_L1 = randn(geo["rows"])
    jvp = F.scattering_chain_ensemble_eval_jvp(
        *_ensemble_positional(args), coef=1.0, threshold=-1.0, frequency_hz=_FREQ,
        tangent_d_i=t_di, tangent_d_o=t_do, tangent_l1=t_L1,
    )

    def forward_at(step):
        shifted = dict(args)
        shifted["d_i"] = args["d_i"] + step * t_di
        shifted["d_o"] = args["d_o"] + step * t_do
        shifted["L1"] = args["L1"] + step * t_L1
        return F.scattering_chain_ensemble_eval(
            *_ensemble_positional(shifted), coef=1.0, threshold=-1.0, frequency_hz=_FREQ
        )

    plus = forward_at(_FD_STEP)
    minus = forward_at(-_FD_STEP)
    for out_name, t_name in (("gain", "tangent_gain"), ("length", "tangent_length")):
        fd = (plus[out_name] - minus[out_name]) / (2.0 * _FD_STEP)
        assert relative_error(jvp[t_name], fd, abs_floor=ABS_TOL) <= _REL_TOL_FD, out_name


def test_chain_ensemble_jvp_vjp_inner_product():
    geo = _ensemble_geo(seed=42)
    f_te, f_tm, dims = _ensemble_table(geo["device"])
    args = _ensemble_native_args(geo, f_te.reshape(-1), f_tm.reshape(-1), dims)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    rows = geo["rows"]

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)

    tangents = {
        "tangent_c1_eps_r": randn(rows, _DMAX),
        "tangent_c2_eps_r": randn(rows, _DMAX),
        "tangent_c1_thickness": randn(rows, _DMAX),
    }
    t_f_te = randn(*args["f_te_flat"].shape)
    t_f_tm = randn(*args["f_tm_flat"].shape)
    t_coef = float(randn(1))
    g_gain, g_amp, g_len = randn(rows), randn(rows), randn(rows)

    jvp = F.scattering_chain_ensemble_eval_jvp(
        *_ensemble_positional(args), coef=1.0, threshold=-1.0, frequency_hz=_FREQ,
        tangent_f_te_flat=t_f_te, tangent_f_tm_flat=t_f_tm, tangent_coef=t_coef,
        **tangents,
    )
    lhs = (
        (g_gain * jvp["tangent_gain"]).sum()
        + (g_amp * jvp["tangent_amplitude"]).sum()
        + (g_len * jvp["tangent_length"]).sum()
    )
    vjp = F.scattering_chain_ensemble_eval_backward(
        *_ensemble_positional(args), coef=1.0, threshold=-1.0, frequency_hz=_FREQ,
        grad_gain=g_gain, grad_amplitude=g_amp, grad_length=g_len,
        need_grad_chain1=True, need_grad_chain2=True, need_grad_tables=True,
        need_grad_geometry=False, need_grad_coef=True, need_grad_frequency=False,
    )
    rhs = (vjp["grad_c1_eps_r"] * tangents["tangent_c1_eps_r"]).double().sum()
    rhs = rhs + (vjp["grad_c2_eps_r"] * tangents["tangent_c2_eps_r"]).double().sum()
    rhs = rhs + (vjp["grad_c1_thickness"] * tangents["tangent_c1_thickness"]).double().sum()
    rhs = rhs + (vjp["grad_f_te"].double() * t_f_te.double()).sum()
    rhs = rhs + (vjp["grad_f_tm"].double() * t_f_tm.double()).sum()
    rhs = rhs + vjp["grad_coef"].double().reshape(()) * t_coef
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_ACCUM


def test_chain_ensemble_ad_wrapper_and_fixed_rejection():
    geo = _ensemble_geo(seed=101)
    f_te, f_tm, dims = _ensemble_table(geo["device"])
    args = _ensemble_native_args(geo, f_te.reshape(-1), f_tm.reshape(-1), dims)
    device = "cuda"

    c1_eps = args["c1"]["eps_r"].clone().requires_grad_(True)
    args["c1"]["eps_r"] = c1_eps
    coef = torch.tensor(1.0, device=device, requires_grad=True)
    freq = torch.tensor(_FREQ, device=device)
    out = chain_autograd.scattering_chain_ensemble_eval_ad(
        *_ensemble_positional(args), coef=coef, threshold=-1.0, frequency=freq
    )
    out["gain"].sum().backward()
    assert c1_eps.grad is not None and torch.isfinite(c1_eps.grad).all()
    assert coef.grad is not None

    # tx_pol is a fixed input; requesting its gradient must fail loudly.
    args2 = _ensemble_native_args(geo, f_te.reshape(-1), f_tm.reshape(-1), dims)
    args2["tx_pol"] = args2["tx_pol"].clone().requires_grad_(True)
    out2 = chain_autograd.scattering_chain_ensemble_eval_ad(
        *_ensemble_positional(args2), coef=1.0, threshold=-1.0, frequency=freq
    )
    with pytest.raises(NotImplementedError):
        out2["gain"].sum().backward()


def test_chain_ensemble_ad_mode_none_has_no_tape():
    geo = _ensemble_geo(seed=105)
    f_te, f_tm, dims = _ensemble_table(geo["device"])
    args = _ensemble_native_args(geo, f_te.reshape(-1), f_tm.reshape(-1), dims)
    out = F.scattering_chain_ensemble_eval(
        *_ensemble_positional(args), coef=1.0, threshold=-1.0, frequency_hz=_FREQ
    )
    assert not out["gain"].requires_grad


# ---------------------------------------------------------------------------
# Op B geometry / native-input construction.
# ---------------------------------------------------------------------------


def _flat_plate(grid, extent, device, dtype):
    xs = torch.linspace(-extent, extent, grid + 1, device=device, dtype=dtype)
    tris, uvs = [], []
    for i in range(grid):
        for j in range(grid):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = xs[j], xs[j + 1]
            z = torch.zeros((), device=device, dtype=dtype)
            tris.append(torch.stack((
                torch.stack((x0, y0, z)), torch.stack((x1, y0, z)), torch.stack((x0, y1, z)),
            )))
            u0 = (x0 + extent) / (2 * extent)
            u1 = (x1 + extent) / (2 * extent)
            v0 = (y0 + extent) / (2 * extent)
            v1 = (y1 + extent) / (2 * extent)
            uvs.append(torch.stack((
                torch.stack((u0, v0)), torch.stack((u1, v0)), torch.stack((u0, v1)),
            )))
    return torch.stack(tris).contiguous(), torch.stack(uvs).contiguous()


def _realization_geo(*, seed, d1=1, d2=1, device="cuda", grid=6):
    dtype = torch.float64
    generator = torch.Generator(device=device).manual_seed(seed)
    patch_tris, patch_uvs = _flat_plate(grid, 0.4, device, dtype)
    p = patch_tris.shape[0]
    rows_idx = torch.arange(p, device=device, dtype=torch.int64)
    centroids = patch_tris.mean(dim=1).contiguous()
    n_rows = torch.zeros(p, 3, device=device, dtype=dtype)
    n_rows[:, 2] = 1.0
    # Non-zero heights + non-specular outgoing direction keep path_gain
    # first-order sensitive to the height/centroid tangents (a flat plate in
    # the exact specular direction makes the per-row gain tangents pure
    # roundoff and the FD check meaningless) and exercise the bilinear
    # height-sampling path in the oracle-vs-native parity tests.
    heights = 0.02 * torch.randn(48, 48, device=device, dtype=dtype, generator=generator)

    theta = math.radians(28.0)
    theta_o = math.radians(34.0)
    d_i = torch.tensor([math.sin(theta), 0.0, -math.cos(theta)], device=device, dtype=dtype).expand(p, 3).contiguous()
    d_o = torch.tensor([math.sin(theta_o), 0.0, math.cos(theta_o)], device=device, dtype=dtype).expand(p, 3).contiguous()
    vertex = centroids
    source = torch.tensor([-1.2, 0.0, 1.5], device=device, dtype=dtype).expand(p, 3).contiguous()
    target = torch.tensor([1.2, 0.0, 1.5], device=device, dtype=dtype).expand(p, 3).contiguous()

    def leg(depth):
        if depth == 0:
            return ref_a._empty_chain(p, device, dtype)
        return {
            "positions": (vertex[:, None, :] + torch.tensor([0.5, 0.0, 0.9], device=device, dtype=dtype)).expand(p, depth, 3).contiguous(),
            "normals": torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(p, depth, 3).contiguous(),
            "eps_r": torch.full((p, depth), 4.0, device=device, dtype=dtype),
            "sigma_e": torch.full((p, depth), 0.02, device=device, dtype=dtype),
            "mu_r": torch.ones(p, depth, device=device, dtype=dtype),
            "gain": torch.ones(p, depth, device=device, dtype=dtype),
            "thickness": torch.full((p, depth), 0.1, device=device, dtype=dtype),
            "sigma_b": torch.zeros(p, depth, device=device, dtype=dtype),
            "rough": torch.zeros(p, depth, device=device, dtype=torch.bool),
        }

    l1 = torch.full((p,), 2.0, device=device, dtype=dtype)
    l2 = torch.full((p,), 1.8, device=device, dtype=dtype)
    cos_spec = torch.full((p,), math.cos(theta), device=device, dtype=dtype)
    return {
        "patch_tris": patch_tris, "patch_uvs": patch_uvs, "rows": rows_idx,
        "centroids": centroids, "n_rows": n_rows, "heights": heights,
        "d_i": d_i, "d_o": d_o, "vertex": vertex, "source": source, "target": target,
        "c1": leg(d1), "c2": leg(d2), "l1": l1, "l2": l2, "cos_spec": cos_spec,
        "tx_pol": torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand(p, 3).contiguous(),
        "rx_pol": torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand(p, 3).contiguous(),
        "p": p, "d1": d1, "d2": d2, "device": device, "generator": generator,
    }


def _layer_csr(device):
    return {
        "material_id": None,  # filled per row below
        "layer_offset": torch.zeros(1, dtype=torch.int32, device=device),
        "layer_count": torch.ones(1, dtype=torch.int32, device=device),
        "layer_thickness_m": torch.tensor([0.1], device=device),
        "layer_eps_r": torch.tensor([4.0], device=device),
        "layer_sigma_e": torch.tensor([0.02], device=device),
        "layer_mu_r": torch.tensor([1.0], device=device),
    }


def _stack_rtte(geo, csr):
    """r_te/r_tm the native Op B computes in-kernel, via em_layer_stack_eval."""

    p = geo["p"]
    device = geo["device"]
    cos_spec = geo["cos_spec"].to(torch.float32).contiguous()
    material_id = torch.zeros(p, dtype=torch.int32, device=device)
    out = materials_functional.em_layer_stack_eval(
        cos_spec, material_id,
        csr["layer_offset"], csr["layer_count"], csr["layer_thickness_m"].to(torch.float32),
        csr["layer_eps_r"].to(torch.float32), csr["layer_sigma_e"].to(torch.float32),
        csr["layer_mu_r"].to(torch.float32),
        frequency_hz=_FREQ,
    )
    r_te = torch.complex(out["r_te_real"], out["r_te_imag"]).to(torch.complex128)
    r_tm = torch.complex(out["r_tm_real"], out["r_tm_imag"]).to(torch.complex128)
    return r_te, r_tm


def _realization_native_args(geo, csr, *, dtype=torch.float32):
    device = geo["device"]
    c1 = _pad_leg(geo["c1"], _DMAX, device)
    c2 = _pad_leg(geo["c2"], _DMAX, device)
    p = geo["p"]

    def f32(x):
        return x.to(dtype).contiguous()

    return {
        "valid": torch.ones(p, dtype=torch.bool, device=device),
        "patch_tris": f32(geo["patch_tris"]), "patch_uvs": f32(geo["patch_uvs"]),
        "rows": geo["rows"], "d_i": f32(geo["d_i"]), "d_o": f32(geo["d_o"]),
        "n_rows": f32(geo["n_rows"]),
        "source": f32(geo["source"]), "vertex": f32(geo["vertex"]),
        "target": f32(geo["target"]),
        "c1": c1, "c2": c2,
        "tx_pol": f32(geo["tx_pol"]), "rx_pol": f32(geo["rx_pol"]),
        "L1": f32(geo["l1"]), "L2": f32(geo["l2"]),
        "sp1": f32(1.0 / geo["l1"]), "sp2": f32(1.0 / geo["l2"]),
        "centroids": f32(geo["centroids"]), "heights": f32(geo["heights"]),
        "cos_spec": f32(geo["cos_spec"]),
        "material_id": torch.zeros(p, dtype=torch.int32, device=device),
        "layer_offset": csr["layer_offset"], "layer_count": csr["layer_count"],
        "layer_thickness_m": csr["layer_thickness_m"].to(dtype).contiguous(),
        "layer_eps_r": csr["layer_eps_r"].to(dtype).contiguous(),
        "layer_sigma_e": csr["layer_sigma_e"].to(dtype).contiguous(),
        "layer_mu_r": csr["layer_mu_r"].to(dtype).contiguous(),
    }


def _realization_positional(args):
    c1, c2 = args["c1"], args["c2"]
    return (
        args["valid"],
        args["patch_tris"], args["patch_uvs"], args["rows"], args["d_i"], args["d_o"],
        args["n_rows"],
        args["source"], args["vertex"], args["target"],
        c1["positions"], c1["normals"], c1["eps_r"], c1["sigma_e"], c1["mu_r"],
        c1["gain"], c1["thickness"], c1["depth"],
        c2["positions"], c2["normals"], c2["eps_r"], c2["sigma_e"], c2["mu_r"],
        c2["gain"], c2["thickness"], c2["depth"],
        args["tx_pol"], args["rx_pol"], args["L1"], args["L2"], args["sp1"],
        args["sp2"], args["centroids"], args["heights"], args["cos_spec"],
        args["material_id"], args["layer_offset"], args["layer_count"],
        args["layer_thickness_m"], args["layer_eps_r"], args["layer_sigma_e"],
        args["layer_mu_r"],
    )


def _oracle_realization(geo, r_te, r_tm):
    quad_a, quad_b, quad_w = ref_b.duffy_gl_nodes(geo["device"], torch.float64)
    k0 = torch.tensor(2.0 * math.pi * _FREQ / _C0, device=geo["device"], dtype=torch.float64)
    return ref_b.chain_realization_eval(
        geo["heights"], geo["patch_tris"], geo["patch_uvs"], geo["rows"], geo["source"],
        geo["tx_pol"], geo["c1"], geo["vertex"], geo["n_rows"], r_te, r_tm,
        geo["d_i"], geo["d_o"], geo["c2"], geo["target"], geo["rx_pol"],
        geo["l1"], geo["l2"], geo["centroids"], quad_a, quad_b, quad_w, k0,
        torch.tensor(_FREQ, device=geo["device"], dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# Op B tests.
# ---------------------------------------------------------------------------


def _k0():
    return 2.0 * math.pi * _FREQ / _C0


def test_chain_realization_forward_matches_oracle():
    geo = _realization_geo(seed=31)
    csr = _layer_csr(geo["device"])
    r_te, r_tm = _stack_rtte(geo, csr)
    args = _realization_native_args(geo, csr)
    native = F.scattering_chain_realization_eval(
        *_realization_positional(args), k0=_k0(), frequency_hz=_FREQ
    )
    ref = _oracle_realization(geo, r_te, r_tm)
    assert relative_error(native["total"], ref["total"], abs_floor=ABS_TOL) <= _REL_TOL_ACCUM
    assert relative_error(native["row_value"], ref["row_value"], abs_floor=ABS_TOL) <= _REL_TOL_ACCUM


def test_chain_realization_backward_heights_and_centroids_lockstep():
    geo = _realization_geo(seed=41)
    csr = _layer_csr(geo["device"])
    r_te, r_tm = _stack_rtte(geo, csr)
    args = _realization_native_args(geo, csr)
    generator = torch.Generator(device="cuda").manual_seed(311)
    grad_total = torch.complex(
        torch.randn((), generator=generator, device="cuda"),
        torch.randn((), generator=generator, device="cuda"),
    ).to(torch.complex64)

    native = F.scattering_chain_realization_eval_backward(
        *_realization_positional(args), k0=_k0(), frequency_hz=_FREQ, grad_total=grad_total,
        need_grad_heights=True, need_grad_layers=False, need_grad_chain1=False,
        need_grad_chain2=False, need_grad_geometry=True, need_grad_k0=False,
        need_grad_frequency=False,
    )

    heights = geo["heights"].clone().requires_grad_(True)
    centroids = geo["centroids"].clone().requires_grad_(True)
    geo_leaf = dict(geo)
    geo_leaf["heights"] = heights
    geo_leaf["centroids"] = centroids
    ref = _oracle_realization(geo_leaf, r_te, r_tm)
    g = grad_total.to(torch.complex128)
    (g.real * ref["total"].real + g.imag * ref["total"].imag).backward()

    assert relative_error(native["grad_heights"], heights.grad, abs_floor=ABS_TOL) <= _REL_TOL_ACCUM
    assert relative_error(native["grad_centroids"], centroids.grad, abs_floor=ABS_TOL) <= _REL_TOL_DIRECT


def test_chain_realization_jvp_matches_forward_fd():
    geo = _realization_geo(seed=51)
    csr = _layer_csr(geo["device"])
    args = _realization_native_args(geo, csr)
    t_heights = torch.randn_like(args["heights"])
    t_centroids = torch.randn_like(args["centroids"])

    jvp = F.scattering_chain_realization_eval_jvp(
        *_realization_positional(args), k0=_k0(), frequency_hz=_FREQ,
        tangent_heights=t_heights, tangent_centroids=t_centroids,
    )

    def forward_at(step):
        shifted = dict(args)
        shifted["heights"] = args["heights"] + step * t_heights
        shifted["centroids"] = args["centroids"] + step * t_centroids
        return F.scattering_chain_realization_eval(
            *_realization_positional(shifted), k0=_k0(), frequency_hz=_FREQ
        )

    plus = forward_at(_FD_STEP)
    minus = forward_at(-_FD_STEP)
    fd_total = (plus["total"] - minus["total"]) / (2.0 * _FD_STEP)
    assert relative_error(jvp["tangent_total"], fd_total, abs_floor=ABS_TOL) <= _REL_TOL_FD
    fd_gain = (plus["path_gain"] - minus["path_gain"]) / (2.0 * _FD_STEP)
    assert relative_error(jvp["tangent_path_gain"], fd_gain, abs_floor=ABS_TOL) <= _REL_TOL_FD


def test_chain_realization_jvp_vjp_inner_product():
    geo = _realization_geo(seed=61)
    csr = _layer_csr(geo["device"])
    args = _realization_native_args(geo, csr)
    generator = torch.Generator(device="cuda").manual_seed(331)
    t_heights = torch.randn_like(args["heights"])
    t_centroids = torch.randn_like(args["centroids"])
    g = torch.complex(
        torch.randn((), generator=generator, device="cuda"),
        torch.randn((), generator=generator, device="cuda"),
    ).to(torch.complex64)

    jvp = F.scattering_chain_realization_eval_jvp(
        *_realization_positional(args), k0=_k0(), frequency_hz=_FREQ,
        tangent_heights=t_heights, tangent_centroids=t_centroids,
    )
    lhs = (g.conj() * jvp["tangent_total"]).real.to(torch.float64)

    vjp = F.scattering_chain_realization_eval_backward(
        *_realization_positional(args), k0=_k0(), frequency_hz=_FREQ, grad_total=g,
        need_grad_heights=True, need_grad_layers=False, need_grad_chain1=False,
        need_grad_chain2=False, need_grad_geometry=True, need_grad_k0=False,
        need_grad_frequency=False,
    )
    rhs = (vjp["grad_heights"].double() * t_heights.double()).sum()
    rhs = rhs + (vjp["grad_centroids"].double() * t_centroids.double()).sum()
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_ACCUM


def test_chain_realization_ad_wrapper_and_fixed_rejection():
    geo = _realization_geo(seed=71)
    csr = _layer_csr(geo["device"])
    args = _realization_native_args(geo, csr)
    device = "cuda"

    heights = args["heights"].clone().requires_grad_(True)
    args["heights"] = heights
    k0 = torch.tensor(_k0(), device=device, requires_grad=True)
    freq = torch.tensor(_FREQ, device=device)
    out = chain_autograd.scattering_chain_realization_eval_ad(
        *_realization_positional(args), k0=k0, frequency=freq
    )
    out["total"].real.backward()
    assert heights.grad is not None and torch.isfinite(heights.grad).all()
    assert k0.grad is not None

    # patch_tris is a fixed input; requesting its gradient must fail loudly.
    args2 = _realization_native_args(geo, csr)
    args2["patch_tris"] = args2["patch_tris"].clone().requires_grad_(True)
    out2 = chain_autograd.scattering_chain_realization_eval_ad(
        *_realization_positional(args2), k0=_k0(), frequency=freq
    )
    with pytest.raises(NotImplementedError):
        out2["total"].real.backward()


def test_chain_realization_ad_mode_none_has_no_tape():
    geo = _realization_geo(seed=73)
    csr = _layer_csr(geo["device"])
    args = _realization_native_args(geo, csr)
    out = F.scattering_chain_realization_eval(
        *_realization_positional(args), k0=_k0(), frequency_hz=_FREQ
    )
    assert not out["total"].requires_grad
