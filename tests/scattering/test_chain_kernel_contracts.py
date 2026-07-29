# Copyright Xingyu Chen.
# Tests chain kernel contracts.

"""Tests chain kernel contracts."""

from __future__ import annotations

import math

import pytest
import torch

from witwin.channel import runtime
from witwin.channel.kernels import scattering as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for chain scattering"
)

_DMAX = 8
_C0 = 299792458.0


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _leg(generator, rows, depth, device, *, dmax=_DMAX):
    """One padded specular leg block filled in its first ``depth`` slots."""

    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=device, dtype=torch.float32)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=torch.float32)

    positions = torch.zeros(rows, dmax, 3, device=device)
    normals = torch.zeros(rows, dmax, 3, device=device)
    if depth > 0:
        positions[:, :depth] = randn(rows, depth, 3)
        normals[:, :depth] = torch.nn.functional.normalize(randn(rows, depth, 3), dim=-1)
    eps_r = 1.0 + 3.0 * rand(rows, dmax)
    sigma_e = 0.01 + 0.05 * rand(rows, dmax)
    mu_r = torch.ones(rows, dmax, device=device)
    gain = torch.ones(rows, dmax, device=device)
    thickness = 0.05 + 0.1 * rand(rows, dmax)
    depth_t = torch.full((rows,), depth, dtype=torch.int32, device=device)
    return {
        "positions": positions.contiguous(),
        "normals": normals.contiguous(),
        "eps_r": eps_r.contiguous(),
        "sigma_e": sigma_e.contiguous(),
        "mu_r": mu_r,
        "gain": gain,
        "thickness": thickness.contiguous(),
        "depth": depth_t,
    }


def _chain_ensemble_case(
    *,
    device: str = "cuda",
    seed: int,
    rows: int = 12,
    d1: int = 1,
    d2: int = 1,
    nti: int = 6,
    npi: int = 1,
    nto: int = 6,
    npo: int = 8,
    coef: float = 3.0e-4,
    frequency_hz: float = 3.0e9,
):
    generator = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=device, dtype=torch.float32)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=torch.float32)

    def unit(*shape):
        return torch.nn.functional.normalize(randn(*shape), dim=-1)

    n_o = unit(rows, 3)
    t1r = torch.nn.functional.normalize(
        torch.cross(n_o, randn(rows, 3), dim=-1), dim=-1
    )
    t2r = torch.cross(n_o, t1r, dim=-1)
    backup_axis = t1r.clone()

    # Incident direction into the surface, outgoing away from it (above horizon).
    d_i = torch.nn.functional.normalize(
        randn(rows, 3) * 0.3 - n_o, dim=-1
    ).contiguous()
    d_o = torch.nn.functional.normalize(
        randn(rows, 3) * 0.3 + n_o, dim=-1
    ).contiguous()
    wi_hat = -d_i
    cos_i = (wi_hat * n_o).sum(-1).clamp_min(0.2)
    cos_o = (d_o * n_o).sum(-1).clamp_min(0.2)
    wi_local = torch.stack(
        ((wi_hat * t1r).sum(-1), (wi_hat * t2r).sum(-1), cos_i), dim=-1
    ).contiguous()

    # Endpoint positions feed the C1/C2 transport (source=tx, vertex=v_s,
    # target=rx). weights is the per-vertex patch area A_patch (op-1 convention).
    source = randn(rows, 3).contiguous()
    vertex = randn(rows, 3).contiguous()
    target = randn(rows, 3).contiguous()
    weights = (0.5 + rand(rows)).contiguous()

    L1 = 1.0 + rand(rows)
    L2 = 1.0 + rand(rows)

    f_te = 0.2 + rand(nti, npi, nto, npo)
    f_tm = 0.2 + rand(nti, npi, nto, npo)
    return {
        "valid": torch.ones(rows, dtype=torch.bool, device=device),
        "tx_pol": unit(rows, 3).contiguous(),
        "rx_pol": unit(rows, 3).contiguous(),
        "source": source,
        "vertex": vertex,
        "target": target,
        "c1": _leg(generator, rows, d1, device),
        "c2": _leg(generator, rows, d2, device),
        "d_i": d_i,
        "d_o": d_o,
        "n_o": n_o.contiguous(),
        "t1r": t1r.contiguous(),
        "t2r": t2r.contiguous(),
        "backup_axis": backup_axis.contiguous(),
        "wi_local": wi_local,
        "cos_i": cos_i.contiguous(),
        "cos_o": cos_o.contiguous(),
        "L1": L1.contiguous(),
        "L2": L2.contiguous(),
        "weights": weights,
        "material_id": torch.zeros(rows, dtype=torch.int32, device=device),
        "f_te_flat": f_te.reshape(-1).contiguous(),
        "f_tm_flat": f_tm.reshape(-1).contiguous(),
        "table_offset": torch.zeros(1, dtype=torch.int64, device=device),
        "table_dims": torch.tensor([[nti, npi, nto, npo]], dtype=torch.int32, device=device),
        "material_slot": torch.zeros(1, dtype=torch.int32, device=device),
        "coef": coef,
        "frequency_hz": frequency_hz,
        "rows": rows,
    }


def _ensemble_forward_args(case):
    c1, c2 = case["c1"], case["c2"]
    return (
        case["valid"],
        case["tx_pol"],
        case["rx_pol"],
        case["source"], case["vertex"], case["target"],
        c1["positions"], c1["normals"], c1["eps_r"], c1["sigma_e"], c1["mu_r"],
        c1["gain"], c1["thickness"], c1["depth"],
        c2["positions"], c2["normals"], c2["eps_r"], c2["sigma_e"], c2["mu_r"],
        c2["gain"], c2["thickness"], c2["depth"],
        case["n_o"], case["t1r"], case["t2r"], case["backup_axis"],
        case["wi_local"], case["cos_i"], case["cos_o"], case["d_i"], case["d_o"],
        case["L1"], case["L2"], case["weights"],
        case["material_id"], case["f_te_flat"], case["f_tm_flat"],
        case["table_offset"], case["table_dims"], case["material_slot"],
    )


def _ensemble_forward(case, threshold=-1.0):
    return F.scattering_chain_ensemble_eval(
        *_ensemble_forward_args(case),
        coef=case["coef"],
        threshold=threshold,
        frequency_hz=case["frequency_hz"],
    )


def _chain_realization_case(
    *,
    device: str = "cuda",
    seed: int,
    rows: int = 6,
    patches: int = 8,
    grid: int = 16,
    d1: int = 1,
    d2: int = 1,
    frequency_hz: float = 3.0e9,
):
    generator = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=device, dtype=torch.float32)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=torch.float32)

    tris, uvs = [], []
    for _ in range(patches):
        origin = rand(2)
        size = 0.1 + 0.2 * rand(2)
        z = torch.zeros((), device=device)
        p0 = torch.stack((origin[0], origin[1], z))
        p1 = torch.stack((origin[0] + size[0], origin[1], z))
        p2 = torch.stack((origin[0], origin[1] + size[1], z))
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
    d_i = torch.nn.functional.normalize(
        torch.stack((0.3 * randn(rows), 0.3 * randn(rows), -(0.4 + 0.4 * rand(rows))), dim=-1),
        dim=-1,
    ).contiguous()
    d_o = torch.nn.functional.normalize(
        torch.stack((0.3 * randn(rows), 0.3 * randn(rows), 0.4 + 0.4 * rand(rows)), dim=-1),
        dim=-1,
    ).contiguous()

    L1 = 1.0 + rand(rows)
    L2 = 1.0 + rand(rows)
    heights = (1.0e-3 * randn(grid, grid)).contiguous()
    k0 = 2.0 * math.pi * frequency_hz / _C0
    return {
        "valid": torch.ones(rows, dtype=torch.bool, device=device),
        "patch_tris": patch_tris,
        "patch_uvs": patch_uvs,
        "rows": row_index,
        "d_i": d_i,
        "d_o": d_o,
        "n_rows": n_rows.contiguous(),
        "source": randn(rows, 3).contiguous(),
        "vertex": centroids.clone(),
        "target": randn(rows, 3).contiguous(),
        "c1": _leg(generator, rows, d1, device),
        "c2": _leg(generator, rows, d2, device),
        "tx_pol": torch.nn.functional.normalize(randn(rows, 3), dim=-1).contiguous(),
        "rx_pol": torch.nn.functional.normalize(randn(rows, 3), dim=-1).contiguous(),
        "L1": L1.contiguous(),
        "L2": L2.contiguous(),
        "sp1": (1.0 / L1).contiguous(),
        "sp2": (1.0 / L2).contiguous(),
        "centroids": centroids,
        "heights": heights,
        "cos_spec": (0.3 + 0.6 * rand(rows)).contiguous(),
        "material_id": torch.zeros(rows, dtype=torch.int32, device=device),
        "layer_offset": torch.zeros(1, dtype=torch.int32, device=device),
        "layer_count": torch.ones(1, dtype=torch.int32, device=device),
        "layer_thickness_m": torch.tensor([0.1], device=device),
        "layer_eps_r": torch.tensor([4.0], device=device),
        "layer_sigma_e": torch.tensor([0.02], device=device),
        "layer_mu_r": torch.tensor([1.0], device=device),
        "k0": k0,
        "frequency_hz": frequency_hz,
        "n": rows,
    }


def _realization_forward_args(case):
    c1, c2 = case["c1"], case["c2"]
    return (
        case["valid"],
        case["patch_tris"], case["patch_uvs"], case["rows"], case["d_i"],
        case["d_o"], case["n_rows"],
        case["source"], case["vertex"], case["target"],
        c1["positions"], c1["normals"], c1["eps_r"], c1["sigma_e"], c1["mu_r"],
        c1["gain"], c1["thickness"], c1["depth"],
        c2["positions"], c2["normals"], c2["eps_r"], c2["sigma_e"], c2["mu_r"],
        c2["gain"], c2["thickness"], c2["depth"],
        case["tx_pol"], case["rx_pol"], case["L1"], case["L2"], case["sp1"],
        case["sp2"], case["centroids"], case["heights"], case["cos_spec"],
        case["material_id"], case["layer_offset"], case["layer_count"],
        case["layer_thickness_m"], case["layer_eps_r"], case["layer_sigma_e"],
        case["layer_mu_r"],
    )


def _realization_forward(case):
    return F.scattering_chain_realization_eval(
        *_realization_forward_args(case), k0=case["k0"], frequency_hz=case["frequency_hz"]
    )


# ---------------------------------------------------------------------------
# Op A: forward contract.
# ---------------------------------------------------------------------------


def test_chain_ensemble_forward_schema():
    case = _chain_ensemble_case(seed=11)
    out = _ensemble_forward(case, threshold=-1.0)
    assert set(out) == {"gain", "amplitude", "length", "keep"}
    rows = case["rows"]
    for name, dtype in (("gain", torch.float32), ("amplitude", torch.float32), ("length", torch.float32), ("keep", torch.bool)):
        assert out[name].shape == (rows,), name
        assert out[name].dtype == dtype, name
    torch.testing.assert_close(out["length"], case["L1"] + case["L2"])
    torch.testing.assert_close(out["amplitude"], out["gain"].clamp_min(0.0).sqrt())
    assert bool(torch.equal(out["keep"], out["gain"] > -1.0))


def test_chain_ensemble_valid_masks_outputs_inert():
    case = _chain_ensemble_case(seed=111, rows=6)
    case["valid"][1::2] = False
    out = _ensemble_forward(case, threshold=-1.0)
    invalid = ~case["valid"]
    for name in ("gain", "amplitude", "length"):
        assert torch.count_nonzero(out[name][invalid]) == 0
    assert not bool(out["keep"][invalid].any())


def test_chain_ensemble_rejects_non_bool_valid():
    case = _chain_ensemble_case(seed=112)
    case["valid"] = case["valid"].to(torch.int32)
    with pytest.raises((TypeError, ValueError), match="valid"):
        _ensemble_forward(case)


def test_chain_ensemble_forward_keep_threshold():
    case = _chain_ensemble_case(seed=12)
    big = float(_ensemble_forward(case)["gain"].max()) + 1.0
    out = _ensemble_forward(case, threshold=big)
    assert not bool(out["keep"].any())


def test_chain_ensemble_degenerate_rows_run():
    # d1 = d2 = 0: both legs collapse to identity transport (the op-1 limit).
    case = _chain_ensemble_case(seed=13, d1=0, d2=0)
    out = _ensemble_forward(case)
    assert set(out) == {"gain", "amplitude", "length", "keep"}
    assert torch.isfinite(out["gain"]).all()


def test_chain_ensemble_forward_rejects_bad_dtype():
    case = _chain_ensemble_case(seed=14)
    args = list(_ensemble_forward_args(case))
    args[1] = args[1].double()  # tx_pol wrong dtype
    with pytest.raises((TypeError, ValueError)):
        F.scattering_chain_ensemble_eval(
            *args, coef=case["coef"], threshold=-1.0, frequency_hz=case["frequency_hz"]
        )


def test_chain_ensemble_forward_rejects_depth_over_max():
    case = _chain_ensemble_case(seed=15)
    c1 = case["c1"]
    # A padded block wider than kMaxAdDepth must be rejected by the facade.
    c1["positions"] = torch.zeros(case["rows"], _DMAX + 1, 3, device="cuda")
    c1["normals"] = torch.zeros(case["rows"], _DMAX + 1, 3, device="cuda")
    c1["eps_r"] = torch.ones(case["rows"], _DMAX + 1, device="cuda")
    c1["sigma_e"] = torch.ones(case["rows"], _DMAX + 1, device="cuda")
    c1["mu_r"] = torch.ones(case["rows"], _DMAX + 1, device="cuda")
    c1["gain"] = torch.ones(case["rows"], _DMAX + 1, device="cuda")
    c1["thickness"] = torch.ones(case["rows"], _DMAX + 1, device="cuda")
    with pytest.raises(ValueError, match="kMaxAdDepth"):
        _ensemble_forward(case)


def test_chain_ensemble_forward_rejects_row_mismatch():
    case = _chain_ensemble_case(seed=16)
    case["cos_i"] = case["cos_i"][:-1]  # wrong row count
    with pytest.raises(ValueError):
        _ensemble_forward(case)


def test_chain_ensemble_requires_native_kernel(monkeypatch):
    case = _chain_ensemble_case(seed=17)
    monkeypatch.setattr(runtime, "native_extension", lambda: None)
    with pytest.raises(RuntimeError, match="CUDA kernel is required"):
        _ensemble_forward(case)


# ---------------------------------------------------------------------------
# Op A: backward / jvp contract.
# ---------------------------------------------------------------------------


def test_chain_ensemble_backward_schema_and_gating():
    case = _chain_ensemble_case(seed=21)
    rows = case["rows"]
    out = F.scattering_chain_ensemble_eval_backward(
        *_ensemble_forward_args(case),
        coef=case["coef"], threshold=-1.0, frequency_hz=case["frequency_hz"],
        grad_gain=torch.ones(rows, device="cuda"),
        need_grad_chain1=True, need_grad_chain2=False, need_grad_tables=False,
        need_grad_geometry=False, need_grad_coef=False, need_grad_frequency=False,
    )
    expected = set(F._CHAIN_ENSEMBLE_BACKWARD_FIELDS)
    assert set(out) == expected
    for key in ("grad_c1_eps_r", "grad_c1_sigma_e", "grad_c1_gain", "grad_c1_thickness"):
        assert out[key] is not None and out[key].shape == (rows, _DMAX), key
    # Reverse-mode chain geometry is not emitted this wave; only the twelve native
    # VJP fields exist, and the off-flag ones are None.
    for key in ("grad_c2_eps_r", "grad_f_te", "grad_coef", "grad_frequency"):
        assert out[key] is None, key


def test_chain_ensemble_backward_requires_native_kernel(monkeypatch):
    case = _chain_ensemble_case(seed=22)
    rows = case["rows"]
    monkeypatch.setattr(runtime, "native_extension", lambda: None)
    with pytest.raises(RuntimeError, match="CUDA kernel is required"):
        F.scattering_chain_ensemble_eval_backward(
            *_ensemble_forward_args(case),
            coef=case["coef"], threshold=-1.0, frequency_hz=case["frequency_hz"],
            grad_gain=torch.ones(rows, device="cuda"),
            need_grad_chain1=True,
        )


def test_chain_ensemble_jvp_schema():
    case = _chain_ensemble_case(seed=23)
    out = F.scattering_chain_ensemble_eval_jvp(
        *_ensemble_forward_args(case),
        coef=case["coef"], threshold=-1.0, frequency_hz=case["frequency_hz"],
        tangent_c1_eps_r=torch.randn(case["rows"], _DMAX, device="cuda"),
        tangent_coef=0.5,
    )
    assert set(out) == {"tangent_gain", "tangent_amplitude", "tangent_length"}
    for value in out.values():
        assert value.shape == (case["rows"],)
        assert value.dtype == torch.float32


# ---------------------------------------------------------------------------
# Op B: forward contract.
# ---------------------------------------------------------------------------


def test_chain_realization_forward_schema():
    case = _chain_realization_case(seed=31)
    out = _realization_forward(case)
    assert set(out) == {"total", "path_field", "path_gain", "integral", "row_value"}
    n = case["n"]
    assert out["total"].shape == () and out["total"].dtype == torch.complex64
    assert out["path_field"].shape == (n,) and out["path_field"].dtype == torch.complex64
    assert out["path_gain"].shape == (n,) and out["path_gain"].dtype == torch.float32
    assert out["integral"].shape == (n,) and out["integral"].dtype == torch.complex64
    assert out["row_value"].shape == (n,) and out["row_value"].dtype == torch.complex64
    torch.testing.assert_close(out["path_gain"], out["path_field"].abs().square())


def test_chain_realization_valid_masks_rows_inert():
    case = _chain_realization_case(seed=131, rows=6)
    case["valid"][1::2] = False
    out = _realization_forward(case)
    invalid = ~case["valid"]
    for name in ("path_field", "path_gain", "integral", "row_value"):
        assert torch.count_nonzero(out[name][invalid]) == 0


def test_chain_realization_rejects_wrong_valid_shape():
    case = _chain_realization_case(seed=132)
    case["valid"] = case["valid"][:-1]
    with pytest.raises(ValueError, match="valid must have shape"):
        _realization_forward(case)


def test_chain_realization_degenerate_rows_run():
    case = _chain_realization_case(seed=32, d1=0, d2=0)
    out = _realization_forward(case)
    assert torch.isfinite(out["path_gain"]).all()
    assert torch.isfinite(out["total"].real) and torch.isfinite(out["total"].imag)


def test_chain_realization_forward_rejects_bad_dtype():
    case = _chain_realization_case(seed=33)
    args = list(_realization_forward_args(case))
    args[33] = args[33].double()  # heights wrong dtype
    with pytest.raises((TypeError, ValueError)):
        F.scattering_chain_realization_eval(
            *args, k0=case["k0"], frequency_hz=case["frequency_hz"]
        )


def test_chain_realization_forward_rejects_depth_over_max():
    case = _chain_realization_case(seed=34)
    c2 = case["c2"]
    c2["positions"] = torch.zeros(case["n"], _DMAX + 1, 3, device="cuda")
    c2["normals"] = torch.zeros(case["n"], _DMAX + 1, 3, device="cuda")
    c2["eps_r"] = torch.ones(case["n"], _DMAX + 1, device="cuda")
    c2["sigma_e"] = torch.ones(case["n"], _DMAX + 1, device="cuda")
    c2["mu_r"] = torch.ones(case["n"], _DMAX + 1, device="cuda")
    c2["gain"] = torch.ones(case["n"], _DMAX + 1, device="cuda")
    c2["thickness"] = torch.ones(case["n"], _DMAX + 1, device="cuda")
    with pytest.raises(ValueError, match="kMaxAdDepth"):
        _realization_forward(case)


def test_chain_realization_requires_native_kernel(monkeypatch):
    case = _chain_realization_case(seed=35)
    monkeypatch.setattr(runtime, "native_extension", lambda: None)
    with pytest.raises(RuntimeError, match="CUDA kernel is required"):
        _realization_forward(case)


# ---------------------------------------------------------------------------
# Op B: backward / jvp contract.
# ---------------------------------------------------------------------------


def test_chain_realization_backward_schema_and_gating():
    case = _chain_realization_case(seed=41)
    n = case["n"]
    grad_total = torch.ones((), device="cuda", dtype=torch.complex64)
    out = F.scattering_chain_realization_eval_backward(
        *_realization_forward_args(case),
        k0=case["k0"], frequency_hz=case["frequency_hz"], grad_total=grad_total,
        need_grad_heights=True, need_grad_layers=False, need_grad_chain1=False,
        need_grad_chain2=False, need_grad_geometry=False, need_grad_k0=False,
        need_grad_frequency=False,
    )
    assert set(out) == set(F._CHAIN_REALIZATION_BACKWARD_FIELDS)
    assert out["grad_heights"] is not None
    assert out["grad_heights"].shape == case["heights"].shape
    for key in ("grad_layer_thickness", "grad_c1_eps_r", "grad_d_i", "grad_k0", "grad_frequency"):
        assert out[key] is None, key
    _ = n


def test_chain_realization_backward_requires_grad_total_dtype():
    case = _chain_realization_case(seed=42)
    with pytest.raises((TypeError, ValueError)):
        F.scattering_chain_realization_eval_backward(
            *_realization_forward_args(case),
            k0=case["k0"], frequency_hz=case["frequency_hz"],
            grad_total=torch.ones((), device="cuda"),  # float, not complex64
            need_grad_heights=True,
        )


def test_chain_realization_backward_requires_native_kernel(monkeypatch):
    case = _chain_realization_case(seed=43)
    monkeypatch.setattr(runtime, "native_extension", lambda: None)
    with pytest.raises(RuntimeError, match="CUDA kernel is required"):
        F.scattering_chain_realization_eval_backward(
            *_realization_forward_args(case),
            k0=case["k0"], frequency_hz=case["frequency_hz"],
            grad_total=torch.ones((), device="cuda", dtype=torch.complex64),
            need_grad_heights=True,
        )


def test_chain_realization_jvp_schema():
    case = _chain_realization_case(seed=44)
    out = F.scattering_chain_realization_eval_jvp(
        *_realization_forward_args(case),
        k0=case["k0"], frequency_hz=case["frequency_hz"],
        tangent_heights=torch.randn_like(case["heights"]),
        tangent_k0=0.5,
    )
    assert set(out) == {"tangent_total", "tangent_path_field", "tangent_path_gain"}
    n = case["n"]
    assert out["tangent_total"].shape == () and out["tangent_total"].dtype == torch.complex64
    assert out["tangent_path_field"].shape == (n,) and out["tangent_path_field"].dtype == torch.complex64
    assert out["tangent_path_gain"].shape == (n,) and out["tangent_path_gain"].dtype == torch.float32