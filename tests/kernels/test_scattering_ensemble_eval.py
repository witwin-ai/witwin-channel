"""Lockstep and contract tests for ADR-010 op 1 (Kirchhoff ensemble rows)."""

import pytest
import torch

from witwin.core import SurfaceRoughness
from witwin.channel.scene.resources import build_kirchhoff_table
from witwin.channel.kernels import scattering as ops
from witwin.channel.scene.resources import (
    build_kirchhoff_table_stack,
)
from witwin.channel import runtime

from tests.reference import kirchhoff_ensemble as reference


def _build_tables(device):
    layers = ((0.1, 4.0, 0.02, 1.0),)
    frequency = 3.0e9
    tables = {
        0: build_kirchhoff_table(
            SurfaceRoughness(rms_height_m=0.01, correlation_length_x_m=0.15, correlation_length_y_m=0.15),
            layers, frequency, device=device,
        ),
        1: build_kirchhoff_table(
            SurfaceRoughness(
                rms_height_m=0.012, correlation_length_x_m=0.2, correlation_length_y_m=0.1,
                principal_axis_rad=0.3,
            ),
            layers, frequency, device=device,
        ),
    }
    return tables


def _orthonormal_frame(n):
    ref = torch.zeros_like(n)
    pick = n.abs().argmin(dim=-1)
    ref.scatter_(-1, pick.unsqueeze(-1), 1.0)
    t1 = torch.nn.functional.normalize(
        ref - (ref * n).sum(-1, keepdim=True) * n, dim=-1
    )
    t2 = torch.cross(n, t1, dim=-1)
    return t1, t2


def _random_case(samples, rx_count, rows, *, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32)

    points = rn(samples, 3) * 0.5
    n_o = torch.nn.functional.normalize(rn(samples, 3) * 0.25 + torch.tensor(
        [0.0, 0.0, 1.0], device=device), dim=-1)
    t1r, t2r = _orthonormal_frame(n_o)
    backup_axis = t1r.clone()
    wi = torch.nn.functional.normalize(rn(samples, 3) * 0.3 + torch.tensor(
        [0.0, 0.0, 1.0], device=device), dim=-1)
    cos_i = (wi * n_o).sum(-1).clamp_min(1.0e-3)
    wi_local = torch.stack(((wi * t1r).sum(-1), (wi * t2r).sum(-1), cos_i), dim=-1)
    r1 = rn(samples).abs() + 0.5
    a_te2 = rn(samples).abs().clamp(0.0, 1.0)
    a_tm2 = rn(samples).abs().clamp(0.0, 1.0)
    weights = rn(samples).abs() + 0.01
    material_id = (torch.rand(samples, generator=g, device=device) < 0.5).to(torch.int32)
    rx_positions = rn(rx_count, 3) * 2.0 + torch.tensor([0.0, 0.0, 10.0], device=device)
    rx_pol = torch.nn.functional.normalize(rn(rx_count, 3), dim=-1)
    rc = torch.randint(0, rx_count, (rows,), generator=g, device=device, dtype=torch.int64)
    sc = torch.randint(0, samples, (rows,), generator=g, device=device, dtype=torch.int64)
    return dict(
        points=points, n_o=n_o, t1r=t1r, t2r=t2r, wi_local=wi_local, cos_i=cos_i,
        r1=r1, a_te2=a_te2, a_tm2=a_tm2, weights=weights, material_id=material_id,
        backup_axis=backup_axis, rx_positions=rx_positions, rx_pol=rx_pol, rc=rc, sc=sc,
    )


def _grid_rows(case):
    """Torch candidate-grid values for the surviving rows (production path)."""

    to_rx = case["rx_positions"][case["rc"]] - case["points"][case["sc"]]
    r2 = torch.linalg.vector_norm(to_rx, dim=-1).clamp_min(1.0e-6)
    wo = to_rx / r2[:, None]
    cos_o = (wo * case["n_o"][case["sc"]]).sum(-1)
    return wo.contiguous(), r2.contiguous(), cos_o.contiguous()


def _native_eval(case, stack, coef, threshold):
    wo, r2, cos_o = _grid_rows(case)
    return ops.scattering_ensemble_eval(
        torch.ones(wo.shape[0], dtype=torch.bool, device=wo.device),
        wo, r2, cos_o, case["n_o"], case["t1r"], case["t2r"], case["wi_local"],
        case["cos_i"], case["r1"], case["a_te2"], case["a_tm2"], case["weights"],
        case["material_id"], case["backup_axis"], case["rx_pol"],
        case["rc"], case["sc"], stack.f_te_flat, stack.f_tm_flat, stack.table_offset,
        stack.table_dims, stack.material_slot, coef=coef, threshold=threshold,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("seed", [1, 7, 42])
def test_scattering_ensemble_eval_matches_reference(seed):
    device = torch.device("cuda")
    tables = _build_tables(device)
    stack = build_kirchhoff_table_stack(tables, 2, device)
    case = _random_case(96, 8, 200, device=device, seed=seed)
    coef = 1.7e-4
    threshold = 0.0

    native = _native_eval(case, stack, coef, threshold)
    ref = reference.kirchhoff_ensemble_rows(
        case["points"], case["n_o"], case["t1r"], case["t2r"], case["wi_local"],
        case["cos_i"], case["r1"], case["a_te2"], case["a_tm2"], case["weights"],
        case["material_id"], case["backup_axis"], case["rx_positions"], case["rx_pol"],
        case["rc"], case["sc"], tables, coef, threshold,
    )
    # ADR-010 gates: max-rel <= 1e-6 with an absolute floor of
    # 1e-9 * max|baseline| absorbing denormal-scale rows.
    for name, rtol in (("gain", 1.0e-6), ("amplitude", 1.0e-6), ("length", 1.0e-6)):
        atol = 1.0e-9 * float(ref[name].abs().max())
        torch.testing.assert_close(
            native[name], ref[name], rtol=rtol, atol=atol, msg=f"{name} mismatch"
        )
    assert torch.equal(native["keep"], ref["keep"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scattering_ensemble_eval_contract_shapes_and_dtypes():
    device = torch.device("cuda")
    tables = _build_tables(device)
    stack = build_kirchhoff_table_stack(tables, 2, device)
    case = _random_case(4, 2, 6, device=device, seed=3)
    out = _native_eval(case, stack, 1.0, 0.0)
    assert out["gain"].shape == (6,)
    assert out["gain"].dtype == torch.float32
    assert out["gain"].is_cuda
    assert torch.isfinite(out["gain"]).all()
    assert (out["gain"] >= 0.0).all()
    assert out["amplitude"].shape == (6,)
    assert out["length"].shape == (6,)
    assert out["keep"].dtype == torch.bool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scattering_ensemble_eval_requires_native_kernel(monkeypatch):
    device = torch.device("cuda")
    tables = _build_tables(device)
    stack = build_kirchhoff_table_stack(tables, 2, device)
    case = _random_case(4, 2, 3, device=device, seed=5)
    monkeypatch.setattr(runtime, "native_extension", lambda: None)
    with pytest.raises(RuntimeError, match="scattering_ensemble_eval CUDA kernel is required"):
        _native_eval(case, stack, 1.0, 0.0)
