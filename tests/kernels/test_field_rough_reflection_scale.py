"""Lockstep and contract tests for ADR-010 op 3 (rough-reflection C_r scale)."""

import pytest
import torch

from witwin.channel.propagation.fields.kernels import (
    functional as field_functional,
    rough_scale as field_rough_scale,
)
from witwin.channel.runtime import symbols

from tests.reference import rough_reflection as reference


def _random_case(
    rows, depth, *, device, seed, rough_fraction=0.7, replaced_fraction=0.0,
    sigma_scale=0.005,
):
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=torch.float32)

    positions = randn(rows, depth, 3)
    normals = torch.nn.functional.normalize(randn(rows, depth, 3), dim=-1)
    source = randn(rows, 3)
    # Physical roughness scale: the frozen rough-reflection-cr cell uses
    # rms_height ~ 0.015 m; keep sigma in that band so the exp(-2u^2) factor
    # stays in the ADR 1e-6 float32 envelope.
    sigma_b = randn(rows, depth).abs() * sigma_scale
    rough_b = (
        torch.rand(rows, depth, generator=generator, device=device) < rough_fraction
    )
    replaced = (
        torch.rand(rows, generator=generator, device=device) < replaced_fraction
    )
    field_vector = torch.complex(randn(rows, 3), randn(rows, 3))
    coefficient = torch.complex(randn(rows), randn(rows))
    path_field = torch.complex(randn(rows), randn(rows))
    path_gain = randn(rows).abs() + 0.1
    return {
        "field_vector": field_vector,
        "coefficient": coefficient,
        "path_field": path_field,
        "path_gain": path_gain,
        "positions": positions,
        "normals": normals,
        "source": source,
        "sigma_b": sigma_b,
        "rough_b": rough_b,
        "replaced": replaced,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("depth", [1, 2, 3, 5])
@pytest.mark.parametrize("frequency_hz", [1.0e9, 3.0e9])
def test_rough_reflection_scale_forward_matches_reference(depth, frequency_hz):
    # The ADR-010 C_r gate (max-rel <= 1e-6) is validated at the frozen-cell
    # operating point (~GHz carriers, physical roughness). Deep into the
    # exp(-2u^2) tail (tens of GHz and/or large roughness) the single-precision
    # exp intrinsic vs Torch's exp can amplify above 1e-6; that regime is
    # outside the frozen cells and the contract envelope.
    case = _random_case(64, depth, device="cuda", seed=depth * 100 + 7, replaced_fraction=0.2)
    native = field_functional.field_rough_reflection_scale(
        *(case[name] for name in (
            "field_vector", "coefficient", "path_field", "path_gain",
            "positions", "normals", "source", "sigma_b", "rough_b", "replaced",
        )),
        frequency_hz=frequency_hz,
    )
    ref = reference.rough_reflection_scale(
        *(case[name] for name in (
            "field_vector", "coefficient", "path_field", "path_gain",
            "positions", "normals", "source", "sigma_b", "rough_b", "replaced",
        )),
        frequency_hz=frequency_hz,
    )
    for name in ("field_vector", "coefficient", "path_field", "path_gain", "factor"):
        torch.testing.assert_close(native[name], ref[name], rtol=1.0e-6, atol=1.0e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rough_reflection_scale_contract_shapes_and_dtypes():
    case = _random_case(8, 2, device="cuda", seed=11)
    out = field_functional.field_rough_reflection_scale(
        *(case[name] for name in (
            "field_vector", "coefficient", "path_field", "path_gain",
            "positions", "normals", "source", "sigma_b", "rough_b", "replaced",
        )),
        frequency_hz=3.0e9,
    )
    assert out["field_vector"].shape == (8, 3)
    assert out["field_vector"].dtype == torch.complex64
    assert out["field_vector"].is_cuda
    assert out["coefficient"].shape == (8,)
    assert out["path_gain"].dtype == torch.float32
    assert out["factor"].shape == (8,)
    with pytest.raises((TypeError, ValueError)):
        field_functional.field_rough_reflection_scale(
            case["field_vector"].real,  # wrong dtype (real not complex)
            case["coefficient"], case["path_field"], case["path_gain"],
            case["positions"], case["normals"], case["source"],
            case["sigma_b"], case["rough_b"], case["replaced"],
            frequency_hz=3.0e9,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rough_reflection_scale_requires_native_kernel(monkeypatch):
    case = _random_case(4, 1, device="cuda", seed=3)
    monkeypatch.setattr(symbols, "native_extension", lambda: None)
    with pytest.raises(RuntimeError, match="field_rough_reflection_scale CUDA kernel is required"):
        field_functional.field_rough_reflection_scale(
            *(case[name] for name in (
                "field_vector", "coefficient", "path_field", "path_gain",
                "positions", "normals", "source", "sigma_b", "rough_b", "replaced",
            )),
            frequency_hz=3.0e9,
        )


def _ad_loss(outputs):
    fv, coef, pf, pg = outputs
    return (
        fv.real.sum() + 0.5 * fv.imag.sum()
        + 0.7 * coef.real.sum() - 0.3 * coef.imag.sum()
        + 0.2 * pf.real.sum() + 0.9 * pf.imag.sum()
        + 1.3 * pg.sum()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("depth", [1, 3])
def test_rough_reflection_scale_vjp_matches_reference_autograd(depth):
    frequency_hz = 3.0e9
    case = _random_case(32, depth, device="cuda", seed=depth * 17 + 5, replaced_fraction=0.15)
    names = (
        "field_vector", "coefficient", "path_field", "path_gain",
        "positions", "normals", "source",
    )

    def leaves():
        out = {name: case[name].clone().requires_grad_(True) for name in names}
        freq = torch.tensor(frequency_hz, dtype=torch.float64, device="cuda", requires_grad=True)
        return out, freq

    ref_in, ref_freq = leaves()
    ref = reference.rough_reflection_scale(
        ref_in["field_vector"], ref_in["coefficient"], ref_in["path_field"],
        ref_in["path_gain"], ref_in["positions"], ref_in["normals"],
        ref_in["source"], case["sigma_b"], case["rough_b"], case["replaced"],
        ref_freq,
    )
    _ad_loss((ref["field_vector"], ref["coefficient"], ref["path_field"], ref["path_gain"])).backward()

    nat_in, nat_freq = leaves()
    scaled = field_rough_scale.field_rough_reflection_scale_ad(
        nat_in["field_vector"], nat_in["coefficient"], nat_in["path_field"],
        nat_in["path_gain"], nat_in["positions"], nat_in["normals"],
        nat_in["source"], case["sigma_b"], case["rough_b"], case["replaced"],
        frequency=nat_freq,
    )
    _ad_loss((scaled["field_vector"], scaled["coefficient"], scaled["path_field"], scaled["path_gain"])).backward()

    for name in names:
        torch.testing.assert_close(
            nat_in[name].grad, ref_in[name].grad, rtol=2.0e-4, atol=1.0e-6,
            msg=f"grad mismatch for {name}",
        )
    torch.testing.assert_close(nat_freq.grad, ref_freq.grad, rtol=2.0e-4, atol=1.0e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("depth", [1, 3])
def test_rough_reflection_scale_jvp_matches_reference_autograd(depth):
    frequency_hz = 3.0e9
    case = _random_case(32, depth, device="cuda", seed=depth * 23 + 1, replaced_fraction=0.15)
    names = (
        "field_vector", "coefficient", "path_field", "path_gain",
        "positions", "normals", "source",
    )
    generator = torch.Generator(device="cuda").manual_seed(depth * 91)

    def tangent_like(t):
        if t.is_complex():
            return torch.complex(
                torch.randn(*t.shape, generator=generator, device="cuda"),
                torch.randn(*t.shape, generator=generator, device="cuda"),
            )
        return torch.randn(*t.shape, generator=generator, device="cuda", dtype=torch.float32)

    tangents = {name: tangent_like(case[name]) for name in names}
    freq_tangent = torch.tensor(1.0, dtype=torch.float64, device="cuda")

    def run(scale_fn):
        with torch.autograd.forward_ad.dual_level():
            duals = {
                name: torch.autograd.forward_ad.make_dual(case[name], tangents[name])
                for name in names
            }
            freq_dual = torch.autograd.forward_ad.make_dual(
                torch.tensor(frequency_hz, dtype=torch.float64, device="cuda"),
                freq_tangent,
            )
            out = scale_fn(duals, freq_dual)
            return {
                key: torch.autograd.forward_ad.unpack_dual(value).tangent
                for key, value in out.items()
                if key != "factor"
            }

    def ref_fn(duals, freq_dual):
        return reference.rough_reflection_scale(
            duals["field_vector"], duals["coefficient"], duals["path_field"],
            duals["path_gain"], duals["positions"], duals["normals"],
            duals["source"], case["sigma_b"], case["rough_b"], case["replaced"],
            freq_dual,
        )

    def native_fn(duals, freq_dual):
        return field_rough_scale.field_rough_reflection_scale_ad(
            duals["field_vector"], duals["coefficient"], duals["path_field"],
            duals["path_gain"], duals["positions"], duals["normals"],
            duals["source"], case["sigma_b"], case["rough_b"], case["replaced"],
            frequency=freq_dual,
        )

    ref_t = run(ref_fn)
    nat_t = run(native_fn)
    for name in ("field_vector", "coefficient", "path_field", "path_gain"):
        torch.testing.assert_close(
            nat_t[name], ref_t[name], rtol=2.0e-4, atol=1.0e-6,
            msg=f"tangent mismatch for {name}",
        )
