# Copyright Xingyu Chen.
# Tests forward mode liveness.

"""Tests forward mode liveness."""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

from witwin.channel.kernels import fields as field_kernels

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native field companions require CUDA"
)

_FREQUENCY_HZ = 3.5e9
_C0 = 299792458.0


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(2038)

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to("cuda", torch.float32)

    source = rand(4, 3) * 2.0
    target = source + rand(4, 3) * 3.0 + torch.tensor(
        [5.0, 0.0, 0.0], device="cuda"
    )
    return {
        "source": source.contiguous(),
        "target": target.contiguous(),
        "tx_power": (torch.rand(4, generator=generator) + 0.5).to(
            "cuda", torch.float32
        ),
        "tx_polarization": rand(4, 3).contiguous(),
        "rx_polarization": rand(4, 3).contiguous(),
    }


def test_forward_only_dual_carries_geometry_tangents_matching_fd():
    batch = _batch()
    tangent = torch.randn(
        4, 3, generator=torch.Generator(device="cpu").manual_seed(7)
    ).to("cuda", torch.float32)

    def path_length(target: torch.Tensor) -> torch.Tensor:
        out = field_kernels.field_free_space_ad(
            batch["source"],
            target,
            batch["tx_power"],
            batch["tx_polarization"],
            batch["rx_polarization"],
            frequency=_FREQUENCY_HZ,
        )
        return out["path_length_m"]

    step = 1.0e-3
    reference = (
        path_length(batch["target"] + step * tangent)
        - path_length(batch["target"] - step * tangent)
    ) / (2.0 * step)

    # The tangent is the only derivative request: no requires_grad anywhere.
    assert not any(value.requires_grad for value in batch.values())
    with forward_ad.dual_level():
        dual_target = forward_ad.make_dual(batch["target"], tangent)
        out = field_kernels.field_free_space_ad(
            batch["source"],
            dual_target,
            batch["tx_power"],
            batch["tx_polarization"],
            batch["rx_polarization"],
            frequency=_FREQUENCY_HZ,
        )
        path_tangent = forward_ad.unpack_dual(out["path_length_m"]).tangent
        delay_tangent = forward_ad.unpack_dual(out["delay_s"]).tangent
        coefficient_tangent = forward_ad.unpack_dual(out["coefficient"]).tangent
        assert path_tangent is not None, "geometry tangent silently dropped"
        assert delay_tangent is not None
        assert coefficient_tangent is not None
        path_tangent = path_tangent.clone()
        delay_tangent = delay_tangent.clone()

    torch.testing.assert_close(path_tangent, reference, rtol=1.0e-3, atol=1.0e-4)
    # delay_s is path_length_m / c exactly, so the tangents share the ratio.
    torch.testing.assert_close(
        delay_tangent * _C0, path_tangent, rtol=1.0e-5, atol=1.0e-9
    )


def test_forward_only_dual_and_requires_grad_convention_agree():
    """The fixed-topology replay interim convention resolves to the identical tangent."""

    batch = _batch()
    tangent = torch.randn(
        4, 3, generator=torch.Generator(device="cpu").manual_seed(11)
    ).to("cuda", torch.float32)

    def run(target_primal: torch.Tensor) -> torch.Tensor:
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(target_primal, tangent)
            out = field_kernels.field_free_space_ad(
                batch["source"],
                dual,
                batch["tx_power"],
                batch["tx_polarization"],
                batch["rx_polarization"],
                frequency=_FREQUENCY_HZ,
            )
            result = forward_ad.unpack_dual(out["delay_s"]).tangent
            assert result is not None
            return result.clone()

    dual_only = run(batch["target"])
    with_grad = run(batch["target"].clone().requires_grad_())
    torch.testing.assert_close(dual_only, with_grad, rtol=0.0, atol=0.0)


def test_materials_only_request_keeps_geometry_detached():
    """The AD exactness contract survives the liveness move.

 With no geometry gradient or tangent requested anywhere, the conditional
 outputs stay detached so a materials-only graph never pays for geometry
 adjoints it did not ask for.
 """

    batch = _batch()
    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float32, device="cuda", requires_grad=True
    )
    out = field_kernels.field_free_space_ad(
        batch["source"],
        batch["target"],
        batch["tx_power"],
        batch["tx_polarization"],
        batch["rx_polarization"],
        frequency=frequency,
    )
    assert not out["path_length_m"].requires_grad
    assert not out["delay_s"].requires_grad
    assert out["coefficient"].requires_grad


def test_reverse_mode_geometry_gradients_are_unchanged():
    """requires_grad liveness resolves identically from the wrapper."""

    batch = _batch()
    target = batch["target"].clone().requires_grad_()
    out = field_kernels.field_free_space_ad(
        batch["source"],
        target,
        batch["tx_power"],
        batch["tx_polarization"],
        batch["rx_polarization"],
        frequency=_FREQUENCY_HZ,
    )
    assert out["path_length_m"].requires_grad
    out["path_length_m"].sum().backward()
    assert target.grad is not None

    direction = (batch["target"] - batch["source"]) / (
        batch["target"] - batch["source"]
    ).norm(dim=-1, keepdim=True)
    torch.testing.assert_close(target.grad, direction, rtol=1.0e-5, atol=1.0e-6)