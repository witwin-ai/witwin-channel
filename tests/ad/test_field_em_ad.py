# Copyright Xingyu Chen.
# AD-1 kernel-level tests: EM-response derivatives of the field kernels.

"""AD-1 kernel-level tests: EM-response derivatives of the field kernels.

Covers forward parity of the native float32 kernels against the pure-torch
complex128 references, the gradient oracle (torch autograd through the
reference versus the native backward/jvp companions, which pins the Wirtinger
convention), JVP-vs-VJP duality, torch.autograd.gradcheck, and the explicit
failure contract for fixed inputs (geometry, mu_r).
"""

from __future__ import annotations

import pytest
import torch

from tests.ad._fd import central_difference_directional, relative_error
from tests.ad._reference_fields import (
    free_space_reference,
    reflection_sequence_reference,
    transmission_sequence_reference,
)
from tests.ad._tolerances import ABS_TOL, REL_TOL_PATH
from witwin.channel.kernels import fields as field_kernels

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for field EM AD"
)

_FREQUENCY_HZ = 3.0e9

# Forward-parity floor for the float32 kernels against the complex128
# reference: the carrier phase k*L (~4e2 rad for these fixtures) amplifies
# float32 rounding of k and L to ~2.5e-5 rad of phase, i.e. ~2.5e-5 relative
# error on complex fields even for a bit-exact formula mirror.
_FORWARD_PARITY_TOL = 2.0e-4


def _free_space_batch(dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(101)

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to("cuda", dtype)

    source = rand(4, 3) * 2.0
    target = source + rand(4, 3) * 3.0 + torch.tensor(
        [5.0, 0.0, 0.0], device="cuda", dtype=dtype
    )
    return {
        "source": source.contiguous(),
        "target": target.contiguous(),
        "tx_power": (torch.rand(4, generator=generator) + 0.5).to("cuda", dtype),
        "tx_polarization": rand(4, 3).contiguous(),
        "rx_polarization": rand(4, 3).contiguous(),
    }


def _reflection_batch(depth: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(103)
    count = 3
    source = torch.zeros(count, 3, device="cuda")
    hit_templates = (
        torch.tensor([2.5, 0.1, 0.4]),
        torch.tensor([0.5, 2.5, 0.3]),
    )
    normal_templates = (
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0.0, -1.0, 0.0]),
    )
    positions = torch.stack(
        [
            torch.stack(
                [
                    hit_templates[b]
                    + 0.3 * torch.randn(3, generator=generator)
                    for b in range(depth)
                ]
            )
            for _ in range(count)
        ]
    ).to("cuda")
    normals = torch.stack(
        [torch.stack([normal_templates[b] for b in range(depth)])] * count
    ).to("cuda")
    materials = {
        "eps_r": torch.full((count, depth), 4.0, device="cuda")
        + 0.5 * torch.rand(count, depth, generator=generator).to("cuda"),
        "sigma_e": torch.full((count, depth), 0.02, device="cuda"),
        "mu_r": torch.ones(count, depth, device="cuda"),
        "gain": torch.ones(count, depth, device="cuda"),
        "thickness": torch.full((count, depth), 0.1, device="cuda"),
    }
    return {
        "source": source.contiguous(),
        "target": torch.tensor([[0.5, 1.0, 0.3]] * count, device="cuda"),
        "interaction_positions": positions.contiguous(),
        "interaction_normals": normals.contiguous(),
        "tx_power": torch.full((count,), 1.5, device="cuda"),
        "tx_polarization": torch.tensor([[0.0, 0.0, 1.0]] * count, device="cuda"),
        "rx_polarization": torch.tensor([[0.0, 0.0, 1.0]] * count, device="cuda"),
        **{key: value.contiguous() for key, value in materials.items()},
    }


def _transmission_batch() -> dict[str, torch.Tensor]:
    count, depth = 2, 2
    return {
        "path_valid": torch.ones(count, dtype=torch.bool, device="cuda"),
        "source": torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], device="cuda"
        ).contiguous(),
        "target": torch.tensor(
            [[6.0, 0.4, 0.2], [6.0, -0.3, 0.1]], device="cuda"
        ).contiguous(),
        "interaction_positions": torch.zeros(count, depth, 3, device="cuda"),
        "interaction_normals": torch.tensor(
            [[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]] * count, device="cuda"
        ).contiguous(),
        # Path 0 crosses material 0 then material 1; path 1 crosses only
        # material 0 (second slot skipped), exercising the CSR scatter.
        "interaction_material_id": torch.tensor(
            [[0, 1], [0, -1]], dtype=torch.int32, device="cuda"
        ),
        "interaction_valid": torch.tensor(
            [[True, True], [True, False]], device="cuda"
        ),
        "tx_power": torch.full((count,), 2.0, device="cuda"),
        "tx_polarization": torch.tensor([[0.0, 0.0, 1.0]] * count, device="cuda"),
        "rx_polarization": torch.tensor([[0.0, 0.0, 1.0]] * count, device="cuda"),
        "layer_offset": torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
        "layer_count": torch.tensor([2, 1], dtype=torch.int32, device="cuda"),
        "layer_thickness_m": torch.tensor([0.05, 0.08, 0.1], device="cuda"),
        "layer_eps_r": torch.tensor([4.0, 2.5, 3.0], device="cuda"),
        "layer_sigma_e": torch.tensor([0.02, 0.01, 0.05], device="cuda"),
        "layer_mu_r": torch.ones(3, device="cuda"),
    }


def _loss_weights(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.complex(
        torch.randn(count, generator=generator), torch.randn(count, generator=generator)
    ).to("cuda")


def _real_pair_loss(coefficient: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (
        weights.real.to(coefficient.real.dtype) * coefficient.real
        + weights.imag.to(coefficient.real.dtype) * coefficient.imag
    ).sum()


def _reference_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for name, value in batch.items():
        if value.dtype == torch.float32:
            converted[name] = value.double()
        else:
            converted[name] = value
    return converted


# ---------------------------------------------------------------------------
# Forward parity: native float32 kernels versus the complex128 references.
# ---------------------------------------------------------------------------


def test_free_space_forward_parity_vs_reference():
    batch = _free_space_batch()
    native = field_kernels.field_free_space(*batch.values(), frequency_hz=_FREQUENCY_HZ)
    reference = free_space_reference(
        **_reference_inputs(batch),
        frequency=torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda"),
    )
    for name in ("field_vector", "coefficient", "path_field", "path_gain"):
        assert (
            relative_error(native[name], reference[name], abs_floor=ABS_TOL)
            <= _FORWARD_PARITY_TOL
        ), name


def test_reflection_forward_parity_vs_reference():
    batch = _reflection_batch(depth=2)
    native = field_kernels.field_reflection_sequence(*batch.values(), frequency_hz=_FREQUENCY_HZ)
    reference = reflection_sequence_reference(
        **_reference_inputs(batch),
        frequency=torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda"),
    )
    for name in ("field_vector", "coefficient", "path_field", "path_gain"):
        assert (
            relative_error(native[name], reference[name], abs_floor=ABS_TOL)
            <= _FORWARD_PARITY_TOL
        ), name


def test_transmission_forward_parity_vs_reference():
    batch = _transmission_batch()
    native = field_kernels.field_transmission_sequence(*batch.values(), frequency_hz=_FREQUENCY_HZ)
    reference_batch = _reference_inputs(batch)
    del reference_batch["path_valid"]
    del reference_batch["interaction_positions"]
    reference = transmission_sequence_reference(
        **reference_batch,
        frequency=torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda"),
    )
    for name in ("field_vector", "coefficient", "path_field", "path_gain"):
        assert (
            relative_error(native[name], reference[name], abs_floor=ABS_TOL)
            <= _FORWARD_PARITY_TOL
        ), name


# ---------------------------------------------------------------------------
# Gradient oracle: torch autograd through the complex128 reference versus the
# native backward/jvp companions. This catches Wirtinger-convention errors
# (sign or conjugation) exactly.
# ---------------------------------------------------------------------------


def test_free_space_frequency_vjp_matches_reference_oracle():
    batch = _free_space_batch()
    weights = _loss_weights(4, seed=7)
    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float32, device="cuda", requires_grad=True
    )
    out = field_kernels.field_free_space_ad(*batch.values(), frequency=frequency)
    loss = _real_pair_loss(out["coefficient"], weights) + out["path_gain"].sum()
    loss.backward()
    assert frequency.grad is not None

    reference_frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    reference = free_space_reference(
        **_reference_inputs(batch), frequency=reference_frequency
    )
    reference_loss = _real_pair_loss(reference["coefficient"], weights)
    reference_loss = reference_loss + reference["path_gain"].sum()
    (expected,) = torch.autograd.grad(reference_loss, reference_frequency)
    assert relative_error(frequency.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_free_space_jvp_matches_reference_oracle():
    batch = _free_space_batch()
    tangents = field_kernels.field_free_space_jvp(
        *batch.values(), frequency_hz=_FREQUENCY_HZ, tangent_frequency=1.0
    )
    reference_frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    reference = free_space_reference(
        **_reference_inputs(batch), frequency=reference_frequency
    )
    for name in ("coefficient", "path_gain"):
        value = reference[name]
        if value.is_complex():
            expected_real = torch.autograd.grad(
                value.real.sum(), reference_frequency, retain_graph=True
            )[0]
            expected_imag = torch.autograd.grad(
                value.imag.sum(), reference_frequency, retain_graph=True
            )[0]
            actual = tangents[name].sum()
            assert (
                relative_error(actual.real, expected_real, abs_floor=ABS_TOL)
                <= REL_TOL_PATH
            )
            assert (
                relative_error(actual.imag, expected_imag, abs_floor=ABS_TOL)
                <= REL_TOL_PATH
            )
        else:
            (expected,) = torch.autograd.grad(
                value.sum(), reference_frequency, retain_graph=True
            )
            assert (
                relative_error(tangents[name].sum(), expected, abs_floor=ABS_TOL)
                <= REL_TOL_PATH
            )


@pytest.mark.parametrize("depth", [1, 2])
def test_reflection_material_vjp_matches_reference_oracle(depth):
    batch = _reflection_batch(depth=depth)
    weights = _loss_weights(3, seed=11)
    leaves = {}
    ad_batch = dict(batch)
    for name in ("eps_r", "sigma_e", "gain", "thickness"):
        leaves[name] = batch[name].clone().requires_grad_(True)
        ad_batch[name] = leaves[name]
    out = field_kernels.field_reflection_sequence_ad(*ad_batch.values(), frequency=_FREQUENCY_HZ)
    loss = _real_pair_loss(out["coefficient"], weights) + out["path_gain"].sum()
    loss.backward()

    reference_batch = _reference_inputs(batch)
    reference_leaves = {}
    for name in ("eps_r", "sigma_e", "gain", "thickness"):
        reference_leaves[name] = reference_batch[name].clone().requires_grad_(True)
        reference_batch[name] = reference_leaves[name]
    reference = reflection_sequence_reference(
        **reference_batch,
        frequency=torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda"),
    )
    reference_loss = _real_pair_loss(reference["coefficient"], weights)
    reference_loss = reference_loss + reference["path_gain"].sum()
    expected = torch.autograd.grad(reference_loss, tuple(reference_leaves.values()))
    for (name, leaf), expected_grad in zip(leaves.items(), expected, strict=True):
        assert leaf.grad is not None, name
        assert (
            relative_error(leaf.grad, expected_grad, abs_floor=ABS_TOL) <= REL_TOL_PATH
        ), name


def test_reflection_frequency_vjp_matches_reference_oracle():
    batch = _reflection_batch(depth=1)
    weights = _loss_weights(3, seed=13)
    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    out = field_kernels.field_reflection_sequence_ad(*batch.values(), frequency=frequency)
    loss = _real_pair_loss(out["coefficient"], weights)
    loss.backward()
    assert frequency.grad is not None

    reference_frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    reference = reflection_sequence_reference(
        **_reference_inputs(batch), frequency=reference_frequency
    )
    (expected,) = torch.autograd.grad(
        _real_pair_loss(reference["coefficient"], weights), reference_frequency
    )
    assert relative_error(frequency.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_transmission_layer_vjp_matches_reference_oracle():
    batch = _transmission_batch()
    weights = _loss_weights(2, seed=17)
    leaves = {}
    ad_batch = dict(batch)
    for name in ("layer_thickness_m", "layer_eps_r", "layer_sigma_e"):
        leaves[name] = batch[name].clone().requires_grad_(True)
        ad_batch[name] = leaves[name]
    out = field_kernels.field_transmission_sequence_ad(
        *ad_batch.values(), frequency=_FREQUENCY_HZ
    )
    loss = _real_pair_loss(out["coefficient"], weights) + out["path_gain"].sum()
    loss.backward()

    reference_batch = _reference_inputs(batch)
    del reference_batch["path_valid"]
    del reference_batch["interaction_positions"]
    reference_leaves = {}
    for name in ("layer_thickness_m", "layer_eps_r", "layer_sigma_e"):
        reference_leaves[name] = reference_batch[name].clone().requires_grad_(True)
        reference_batch[name] = reference_leaves[name]
    reference = transmission_sequence_reference(
        **reference_batch,
        frequency=torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda"),
    )
    reference_loss = _real_pair_loss(reference["coefficient"], weights)
    reference_loss = reference_loss + reference["path_gain"].sum()
    expected = torch.autograd.grad(reference_loss, tuple(reference_leaves.values()))
    for (name, leaf), expected_grad in zip(leaves.items(), expected, strict=True):
        assert leaf.grad is not None, name
        assert (
            relative_error(leaf.grad, expected_grad, abs_floor=ABS_TOL) <= REL_TOL_PATH
        ), name


def test_transmission_frequency_vjp_matches_reference_oracle():
    batch = _transmission_batch()
    weights = _loss_weights(2, seed=19)
    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float32, device="cuda", requires_grad=True
    )
    out = field_kernels.field_transmission_sequence_ad(*batch.values(), frequency=frequency)
    loss = _real_pair_loss(out["coefficient"], weights)
    loss.backward()
    assert frequency.grad is not None

    reference_batch = _reference_inputs(batch)
    del reference_batch["path_valid"]
    del reference_batch["interaction_positions"]
    reference_frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    reference = transmission_sequence_reference(
        **reference_batch, frequency=reference_frequency
    )
    (expected,) = torch.autograd.grad(
        _real_pair_loss(reference["coefficient"], weights), reference_frequency
    )
    assert relative_error(frequency.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_PATH


# ---------------------------------------------------------------------------
# Clamp-boundary regressions: the forward fmaxf(x, 0) passes x through at
# x == 0, so the dual/adjoint gates must pass the gradient there too
# (>=, matching the oracle's clamp_min autograd). sigma_e = 0 is the default
# material initialization; a dead gradient would stall conductivity
# optimization at 0 forever.
# ---------------------------------------------------------------------------


def test_reflection_sigma_zero_boundary_grad_matches_oracle():
    batch = _reflection_batch(depth=1)
    batch["sigma_e"] = torch.zeros_like(batch["sigma_e"])
    weights = _loss_weights(3, seed=53)
    sigma = batch["sigma_e"].clone().requires_grad_(True)
    ad_batch = dict(batch)
    ad_batch["sigma_e"] = sigma
    out = field_kernels.field_reflection_sequence_ad(*ad_batch.values(), frequency=_FREQUENCY_HZ)
    loss = _real_pair_loss(out["coefficient"], weights) + out["path_gain"].sum()
    loss.backward()
    assert sigma.grad is not None
    assert float(sigma.grad.abs().max()) > 0.0

    reference_batch = _reference_inputs(batch)
    reference_sigma = reference_batch["sigma_e"].clone().requires_grad_(True)
    reference_batch["sigma_e"] = reference_sigma
    reference = reflection_sequence_reference(
        **reference_batch,
        frequency=torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda"),
    )
    reference_loss = _real_pair_loss(reference["coefficient"], weights)
    reference_loss = reference_loss + reference["path_gain"].sum()
    (expected,) = torch.autograd.grad(reference_loss, reference_sigma)
    assert relative_error(sigma.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_transmission_sigma_zero_boundary_grad_matches_oracle():
    batch = _transmission_batch()
    batch["layer_sigma_e"] = torch.zeros_like(batch["layer_sigma_e"])
    weights = _loss_weights(2, seed=59)
    sigma = batch["layer_sigma_e"].clone().requires_grad_(True)
    ad_batch = dict(batch)
    ad_batch["layer_sigma_e"] = sigma
    out = field_kernels.field_transmission_sequence_ad(
        *ad_batch.values(), frequency=_FREQUENCY_HZ
    )
    loss = _real_pair_loss(out["coefficient"], weights) + out["path_gain"].sum()
    loss.backward()
    assert sigma.grad is not None
    assert float(sigma.grad.abs().max()) > 0.0

    reference_batch = _reference_inputs(batch)
    del reference_batch["path_valid"]
    del reference_batch["interaction_positions"]
    reference_sigma = reference_batch["layer_sigma_e"].clone().requires_grad_(True)
    reference_batch["layer_sigma_e"] = reference_sigma
    reference = transmission_sequence_reference(
        **reference_batch,
        frequency=torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda"),
    )
    reference_loss = _real_pair_loss(reference["coefficient"], weights)
    reference_loss = reference_loss + reference["path_gain"].sum()
    (expected,) = torch.autograd.grad(reference_loss, reference_sigma)
    assert relative_error(sigma.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_transmission_thickness_zero_boundary_grads():
    batch = _transmission_batch()
    batch["layer_thickness_m"] = batch["layer_thickness_m"].clone()
    batch["layer_thickness_m"][1] = 0.0
    weights = _loss_weights(2, seed=61)
    thickness = batch["layer_thickness_m"].clone().requires_grad_(True)
    ad_batch = dict(batch)
    ad_batch["layer_thickness_m"] = thickness
    out = field_kernels.field_transmission_sequence_ad(
        *ad_batch.values(), frequency=_FREQUENCY_HZ
    )
    loss = _real_pair_loss(out["coefficient"], weights)
    loss.backward()
    assert thickness.grad is not None
    assert float(thickness.grad[1].abs()) > 0.0

    reference_batch = _reference_inputs(batch)
    del reference_batch["path_valid"]
    del reference_batch["interaction_positions"]
    reference_thickness = (
        reference_batch["layer_thickness_m"].clone().requires_grad_(True)
    )
    reference_batch["layer_thickness_m"] = reference_thickness
    reference = transmission_sequence_reference(
        **reference_batch,
        frequency=torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda"),
    )
    (expected,) = torch.autograd.grad(
        _real_pair_loss(reference["coefficient"], weights), reference_thickness
    )
    assert relative_error(thickness.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_PATH

    # jvp side of the same boundary: the forward tangent must agree with the
    # backward through the inner-product duality on a thickness-only seed.
    generator = torch.Generator(device="cpu").manual_seed(67)
    v_thickness = torch.randn(3, generator=generator).to("cuda") * 0.01
    tangents = field_kernels.field_transmission_sequence_jvp(
        *batch.values(),
        frequency_hz=_FREQUENCY_HZ,
        tangent_layer_thickness_m=v_thickness,
    )
    lhs = _real_pair_loss(tangents["coefficient"], weights).double()
    grads = field_kernels.field_transmission_sequence_backward(
        *batch.values(),
        frequency_hz=_FREQUENCY_HZ,
        grad_coefficient=weights,
        need_grad_layer_eps_r=False,
        need_grad_layer_sigma_e=False,
        need_grad_frequency=False,
    )
    rhs = (grads["grad_layer_thickness_m"].double() * v_thickness.double()).sum()
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= REL_TOL_PATH


# ---------------------------------------------------------------------------
# JVP-vs-VJP self consistency (inner-product duality and vjp == sum of jvp).
# ---------------------------------------------------------------------------


def test_jvp_vjp_inner_product_duality_reflection():
    batch = _reflection_batch(depth=2)
    generator = torch.Generator(device="cpu").manual_seed(23)
    u_coefficient = _loss_weights(3, seed=29)
    u_gain = torch.randn(3, generator=generator).to("cuda")
    v_eps = torch.randn(3, 2, generator=generator).to("cuda")
    v_sigma = torch.randn(3, 2, generator=generator).to("cuda") * 0.01
    v_thickness = torch.randn(3, 2, generator=generator).to("cuda") * 0.01
    v_gain = torch.randn(3, 2, generator=generator).to("cuda") * 0.01
    v_frequency = 1.0e6

    tangents = field_kernels.field_reflection_sequence_jvp(
        *batch.values(),
        frequency_hz=_FREQUENCY_HZ,
        tangent_eps_r=v_eps,
        tangent_sigma_e=v_sigma,
        tangent_gain=v_gain,
        tangent_thickness=v_thickness,
        tangent_frequency=v_frequency,
    )
    lhs = _real_pair_loss(tangents["coefficient"], u_coefficient).double()
    lhs = lhs + (tangents["path_gain"].double() * u_gain.double()).sum()

    grads = field_kernels.field_reflection_sequence_backward(
        *batch.values(),
        frequency_hz=_FREQUENCY_HZ,
        grad_coefficient=u_coefficient,
        grad_path_gain=u_gain,
        need_grad_eps_r=True,
        need_grad_sigma_e=True,
        need_grad_gain=True,
        need_grad_thickness=True,
        need_grad_frequency=True,
    )
    rhs = (grads["grad_eps_r"].double() * v_eps.double()).sum()
    rhs = rhs + (grads["grad_sigma_e"].double() * v_sigma.double()).sum()
    rhs = rhs + (grads["grad_gain"].double() * v_gain.double()).sum()
    rhs = rhs + (grads["grad_thickness"].double() * v_thickness.double()).sum()
    rhs = rhs + grads["grad_frequency"].double().sum() * v_frequency
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_jvp_vjp_inner_product_duality_transmission():
    batch = _transmission_batch()
    generator = torch.Generator(device="cpu").manual_seed(31)
    u_coefficient = _loss_weights(2, seed=37)
    u_gain = torch.randn(2, generator=generator).to("cuda")
    v_thickness = torch.randn(3, generator=generator).to("cuda") * 0.01
    v_eps = torch.randn(3, generator=generator).to("cuda")
    v_sigma = torch.randn(3, generator=generator).to("cuda") * 0.01
    v_frequency = 1.0e6

    tangents = field_kernels.field_transmission_sequence_jvp(
        *batch.values(),
        frequency_hz=_FREQUENCY_HZ,
        tangent_layer_thickness_m=v_thickness,
        tangent_layer_eps_r=v_eps,
        tangent_layer_sigma_e=v_sigma,
        tangent_frequency=v_frequency,
    )
    lhs = _real_pair_loss(tangents["coefficient"], u_coefficient).double()
    lhs = lhs + (tangents["path_gain"].double() * u_gain.double()).sum()

    grads = field_kernels.field_transmission_sequence_backward(
        *batch.values(),
        frequency_hz=_FREQUENCY_HZ,
        grad_coefficient=u_coefficient,
        grad_path_gain=u_gain,
    )
    rhs = (grads["grad_layer_thickness_m"].double() * v_thickness.double()).sum()
    rhs = rhs + (grads["grad_layer_eps_r"].double() * v_eps.double()).sum()
    rhs = rhs + (grads["grad_layer_sigma_e"].double() * v_sigma.double()).sum()
    rhs = rhs + grads["grad_frequency"].double().sum() * v_frequency
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_reflection_gain_jvp_matches_fd():
    """Gain-tangent seeding through the reflection jvp (forward mode)."""

    batch = _reflection_batch(depth=2)
    generator = torch.Generator(device="cpu").manual_seed(47)
    v_gain = torch.randn(3, 2, generator=generator).to("cuda") * 0.1
    tangents = field_kernels.field_reflection_sequence_jvp(
        *batch.values(), frequency_hz=_FREQUENCY_HZ, tangent_gain=v_gain
    )
    assert float(tangents["coefficient"].abs().max()) > 0.0

    def evaluate(gain: torch.Tensor) -> torch.Tensor:
        fd_batch = dict(batch)
        fd_batch["gain"] = gain
        return field_kernels.field_reflection_sequence(
            *fd_batch.values(), frequency_hz=_FREQUENCY_HZ
        )["coefficient"]

    expected = central_difference_directional(
        evaluate, batch["gain"], v_gain, 1.0e-2
    )
    assert (
        relative_error(tangents["coefficient"], expected, abs_floor=ABS_TOL)
        <= REL_TOL_PATH
    )


def _direction_adjoint_identity(jvp_call, backward_call, seeds, tangents) -> None:
    """``<w, J v> == <J^T w, v>`` for the arrival-direction seam (ADR-043).

    An adjoint identity, not a finite difference: it is exact up to float32
    rounding and it falsifies a transposed, dropped, or double-counted seed,
    which is precisely what a hand-added cotangent input can get wrong.
    """

    forward = jvp_call()
    reverse = backward_call()
    left = float((seeds * forward["direction"]).sum())
    right = sum(
        float((reverse[f"grad_{name}"] * tangent).sum())
        for name, tangent in tangents.items()
    )
    assert relative_error(
        torch.tensor(left), torch.tensor(right), abs_floor=ABS_TOL
    ) <= REL_TOL_PATH
    assert abs(left) > 1.0e-6, "the identity must not be checked at zero"


def test_free_space_direction_seed_satisfies_the_adjoint_identity():
    batch = _free_space_batch()
    generator = torch.Generator(device="cpu").manual_seed(211)

    def rand():
        return torch.randn(4, 3, generator=generator).to("cuda").contiguous()

    seeds = rand()
    tangent_source = rand()
    tangent_target = rand()
    _direction_adjoint_identity(
        lambda: field_kernels.field_free_space_jvp(
            *batch.values(),
            frequency_hz=_FREQUENCY_HZ,
            tangent_frequency=0.0,
            tangent_source=tangent_source,
            tangent_target=tangent_target,
        ),
        lambda: field_kernels.field_free_space_backward(
            *batch.values(),
            frequency_hz=_FREQUENCY_HZ,
            grad_direction=seeds,
            need_grad_frequency=False,
            need_grad_geometry=True,
        ),
        seeds,
        {"source": tangent_source, "target": tangent_target},
    )


def test_reflection_direction_seed_satisfies_the_adjoint_identity():
    batch = _reflection_batch(depth=2)
    generator = torch.Generator(device="cpu").manual_seed(213)
    count = batch["source"].shape[0]
    depth = batch["interaction_positions"].shape[1]

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to("cuda").contiguous()

    seeds = rand(count, 3)
    tangent_source = rand(count, 3)
    tangent_target = rand(count, 3)
    tangent_positions = rand(count, depth, 3)
    _direction_adjoint_identity(
        lambda: field_kernels.field_reflection_sequence_jvp(
            *batch.values(),
            frequency_hz=_FREQUENCY_HZ,
            tangent_source=tangent_source,
            tangent_target=tangent_target,
            tangent_interaction_positions=tangent_positions,
        ),
        lambda: field_kernels.field_reflection_sequence_backward(
            *batch.values(),
            frequency_hz=_FREQUENCY_HZ,
            grad_direction=seeds,
            need_grad_eps_r=False,
            need_grad_sigma_e=False,
            need_grad_thickness=False,
            need_grad_frequency=False,
            need_grad_geometry=True,
        ),
        seeds,
        {
            "source": tangent_source,
            "target": tangent_target,
            "interaction_positions": tangent_positions,
        },
    )


def test_a_direction_seed_of_the_wrong_shape_fails_loudly():
    """No silent broadcast, no reshape, no ignored seed."""

    batch = _free_space_batch()
    with pytest.raises(RuntimeError):
        field_kernels.field_free_space_backward(
            *batch.values(),
            frequency_hz=_FREQUENCY_HZ,
            grad_direction=torch.zeros(4, device="cuda"),
            need_grad_geometry=True,
        )
    with pytest.raises(RuntimeError):
        field_kernels.field_reflection_sequence_backward(
            *_reflection_batch(depth=1).values(),
            frequency_hz=_FREQUENCY_HZ,
            grad_direction=torch.zeros(3, 4, device="cuda"),
            need_grad_geometry=True,
        )


def test_a_direction_seed_alone_still_launches_the_geometry_adjoint():
    """The seed is a real cotangent input, not a passenger of another one.

    With every other cotangent absent the launch used to be skipped entirely,
    so a direction-only loss would have come back as an exact zero.
    """

    batch = _free_space_batch()
    seeds = torch.ones(4, 3, device="cuda")
    grads = field_kernels.field_free_space_backward(
        *batch.values(),
        frequency_hz=_FREQUENCY_HZ,
        grad_direction=seeds,
        need_grad_frequency=False,
        need_grad_geometry=True,
    )
    assert float(grads["grad_source"].abs().sum()) > 0.0
    assert float(grads["grad_target"].abs().sum()) > 0.0


def test_free_space_vjp_matches_sum_of_jvp():
    batch = _free_space_batch()
    ones = torch.ones(4, device="cuda")
    grads = field_kernels.field_free_space_backward(
        *batch.values(), frequency_hz=_FREQUENCY_HZ, grad_path_gain=ones
    )
    tangents = field_kernels.field_free_space_jvp(
        *batch.values(), frequency_hz=_FREQUENCY_HZ, tangent_frequency=1.0
    )
    assert (
        relative_error(
            grads["grad_frequency"].sum(), tangents["path_gain"].sum(), abs_floor=ABS_TOL
        )
        <= REL_TOL_PATH
    )


# ---------------------------------------------------------------------------
# torch.autograd.gradcheck.
# ---------------------------------------------------------------------------


def test_free_space_gradcheck_strict_float64():
    batch = _free_space_batch(dtype=torch.float64)

    def func(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = field_kernels.field_free_space_ad(*batch.values(), frequency=scale * _FREQUENCY_HZ)
        return out["coefficient"], out["path_gain"]

    scale = torch.tensor(1.0, dtype=torch.float64, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(
        func, (scale,), eps=1.0e-6, atol=1.0e-9, rtol=1.0e-5, nondet_tol=1.0e-10
    )
    assert torch.autograd.gradcheck(
        func,
        (scale,),
        eps=1.0e-6,
        atol=1.0e-9,
        rtol=1.0e-5,
        nondet_tol=1.0e-10,
        check_forward_ad=True,
        check_backward_ad=False,
        check_undefined_grad=False,
        check_batched_grad=False,
    )


def test_reflection_gradcheck_float32_relaxed():
    batch = _reflection_batch(depth=1)

    def func(eps_r: torch.Tensor) -> torch.Tensor:
        ad_batch = dict(batch)
        ad_batch["eps_r"] = eps_r
        return field_kernels.field_reflection_sequence_ad(
            *ad_batch.values(), frequency=_FREQUENCY_HZ
        )["coefficient"]

    eps_r = batch["eps_r"].clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        func, (eps_r,), eps=1.0e-2, atol=1.0e-5, rtol=5.0e-2, nondet_tol=1.0e-6
    )


def test_transmission_gradcheck_float32_relaxed():
    batch = _transmission_batch()

    def func(layer_eps_r: torch.Tensor) -> torch.Tensor:
        ad_batch = dict(batch)
        ad_batch["layer_eps_r"] = layer_eps_r
        return field_kernels.field_transmission_sequence_ad(
            *ad_batch.values(), frequency=_FREQUENCY_HZ
        )["coefficient"]

    layer_eps_r = batch["layer_eps_r"].clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        func, (layer_eps_r,), eps=1.0e-2, atol=1.0e-5, rtol=5.0e-2, nondet_tol=1.0e-6
    )


# ---------------------------------------------------------------------------
# Forward-mode entry points (torch.func.jvp and dual tensors).
# ---------------------------------------------------------------------------


def test_functorch_jvp_reflection_matches_native():
    batch = _reflection_batch(depth=1)
    generator = torch.Generator(device="cpu").manual_seed(41)
    tangent = torch.randn(3, 1, generator=generator).to("cuda")

    expected = field_kernels.field_reflection_sequence_jvp(
        *batch.values(), frequency_hz=_FREQUENCY_HZ, tangent_eps_r=tangent
    )

    def f(eps_r: torch.Tensor) -> torch.Tensor:
        ad_batch = dict(batch)
        ad_batch["eps_r"] = eps_r
        return field_kernels.field_reflection_sequence_ad(
            *ad_batch.values(), frequency=_FREQUENCY_HZ
        )["coefficient"]

    _, tangent_out = torch.func.jvp(f, (batch["eps_r"],), (tangent,))
    assert float(expected["coefficient"].abs().max()) > 0.0
    assert (
        relative_error(tangent_out, expected["coefficient"], abs_floor=ABS_TOL)
        <= REL_TOL_PATH
    )


def test_forward_dual_free_space_matches_native_jvp():
    batch = _free_space_batch()
    expected = field_kernels.field_free_space_jvp(
        *batch.values(), frequency_hz=_FREQUENCY_HZ, tangent_frequency=1.0e6
    )
    with torch.autograd.forward_ad.dual_level():
        frequency = torch.autograd.forward_ad.make_dual(
            torch.tensor(_FREQUENCY_HZ, device="cuda"),
            torch.tensor(1.0e6, device="cuda"),
        )
        out = field_kernels.field_free_space_ad(*batch.values(), frequency=frequency)
        tangent = torch.autograd.forward_ad.unpack_dual(out["coefficient"]).tangent
    assert tangent is not None
    assert (
        relative_error(tangent, expected["coefficient"], abs_floor=ABS_TOL)
        <= REL_TOL_PATH
    )


# ---------------------------------------------------------------------------
# Explicit failure contract: fixed inputs never return silent zeros.
# ---------------------------------------------------------------------------


def test_fixed_inputs_fail_loudly():
    """tx_power and the polarizations stay fixed under AD-2 (plan 07)."""

    batch = _free_space_batch()
    tx_power = batch["tx_power"].clone().requires_grad_(True)
    out = field_kernels.field_free_space_ad(
        batch["source"],
        batch["target"],
        tx_power,
        batch["tx_polarization"],
        batch["rx_polarization"],
        frequency=_FREQUENCY_HZ,
    )
    with pytest.raises(NotImplementedError, match="tx_power"):
        out["coefficient"].real.sum().backward()

    reflection = _reflection_batch(depth=1)
    tx_pol = reflection["tx_polarization"].clone().requires_grad_(True)
    ad_batch = dict(reflection)
    ad_batch["tx_polarization"] = tx_pol
    out = field_kernels.field_reflection_sequence_ad(*ad_batch.values(), frequency=_FREQUENCY_HZ)
    with pytest.raises(NotImplementedError, match="tx_polarization"):
        out["coefficient"].real.sum().backward()

    transmission = _transmission_batch()
    rx_pol = transmission["rx_polarization"].clone().requires_grad_(True)
    ad_batch = dict(transmission)
    ad_batch["rx_polarization"] = rx_pol
    out = field_kernels.field_transmission_sequence_ad(
        *ad_batch.values(), frequency=_FREQUENCY_HZ
    )
    with pytest.raises(NotImplementedError, match="rx_polarization"):
        out["coefficient"].real.sum().backward()


def test_mu_r_gradients_fail_loudly():
    batch = _reflection_batch(depth=1)
    mu_r = batch["mu_r"].clone().requires_grad_(True)
    ad_batch = dict(batch)
    ad_batch["mu_r"] = mu_r
    out = field_kernels.field_reflection_sequence_ad(*ad_batch.values(), frequency=_FREQUENCY_HZ)
    with pytest.raises(NotImplementedError, match="mu_r"):
        out["coefficient"].real.sum().backward()

    transmission = _transmission_batch()
    layer_mu_r = transmission["layer_mu_r"].clone().requires_grad_(True)
    ad_batch = dict(transmission)
    ad_batch["layer_mu_r"] = layer_mu_r
    out = field_kernels.field_transmission_sequence_ad(
        *ad_batch.values(), frequency=_FREQUENCY_HZ
    )
    with pytest.raises(NotImplementedError, match="layer_mu_r"):
        out["coefficient"].real.sum().backward()


def test_non_differentiable_outputs_stay_detached():
    batch = _reflection_batch(depth=1)
    eps_r = batch["eps_r"].clone().requires_grad_(True)
    ad_batch = dict(batch)
    ad_batch["eps_r"] = eps_r
    out = field_kernels.field_reflection_sequence_ad(*ad_batch.values(), frequency=_FREQUENCY_HZ)
    assert out["coefficient"].requires_grad
    assert out["path_gain"].requires_grad
    for name in ("path_length_m", "delay_s", "direction"):
        assert not out[name].requires_grad
        assert out[name].grad_fn is None


def test_double_backward_raises():
    """ADR-043: the second-order request fails at the request, by owner name.

    Before ADR-043 this returned a silently detached first gradient and only
    failed one step later, with a generic Torch message that named Torch rather
    than the owner that cannot answer. The raise now happens inside the very
    backward that ``create_graph=True`` asked to be differentiable, before any
    native companion launches, so no partial second-order result exists.
    """

    batch = _reflection_batch(depth=1)
    eps_r = batch["eps_r"].clone().requires_grad_(True)
    ad_batch = dict(batch)
    ad_batch["eps_r"] = eps_r
    out = field_kernels.field_reflection_sequence_ad(*ad_batch.values(), frequency=_FREQUENCY_HZ)
    with pytest.raises(NotImplementedError, match="first-order only") as raised:
        torch.autograd.grad(out["path_gain"].sum(), eps_r, create_graph=True)
    assert "_FieldReflectionSequenceAdFunction.backward" in str(raised.value)
    # The first-order request over the same graph still works, unchanged.
    (grad,) = torch.autograd.grad(out["path_gain"].sum(), eps_r)
    assert grad.requires_grad is False


def test_composed_functorch_transforms_raise():
    batch = _reflection_batch(depth=1)
    generator = torch.Generator(device="cpu").manual_seed(43)
    direction = torch.randn(3, 1, generator=generator).to("cuda")

    def scalar(eps_r: torch.Tensor) -> torch.Tensor:
        ad_batch = dict(batch)
        ad_batch["eps_r"] = eps_r
        return field_kernels.field_reflection_sequence_ad(
            *ad_batch.values(), frequency=_FREQUENCY_HZ
        )["path_gain"].sum()

    def jvp_scalar(eps_r: torch.Tensor) -> torch.Tensor:
        _, tangent = torch.func.jvp(scalar, (eps_r,), (direction,))
        return tangent

    with pytest.raises(NotImplementedError):
        torch.func.grad(jvp_scalar)(batch["eps_r"])


# ---------------------------------------------------------------------------
# Geometry gradients (plan 07 AD-2): source / target / interaction_positions /
# interaction_normals against the complex128 reference oracle, forward-mode
# tangents against the oracle jvp, inner-product duality on geometry seeds,
# and the differentiable path_length_m / delay_s outputs.
# ---------------------------------------------------------------------------

_FREE_SPACE_GEOMETRY = ("source", "target")
_REFLECTION_GEOMETRY = (
    "source",
    "target",
    "interaction_positions",
    "interaction_normals",
)
_TRANSMISSION_GEOMETRY = ("source", "target", "interaction_normals")


def _reference_reflection_length(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    previous = batch["source"]
    depth = batch["interaction_positions"].shape[1]
    total = torch.zeros(
        batch["source"].shape[0], dtype=torch.float64, device="cuda"
    )
    for bounce in range(depth):
        hit = batch["interaction_positions"][:, bounce]
        total = total + torch.linalg.vector_norm(hit - previous, dim=-1)
        previous = hit
    return total + torch.linalg.vector_norm(batch["target"] - previous, dim=-1)


def test_free_space_geometry_vjp_matches_reference_oracle():
    batch = _free_space_batch()
    weights = _loss_weights(4, seed=71)
    leaves = {}
    ad_batch = dict(batch)
    for name in _FREE_SPACE_GEOMETRY:
        leaves[name] = batch[name].clone().requires_grad_(True)
        ad_batch[name] = leaves[name]
    out = field_kernels.field_free_space_ad(*ad_batch.values(), frequency=_FREQUENCY_HZ)
    loss = _real_pair_loss(out["coefficient"], weights) + out["path_gain"].sum()
    loss.backward()

    reference_batch = _reference_inputs(batch)
    reference_leaves = {}
    for name in _FREE_SPACE_GEOMETRY:
        reference_leaves[name] = reference_batch[name].clone().requires_grad_(True)
        reference_batch[name] = reference_leaves[name]
    reference = free_space_reference(
        **reference_batch,
        frequency=torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda"),
    )
    reference_loss = _real_pair_loss(reference["coefficient"], weights)
    reference_loss = reference_loss + reference["path_gain"].sum()
    expected = torch.autograd.grad(reference_loss, tuple(reference_leaves.values()))
    for (name, leaf), expected_grad in zip(leaves.items(), expected, strict=True):
        assert leaf.grad is not None, name
        assert (
            relative_error(leaf.grad, expected_grad, abs_floor=ABS_TOL) <= REL_TOL_PATH
        ), name


@pytest.mark.parametrize("depth", [1, 2])
def test_reflection_geometry_vjp_matches_reference_oracle(depth):
    batch = _reflection_batch(depth=depth)
    weights = _loss_weights(3, seed=73)
    leaves = {}
    ad_batch = dict(batch)
    for name in _REFLECTION_GEOMETRY:
        leaves[name] = batch[name].clone().requires_grad_(True)
        ad_batch[name] = leaves[name]
    out = field_kernels.field_reflection_sequence_ad(*ad_batch.values(), frequency=_FREQUENCY_HZ)
    loss = _real_pair_loss(out["coefficient"], weights) + out["path_gain"].sum()
    loss.backward()

    reference_batch = _reference_inputs(batch)
    reference_leaves = {}
    for name in _REFLECTION_GEOMETRY:
        reference_leaves[name] = reference_batch[name].clone().requires_grad_(True)
        reference_batch[name] = reference_leaves[name]
    reference = reflection_sequence_reference(
        **reference_batch,
        frequency=torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda"),
    )
    reference_loss = _real_pair_loss(reference["coefficient"], weights)
    reference_loss = reference_loss + reference["path_gain"].sum()
    expected = torch.autograd.grad(reference_loss, tuple(reference_leaves.values()))
    for (name, leaf), expected_grad in zip(leaves.items(), expected, strict=True):
        assert leaf.grad is not None, name
        assert (
            relative_error(leaf.grad, expected_grad, abs_floor=ABS_TOL) <= REL_TOL_PATH
        ), name


def test_transmission_geometry_vjp_matches_reference_oracle():
    batch = _transmission_batch()
    weights = _loss_weights(2, seed=79)
    leaves = {}
    ad_batch = dict(batch)
    for name in _TRANSMISSION_GEOMETRY:
        leaves[name] = batch[name].clone().requires_grad_(True)
        ad_batch[name] = leaves[name]
    # The crossing points do not enter the straight-path field; their
    # gradient is exactly zero (delivered as None by the Function).
    positions = batch["interaction_positions"].clone().requires_grad_(True)
    ad_batch["interaction_positions"] = positions
    out = field_kernels.field_transmission_sequence_ad(
        *ad_batch.values(), frequency=_FREQUENCY_HZ
    )
    loss = _real_pair_loss(out["coefficient"], weights) + out["path_gain"].sum()
    loss.backward()
    assert positions.grad is None or float(positions.grad.abs().max()) == 0.0

    reference_batch = _reference_inputs(batch)
    del reference_batch["path_valid"]
    del reference_batch["interaction_positions"]
    reference_leaves = {}
    for name in _TRANSMISSION_GEOMETRY:
        reference_leaves[name] = reference_batch[name].clone().requires_grad_(True)
        reference_batch[name] = reference_leaves[name]
    reference = transmission_sequence_reference(
        **reference_batch,
        frequency=torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda"),
    )
    reference_loss = _real_pair_loss(reference["coefficient"], weights)
    reference_loss = reference_loss + reference["path_gain"].sum()
    expected = torch.autograd.grad(reference_loss, tuple(reference_leaves.values()))
    for (name, leaf), expected_grad in zip(leaves.items(), expected, strict=True):
        assert leaf.grad is not None, name
        assert (
            relative_error(leaf.grad, expected_grad, abs_floor=ABS_TOL) <= REL_TOL_PATH
        ), name


def test_free_space_geometry_jvp_matches_reference_oracle():
    batch = _free_space_batch()
    generator = torch.Generator(device="cpu").manual_seed(83)
    v_source = (torch.randn(4, 3, generator=generator) * 0.01).cuda()
    v_target = (torch.randn(4, 3, generator=generator) * 0.01).cuda()
    tangents = field_kernels.field_free_space_jvp(
        *batch.values(),
        frequency_hz=_FREQUENCY_HZ,
        tangent_frequency=0.0,
        tangent_source=v_source,
        tangent_target=v_target,
    )
    reference_batch = _reference_inputs(batch)

    def f(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        local = dict(reference_batch)
        local["source"] = source
        local["target"] = target
        return free_space_reference(
            **local,
            frequency=torch.tensor(
                _FREQUENCY_HZ, dtype=torch.float64, device="cuda"
            ),
        )["coefficient"]

    _, expected = torch.func.jvp(
        f,
        (reference_batch["source"], reference_batch["target"]),
        (v_source.double(), v_target.double()),
    )
    assert (
        relative_error(tangents["coefficient"], expected, abs_floor=ABS_TOL)
        <= REL_TOL_PATH
    )


@pytest.mark.parametrize("depth", [1, 2])
def test_reflection_geometry_jvp_matches_reference_oracle(depth):
    batch = _reflection_batch(depth=depth)
    generator = torch.Generator(device="cpu").manual_seed(89)
    seeds = {
        "tangent_source": (torch.randn(3, 3, generator=generator) * 0.01).cuda(),
        "tangent_target": (torch.randn(3, 3, generator=generator) * 0.01).cuda(),
        "tangent_interaction_positions": (
            torch.randn(3, depth, 3, generator=generator) * 0.01
        ).cuda(),
        "tangent_interaction_normals": (
            torch.randn(3, depth, 3, generator=generator) * 0.01
        ).cuda(),
    }
    tangents = field_kernels.field_reflection_sequence_jvp(
        *batch.values(), frequency_hz=_FREQUENCY_HZ, **seeds
    )
    assert float(tangents["coefficient"].abs().max()) > 0.0
    reference_batch = _reference_inputs(batch)

    def f(source, target, positions, normals) -> torch.Tensor:
        local = dict(reference_batch)
        local["source"] = source
        local["target"] = target
        local["interaction_positions"] = positions
        local["interaction_normals"] = normals
        return reflection_sequence_reference(
            **local,
            frequency=torch.tensor(
                _FREQUENCY_HZ, dtype=torch.float64, device="cuda"
            ),
        )["coefficient"]

    _, expected = torch.func.jvp(
        f,
        (
            reference_batch["source"],
            reference_batch["target"],
            reference_batch["interaction_positions"],
            reference_batch["interaction_normals"],
        ),
        (
            seeds["tangent_source"].double(),
            seeds["tangent_target"].double(),
            seeds["tangent_interaction_positions"].double(),
            seeds["tangent_interaction_normals"].double(),
        ),
    )
    assert (
        relative_error(tangents["coefficient"], expected, abs_floor=ABS_TOL)
        <= REL_TOL_PATH
    )


def test_transmission_geometry_jvp_matches_reference_oracle():
    """Native geometry jvp against the oracle through random projections.

    The transmission reference reads discrete winner data with Python scalar
    conversions, so instead of torch.func.jvp the oracle directional
    derivative is evaluated as <grad(loss_u), v> for a random output
    projection u; the native side is the same projection of the jvp output.
    """

    batch = _transmission_batch()
    generator = torch.Generator(device="cpu").manual_seed(97)
    weights = _loss_weights(2, seed=101)
    seeds = {
        "tangent_source": (torch.randn(2, 3, generator=generator) * 0.01).cuda(),
        "tangent_target": (torch.randn(2, 3, generator=generator) * 0.01).cuda(),
        "tangent_interaction_normals": (
            torch.randn(2, 2, 3, generator=generator) * 0.01
        ).cuda(),
    }
    tangents = field_kernels.field_transmission_sequence_jvp(
        *batch.values(), frequency_hz=_FREQUENCY_HZ, **seeds
    )
    lhs = _real_pair_loss(tangents["coefficient"], weights).double()

    reference_batch = _reference_inputs(batch)
    del reference_batch["path_valid"]
    del reference_batch["interaction_positions"]
    reference_leaves = {}
    for name in _TRANSMISSION_GEOMETRY:
        reference_leaves[name] = reference_batch[name].clone().requires_grad_(True)
        reference_batch[name] = reference_leaves[name]
    reference = transmission_sequence_reference(
        **reference_batch,
        frequency=torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda"),
    )
    expected_grads = torch.autograd.grad(
        _real_pair_loss(reference["coefficient"], weights),
        tuple(reference_leaves.values()),
    )
    rhs = torch.zeros((), dtype=torch.float64)
    seed_values = (
        seeds["tangent_source"],
        seeds["tangent_target"],
        seeds["tangent_interaction_normals"],
    )
    for expected_grad, seed in zip(expected_grads, seed_values, strict=True):
        rhs = rhs + (expected_grad.double().cpu() * seed.double().cpu()).sum()
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_jvp_vjp_inner_product_duality_geometry_free_space():
    batch = _free_space_batch()
    generator = torch.Generator(device="cpu").manual_seed(103)
    u_coefficient = _loss_weights(4, seed=107)
    u_gain = torch.randn(4, generator=generator).to("cuda")
    u_length = torch.randn(4, generator=generator).to("cuda")
    v_source = (torch.randn(4, 3, generator=generator) * 0.01).cuda()
    v_target = (torch.randn(4, 3, generator=generator) * 0.01).cuda()

    tangents = field_kernels.field_free_space_jvp(
        *batch.values(),
        frequency_hz=_FREQUENCY_HZ,
        tangent_frequency=0.0,
        tangent_source=v_source,
        tangent_target=v_target,
    )
    lhs = _real_pair_loss(tangents["coefficient"], u_coefficient).double()
    lhs = lhs + (tangents["path_gain"].double() * u_gain.double()).sum()
    lhs = lhs + (tangents["path_length_m"].double() * u_length.double()).sum()

    grads = field_kernels.field_free_space_backward(
        *batch.values(),
        frequency_hz=_FREQUENCY_HZ,
        grad_coefficient=u_coefficient,
        grad_path_gain=u_gain,
        grad_path_length=u_length,
        need_grad_frequency=False,
        need_grad_geometry=True,
    )
    rhs = (grads["grad_source"].double() * v_source.double()).sum()
    rhs = rhs + (grads["grad_target"].double() * v_target.double()).sum()
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_jvp_vjp_inner_product_duality_geometry_reflection():
    batch = _reflection_batch(depth=2)
    generator = torch.Generator(device="cpu").manual_seed(109)
    u_coefficient = _loss_weights(3, seed=113)
    u_gain = torch.randn(3, generator=generator).to("cuda")
    u_length = torch.randn(3, generator=generator).to("cuda")
    seeds = {
        "tangent_source": (torch.randn(3, 3, generator=generator) * 0.01).cuda(),
        "tangent_target": (torch.randn(3, 3, generator=generator) * 0.01).cuda(),
        "tangent_interaction_positions": (
            torch.randn(3, 2, 3, generator=generator) * 0.01
        ).cuda(),
        "tangent_interaction_normals": (
            torch.randn(3, 2, 3, generator=generator) * 0.01
        ).cuda(),
    }

    tangents = field_kernels.field_reflection_sequence_jvp(
        *batch.values(), frequency_hz=_FREQUENCY_HZ, **seeds
    )
    lhs = _real_pair_loss(tangents["coefficient"], u_coefficient).double()
    lhs = lhs + (tangents["path_gain"].double() * u_gain.double()).sum()
    lhs = lhs + (tangents["path_length_m"].double() * u_length.double()).sum()

    grads = field_kernels.field_reflection_sequence_backward(
        *batch.values(),
        frequency_hz=_FREQUENCY_HZ,
        grad_coefficient=u_coefficient,
        grad_path_gain=u_gain,
        grad_path_length=u_length,
        need_grad_eps_r=False,
        need_grad_sigma_e=False,
        need_grad_gain=False,
        need_grad_thickness=False,
        need_grad_frequency=False,
        need_grad_geometry=True,
    )
    rhs = (grads["grad_source"].double() * seeds["tangent_source"].double()).sum()
    rhs = rhs + (grads["grad_target"].double() * seeds["tangent_target"].double()).sum()
    rhs = rhs + (
        grads["grad_interaction_positions"].double()
        * seeds["tangent_interaction_positions"].double()
    ).sum()
    rhs = rhs + (
        grads["grad_interaction_normals"].double()
        * seeds["tangent_interaction_normals"].double()
    ).sum()
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_jvp_vjp_inner_product_duality_geometry_transmission():
    batch = _transmission_batch()
    generator = torch.Generator(device="cpu").manual_seed(127)
    u_coefficient = _loss_weights(2, seed=131)
    u_gain = torch.randn(2, generator=generator).to("cuda")
    u_length = torch.randn(2, generator=generator).to("cuda")
    seeds = {
        "tangent_source": (torch.randn(2, 3, generator=generator) * 0.01).cuda(),
        "tangent_target": (torch.randn(2, 3, generator=generator) * 0.01).cuda(),
        "tangent_interaction_normals": (
            torch.randn(2, 2, 3, generator=generator) * 0.01
        ).cuda(),
    }

    tangents = field_kernels.field_transmission_sequence_jvp(
        *batch.values(), frequency_hz=_FREQUENCY_HZ, **seeds
    )
    lhs = _real_pair_loss(tangents["coefficient"], u_coefficient).double()
    lhs = lhs + (tangents["path_gain"].double() * u_gain.double()).sum()
    lhs = lhs + (tangents["path_length_m"].double() * u_length.double()).sum()

    grads = field_kernels.field_transmission_sequence_backward(
        *batch.values(),
        frequency_hz=_FREQUENCY_HZ,
        grad_coefficient=u_coefficient,
        grad_path_gain=u_gain,
        grad_path_length=u_length,
        need_grad_layer_thickness=False,
        need_grad_layer_eps_r=False,
        need_grad_layer_sigma_e=False,
        need_grad_frequency=False,
        need_grad_geometry=True,
    )
    rhs = (grads["grad_source"].double() * seeds["tangent_source"].double()).sum()
    rhs = rhs + (grads["grad_target"].double() * seeds["tangent_target"].double()).sum()
    rhs = rhs + (
        grads["grad_interaction_normals"].double()
        * seeds["tangent_interaction_normals"].double()
    ).sum()
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_free_space_path_length_grads_match_reference():
    batch = _free_space_batch()
    leaves = {}
    ad_batch = dict(batch)
    for name in _FREE_SPACE_GEOMETRY:
        leaves[name] = batch[name].clone().requires_grad_(True)
        ad_batch[name] = leaves[name]
    out = field_kernels.field_free_space_ad(*ad_batch.values(), frequency=_FREQUENCY_HZ)
    assert out["path_length_m"].requires_grad
    assert out["delay_s"].requires_grad
    loss = out["path_length_m"].sum() + out["delay_s"].sum() * 2.99792458e8
    loss.backward()

    reference_batch = _reference_inputs(batch)
    source = reference_batch["source"].clone().requires_grad_(True)
    target = reference_batch["target"].clone().requires_grad_(True)
    length = torch.linalg.vector_norm(target - source, dim=-1)
    reference_loss = length.sum() + (length / 299792458.0).sum() * 2.99792458e8
    expected = torch.autograd.grad(reference_loss, (source, target))
    for (name, leaf), expected_grad in zip(leaves.items(), expected, strict=True):
        assert (
            relative_error(leaf.grad, expected_grad, abs_floor=ABS_TOL) <= REL_TOL_PATH
        ), name


def test_reflection_path_length_grads_match_reference():
    batch = _reflection_batch(depth=2)
    leaves = {}
    ad_batch = dict(batch)
    for name in ("source", "target", "interaction_positions"):
        leaves[name] = batch[name].clone().requires_grad_(True)
        ad_batch[name] = leaves[name]
    out = field_kernels.field_reflection_sequence_ad(*ad_batch.values(), frequency=_FREQUENCY_HZ)
    assert out["path_length_m"].requires_grad
    out["path_length_m"].sum().backward()

    reference_batch = _reference_inputs(batch)
    reference_leaves = {}
    for name in ("source", "target", "interaction_positions"):
        reference_leaves[name] = reference_batch[name].clone().requires_grad_(True)
        reference_batch[name] = reference_leaves[name]
    expected = torch.autograd.grad(
        _reference_reflection_length(reference_batch).sum(),
        tuple(reference_leaves.values()),
    )
    for (name, leaf), expected_grad in zip(leaves.items(), expected, strict=True):
        assert (
            relative_error(leaf.grad, expected_grad, abs_floor=ABS_TOL) <= REL_TOL_PATH
        ), name

    # Materials receive an exactly zero cotangent from the length outputs.
    eps_r = batch["eps_r"].clone().requires_grad_(True)
    material_batch = dict(batch)
    material_batch["eps_r"] = eps_r
    material_batch["source"] = batch["source"].clone().requires_grad_(True)
    out = field_kernels.field_reflection_sequence_ad(
        *material_batch.values(), frequency=_FREQUENCY_HZ
    )
    (grad_eps,) = torch.autograd.grad(
        out["path_length_m"].sum(), eps_r, allow_unused=True
    )
    assert grad_eps is None or float(grad_eps.abs().max()) == 0.0


def test_transmission_path_length_grads_match_reference():
    batch = _transmission_batch()
    leaves = {}
    ad_batch = dict(batch)
    for name in ("source", "target"):
        leaves[name] = batch[name].clone().requires_grad_(True)
        ad_batch[name] = leaves[name]
    out = field_kernels.field_transmission_sequence_ad(
        *ad_batch.values(), frequency=_FREQUENCY_HZ
    )
    assert out["path_length_m"].requires_grad
    out["path_length_m"].sum().backward()

    reference_batch = _reference_inputs(batch)
    source = reference_batch["source"].clone().requires_grad_(True)
    target = reference_batch["target"].clone().requires_grad_(True)
    expected = torch.autograd.grad(
        torch.linalg.vector_norm(target - source, dim=-1).sum(), (source, target)
    )
    for (name, leaf), expected_grad in zip(leaves.items(), expected, strict=True):
        assert (
            relative_error(leaf.grad, expected_grad, abs_floor=ABS_TOL) <= REL_TOL_PATH
        ), name


def test_path_outputs_differentiable_only_with_geometry():
    """path_length_m / delay_s join the graph exactly when geometry does."""

    batch = _reflection_batch(depth=1)
    source = batch["source"].clone().requires_grad_(True)
    ad_batch = dict(batch)
    ad_batch["source"] = source
    out = field_kernels.field_reflection_sequence_ad(*ad_batch.values(), frequency=_FREQUENCY_HZ)
    assert out["path_length_m"].requires_grad
    assert out["delay_s"].requires_grad
    assert not out["direction"].requires_grad
    assert out["direction"].grad_fn is None