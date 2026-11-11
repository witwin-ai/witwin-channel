# Copyright Xingyu Chen.
# Reflection off multilayer materials: the layer store, its derivatives and the refused ones.

from __future__ import annotations

import pytest
import torch

from tests.ad.test_field_em_ad import _FREQUENCY_HZ, _reflection_batch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="field kernels need CUDA")


def _layers(count: int, depth: int, *, multilayer: bool):
    from witwin.channel.kernels.fields import ReflectionLayers

    # Material 0: one layer; material 1: a three-layer stack.
    thickness = torch.tensor([0.1, 0.02, 0.15, 0.02], device="cuda")
    eps = torch.tensor([4.0, 3.0, 6.5, 3.0], device="cuda")
    sigma = torch.tensor([0.02, 0.001, 0.05, 0.001], device="cuda")
    mu = torch.ones(4, device="cuda")
    material = torch.full((count, depth), 1 if multilayer else 0, device="cuda", dtype=torch.int32)
    return ReflectionLayers(
        bounce_material=material,
        layer_offset=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        layer_count=torch.tensor([1, 3], device="cuda", dtype=torch.int32),
        layer_thickness_m=thickness, layer_eps_r=eps, layer_sigma_e=sigma, layer_mu_r=mu,
        multilayer=multilayer,
    )


def test_single_layer_store_is_bitwise_the_scalar_slab():
    from witwin.channel.kernels import fields as field_kernels

    batch = _reflection_batch(depth=2)
    plain = field_kernels.field_reflection_sequence(*batch.values(), frequency_hz=_FREQUENCY_HZ)
    layered = field_kernels.field_reflection_sequence(
        *batch.values(), layers=_layers(3, 2, multilayer=False), frequency_hz=_FREQUENCY_HZ)
    for name in ("field_vector", "coefficient", "path_gain"):
        assert torch.equal(plain[name], layered[name]), name


def test_multilayer_store_changes_the_coefficient():
    from witwin.channel.kernels import fields as field_kernels

    batch = _reflection_batch(depth=1)
    plain = field_kernels.field_reflection_sequence(*batch.values(), frequency_hz=_FREQUENCY_HZ)
    layered = field_kernels.field_reflection_sequence(
        *batch.values(), layers=_layers(3, 1, multilayer=True), frequency_hz=_FREQUENCY_HZ)
    assert not torch.allclose(plain["coefficient"], layered["coefficient"])


def test_multilayer_material_derivatives_are_refused():
    from witwin.channel.kernels import fields as field_kernels

    batch = _reflection_batch(depth=1)
    with pytest.raises(NotImplementedError, match="multilayer"):
        field_kernels.field_reflection_sequence_backward(
            *batch.values(), layers=_layers(3, 1, multilayer=True), frequency_hz=_FREQUENCY_HZ,
            grad_path_gain=torch.ones(3, device="cuda"), need_grad_eps_r=True)


def test_multilayer_geometry_jvp_matches_a_central_difference():
    """The stack's incidence-angle derivative flows into the reflection-point tangent."""
    from witwin.channel.kernels import fields as field_kernels

    batch = {k: (v.double().float() if v.dtype == torch.float32 else v)
             for k, v in _reflection_batch(depth=1).items()}
    layers = _layers(3, 1, multilayer=True)
    direction = torch.zeros_like(batch["interaction_positions"])
    direction[..., 1] = 1.0
    tangent = field_kernels.field_reflection_sequence_jvp(
        *batch.values(), layers=layers, frequency_hz=_FREQUENCY_HZ,
        tangent_interaction_positions=direction)
    step = 1.0e-3

    def gain(offset: float) -> torch.Tensor:
        moved = dict(batch)
        moved["interaction_positions"] = batch["interaction_positions"] + offset * direction
        return field_kernels.field_reflection_sequence(
            *moved.values(), layers=layers, frequency_hz=_FREQUENCY_HZ)["path_gain"].double()

    fd = (gain(step) - gain(-step)) / (2.0 * step)
    jvp = tangent["path_gain"].double()
    assert torch.allclose(jvp, fd, rtol=2e-2, atol=1e-3 * float(fd.abs().max()))
