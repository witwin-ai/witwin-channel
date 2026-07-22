from __future__ import annotations

import pytest
import torch

from witwin.channel.runtime import autograd_contracts


CONTRACT_NAMES = (
    "_ad_active_ctx",
    "_ad_check_active",
    "_ad_check_optional_grad",
    "_ad_check_rows",
    "_ad_check_tangent_vec3",
    "_ad_checked_tangent",
    "_ad_frequency_grad",
    "_ad_frequency_tangent",
    "_ad_frequency_value",
    "_ad_geometry_live",
    "_ad_geometry_tangent",
    "_ad_native_tangent_or_none",
    "_ad_native_tensor",
    "_ad_raise_composed_transforms",
    "_ad_reject_fixed_inputs",
    "_ad_reject_fixed_tangents",
    "_ad_still_wrapped",
)


@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_autograd_contract_has_the_canonical_owner(name: str):
    owner = getattr(autograd_contracts, name)

    assert owner.__module__ == autograd_contracts.__name__


def test_active_context_preserves_or_creates_the_expected_mask():
    like = torch.empty((2, 3))
    active = torch.tensor([True, False])

    assert autograd_contracts._ad_active_ctx(active, like) is active
    empty = autograd_contracts._ad_active_ctx(None, like)
    assert empty.shape == (0,)
    assert empty.dtype == torch.bool
    assert empty.device == like.device


def test_composed_transform_error_remains_fail_loud():
    with pytest.raises(NotImplementedError, match="composed functorch transforms"):
        autograd_contracts._ad_raise_composed_transforms()
