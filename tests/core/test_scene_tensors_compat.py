from __future__ import annotations

from types import SimpleNamespace

import torch

from witwin.channel_native.core import scene_tensors
from witwin.channel_native.deterministic import solver


def test_frequency_scalar_preserves_detached_scalar_contract():
    assert scene_tensors._frequency_scalar(SimpleNamespace(frequency=2.4)) == 2.4

    frequency = torch.tensor(2.4, dtype=torch.float64, requires_grad=True)
    assert scene_tensors._frequency_scalar(SimpleNamespace(frequency=frequency)) == 2.4


def test_frequency_scalar_has_canonical_owner():
    owner = scene_tensors._frequency_scalar

    assert owner.__module__ == scene_tensors.__name__
    assert solver._frequency_scalar is owner
