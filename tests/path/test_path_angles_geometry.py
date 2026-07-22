import math

import torch

from witwin.channel.path.result import endpoint_angles


def test_endpoint_angles_use_standard_spherical_convention():
    theta, phi = endpoint_angles(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    )

    assert torch.allclose(theta, torch.tensor([math.pi / 2, math.pi / 2, 0.0]))
    assert torch.allclose(phi, torch.tensor([0.0, math.pi / 2, 0.0]))
