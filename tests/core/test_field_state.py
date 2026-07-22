import pytest
import torch

from witwin.channel import Complex3State, JonesState


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_complex3_and_jones_state_abi_is_explicit():
    field = torch.zeros((2, 3), device="cuda", dtype=torch.complex64)
    direction = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda"
    )
    complex3 = Complex3State(field=field, direction=direction)
    jones = JonesState(
        value=torch.zeros((2, 2), device="cuda", dtype=torch.complex64),
        basis=torch.tensor(
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            device="cuda",
        ),
        direction=direction,
    )

    assert complex3.field.shape == (2, 3)
    assert jones.value.shape == (2, 2)
    assert jones.basis.shape == (2, 2, 3)
