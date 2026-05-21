from __future__ import annotations

import torch

import witwin.channel as wc


def _payload(*, num_rx=2, num_rx_ant=3, num_tx=2, num_tx_ant=4, max_num_paths=5, max_depth=2):
    path_shape = (num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths)
    coeff_shape = (*path_shape, 1)
    depth_shape = (*path_shape, max_depth)
    num_entries = int(torch.tensor(coeff_shape).prod().item())
    coeff = torch.arange(num_entries, dtype=torch.float32).reshape(coeff_shape).to(torch.complex64)
    tau = torch.zeros(path_shape, dtype=torch.float32)
    tau[..., 1] = 1.0e-9
    valid = torch.zeros(path_shape, dtype=torch.bool)
    valid[..., 0] = True
    valid[..., 1] = True
    return {
        "name": "rx",
        "num_rx": num_rx,
        "num_rx_ant": num_rx_ant,
        "num_tx": num_tx,
        "num_tx_ant": num_tx_ant,
        "num_time_steps": 1,
        "max_num_paths": max_num_paths,
        "max_depth": max_depth,
        "tx_pos": (0.0, 0.0, 0.0),
        "tx_positions": torch.zeros((num_tx, 3), dtype=torch.float32),
        "rx_positions": torch.zeros((num_rx, 3), dtype=torch.float32),
        "frequency": 3.5e9,
        "wavelength": 0.0857,
        "a": coeff,
        "tau": tau,
        "theta_t": torch.zeros(path_shape, dtype=torch.float32),
        "phi_t": torch.zeros(path_shape, dtype=torch.float32),
        "theta_r": torch.zeros(path_shape, dtype=torch.float32),
        "phi_r": torch.zeros(path_shape, dtype=torch.float32),
        "valid": valid,
        "types": torch.zeros(depth_shape, dtype=torch.int32),
        "num_paths": valid.to(torch.int32).sum(dim=-1),
        "vertices": torch.zeros((*depth_shape, 3), dtype=torch.float32),
        "normals": torch.zeros((*depth_shape, 3), dtype=torch.float32),
        "objects": torch.full(depth_shape, -1, dtype=torch.int32),
        "metadata": {},
    }


def test_path_result_uses_sionna_antenna_and_time_axes():
    result = wc.path.PathResult._from_payload(_payload())

    assert result.path_shape == (2, 3, 2, 4, 5)
    assert result.coeff_shape == (2, 3, 2, 4, 5, 1)
    assert result.depth_shape == (2, 3, 2, 4, 5, 2)

    a, tau = result.cir()
    assert tuple(a.shape) == result.coeff_shape
    assert tuple(tau.shape) == result.path_shape

    response = result.cfr(torch.linspace(3.49e9, 3.51e9, 7))
    assert tuple(response.shape) == (2, 3, 2, 4, 1, 7)

    taps = result.taps(bandwidth=1.0e9, num_taps=4)
    assert tuple(taps.shape) == (2, 3, 2, 4, 1, 4)
