import pytest
import torch

from witwin.channel_native.propagation.geometry.kernels import bridge as ops
from witwin.channel_native.runtime import symbols


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_diffraction_discover_edges_uses_prim_id_and_best_edge_filter():
    symbols.native_extension()
    device = torch.device("cuda")
    tx_pos = torch.tensor([0.0, -1.0, 0.5], device=device, dtype=torch.float32)
    ray_dir = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=torch.float32)
    prim_index = torch.tensor([0], device=device, dtype=torch.int32)
    hit_p = torch.tensor([[2.0, 0.0, 0.5]], device=device, dtype=torch.float32)
    hit_n = torch.tensor([[0.0, -1.0, 0.0]], device=device, dtype=torch.float32)
    hit_geo_n = hit_n.clone()
    triangle_edge_count = torch.tensor([3, 0], device=device, dtype=torch.int32)
    triangle_edge_indices = torch.tensor([[0, 1, 2], [-1, -1, -1]], device=device, dtype=torch.int32)
    edge_pos = torch.tensor(
        [
            [2.0, 0.0, -1.0],
            [2.0, 0.0, -1.0],
            [2.0, 2.0, -1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    edge_dir = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, -0.70710678, 0.70710678],
        ],
        device=device,
        dtype=torch.float32,
    )
    edge_n0 = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    edge_nn = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    edge_line_min = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    edge_line_max = torch.tensor([3.0, 2.0, 2.8284271], device=device, dtype=torch.float32)
    adjacent_face1 = torch.tensor([1, -1, -1], device=device, dtype=torch.int32)

    edge_idx = ops.raydn_diffraction_discover_edges(
        tx_pos,
        ray_dir,
        prim_index,
        hit_p,
        hit_n,
        hit_geo_n,
        triangle_edge_count,
        triangle_edge_indices,
        edge_pos,
        edge_dir,
        edge_n0,
        edge_nn,
        edge_line_min,
        edge_line_max,
        adjacent_face1,
    )

    torch.cuda.synchronize()
    assert edge_idx.dtype == torch.int32
    assert edge_idx.tolist() == [0]
