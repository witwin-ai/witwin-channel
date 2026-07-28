import pytest
import torch

from witwin.channel.montecarlo.basic.kernels import sampling as ops
from witwin.channel import runtime


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_diffraction_discover_edges_uses_prim_id_and_best_edge_filter():
    runtime.native_extension()
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

    edge_idx = ops.mc_diffraction_discover_edges(
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_diffraction_discover_edges_rejects_non_vector_tx_position():
    bad_tx_pos = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
    placeholder = torch.empty(0, device="cuda")

    with pytest.raises(ValueError, match="tx_pos"):
        ops.mc_diffraction_discover_edges(bad_tx_pos, *(placeholder,) * 14)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_counted_diffraction_discovery_requires_int32_hit_count():
    tx_pos = torch.zeros(3, device="cuda", dtype=torch.float32)
    ray_dir = torch.empty((0, 3), device="cuda", dtype=torch.float32)
    prim_index = torch.empty(0, device="cuda", dtype=torch.int32)
    hit = torch.empty((0, 3), device="cuda", dtype=torch.float32)
    bad_hit_count = torch.zeros(1, device="cuda", dtype=torch.float32)
    triangle_edge_count = torch.empty(0, device="cuda", dtype=torch.int32)
    triangle_edge_indices = torch.empty((0, 3), device="cuda", dtype=torch.int32)
    edge = torch.empty((0, 3), device="cuda", dtype=torch.float32)
    edge_extent = torch.empty(0, device="cuda", dtype=torch.float32)
    adjacent_face = torch.empty(0, device="cuda", dtype=torch.int32)

    with pytest.raises(TypeError, match="hit_count"):
        ops.mc_diffraction_discover_edges_counted(
            tx_pos,
            ray_dir,
            prim_index,
            hit,
            hit,
            hit,
            bad_hit_count,
            triangle_edge_count,
            triangle_edge_indices,
            edge,
            edge,
            edge,
            edge,
            edge_extent,
            edge_extent,
            adjacent_face,
        )
