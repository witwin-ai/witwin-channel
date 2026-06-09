from __future__ import annotations

import argparse
import json
import time

import torch
import raydtorch as rt

from .benchmark_support import synchronize, time_ms


def _dfr_case():
    states = rt.DfrStates(
        edge_index=torch.tensor([0], device="cuda", dtype=torch.int32),
        edge_pos=torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        edge_dir=torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        edge_t_min=torch.tensor([-0.5], device="cuda", dtype=torch.float32),
        edge_t_max=torch.tensor([0.5], device="cuda", dtype=torch.float32),
        n0=torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
        n1=torch.tensor([[0.0, -1.0, 0.0]], device="cuda", dtype=torch.float32),
        prim0=torch.tensor([-1], device="cuda", dtype=torch.int32),
        prim1=torch.tensor([-1], device="cuda", dtype=torch.int32),
        exterior_angle=torch.tensor([1.5 * torch.pi], device="cuda", dtype=torch.float32),
        src=torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        src_power=torch.tensor([2.0], device="cuda", dtype=torch.float32),
        wi=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        d0=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        count=1,
    )
    grid = rt.DfrGrid(axis=2, position=-1.0, resolution0=1, resolution1=1, cell_area=4.0)
    material = rt.DfrMaterial.default(1, device=torch.device("cuda"), dtype=torch.float32)
    return states, grid, material


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=192)
    parser.add_argument("--queries", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--max-bounces", type=int, default=1)
    parser.add_argument("--direct-samples", type=int, default=64)
    args = parser.parse_args()

    n = args.grid
    xs, ys = torch.meshgrid(
        torch.linspace(0, 1, n, device="cuda"),
        torch.linspace(0, 1, n, device="cuda"),
        indexing="ij",
    )
    verts = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros(n * n, device="cuda")], dim=1).contiguous()
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces = torch.tensor(faces, device="cuda", dtype=torch.int32)

    scene = rt.Scene()
    t0 = time.perf_counter()
    scene.add_mesh(rt.Mesh(verts, faces, edges_enabled=True), dynamic=True)
    scene.build()
    synchronize()
    build_ms = (time.perf_counter() - t0) * 1000.0

    updated = verts.clone()
    updated[:, 2] = updated[:, 2] + 0.001
    sync_start = time.perf_counter()
    scene.update_mesh_vertices(0, updated)
    scene.sync()
    synchronize()
    dynamic_sync_ms = (time.perf_counter() - sync_start) * 1000.0

    ray = rt.Ray(
        torch.rand((args.queries, 3), device="cuda", dtype=torch.float32),
        torch.randn((args.queries, 3), device="cuda", dtype=torch.float32),
    )
    ray_flags_none = getattr(rt.RayFlags, "None")
    points = torch.rand((args.queries, 3), device="cuda", dtype=torch.float32)
    dfr_states, dfr_grid, dfr_material = _dfr_case()
    dfr_tx = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    dfr_rx = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
    dfr_active = torch.ones((dfr_states.state_count,), device="cuda", dtype=torch.bool)

    result = {
        "grid": n,
        "queries": args.queries,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "build_ms": build_ms,
        "dynamic_sync_ms": dynamic_sync_ms,
        "intersect_flags_none_ms": time_ms(
            lambda: scene.intersect(ray, flags=ray_flags_none).t,
            args.warmup,
            args.repeat,
        ),
        "intersect_ms": time_ms(lambda: scene.intersect(ray).t, args.warmup, args.repeat),
        "nearest_edge_ms": time_ms(lambda: scene.nearest_edge(points).distance, args.warmup, args.repeat),
        "reflection_trace_ms": time_ms(
            lambda: scene.trace_reflections(ray, max_bounces=args.max_bounces).t,
            args.warmup,
            args.repeat,
        ),
        "diffraction_direct_ms": time_ms(
            lambda: scene.accum_dfr_direct(
                states=dfr_states,
                grid=dfr_grid,
                material=dfr_material,
                wavelength=0.125,
                direct_samples=args.direct_samples,
                seed=17,
            ).power,
            args.warmup,
            args.repeat,
        ),
        "diffraction_paths_ms": time_ms(
            lambda: scene.trace_dfr_paths(
                tx_positions=dfr_tx,
                rx_positions=dfr_rx,
                states=dfr_states,
                material=dfr_material,
                active=dfr_active,
                max_paths=4,
                wavelength=0.125,
            ).count,
            args.warmup,
            args.repeat,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
