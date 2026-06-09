from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import torch

import raydtorch as rt


RAYDI_ROOT = Path(r"E:\Code\RayDi")


def _sync_torch() -> None:
    torch.cuda.synchronize()


def _time_torch(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    _sync_torch()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    _sync_torch()
    return (time.perf_counter() - t0) * 1000.0 / repeat


def _time_dr(fn, dr, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        dr.eval(fn())
    dr.sync_thread()
    t0 = time.perf_counter()
    for _ in range(repeat):
        dr.eval(fn())
    dr.sync_thread()
    return (time.perf_counter() - t0) * 1000.0 / repeat


def _load_rayd(source: str, root: Path):
    if source == "local":
        sys.path.insert(0, str(root))
    import rayd as rayd

    cuda = importlib.import_module("dr" + "jit.cuda")
    dr = importlib.import_module("dr" + "jit")
    return rayd, cuda, dr


def _grid_data(grid: int):
    xs, ys = torch.meshgrid(
        torch.linspace(0, 1, grid, device="cuda"),
        torch.linspace(0, 1, grid, device="cuda"),
        indexing="ij",
    )
    verts = torch.stack(
        [xs.reshape(-1), ys.reshape(-1), torch.zeros(grid * grid, device="cuda")],
        dim=1,
    ).contiguous()
    faces = []
    for i in range(grid - 1):
        for j in range(grid - 1):
            a = i * grid + j
            b = a + 1
            c = a + grid
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces_t = torch.tensor(faces, device="cuda", dtype=torch.int32)
    return verts, faces_t


def _torch_scene(verts: torch.Tensor, faces: torch.Tensor, dynamic: bool):
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(verts, faces, edges_enabled=True), dynamic=dynamic)
    t0 = time.perf_counter()
    scene.build()
    _sync_torch()
    return scene, (time.perf_counter() - t0) * 1000.0


def _rayd_scene(rayd, cuda, dr, verts: torch.Tensor, faces: torch.Tensor, dynamic: bool):
    v = verts.detach().cpu()
    f = faces.detach().cpu()
    mesh = rayd.Mesh(
        cuda.Array3f(v[:, 0].tolist(), v[:, 1].tolist(), v[:, 2].tolist()),
        cuda.Array3i(f[:, 0].tolist(), f[:, 1].tolist(), f[:, 2].tolist()),
    )
    scene = rayd.Scene()
    scene.add_mesh(mesh, dynamic)
    t0 = time.perf_counter()
    scene.build()
    dr.sync_thread()
    return scene, (time.perf_counter() - t0) * 1000.0


def _torch_ray(origins: torch.Tensor, directions: torch.Tensor):
    return rt.Ray(origins.contiguous(), directions.contiguous())


def _rayd_ray(rayd, cuda, origins: torch.Tensor, directions: torch.Tensor):
    o = origins.detach().cpu()
    d = directions.detach().cpu()
    return rayd.Ray(
        cuda.Array3f(o[:, 0].tolist(), o[:, 1].tolist(), o[:, 2].tolist()),
        cuda.Array3f(d[:, 0].tolist(), d[:, 1].tolist(), d[:, 2].tolist()),
    )


def _rayd_points(cuda, points: torch.Tensor):
    p = points.detach().cpu()
    return cuda.Array3f(p[:, 0].tolist(), p[:, 1].tolist(), p[:, 2].tolist())


def _torch_dfr_case():
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


def _rayd_dfr_case(rayd, cuda):
    states = rayd.DfrStates()
    states.count = 1
    states.edge_index = cuda.Int([0])
    states.edge_pos = cuda.Array3f([0.0], [0.0], [0.0])
    states.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
    states.edge_t_min = cuda.Float([-0.5])
    states.edge_t_max = cuda.Float([0.5])
    states.n0 = cuda.Array3f([0.0], [1.0], [0.0])
    states.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
    states.prim0 = cuda.Int([-1])
    states.prim1 = cuda.Int([-1])
    states.exterior_angle = cuda.Float([1.5 * 3.141592653589793])
    states.src = cuda.Array3f([0.0], [0.0], [1.0])
    states.src_power = cuda.Float([2.0])
    states.wi = cuda.Array3f([0.0], [0.0], [-1.0])
    states.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
    states.prefix_depth = cuda.Int([0])

    grid = rayd.DfrGrid()
    grid.axis = 2
    grid.position = -1.0
    grid.coord0_min = -1.0
    grid.coord0_max = 1.0
    grid.coord1_min = -1.0
    grid.coord1_max = 1.0
    grid.resolution0 = 1
    grid.resolution1 = 1
    grid.cell_area = 4.0

    material = rayd.DfrMaterial()
    material.eta_r = cuda.Float([4.0])
    material.sigma = cuda.Float([0.0])
    material.mu_r = cuda.Float([1.0])
    material.gain = cuda.Float([1.0])
    material.valid = cuda.Bool([True])

    options = rayd.DfrOptions()
    options.wavelength = 0.125
    options.k = 50.26548245743669
    options.seed = 17
    options.samples = 64
    options.max_order = 1
    options.direct_samples = 64
    options.keller_samples = 0
    options.strategy_mask = rayd.RAYD_DFR_DIRECT
    options.sample_sequence = rayd.RAYD_DFR_HASH
    options.receiver_model = rayd.RAYD_DFR_MATCHED_ISO
    options.collect_edge_use = True
    options.collect_debug_counts = True
    return states, grid, material, options


def _rayd_dfr_path_options(rayd):
    options = rayd.DfrPathOptions()
    options.wavelength = 0.125
    options.k = 50.26548245743669
    options.seed = 17
    options.max_order = 1
    options.max_paths = 4
    options.max_rx = 1
    options.strategy_mask = rayd.RAYD_DFR_DIRECT
    options.sample_count = 1
    options.return_geom = 1
    options.receiver_model = rayd.RAYD_DFR_MATCHED_ISO
    return options


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument(
        "--rayd-source",
        choices=("package", "local"),
        default="package",
        help="Use the installed RayD package by default, or a local checkout via --rayd-root.",
    )
    parser.add_argument("--rayd-root", type=Path, default=RAYDI_ROOT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA torch is required")

    rayd, cuda, dr = _load_rayd(args.rayd_source, args.rayd_root)
    rayd_flags_none = getattr(rayd.RayFlags, "None")
    raydtorch_flags_none = getattr(rt.RayFlags, "None")
    torch.manual_seed(17)
    verts, faces = _grid_data(args.grid)
    scene_t, torch_build_ms = _torch_scene(verts, faces, args.dynamic)
    scene_d, rayd_build_ms = _rayd_scene(rayd, cuda, dr, verts, faces, args.dynamic)

    ray_o = torch.rand((args.queries, 3), device="cuda", dtype=torch.float32)
    ray_o[:, 2] = -1.0
    ray_d = torch.zeros_like(ray_o)
    ray_d[:, 2] = 1.0
    points = torch.rand((args.queries, 3), device="cuda", dtype=torch.float32)
    points[:, 2] = 0.0

    ray_t = _torch_ray(ray_o, ray_d)
    ray_djit = _rayd_ray(rayd, cuda, ray_o, ray_d)
    points_d = _rayd_points(cuda, points)

    dfr_states_t, dfr_grid_t, dfr_material_t = _torch_dfr_case()
    dfr_states_d, dfr_grid_d, dfr_material_d, dfr_options_d = _rayd_dfr_case(rayd, cuda)
    dfr_tx_t = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    dfr_rx_t = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
    dfr_active_t = torch.ones((dfr_states_t.state_count,), device="cuda", dtype=torch.bool)
    dfr_tx_d = cuda.Array3f([0.0], [0.0], [1.0])
    dfr_rx_d = cuda.Array3f([0.0], [0.0], [-1.0])
    dfr_active_d = cuda.Bool([True])
    dfr_path_options_d = _rayd_dfr_path_options(rayd)

    torch_result = {
        "build_ms": torch_build_ms,
        "intersect_flags_none_ms": _time_torch(
            lambda: scene_t.intersect(ray_t, flags=raydtorch_flags_none).t,
            args.warmup,
            args.repeat,
        ),
        "intersect_ms": _time_torch(lambda: scene_t.intersect(ray_t).t, args.warmup, args.repeat),
        "nearest_edge_ms": _time_torch(lambda: scene_t.nearest_edge(points).distance, args.warmup, args.repeat),
        "reflection_trace_ms": _time_torch(
            lambda: scene_t.trace_reflections(ray_t, max_bounces=1).t,
            args.warmup,
            args.repeat,
        ),
        "diffraction_direct_ms": _time_torch(
            lambda: scene_t.accum_dfr_direct(
                states=dfr_states_t,
                grid=dfr_grid_t,
                material=dfr_material_t,
                wavelength=0.125,
                direct_samples=64,
                seed=17,
            ).power,
            args.warmup,
            args.repeat,
        ),
        "diffraction_paths_ms": _time_torch(
            lambda: scene_t.trace_dfr_paths(
                tx_positions=dfr_tx_t,
                rx_positions=dfr_rx_t,
                states=dfr_states_t,
                material=dfr_material_t,
                active=dfr_active_t,
                max_paths=4,
                wavelength=0.125,
            ).count,
            args.warmup,
            args.repeat,
        ),
    }

    rayd_result = {
        "build_ms": rayd_build_ms,
        "intersect_flags_none_ms": _time_dr(
            lambda: scene_d.intersect(ray_djit, flags=rayd_flags_none).t,
            dr,
            args.warmup,
            args.repeat,
        ),
        "intersect_ms": _time_dr(lambda: scene_d.intersect(ray_djit).t, dr, args.warmup, args.repeat),
        "nearest_edge_ms": _time_dr(lambda: scene_d.nearest_edge(points_d).distance, dr, args.warmup, args.repeat),
        "reflection_trace_ms": _time_dr(
            lambda: scene_d.trace_reflections(ray_djit, max_bounces=1, symbolic=False).t,
            dr,
            args.warmup,
            args.repeat,
        ),
        "diffraction_direct_ms": _time_dr(
            lambda: scene_d.accum_dfr_direct(
                dfr_states_d,
                dfr_grid_d,
                dfr_material_d,
                dfr_options_d,
                cuda.Bool([True]),
            ).power,
            dr,
            args.warmup,
            args.repeat,
        ),
        "diffraction_paths_ms": _time_dr(
            lambda: scene_d.trace_dfr_paths(
                dfr_tx_d,
                dfr_rx_d,
                dfr_states_d,
                dfr_material_d,
                dfr_path_options_d,
                dfr_active_d,
            ).count,
            dr,
            args.warmup,
            args.repeat,
        ),
    }

    print(
        json.dumps(
            {
                "grid": args.grid,
                "queries": args.queries,
                "dynamic": args.dynamic,
                "rayd_source": args.rayd_source,
                "rayd_root": str(args.rayd_root) if args.rayd_source == "local" else None,
                "rayd_module": getattr(rayd, "__file__", None),
                "warmup": args.warmup,
                "repeat": args.repeat,
                "raydtorch": torch_result,
                "rayd": rayd_result,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
