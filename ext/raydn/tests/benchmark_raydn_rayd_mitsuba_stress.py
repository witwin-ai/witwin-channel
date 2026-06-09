from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

import raydn as rt


RAYDI_ROOT = Path(r"E:\Code\RayDi")


@dataclass(frozen=True)
class Scenario:
    label: str
    mesh_resolution: int
    ray_grid_side: int

    @property
    def ray_count(self) -> int:
        return self.ray_grid_side * self.ray_grid_side

    @property
    def triangle_count(self) -> int:
        return self.mesh_resolution * self.mesh_resolution * 2

    @property
    def vertex_count(self) -> int:
        side = self.mesh_resolution + 1
        return side * side

    def config(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mesh_resolution": self.mesh_resolution,
            "triangle_count": self.triangle_count,
            "vertex_count": self.vertex_count,
            "ray_grid_side": self.ray_grid_side,
            "ray_count": self.ray_count,
        }


def _parse_scenario(spec: str) -> Scenario:
    parts = [part.strip() for part in spec.split(":") if part.strip()]
    if len(parts) == 2:
        mesh_resolution = int(parts[0])
        ray_grid_side = int(parts[1])
        label = f"{mesh_resolution}x{mesh_resolution} mesh / {ray_grid_side}x{ray_grid_side} rays"
    elif len(parts) == 3:
        label = parts[0]
        mesh_resolution = int(parts[1])
        ray_grid_side = int(parts[2])
    else:
        raise ValueError("Use --scenario mesh_resolution:ray_grid_side or label:mesh_resolution:ray_grid_side.")
    if mesh_resolution <= 0 or ray_grid_side <= 0:
        raise ValueError("Scenario dimensions must be positive.")
    return Scenario(label, mesh_resolution, ray_grid_side)


def _make_grid_mesh_data(
    resolution: int,
    *,
    x_offset: float = 0.0,
    z_offset: float = 0.0,
) -> dict[str, list[float] | list[int]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for y in range(resolution + 1):
        fy = y / resolution
        for x in range(resolution + 1):
            fx = x / resolution
            xs.append(x_offset + fx)
            ys.append(fy)
            zs.append(z_offset)

    i0: list[int] = []
    i1: list[int] = []
    i2: list[int] = []
    stride = resolution + 1
    for y in range(resolution):
        for x in range(resolution):
            v00 = y * stride + x
            v10 = v00 + 1
            v01 = v00 + stride
            v11 = v01 + 1
            i0.extend([v00, v00])
            i1.extend([v10, v11])
            i2.extend([v11, v01])
    return {"x": xs, "y": ys, "z": zs, "i0": i0, "i1": i1, "i2": i2}


def _make_ray_data(
    side: int,
    *,
    x_offset: float = 0.0,
    z_origin: float = -1.0,
) -> dict[str, list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for iy in range(side):
        for ix in range(side):
            xs.append(x_offset + (ix + 0.5) / side)
            ys.append((iy + 0.5) / side)
            zs.append(z_origin)
    count = len(xs)
    return {
        "ox": xs,
        "oy": ys,
        "oz": zs,
        "dx": [0.0] * count,
        "dy": [0.0] * count,
        "dz": [1.0] * count,
    }


def _summarize(times_s: list[float], query_count: int) -> dict[str, float]:
    avg_s = statistics.fmean(times_s)
    return {
        "min_ms": min(times_s) * 1000.0,
        "avg_ms": avg_s * 1000.0,
        "qps_m": query_count / avg_s / 1.0e6,
    }


def _measure(
    fn: Callable[[], object],
    sync: Callable[[], None],
    repeats: int,
    warmup: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
        sync()
    times_s: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync()
        times_s.append(time.perf_counter() - start)
    return times_s


def _time_build(fn: Callable[[], object], sync: Callable[[], None]) -> tuple[object, float]:
    start = time.perf_counter()
    value = fn()
    sync()
    return value, (time.perf_counter() - start) * 1000.0


def _cleanup_torch() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _sync_torch_drjit(dr: Any) -> None:
    torch.cuda.synchronize()
    dr.sync_thread()


def _load_rayd(source: str, root: Path):
    if source == "local":
        sys.path.insert(0, str(root))
    import rayd as rayd

    cuda = importlib.import_module("drjit.cuda")
    dr = importlib.import_module("drjit")
    return rayd, cuda, dr


def _cleanup_drjit(dr: Any) -> None:
    gc.collect()
    dr.sync_thread()
    dr.flush_malloc_cache()
    dr.flush_kernel_cache()
    dr.sync_thread()


def _try_import_mitsuba(variant: str):
    tests_dir = os.path.normcase(os.path.abspath(os.path.dirname(__file__)))
    saved_path = sys.path[:]
    sys.path = [path for path in sys.path if os.path.normcase(os.path.abspath(path)) != tests_dir]
    try:
        import mitsuba as mi  # type: ignore
    except ImportError:
        return None
    finally:
        sys.path = saved_path
    mi.set_variant(variant)
    return mi


def _torch_mesh(mesh_data: dict[str, list[float] | list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    verts = torch.tensor(
        list(zip(mesh_data["x"], mesh_data["y"], mesh_data["z"])),
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    faces = torch.tensor(
        list(zip(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"])),
        device="cuda",
        dtype=torch.int32,
    ).contiguous()
    return verts, faces


def _torch_ray(ray_data: dict[str, list[float]]) -> rt.Ray:
    origins = torch.tensor(
        list(zip(ray_data["ox"], ray_data["oy"], ray_data["oz"])),
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    directions = torch.tensor(
        list(zip(ray_data["dx"], ray_data["dy"], ray_data["dz"])),
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    return rt.Ray(origins, directions)


def _torch_scene(
    mesh_data: dict[str, list[float] | list[int]],
    *,
    dynamic: bool,
    edges_enabled: bool,
) -> tuple[rt.Scene, int, float]:
    verts, faces = _torch_mesh(mesh_data)
    scene = rt.Scene()
    mesh_id = scene.add_mesh(rt.Mesh(verts, faces, edges_enabled=edges_enabled), dynamic=dynamic)
    _, build_ms = _time_build(scene.build, torch.cuda.synchronize)
    return scene, mesh_id, build_ms


def _torch_forward_performance(
    mesh_data: dict[str, list[float] | list[int]],
    updated_mesh_data: dict[str, list[float] | list[int]],
    ray_data: dict[str, list[float]],
    updated_ray_data: dict[str, list[float]],
    *,
    dynamic: bool,
    edges_enabled: bool,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    scene, mesh_id, build_ms = _torch_scene(mesh_data, dynamic=dynamic, edges_enabled=edges_enabled)
    rays = _torch_ray(ray_data)
    updated_rays = _torch_ray(updated_ray_data)
    base_positions, _ = _torch_mesh(mesh_data)
    updated_positions, _ = _torch_mesh(updated_mesh_data)
    flags_none = getattr(rt.RayFlags, "None")

    def make_run(mode: str):
        use_updated = False

        def run():
            nonlocal use_updated
            current_rays = rays
            if dynamic:
                use_updated = not use_updated
                scene.update_mesh_vertices(mesh_id, updated_positions if use_updated else base_positions)
                scene.sync()
                current_rays = updated_rays if use_updated else rays
            if mode == "full":
                its = scene.intersect(current_rays, flags=rt.RayFlags.All)
                return its.t, its.p, its.n, its.uv, its.barycentric, its.prim_id
            its = scene.intersect(current_rays, flags=flags_none)
            return its.t

        return run

    query_count = len(updated_ray_data["ox"] if dynamic else ray_data["ox"])
    return {
        "build_ms": build_ms,
        "performance": {
            mode: _summarize(_measure(make_run(mode), torch.cuda.synchronize, repeats, warmup), query_count)
            for mode in ("full", "reduced")
        },
    }


def _torch_backward_performance(
    mesh_data: dict[str, list[float] | list[int]],
    updated_mesh_data: dict[str, list[float] | list[int]],
    ray_data: dict[str, list[float]],
    updated_ray_data: dict[str, list[float]],
    *,
    dynamic: bool,
    edges_enabled: bool,
    loss_backward: bool,
    materialize_full: bool,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    base_positions, faces = _torch_mesh(mesh_data)
    updated_positions, _ = _torch_mesh(updated_mesh_data)
    base_positions.requires_grad_(True)
    updated_positions.requires_grad_(True)
    scene = rt.Scene()
    mesh_id = scene.add_mesh(rt.Mesh(base_positions, faces, edges_enabled=edges_enabled), dynamic=dynamic)
    _, build_ms = _time_build(scene.build, torch.cuda.synchronize)
    rays = _torch_ray(ray_data)
    updated_rays = _torch_ray(updated_ray_data)
    weights = torch.linspace(0.5, 1.5, rays.o.shape[0], device="cuda", dtype=torch.float32)
    updated_weights = torch.linspace(1.5, 0.5, updated_rays.o.shape[0], device="cuda", dtype=torch.float32)
    flags_none = getattr(rt.RayFlags, "None")

    def make_run(mode: str):
        use_updated = False

        def run():
            nonlocal use_updated
            current_rays = rays
            current_weights = weights
            base_positions.grad = None
            updated_positions.grad = None
            if dynamic:
                use_updated = not use_updated
                scene.update_mesh_vertices(mesh_id, updated_positions if use_updated else base_positions)
                scene.sync()
                current_rays = updated_rays if use_updated else rays
                current_weights = updated_weights if use_updated else weights
            flags = rt.RayFlags.All if mode == "vjp_full" else flags_none
            its = scene.intersect(current_rays, flags=flags)
            if materialize_full and mode == "vjp_full":
                _ = (its.p, its.n, its.geo_n, its.uv, its.barycentric, its.prim_id)
            if loss_backward:
                loss = torch.sum(its.t * current_weights)
                loss.backward()
            else:
                its.t.backward(current_weights)
            return updated_positions.grad if dynamic and use_updated else base_positions.grad

        return run

    query_count = len(updated_ray_data["ox"] if dynamic else ray_data["ox"])
    return {
        "build_ms": build_ms,
        "performance": {
            mode: _summarize(_measure(make_run(mode), torch.cuda.synchronize, repeats, warmup), query_count)
            for mode in ("vjp_full", "vjp_reduced")
        },
    }


def _rayd_ray(rayd: Any, cuda: Any, ray_data: dict[str, list[float]]) -> Any:
    return rayd.Ray(
        cuda.Array3f(ray_data["ox"], ray_data["oy"], ray_data["oz"]),
        cuda.Array3f(ray_data["dx"], ray_data["dy"], ray_data["dz"]),
    )


def _rayd_scene(
    rayd: Any,
    cuda: Any,
    dr: Any,
    mesh_data: dict[str, list[float] | list[int]],
    *,
    dynamic: bool,
) -> tuple[Any, int, float]:
    mesh = rayd.Mesh(
        cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
        cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]),
    )
    scene = rayd.Scene()
    mesh_id = scene.add_mesh(mesh, dynamic=dynamic)
    _, build_ms = _time_build(scene.build, dr.sync_thread)
    return scene, mesh_id, build_ms


def _rayd_forward_performance(
    rayd: Any,
    cuda: Any,
    dr: Any,
    mesh_data: dict[str, list[float] | list[int]],
    updated_mesh_data: dict[str, list[float] | list[int]],
    ray_data: dict[str, list[float]],
    updated_ray_data: dict[str, list[float]],
    *,
    dynamic: bool,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    scene, mesh_id, build_ms = _rayd_scene(rayd, cuda, dr, mesh_data, dynamic=dynamic)
    rays = _rayd_ray(rayd, cuda, ray_data)
    updated_rays = _rayd_ray(rayd, cuda, updated_ray_data)
    base_positions = cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"])
    updated_positions = cuda.Array3f(updated_mesh_data["x"], updated_mesh_data["y"], updated_mesh_data["z"])
    flags_none = getattr(rayd.RayFlags, "None")

    def make_run(mode: str):
        use_updated = False

        def run():
            nonlocal use_updated
            current_rays = rays
            if dynamic:
                use_updated = not use_updated
                scene.update_mesh_vertices(mesh_id, updated_positions if use_updated else base_positions)
                scene.sync()
                current_rays = updated_rays if use_updated else rays
            if mode == "full":
                its = scene.intersect(current_rays)
                dr.eval(its.t, its.p, its.n, its.uv, its.barycentric, its.prim_id)
            else:
                its = scene.intersect(current_rays, flags=flags_none)
                dr.eval(its.t)

        return run

    query_count = len(updated_ray_data["ox"] if dynamic else ray_data["ox"])
    return {
        "build_ms": build_ms,
        "performance": {
            mode: _summarize(_measure(make_run(mode), dr.sync_thread, repeats, warmup), query_count)
            for mode in ("full", "reduced")
        },
    }


def _rayd_ray_ad(rayd: Any, ad: Any, ray_data: dict[str, list[float]]) -> Any:
    return rayd.RayAD(
        ad.Array3f(ray_data["ox"], ray_data["oy"], ray_data["oz"]),
        ad.Array3f(ray_data["dx"], ray_data["dy"], ray_data["dz"]),
    )


def _rayd_backward_performance(
    rayd: Any,
    cuda: Any,
    dr: Any,
    mesh_data: dict[str, list[float] | list[int]],
    updated_mesh_data: dict[str, list[float] | list[int]],
    ray_data: dict[str, list[float]],
    updated_ray_data: dict[str, list[float]],
    *,
    dynamic: bool,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    ad = importlib.import_module("drjit.cuda.ad")
    mesh = rayd.Mesh(
        cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
        cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]),
    )
    base_positions = ad.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"])
    updated_positions = ad.Array3f(updated_mesh_data["x"], updated_mesh_data["y"], updated_mesh_data["z"])
    dr.enable_grad(base_positions)
    dr.enable_grad(updated_positions)
    if not dynamic:
        mesh.vertex_positions = base_positions
    scene = rayd.Scene()
    mesh_id = scene.add_mesh(mesh, dynamic=dynamic)
    _, build_ms = _time_build(scene.build, dr.sync_thread)
    rays = _rayd_ray_ad(rayd, ad, ray_data)
    updated_rays = _rayd_ray_ad(rayd, ad, updated_ray_data)
    weight_index = dr.arange(cuda.UInt, len(ray_data["ox"]))
    weights = 0.5 + cuda.Float(weight_index) / max(1, len(ray_data["ox"]) - 1)
    updated_weight_index = dr.arange(cuda.UInt, len(updated_ray_data["ox"]))
    updated_weights = 1.5 - cuda.Float(updated_weight_index) / max(1, len(updated_ray_data["ox"]) - 1)
    flags_none = getattr(rayd.RayFlags, "None")

    def make_run(mode: str):
        use_updated = False

        def run():
            nonlocal use_updated
            dr.set_grad(base_positions, 0)
            dr.set_grad(updated_positions, 0)
            current_positions = base_positions
            current_rays = rays
            current_weights = weights
            if dynamic:
                use_updated = not use_updated
                current_positions = updated_positions if use_updated else base_positions
                current_rays = updated_rays if use_updated else rays
                current_weights = updated_weights if use_updated else weights
                scene.update_mesh_vertices(mesh_id, current_positions)
                scene.sync()
            flags = rayd.RayFlags.All if mode == "vjp_full" else flags_none
            its = scene.intersect(current_rays, flags=flags)
            dr.set_grad(its.t, current_weights)
            dr.enqueue(dr.ADMode.Backward, its.t)
            dr.traverse(dr.ADMode.Backward)
            dr.eval(dr.grad(current_positions))

        return run

    query_count = len(updated_ray_data["ox"] if dynamic else ray_data["ox"])
    return {
        "build_ms": build_ms,
        "performance": {
            mode: _summarize(_measure(make_run(mode), dr.sync_thread, repeats, warmup), query_count)
            for mode in ("vjp_full", "vjp_reduced")
        },
    }


def _torch_positions(mesh_data: dict[str, list[float] | list[int]], *, requires_grad: bool) -> torch.Tensor:
    value = torch.tensor(
        list(zip(mesh_data["x"], mesh_data["y"], mesh_data["z"])),
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    value.requires_grad_(requires_grad)
    return value


def _rayd_torch_loss_backward_performance(
    rayd: Any,
    cuda: Any,
    dr: Any,
    mesh_data: dict[str, list[float] | list[int]],
    updated_mesh_data: dict[str, list[float] | list[int]],
    ray_data: dict[str, list[float]],
    updated_ray_data: dict[str, list[float]],
    *,
    dynamic: bool,
    materialize_full: bool,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    if not dynamic:
        return {
            "unsupported": (
                "RayD static scenes bind the differentiable vertex array at scene build. "
                "The current public API does not reconnect a later Torch vertex tensor "
                "to the warm-started static AD graph without rebuilding the scene."
            )
        }

    ad = importlib.import_module("drjit.cuda.ad")
    mesh = rayd.Mesh(
        cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
        cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]),
    )
    scene = rayd.Scene()
    mesh_id = scene.add_mesh(mesh, dynamic=True)
    _, build_ms = _time_build(scene.build, dr.sync_thread)
    base_positions = _torch_positions(mesh_data, requires_grad=True)
    updated_positions = _torch_positions(updated_mesh_data, requires_grad=True)
    rays = _rayd_ray_ad(rayd, ad, ray_data)
    updated_rays = _rayd_ray_ad(rayd, ad, updated_ray_data)
    weights = torch.linspace(0.5, 1.5, len(ray_data["ox"]), device="cuda", dtype=torch.float32)
    updated_weights = torch.linspace(1.5, 0.5, len(updated_ray_data["ox"]), device="cuda", dtype=torch.float32)
    flags_none = getattr(rayd.RayFlags, "None")

    def make_wrapped(current_rays: Any, flags: Any):
        @dr.wrap(source="torch", target="drjit")
        def wrapped_t(positions: Any) -> Any:
            current_positions = ad.Array3f(positions[:, 0], positions[:, 1], positions[:, 2])
            scene.update_mesh_vertices(mesh_id, current_positions)
            scene.sync()
            its = scene.intersect(current_rays, flags=flags)
            if materialize_full and flags != flags_none:
                dr.eval(its.t, its.p, its.n, its.uv, its.barycentric, its.prim_id)
            return its.t

        return wrapped_t

    wrapped: dict[str, tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]] = {
        "vjp_full": (
            make_wrapped(rays, rayd.RayFlags.All),
            make_wrapped(updated_rays, rayd.RayFlags.All),
        ),
        "vjp_reduced": (
            make_wrapped(rays, flags_none),
            make_wrapped(updated_rays, flags_none),
        ),
    }

    def make_run(mode: str):
        use_updated = False

        def run():
            nonlocal use_updated
            base_positions.grad = None
            updated_positions.grad = None
            use_updated = not use_updated
            current_positions = updated_positions if use_updated else base_positions
            current_weights = updated_weights if use_updated else weights
            current_fn = wrapped[mode][1 if use_updated else 0]
            t = current_fn(current_positions)
            loss = torch.sum(t * current_weights)
            loss.backward()
            return current_positions.grad

        return run

    query_count = len(updated_ray_data["ox"])
    return {
        "build_ms": build_ms,
        "performance": {
            mode: _summarize(_measure(make_run(mode), lambda: _sync_torch_drjit(dr), repeats, warmup), query_count)
            for mode in ("vjp_full", "vjp_reduced")
        },
    }


def _mitsuba_scene_timed(mi: Any, dr: Any, mesh_data: dict[str, list[float] | list[int]]) -> tuple[Any, Any, float]:
    mesh = mi.Mesh(
        "plane",
        vertex_count=len(mesh_data["x"]),
        face_count=len(mesh_data["i0"]),
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    params = mi.traverse(mesh)
    params["vertex_positions"] = dr.ravel(mi.Point3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]))
    params["faces"] = dr.ravel(mi.Vector3u(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]))
    params.update()

    def build():
        return mi.load_dict({"type": "scene", "mesh": mesh})

    scene, build_ms = _time_build(build, dr.sync_thread)
    return scene, mi.traverse(scene), build_ms


def _mitsuba_ray(mi: Any, ray_data: dict[str, list[float]]) -> Any:
    return mi.Ray3f(
        mi.Point3f(ray_data["ox"], ray_data["oy"], ray_data["oz"]),
        mi.Vector3f(ray_data["dx"], ray_data["dy"], ray_data["dz"]),
    )


def _mitsuba_forward_performance(
    mi: Any,
    dr: Any,
    mesh_data: dict[str, list[float] | list[int]],
    updated_mesh_data: dict[str, list[float] | list[int]],
    ray_data: dict[str, list[float]],
    updated_ray_data: dict[str, list[float]],
    *,
    dynamic: bool,
    include_preliminary: bool,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    scene, params, build_ms = _mitsuba_scene_timed(mi, dr, mesh_data)
    rays = _mitsuba_ray(mi, ray_data)
    updated_rays = _mitsuba_ray(mi, updated_ray_data)
    base_positions = dr.ravel(mi.Point3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]))
    updated_positions = dr.ravel(mi.Point3f(updated_mesh_data["x"], updated_mesh_data["y"], updated_mesh_data["z"]))

    def make_run(mode: str):
        use_updated = False

        def run():
            nonlocal use_updated
            current_rays = rays
            if dynamic:
                use_updated = not use_updated
                params["mesh.vertex_positions"] = updated_positions if use_updated else base_positions
                params.update()
                current_rays = updated_rays if use_updated else rays
            if mode == "full":
                its = scene.ray_intersect(current_rays)
                dr.eval(its.t, its.p, its.n, its.uv, its.prim_index)
            elif mode == "reduced":
                its = scene.ray_intersect(current_rays, mi.RayFlags.Minimal, False)
                dr.eval(its.t)
            else:
                pi = scene.ray_intersect_preliminary(current_rays, coherent=False)
                dr.eval(pi.t)

        return run

    modes = ["full", "reduced"]
    if include_preliminary:
        modes.append("preliminary")
    query_count = len(updated_ray_data["ox"] if dynamic else ray_data["ox"])
    return {
        "build_ms": build_ms,
        "performance": {
            mode: _summarize(_measure(make_run(mode), dr.sync_thread, repeats, warmup), query_count)
            for mode in modes
        },
    }


def _mitsuba_backward_performance(
    mi: Any,
    dr: Any,
    mesh_data: dict[str, list[float] | list[int]],
    updated_mesh_data: dict[str, list[float] | list[int]],
    ray_data: dict[str, list[float]],
    updated_ray_data: dict[str, list[float]],
    *,
    dynamic: bool,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    scene, params, build_ms = _mitsuba_scene_timed(mi, dr, mesh_data)
    base_positions = dr.ravel(mi.Point3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]))
    updated_positions = dr.ravel(mi.Point3f(updated_mesh_data["x"], updated_mesh_data["y"], updated_mesh_data["z"]))
    dr.enable_grad(base_positions)
    dr.enable_grad(updated_positions)
    if not dynamic:
        params["mesh.vertex_positions"] = base_positions
        params.set_dirty("mesh.vertex_positions")
        params.update()
    rays = _mitsuba_ray(mi, ray_data)
    updated_rays = _mitsuba_ray(mi, updated_ray_data)
    weight_index = dr.arange(mi.UInt, len(ray_data["ox"]))
    weights = 0.5 + mi.Float(weight_index) / max(1, len(ray_data["ox"]) - 1)
    updated_weight_index = dr.arange(mi.UInt, len(updated_ray_data["ox"]))
    updated_weights = 1.5 - mi.Float(updated_weight_index) / max(1, len(updated_ray_data["ox"]) - 1)
    use_updated = False

    def run():
        nonlocal use_updated
        dr.set_grad(base_positions, 0)
        dr.set_grad(updated_positions, 0)
        current_positions = base_positions
        current_rays = rays
        current_weights = weights
        if dynamic:
            use_updated = not use_updated
            current_positions = updated_positions if use_updated else base_positions
            current_rays = updated_rays if use_updated else rays
            current_weights = updated_weights if use_updated else weights
            params["mesh.vertex_positions"] = current_positions
            params.set_dirty("mesh.vertex_positions")
            params.update()
        its = scene.ray_intersect(current_rays)
        dr.set_grad(its.t, current_weights)
        dr.enqueue(dr.ADMode.Backward, its.t)
        dr.traverse(dr.ADMode.Backward)
        dr.eval(dr.grad(current_positions))

    query_count = len(updated_ray_data["ox"] if dynamic else ray_data["ox"])
    return {
        "build_ms": build_ms,
        "performance": {
            "vjp_full": _summarize(_measure(run, dr.sync_thread, repeats, warmup), query_count),
        },
    }


def _mitsuba_torch_loss_backward_performance(
    mi: Any,
    dr: Any,
    mesh_data: dict[str, list[float] | list[int]],
    updated_mesh_data: dict[str, list[float] | list[int]],
    ray_data: dict[str, list[float]],
    updated_ray_data: dict[str, list[float]],
    *,
    dynamic: bool,
    materialize_full: bool,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    scene, params, build_ms = _mitsuba_scene_timed(mi, dr, mesh_data)
    base_positions = _torch_positions(mesh_data, requires_grad=True)
    updated_positions = _torch_positions(updated_mesh_data, requires_grad=True)
    rays = _mitsuba_ray(mi, ray_data)
    updated_rays = _mitsuba_ray(mi, updated_ray_data)
    weights = torch.linspace(0.5, 1.5, len(ray_data["ox"]), device="cuda", dtype=torch.float32)
    updated_weights = torch.linspace(1.5, 0.5, len(updated_ray_data["ox"]), device="cuda", dtype=torch.float32)

    def make_wrapped(current_rays: Any, *, reduced: bool):
        @dr.wrap(source="torch", target="drjit")
        def wrapped_t(positions: Any) -> Any:
            current_positions = dr.ravel(mi.Point3f(positions[:, 0], positions[:, 1], positions[:, 2]))
            params["mesh.vertex_positions"] = current_positions
            params.set_dirty("mesh.vertex_positions")
            params.update()
            if reduced:
                its = scene.ray_intersect(current_rays, mi.RayFlags.Minimal, False)
                return its.t
            its = scene.ray_intersect(current_rays)
            if materialize_full:
                dr.eval(its.t, its.p, its.n, its.uv, its.prim_index)
            return its.t

        return wrapped_t

    wrapped: dict[str, tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]] = {
        "vjp_full": (
            make_wrapped(rays, reduced=False),
            make_wrapped(updated_rays, reduced=False),
        ),
        "vjp_reduced": (
            make_wrapped(rays, reduced=True),
            make_wrapped(updated_rays, reduced=True),
        ),
    }

    def make_run(mode: str):
        use_updated = False

        def run():
            nonlocal use_updated
            base_positions.grad = None
            updated_positions.grad = None
            if dynamic:
                use_updated = not use_updated
            current_positions = updated_positions if dynamic and use_updated else base_positions
            current_weights = updated_weights if dynamic and use_updated else weights
            current_fn = wrapped[mode][1 if dynamic and use_updated else 0]
            t = current_fn(current_positions)
            loss = torch.sum(t * current_weights)
            loss.backward()
            return current_positions.grad

        return run

    query_count = len(updated_ray_data["ox"] if dynamic else ray_data["ox"])
    return {
        "build_ms": build_ms,
        "performance": {
            mode: _summarize(_measure(make_run(mode), lambda: _sync_torch_drjit(dr), repeats, warmup), query_count)
            for mode in ("vjp_full", "vjp_reduced")
        },
        "note": "Mitsuba Torch-loss wrapper updates scene parameters inside the wrapped op to connect Torch inputs.",
    }


def _speedups(backends: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    torch_backend = backends.get("raydn")
    if not torch_backend:
        return out
    phases = sorted(
        {
            phase
            for backend in backends.values()
            for phase, value in backend.items()
            if isinstance(value, dict) and "performance" in value
        }
    )
    for phase in phases:
        torch_phase = torch_backend.get(phase, {}).get("performance", {})
        phase_out: dict[str, Any] = {}
        for other_name, other_backend in backends.items():
            if other_name == "raydn":
                continue
            other_phase = other_backend.get(phase, {}).get("performance", {})
            mode_out: dict[str, float] = {}
            for mode, torch_stats in torch_phase.items():
                other_stats = other_phase.get(mode)
                if not other_stats:
                    continue
                mode_out[mode] = other_stats["avg_ms"] / torch_stats["avg_ms"]
            if mode_out:
                phase_out[other_name + "_over_raydn"] = mode_out
        if phase_out:
            out[phase] = phase_out
    return out


def _run_scenario(args: argparse.Namespace, scenario: Scenario) -> dict[str, Any]:
    mesh_data = _make_grid_mesh_data(scenario.mesh_resolution)
    updated_mesh_data = _make_grid_mesh_data(scenario.mesh_resolution, x_offset=args.dynamic_x_offset)
    ray_data = _make_ray_data(scenario.ray_grid_side)
    updated_ray_data = _make_ray_data(scenario.ray_grid_side, x_offset=args.dynamic_x_offset)

    backends: dict[str, Any] = {}

    if "raydn" in args.backends:
        backends["raydn"] = {
            "forward_static": _torch_forward_performance(
                mesh_data,
                updated_mesh_data,
                ray_data,
                updated_ray_data,
                dynamic=False,
                edges_enabled=args.edges,
                repeats=args.repeats,
                warmup=args.warmup,
            ),
            "forward_dynamic": _torch_forward_performance(
                mesh_data,
                updated_mesh_data,
                ray_data,
                updated_ray_data,
                dynamic=True,
                edges_enabled=args.edges,
                repeats=args.repeats,
                warmup=args.warmup,
            ),
        }
        if args.include_backward:
            backends["raydn"]["backward_static"] = _torch_backward_performance(
                mesh_data,
                updated_mesh_data,
                ray_data,
                updated_ray_data,
                dynamic=False,
                edges_enabled=args.edges,
                loss_backward=False,
                materialize_full=args.materialize_full_vjp,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            backends["raydn"]["backward_dynamic"] = _torch_backward_performance(
                mesh_data,
                updated_mesh_data,
                ray_data,
                updated_ray_data,
                dynamic=True,
                edges_enabled=args.edges,
                loss_backward=False,
                materialize_full=args.materialize_full_vjp,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            if args.torch_loss_backward:
                backends["raydn"]["backward_static_torch_loss"] = _torch_backward_performance(
                    mesh_data,
                    updated_mesh_data,
                    ray_data,
                    updated_ray_data,
                    dynamic=False,
                    edges_enabled=args.edges,
                    loss_backward=True,
                    materialize_full=args.materialize_full_vjp,
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
                backends["raydn"]["backward_dynamic_torch_loss"] = _torch_backward_performance(
                    mesh_data,
                    updated_mesh_data,
                    ray_data,
                    updated_ray_data,
                    dynamic=True,
                    edges_enabled=args.edges,
                    loss_backward=True,
                    materialize_full=args.materialize_full_vjp,
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
        _cleanup_torch()

    rayd = cuda = dr = None
    if "rayd" in args.backends:
        rayd, cuda, dr = _load_rayd(args.rayd_source, args.rayd_root)
        backends["rayd"] = {
            "forward_static": _rayd_forward_performance(
                rayd,
                cuda,
                dr,
                mesh_data,
                updated_mesh_data,
                ray_data,
                updated_ray_data,
                dynamic=False,
                repeats=args.repeats,
                warmup=args.warmup,
            ),
            "forward_dynamic": _rayd_forward_performance(
                rayd,
                cuda,
                dr,
                mesh_data,
                updated_mesh_data,
                ray_data,
                updated_ray_data,
                dynamic=True,
                repeats=args.repeats,
                warmup=args.warmup,
            ),
        }
        if args.include_backward:
            backends["rayd"]["backward_static"] = _rayd_backward_performance(
                rayd,
                cuda,
                dr,
                mesh_data,
                updated_mesh_data,
                ray_data,
                updated_ray_data,
                dynamic=False,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            backends["rayd"]["backward_dynamic"] = _rayd_backward_performance(
                rayd,
                cuda,
                dr,
                mesh_data,
                updated_mesh_data,
                ray_data,
                updated_ray_data,
                dynamic=True,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            if args.torch_loss_backward:
                backends["rayd"]["backward_static_torch_loss"] = _rayd_torch_loss_backward_performance(
                    rayd,
                    cuda,
                    dr,
                    mesh_data,
                    updated_mesh_data,
                    ray_data,
                    updated_ray_data,
                    dynamic=False,
                    materialize_full=args.materialize_full_vjp,
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
                backends["rayd"]["backward_dynamic_torch_loss"] = _rayd_torch_loss_backward_performance(
                    rayd,
                    cuda,
                    dr,
                    mesh_data,
                    updated_mesh_data,
                    ray_data,
                    updated_ray_data,
                    dynamic=True,
                    materialize_full=args.materialize_full_vjp,
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
        _cleanup_drjit(dr)

    if "mitsuba" in args.backends:
        if dr is None:
            dr = importlib.import_module("drjit")
        mi = _try_import_mitsuba(args.mitsuba_variant)
        if mi is None:
            if args.require_mitsuba:
                raise RuntimeError("Mitsuba is not installed in the current environment.")
            backends["mitsuba"] = {"error": "Mitsuba is not installed in the current environment."}
        else:
            backends["mitsuba"] = {
                "forward_static": _mitsuba_forward_performance(
                    mi,
                    dr,
                    mesh_data,
                    updated_mesh_data,
                    ray_data,
                    updated_ray_data,
                    dynamic=False,
                    include_preliminary=args.mitsuba_preliminary,
                    repeats=args.repeats,
                    warmup=args.warmup,
                ),
                "forward_dynamic": _mitsuba_forward_performance(
                    mi,
                    dr,
                    mesh_data,
                    updated_mesh_data,
                    ray_data,
                    updated_ray_data,
                    dynamic=True,
                    include_preliminary=args.mitsuba_preliminary,
                    repeats=args.repeats,
                    warmup=args.warmup,
                ),
            }
            if args.include_backward:
                backends["mitsuba"]["backward_static"] = _mitsuba_backward_performance(
                    mi,
                    dr,
                    mesh_data,
                    updated_mesh_data,
                    ray_data,
                    updated_ray_data,
                    dynamic=False,
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
                backends["mitsuba"]["backward_dynamic"] = _mitsuba_backward_performance(
                    mi,
                    dr,
                    mesh_data,
                    updated_mesh_data,
                    ray_data,
                    updated_ray_data,
                    dynamic=True,
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
                if args.torch_loss_backward:
                    backends["mitsuba"]["backward_static_torch_loss"] = _mitsuba_torch_loss_backward_performance(
                        mi,
                        dr,
                        mesh_data,
                        updated_mesh_data,
                        ray_data,
                        updated_ray_data,
                        dynamic=False,
                        materialize_full=args.materialize_full_vjp,
                        repeats=args.repeats,
                        warmup=args.warmup,
                    )
                    backends["mitsuba"]["backward_dynamic_torch_loss"] = _mitsuba_torch_loss_backward_performance(
                        mi,
                        dr,
                        mesh_data,
                        updated_mesh_data,
                        ray_data,
                        updated_ray_data,
                        dynamic=True,
                        materialize_full=args.materialize_full_vjp,
                        repeats=args.repeats,
                        warmup=args.warmup,
                    )
            _cleanup_drjit(dr)

    return {
        "config": scenario.config()
        | {
            "repeats": args.repeats,
            "warmup": args.warmup,
            "dynamic_x_offset": args.dynamic_x_offset,
            "edges_enabled_for_raydn": args.edges,
            "forward_modes": {
                "full": "RayDN/RayD RayFlags.All materialized fields; Mitsuba ray_intersect fields.",
                "reduced": "RayDN/RayD RayFlags.None t-only; Mitsuba ray_intersect RayFlags.Minimal t-only.",
                "preliminary": "Mitsuba-only ray_intersect_preliminary t-only when --mitsuba-preliminary is set.",
                "vjp_full": "AD forward plus vector-Jacobian product for intersection.t, using full public intersection outputs.",
                "vjp_reduced": "AD forward plus vector-Jacobian product for intersection.t, using RayFlags.None t-only public outputs where available.",
            },
            "backward_torch_loss": args.torch_loss_backward,
            "materialize_full_vjp": args.materialize_full_vjp,
        },
        "backends": backends,
        "speedups": _speedups(backends),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RayD latest-style intersection stress benchmark with RayDN and Mitsuba backends."
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["raydn", "rayd", "mitsuba"],
        choices=["raydn", "rayd", "mitsuba"],
    )
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--mesh-resolution", type=int, default=64)
    parser.add_argument("--ray-grid-side", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dynamic-x-offset", type=float, default=2.0)
    parser.add_argument("--edges", action="store_true", help="Enable RayDN edge cache during scene build.")
    parser.add_argument("--rayd-source", choices=("package", "local"), default="package")
    parser.add_argument("--rayd-root", type=Path, default=RAYDI_ROOT)
    parser.add_argument("--mitsuba-variant", default="cuda_ad_rgb")
    parser.add_argument("--mitsuba-preliminary", action="store_true")
    parser.add_argument("--include-backward", action="store_true")
    parser.add_argument(
        "--torch-loss-backward",
        action="store_true",
        help="Also measure PyTorch loss.backward wrapper phases where the backend can be exposed through Torch autograd.",
    )
    parser.add_argument(
        "--materialize-full-vjp",
        action="store_true",
        help="Materialize full intersection fields before full-output VJP timing.",
    )
    parser.add_argument("--require-mitsuba", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if "raydn" in args.backends and not torch.cuda.is_available():
        raise SystemExit("RayDN backend requires CUDA torch.")

    scenario_specs = args.scenario or [f"{args.mesh_resolution}:{args.ray_grid_side}"]
    scenarios = [_parse_scenario(spec) for spec in scenario_specs]

    results = {
        "benchmark": "raydn_rayd_mitsuba_intersection_stress",
        "environment": {
            "raydn": {"version": getattr(rt, "__version__", "unknown")},
            "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
            "rayd_source": args.rayd_source,
            "rayd_root": str(args.rayd_root) if args.rayd_source == "local" else None,
            "mitsuba_variant": args.mitsuba_variant if "mitsuba" in args.backends else None,
        },
        "suite_config": {
            "repeats": args.repeats,
            "warmup": args.warmup,
            "dynamic_x_offset": args.dynamic_x_offset,
            "backends": args.backends,
            "include_backward": args.include_backward,
            "torch_loss_backward": args.torch_loss_backward,
            "materialize_full_vjp": args.materialize_full_vjp,
        },
        "scenarios": [_run_scenario(args, scenario) for scenario in scenarios],
    }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
