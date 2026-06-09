from __future__ import annotations

from pathlib import Path

import torch
import raydn as rt

from .baseline_utils import write_baseline


OUT_DIR = Path(__file__).resolve().parent / "baselines" / "raydn_native"


def _as_list(tensor: torch.Tensor):
    return tensor.detach().cpu().tolist()


def _scene():
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(verts, faces))
    scene.build()
    return scene


def intersect_case():
    scene = _scene()
    ray = rt.Ray(
        torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
    )
    out = scene.intersect(ray)
    return {
        "t": _as_list(out.t),
        "p": _as_list(out.p),
        "barycentric": _as_list(out.barycentric),
        "prim_id": _as_list(out.prim_id),
    }


def edge_case():
    scene = _scene()
    point = torch.tensor([[0.25, -0.2, 0.0]], device="cuda", dtype=torch.float32)
    out = scene.nearest_edge(point)
    return {
        "distance": _as_list(out.distance),
        "edge_point": _as_list(out.edge_point),
        "edge_t": _as_list(out.edge_t),
        "edge_id": _as_list(out.edge_id),
    }


def multipath_case():
    scene = _scene()
    ray = rt.Ray(
        torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
    )
    refl = scene.trace_reflections(ray, max_bounces=1)
    source = torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32)
    receiver = torch.tensor([[0.25, 0.25, 1.0]], device="cuda", dtype=torch.float32)
    epc = scene.trace_refl_epc_field(source, receiver, max_bounces=1)
    dfr = scene.accum_dfr_direct(
        edge_pos=torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        edge_dir=torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
        src=torch.tensor([[0.0, -1.0, 0.2]], device="cuda", dtype=torch.float32),
    )
    return {
        "reflection_t": _as_list(refl.t),
        "reflection_valid": _as_list(refl.valid),
        "epc_field_real": _as_list(epc.field_real),
        "epc_field_imag": _as_list(epc.field_imag),
        "dfr_power": _as_list(dfr.power),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA torch is required to generate raydn baselines.")
    write_baseline(OUT_DIR / "intersect.json", intersect_case())
    write_baseline(OUT_DIR / "edge_queries.json", edge_case())
    write_baseline(OUT_DIR / "multipath.json", multipath_case())


if __name__ == "__main__":
    main()
