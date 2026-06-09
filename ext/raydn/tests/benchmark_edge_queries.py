from __future__ import annotations

import argparse
import time

import torch

import raydn as rt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="raydn-native")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.backend != "raydn-native":
        raise SystemExit("only --backend raydn-native is supported")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA torch is required")

    n = 64 if args.quick else 128
    qn = 4096 if args.quick else 32768
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
    faces_t = torch.tensor(faces, device="cuda", dtype=torch.int32)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(verts, faces_t))
    scene.build()
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000.0

    q = torch.rand((qn, 3), device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = scene.nearest_edge(q)
    torch.cuda.synchronize()
    query_ms = (time.perf_counter() - t0) * 1000.0
    if not torch.isfinite(out.distance).all().item():
        raise SystemExit("edge benchmark produced non-finite distances")
    print(f"backend={args.backend} build_ms={build_ms:.3f} query_ms={query_ms:.3f} queries={qn}")


if __name__ == "__main__":
    main()
