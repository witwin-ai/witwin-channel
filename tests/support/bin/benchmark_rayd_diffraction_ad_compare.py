"""Compare RayD handwritten diffraction AD against a DrJit reference graph.

The DrJit reference intentionally models the same continuous fixed-visibility
direct/suffix formulas used by the small RayD validation scene. It is a
benchmark reference only; solver runtime paths still call RayD directly.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable

import drjit as dr
import drjit.cuda as cuda
import drjit.cuda.ad as ad
import rayd as pj


def _make_direct_scene() -> pj.Scene:
    vertices = cuda.Array3f(
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [10.0, 10.0, 10.0],
    )
    scene = pj.Scene()
    scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
    scene.build()
    return scene


def _make_suffix_scene() -> pj.Scene:
    vertices = cuda.Array3f(
        [-2.0, 2.0, -2.0],
        [0.0, 0.0, 0.0],
        [-2.0, -2.0, 2.0],
    )
    scene = pj.Scene()
    scene.add_mesh(pj.Mesh(vertices, cuda.Array3i([0], [1], [2])))
    scene.build()
    return scene


def _make_grid(kind: str) -> pj.DfrGrid:
    grid = pj.DfrGrid()
    if kind == "suffix":
        grid.axis = 1
        grid.position = -2.0
    else:
        grid.axis = 2
        grid.position = -1.0
    grid.coord0_min = -1.0
    grid.coord0_max = 1.0
    grid.coord1_min = -1.0
    grid.coord1_max = 1.0
    grid.resolution0 = 1
    grid.resolution1 = 1
    grid.cell_area = 4.0
    return grid


def _make_options(kind: str, samples: int) -> pj.DfrOptions:
    options = pj.DfrOptions()
    options.wavelength = 0.125
    options.k = 50.26548245743669
    options.seed = 41
    options.samples = samples
    options.max_order = 1
    options.direct_samples = samples if kind == "direct" else 0
    options.keller_samples = 0
    options.suffix_samples = samples if kind == "suffix" else 0
    options.strategy_mask = (
        pj.RAYD_DFR_SUFFIX_REFL if kind == "suffix" else pj.RAYD_DFR_DIRECT
    )
    options.sample_sequence = pj.RAYD_DFR_HASH
    options.receiver_model = pj.RAYD_DFR_MATCHED_ISO
    return options


def _make_material(gain: ad.Float) -> pj.DfrMaterialAD:
    material = pj.DfrMaterialAD()
    material.eta_r = ad.Float([4.0])
    material.sigma = ad.Float([0.0])
    material.mu_r = ad.Float([1.0])
    material.gain = gain
    material.valid = ad.Bool([True])
    return material


def _make_states(kind: str, src_z: ad.Float) -> pj.DfrStatesAD:
    states = pj.DfrStatesAD()
    states.count = 1
    states.edge_index = ad.Int([0])
    if kind == "suffix":
        states.edge_pos = ad.Array3f([0.0], [-1.0], [0.0])
        states.edge_t_min = ad.Float([-0.25])
        states.edge_t_max = ad.Float([0.25])
        states.src = ad.Array3f([0.0], [-1.0], src_z)
    else:
        states.edge_pos = ad.Array3f([0.0], [0.0], [0.0])
        states.edge_t_min = ad.Float([-0.5])
        states.edge_t_max = ad.Float([0.5])
        states.src = ad.Array3f([0.0], [0.0], src_z)
    states.edge_dir = ad.Array3f([1.0], [0.0], [0.0])
    states.n0 = ad.Array3f([0.0], [1.0], [0.0])
    states.n1 = ad.Array3f([0.0], [-1.0], [0.0])
    states.prim0 = ad.Int([0])
    states.prim1 = ad.Int([0])
    states.exterior_angle = ad.Float([1.5 * math.pi])
    states.src_power = ad.Float([1.0])
    states.wi = ad.Array3f([0.0], [0.0], [-1.0])
    states.d0 = ad.Array3f([0.0], [0.0], [-1.0])
    states.prefix_depth = ad.Int([0])
    return states


def rayd_loss(scene: pj.Scene, kind: str, samples: int, src_z: ad.Float, gain: ad.Float):
    result = scene.accum_dfr_direct(
        _make_states(kind, src_z),
        _make_grid(kind),
        _make_material(gain),
        _make_options(kind, samples),
        True,
    )
    return dr.sum(result.power)


def reference_loss(kind: str, samples: int, src_z: ad.Float, gain: ad.Float):
    lane = dr.arange(ad.Float, samples)
    edge_u = (lane + 0.5) / float(samples)
    if kind == "suffix":
        edge_x = -0.25 + edge_u * 0.5
        source_dist2 = edge_x * edge_x + src_z * src_z
        image_x = edge_x
        image_y = ad.Float([1.0])
        ray_dx = -image_x
        ray_dy = -3.0
        ray_len = dr.sqrt(ray_dx * ray_dx + ray_dy * ray_dy)
        ray_dir_x = ray_dx / ray_len
        ray_dir_y = ray_dy / ray_len
        ray_t = -image_y / ray_dir_y
        refl_x = image_x + ray_t * ray_dir_x
        refl_y = image_y + ray_t * ray_dir_y
        target_dist2 = (refl_x - edge_x) * (refl_x - edge_x) + (refl_y + 1.0) * (refl_y + 1.0)
        outgoing_dist2 = refl_x * refl_x + (-2.0 - refl_y) * (-2.0 - refl_y)
        fspl = (0.125 / (4.0 * math.pi)) ** 2 / outgoing_dist2
        material_scale = gain * gain * gain
        edge_length = 0.5
    else:
        edge_x = -0.5 + edge_u
        source_dist2 = edge_x * edge_x + src_z * src_z
        target_dist2 = edge_x * edge_x + 1.0
        fspl = ad.Float([1.0])
        material_scale = gain
        edge_length = 1.0
    wedge_scale = 0.75
    contribution = (
        material_scale
        * edge_length
        * 4.0
        * wedge_scale
        / float(samples)
        * fspl
        / (source_dist2 * target_dist2)
    )
    return dr.sum(contribution)


def _time_ms(fn: Callable[[], None], repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def _run_mode(
    backend: str,
    kind: str,
    samples: int,
    mode: str,
    scene: pj.Scene | None,
) -> None:
    src_z = ad.Float([1.0])
    gain = ad.Float([1.0])
    dr.enable_grad(src_z, gain)
    loss = (
        rayd_loss(scene, kind, samples, src_z, gain)
        if backend == "rayd"
        else reference_loss(kind, samples, src_z, gain)
    )
    if mode == "primal":
        dr.eval(loss)
    elif mode == "jvp":
        dr.set_grad(src_z, ad.Float([1.0]))
        dr.forward(src_z)
        dr.eval(dr.grad(loss))
    elif mode == "vjp":
        dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        dr.eval(dr.grad(src_z), dr.grad(gain))
    else:
        raise ValueError(f"unknown mode: {mode}")
    dr.sync_thread()


def run_case(kind: str, samples: int, repeats: int, warmup: int) -> dict[str, object]:
    scene = _make_suffix_scene() if kind == "suffix" else _make_direct_scene()
    result: dict[str, object] = {"case": kind, "samples": samples, "modes": {}}
    for mode in ("primal", "jvp", "vjp"):
        rayd_times = _time_ms(
            lambda mode=mode: _run_mode("rayd", kind, samples, mode, scene),
            repeats=repeats,
            warmup=warmup,
        )
        ref_times = _time_ms(
            lambda mode=mode: _run_mode("reference", kind, samples, mode, None),
            repeats=repeats,
            warmup=warmup,
        )
        rayd_median = statistics.median(rayd_times)
        ref_median = statistics.median(ref_times)
        result["modes"][mode] = {
            "rayd_handwritten_ms": rayd_median,
            "drjit_reference_ms": ref_median,
            "rayd_over_reference": rayd_median / ref_median if ref_median > 0 else None,
            "rayd_samples_ms": rayd_times,
            "drjit_reference_samples_ms": ref_times,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["direct", "suffix", "all"], default="all")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cases = ["direct", "suffix"] if args.case == "all" else [args.case]
    results = [
        run_case(case, samples=max(1, args.samples), repeats=args.repeats, warmup=args.warmup)
        for case in cases
    ]
    payload = {
        "reference_scope": (
            "DrJit reference is the fixed-visibility continuous direct/suffix "
            "formula; RayD timings include OptiX visibility and forward tape capture."
        ),
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for case in results:
            print(f"{case['case']} samples={case['samples']}")
            for mode, values in case["modes"].items():
                print(
                    f"  {mode}: rayd={values['rayd_handwritten_ms']:.3f} ms "
                    f"drjit_ref={values['drjit_reference_ms']:.3f} ms "
                    f"ratio={values['rayd_over_reference']:.3f}"
                )


if __name__ == "__main__":
    main()
