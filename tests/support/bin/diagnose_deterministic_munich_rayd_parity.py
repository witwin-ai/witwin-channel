"""Compare Munich deterministic diffraction cells for native vs. RayD accumulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from tests.support.bin import profile_deterministic_munich as profile


def _parse_args() -> argparse.Namespace:
    parser = profile.parse_args([])
    parser.grid_size = 16
    parser.max_diffractions = 1
    parser.reflection_n_rays = 1
    parser.reflection_max_bounces = 0
    parser.edge_selection_mode = "all_edges"
    parser.boundary_edge_policy = "half_plane"
    parser.solver_mode = "fast_approximate"
    parser.memory_profile = "memory_safe"
    parser.assert_finite = True
    parser.json = False
    parser.kernel_history = False
    parser.shadow_boundary_correction = False
    parser.munich_xml = Path(
        r"E:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1\src\sionna\rt\scenes\munich\munich.xml"
    )
    return parser


def _solve_components(mode: str) -> np.ndarray:
    args = _parse_args()
    args.diffraction_accumulate_primal = mode

    repo_root = profile._repo_root()
    repo_root_str = str(repo_root)
    sys.path = [path for path in sys.path if str(Path(path or ".").resolve()) != repo_root_str]
    sys.path.insert(0, repo_root_str)

    import drjit as dr
    from witwin.channel.core.scene import ReceiverGrid, Scene, Transmitter
    from witwin.channel.core.scene.edge_policy import EdgePolicy
    from witwin.channel.deterministic import Config, Tuning, solve

    munich_xml = profile._resolve_munich_xml(repo_root, args.munich_xml)
    scene = Scene.load_mitsuba(
        munich_xml,
        device="cuda",
        merge_shapes=True,
        frequency=float(args.frequency_hz),
        source_root=profile._sionna_source_root_from_xml(munich_xml),
    )
    scene.add(Transmitter("tx", (float(args.tx_x), float(args.tx_y), float(args.tx_z))))
    scene.add(
        ReceiverGrid(
            "rm",
            axis="z",
            position=float(args.plane_z),
            bounds=((float(args.xmin), float(args.xmax)), (float(args.ymin), float(args.ymax))),
            grid_shape=(int(args.grid_size), int(args.grid_size)),
        )
    )
    config = Config(
        num_samples=int(args.reflection_n_rays),
        max_bounces=int(args.reflection_max_bounces),
        max_diffraction_order=int(args.max_diffractions),
        shadow_boundary_correction=bool(args.shadow_boundary_correction),
        edge_policy=EdgePolicy(
            edge_selection_mode=str(args.edge_selection_mode),
            edge_diffraction=True,
            boundary_edge_policy=str(args.boundary_edge_policy),
        ),
        tuning=Tuning(
            shadow_boundary_backend=str(args.shadow_boundary_backend),
            shadow_boundary_tile_shape=tuple(int(value) for value in args.shadow_boundary_tile_shape),
            shadow_boundary_band_width_wavelengths=float(args.shadow_boundary_band_width_wavelengths),
            shadow_boundary_max_candidate_factor=float(args.shadow_boundary_max_candidate_factor),
            enable_rd_diffraction=int(args.max_diffractions) > 0,
            solver_mode=str(args.solver_mode),
            memory_profile=str(args.memory_profile),
            diffraction_execution={"accumulate_primal": mode},
        ),
    )
    result = solve(scene=scene, transmitter="tx", receiver="rm", config=config)
    dr.sync_thread()
    return np.asarray(result.components["diffraction"], dtype=np.float64).reshape(
        int(args.grid_size), int(args.grid_size)
    )


def _cell_xy(i: int, j: int, args: argparse.Namespace) -> tuple[float, float]:
    x = float(args.xmin) + (i + 0.5) * (float(args.xmax) - float(args.xmin)) / int(args.grid_size)
    y = float(args.ymin) + (j + 0.5) * (float(args.ymax) - float(args.ymin)) / int(args.grid_size)
    return x, y


def main() -> None:
    args = _parse_args()
    native = _solve_components("drjit")
    rayd = _solve_components("rayd_exact_coherent")
    delta = rayd - native
    abs_delta = np.abs(delta)
    native_nonzero = native > 0.0
    rayd_nonzero = rayd > 0.0

    flat_order = np.argsort(abs_delta.reshape(-1))[::-1][:20]
    top = []
    for flat in flat_order:
        j, i = divmod(int(flat), int(args.grid_size))
        x, y = _cell_xy(i, j, args)
        top.append(
            {
                "cell": [j, i],
                "xy": [x, y],
                "native": float(native[j, i]),
                "rayd": float(rayd[j, i]),
                "delta": float(delta[j, i]),
            }
        )

    payload = {
        "native_sum": float(np.sum(native)),
        "rayd_sum": float(np.sum(rayd)),
        "delta_sum": float(np.sum(delta)),
        "native_nonzero": int(np.count_nonzero(native_nonzero)),
        "rayd_nonzero": int(np.count_nonzero(rayd_nonzero)),
        "rayd_only_count": int(np.count_nonzero(rayd_nonzero & ~native_nonzero)),
        "native_only_count": int(np.count_nonzero(native_nonzero & ~rayd_nonzero)),
        "top_abs_delta": top,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
