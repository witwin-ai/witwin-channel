"""Measure `scene.compile` cost against scene size.

`compile` reads all four Core version properties before it consults its cache,
so a cache hit still walks the whole world model, and every `solve` pays it
again. That cost is linear in structure count and entirely host-side, which
makes it invisible to the solver benchmarks: they compile one scene once,
outside the timing loop.

This harness measures the warm-cache path, which is what an optimization loop
solving repeatedly against one unchanged scene actually pays.

Three budgets are reported, and they fail for different reasons. The regression
that motivated this harness moved compile from an O(1) integer read to an O(N)
walk, so it was linear both before and after; a scaling check alone would have
missed it entirely. The constant factor is what needed a gate.

- `scaling_ratio` compares 4N structures against N. Linear is 4.0; a quadratic
  regression is 16.0. Machine independent, so it gates everywhere. Catches an
  algorithmic regression, not a constant-factor one.
- `calibrated_per_structure` is the per-structure cost divided by a fixed
  Python workload timed in the same process. Dividing out host CPU speed makes
  a constant-factor ceiling portable, which is the check that would have caught
  the original regression.
- `version_share` is the warm compile time divided by the cost of reading the
  four version properties alone. It catches a second O(N) pass being added to
  compile, independently of how fast either one is.

`per_structure_us` is also recorded as a raw number. It depends on host CPU and
Python build, so it is reported rather than gated.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from witwin.channel.scene import compile as compile_scene
from witwin.channel.scene.compiler import clear_compile_cache
from witwin.core import AntennaState, Mesh, PhysicalMaterial, Scene, Structure
from witwin.core.identity import reserve_antenna_id

SCHEMA_NAME = "witwin.channel.compile_scaling"
SCHEMA_VERSION = "1.0.0"

_BOX_FACES = (
    (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
    (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (3, 0, 4), (3, 4, 7),
)


def build_scene(structure_count: int) -> Scene:
    """A deterministic scene whose only varying dimension is structure count."""

    faces = torch.tensor(_BOX_FACES, dtype=torch.int64)
    structures = []
    for index in range(structure_count):
        offset = float(index) * 3.0
        vertices = torch.tensor(
            [
                [offset, 0.0, 0.0], [offset + 2, 0.0, 0.0],
                [offset + 2, 2.0, 0.0], [offset, 2.0, 0.0],
                [offset, 0.0, 3.0], [offset + 2, 0.0, 3.0],
                [offset + 2, 2.0, 3.0], [offset, 2.0, 3.0],
            ],
            dtype=torch.float32,
        )
        structures.append(
            Structure(
                geometry=Mesh(vertices=vertices, faces=faces.clone()),
                material=PhysicalMaterial(
                    name=f"concrete{index % 6}", eps_r=5.24, sigma_e=0.0462
                ),
                structure_id=index,
                material_id=index,
                assignment_id=index,
                surface_id=index,
            )
        )
    transmitter = AntennaState(
        reserve_antenna_id(0), "tx", torch.tensor([0.0, 0.0, 10.0]), power_w=1.0
    )
    return Scene(structures=structures, endpoints=[transmitter])


def _measure(scene: Scene, frequency_hz: float, *, warmup: int, repeats: int):
    compile_scene(scene, reference_frequency_hz=frequency_hz)
    for _ in range(warmup):
        compile_scene(scene, reference_frequency_hz=frequency_hz)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        compile_scene(scene, reference_frequency_hz=frequency_hz)
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return samples


class _CalibrationNode:
    """Shaped like what the traversal actually walks: attributes and a dict."""

    __slots__ = ("left", "right", "payload")

    def __init__(self) -> None:
        self.left = 1.0
        self.right = "leaf"
        self.payload = {"a": 1, "b": 2.0, "c": None}


def _calibration_us(*, repeats: int) -> float:
    """Time a fixed Python workload so the constant-factor budget is portable.

    This is deliberately not the traversal under test. It is a stable mix of
    attribute access, dict lookups, and type checks, so dividing by it removes
    host CPU speed from the budget without tracking the thing being measured.
    """

    nodes = [_CalibrationNode() for _ in range(64)]
    atomic = frozenset({type(None), bool, int, float, str})
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        total = 0
        for _ in range(64):
            for node in nodes:
                for name in _CalibrationNode.__slots__:
                    item = getattr(node, name)
                    if type(item) not in atomic:
                        for key in sorted(item):
                            if type(item[key]) not in atomic:
                                total += 1
        samples.append((time.perf_counter() - start) * 1e6)
    return min(samples)


def _version_only_ms(scene: Scene, *, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        (
            scene.topology_version,
            scene.geometry_version,
            scene.material_version,
            scene.assignment_version,
        )
        samples.append((time.perf_counter() - start) * 1e3)
    return min(samples)


def run(args: argparse.Namespace) -> dict:
    sizes = tuple(int(value) for value in args.sizes.split(","))
    if len(sizes) < 2:
        raise SystemExit("--sizes needs at least two structure counts")

    points = []
    for size in sizes:
        clear_compile_cache()
        scene = build_scene(size)
        samples = _measure(
            scene,
            args.frequency_hz,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        median = statistics.median(samples)
        points.append(
            {
                "structures": size,
                "warm_median_ms": median,
                "warm_p95_ms": samples[min(len(samples) - 1, int(0.95 * len(samples)))],
                "warm_min_ms": samples[0],
                "per_structure_us": samples[0] * 1000.0 / size,
                "version_only_min_ms": _version_only_ms(
                    scene, repeats=args.repeats
                ),
            }
        )
        del scene

    calibration_us = _calibration_us(repeats=args.repeats)

    smallest, largest = points[0], points[-1]
    size_ratio = largest["structures"] / smallest["structures"]
    scaling_ratio = (
        largest["warm_min_ms"] / smallest["warm_min_ms"]
        if smallest["warm_min_ms"] > 0
        else float("inf")
    )
    max_per_structure_us = max(point["per_structure_us"] for point in points)
    version_share = max(
        point["warm_min_ms"] / point["version_only_min_ms"]
        for point in points
        if point["version_only_min_ms"] > 0
    )

    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "frequency_hz": args.frequency_hz,
            "path": "warm cache; the scene object is unchanged between calls",
        },
        "points": points,
        "summary": {
            "size_ratio": size_ratio,
            "scaling_ratio": scaling_ratio,
            "max_per_structure_us": max_per_structure_us,
            "calibration_us": calibration_us,
            "calibrated_per_structure": max_per_structure_us / calibration_us,
            "version_share": version_share,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="256,1024")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--frequency-hz", dest="frequency_hz", type=float, default=2.4e9)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run(args)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
