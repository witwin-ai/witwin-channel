# Copyright Xingyu Chen.
# Benchmarks worker.

"""Benchmarks worker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
from typing import Callable

import numpy as np
import torch

from witwin.channel import build_info
from witwin.channel.importers import load_mitsuba
from witwin.channel.radiomap import Coherent as DeterministicConfig
from witwin.channel.radiomap import solve as solve_deterministic
from witwin.channel.radiomap import Stochastic as BasicConfig
from witwin.channel.radiomap import solve as solve_basic
from witwin.channel.experimental.bdpt import Config as BDPTConfig
from witwin.channel.experimental.bdpt import solve as solve_bdpt
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import solve as solve_path
from witwin.core import AntennaState, Mesh, PhysicalMaterial, ReceiverGrid, Scene, Structure
from witwin.core.identity import new_antenna_id


SCHEMA = {"name": "witwin.channel.phase13-phase12-worker", "version": 2}
PROFILE_SCHEMA = {
    "name": "witwin.channel.phase13-phase12-profile-worker",
    "version": 2,
}


def _group_scenarios() -> dict[str, str]:
    path = Path(__file__).with_name("phase13_phase12_profile_contract.json")
    contract = json.loads(path.read_text(encoding="utf-8"))
    groups = contract.get("groups")
    if not isinstance(groups, dict) or set(groups) != {
        "enumerated_penetration", "montecarlo_penetration", "diffraction",
    }:
        raise RuntimeError("profile contract group set is not canonical")
    result = {name: str(row["scenario"]) for name, row in groups.items()}
    if len(set(result.values())) != len(result):
        raise RuntimeError("profile contract scenarios must be unique")
    return result


GROUP_SCENARIOS = _group_scenarios()
NON_TARGET_NAMES = (
    "receiver_point_end_to_end",
    "path_non_diffraction_end_to_end",
    "deterministic_non_diffraction_end_to_end",
    "bdpt_end_to_end",
)
HASH_FORMAT = "semantic-dtype-shape-little-endian-contiguous"
HASH_SCHEMA_VERSION = 1
FIXTURE_FREQUENCY_HZ = 3.5e9
MUNICH_FREQUENCY_HZ = 2.4e9


def _tensor_bytes(tensor: torch.Tensor) -> tuple[bytes, list[int], str]:
    value = tensor.detach().contiguous().cpu()
    array = value.numpy()
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and np.little_endian is False
    ):
        array = array.byteswap().newbyteorder("<")
    return array.tobytes(order="C"), list(value.shape), str(value.dtype).removeprefix("torch.")


def _hash_tensors(rows: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in rows:
        payload, shape, dtype = _tensor_bytes(tensor)
        header = json.dumps(
            {"name": name, "dtype": dtype, "shape": shape},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
        digest.update(struct.pack("<I", len(header)))
        digest.update(header)
        digest.update(payload)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _grid(size: int = 32) -> ReceiverGrid:
    return ReceiverGrid(
        new_antenna_id(),
        origin=torch.tensor([-12.0, -12.0, 1.5]),
        x_axis=torch.tensor([1.0, 0.0, 0.0]),
        y_axis=torch.tensor([0.0, 1.0, 0.0]),
        shape=(size, size), spacing=(24.0 / size, 24.0 / size),
    )


def _structure(
    vertices: list[list[float]], faces: list[list[int]], material: PhysicalMaterial, *, name: str,
    surface_id: int,
) -> Structure:
    mesh = Mesh(
        torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int32),
        recenter=False, fill_mode="surface", topology_diagnostics=False,
    )
    return Structure(mesh, material, name=name, surface_id=surface_id)


def _transmitter(position: list[float]) -> AntennaState:
    return AntennaState(new_antenna_id(), "tx", torch.tensor(position))


def _wall_scene(*, point_receiver: bool = False) -> Scene:
    wall = _structure(
        [[0.0, -20.0, 0.0], [0.0, 20.0, 0.0], [0.0, -20.0, 20.0], [0.0, 20.0, 20.0]],
        [[0, 1, 2], [1, 3, 2]],
        PhysicalMaterial(eps_r=4.8, sigma_e=0.015),
        name="fixed-wall", surface_id=1,
    )
    receiver: AntennaState | ReceiverGrid
    receiver = (
        AntennaState(new_antenna_id(), "rx", torch.tensor([8.0, 0.0, 1.5]))
        if point_receiver else _grid()
    )
    return Scene(
        structures=[wall],
        endpoints=[_transmitter([-8.0, 0.0, 2.0]), receiver],
        metadata={"fixture": "phase13-phase12-fixed-wall"},
    )


def _wedge_scene() -> Scene:
    metal = PhysicalMaterial.perfect_conductor()
    structures = [
        _structure(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 8.0], [0.0, 12.0, 0.0]], [[0, 1, 2]], metal,
            name="wedge-a", surface_id=11,
        ),
        _structure(
            [[0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [0.0, 0.0, 8.0]], [[0, 1, 2]], metal,
            name="wedge-b", surface_id=12,
        ),
    ]
    return Scene(
        structures=structures,
        endpoints=[_transmitter([-4.0, -4.0, 3.0]), _grid()],
        metadata={"fixture": "phase13-phase12-fixed-wedge"},
    )


def _munich_scene(scene_xml: Path, sionna_root: Path) -> Scene:
    # The importer resolves shape files relative to the XML; the canonical argv still names the
    # Sionna source root, which must exist.
    if not Path(sionna_root).is_dir():
        raise RuntimeError(f"Sionna source root is not a directory: {sionna_root}")
    imported = load_mitsuba(scene_xml, frequency_hz=MUNICH_FREQUENCY_HZ, merge_shapes=True)
    return Scene(
        structures=list(imported.structures),
        endpoints=[
            _transmitter([8.5, 21.0, 27.0]),
            ReceiverGrid(
                new_antenna_id(),
                origin=torch.tensor([-116.25, -115.9375, 1.5]),
                x_axis=torch.tensor([1.0, 0.0, 0.0]),
                y_axis=torch.tensor([0.0, 1.0, 0.0]),
                shape=(32, 32), spacing=(7.5, 8.125),
            ),
        ],
        # The solvers keep a merged Mitsuba import's edge table as imported.
        metadata={"mitsuba": {"merge_shapes": True, "source": str(scene_xml)}},
    )


def _target(group: str, munich_xml: Path, sionna_root: Path) -> Callable[[], object]:
    if group == "enumerated_penetration":
        scene = _wall_scene()
        config = DeterministicConfig(
            max_depth=2, components={"transmission"},
            export_paths=True,
        )
        return lambda: solve_deterministic(
            scene, config, reference_frequency_hz=FIXTURE_FREQUENCY_HZ
        )
    if group == "montecarlo_penetration":
        scene = _wall_scene()
        config = BasicConfig(
            samples=4096, max_depth=2, seed=13027,
            components={"transmission"},
        )
        return lambda: solve_basic(scene, config, reference_frequency_hz=FIXTURE_FREQUENCY_HZ)
    scene = _munich_scene(munich_xml, sionna_root)
    config = DeterministicConfig(
        max_depth=1, components={"diffraction"},
        combine="field", return_field=True, export_paths=True,
    )
    return lambda: solve_deterministic(scene, config, reference_frequency_hz=MUNICH_FREQUENCY_HZ)


def pair_field_sum(
    tx_id: torch.Tensor, rx_id: torch.Tensor, field_xyz: torch.Tensor, *, tx_count: int,
    rx_count: int,
) -> torch.Tensor:
    """Per (transmitter, receiver) sum of row fields in row order, bitwise repeatable."""

    field = torch.zeros((tx_count * rx_count, 3), device=field_xyz.device, dtype=torch.complex64)
    pair = tx_id.to(dtype=torch.long) * rx_count + rx_id.to(dtype=torch.long)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        field.index_add_(0, pair, field_xyz)
    finally:
        torch.use_deterministic_algorithms(previous)
    return field


def _pair_vector_field(result: object) -> torch.Tensor:
    """Per (transmitter, receiver) sum of the exported world-Cartesian path fields."""

    tx_count = int(result.path_gain.shape[0])
    paths = result.paths
    return pair_field_sum(
        paths.tx_id, paths.rx_id, paths.field_xyz, tx_count=tx_count,
        rx_count=int(result.path_gain.numel()) // tx_count,
    )


def _target_tensor(group: str, result: object) -> torch.Tensor:
    if group == "diffraction":
        # The diffraction target is the per-pair vector field (six float32 components).
        return torch.view_as_real(_pair_vector_field(result)).reshape(-1, 6).contiguous()
    return result.path_gain.detach().contiguous()


def _result_tensors(result: object) -> list[tuple[str, torch.Tensor]]:
    rows: list[tuple[str, torch.Tensor]] = []
    path_gain = getattr(result, "path_gain", None)
    if isinstance(path_gain, torch.Tensor):
        rows.append(("path_gain", path_gain))
    else:
        for name in ("valid", "a", "tau", "primitive_id", "interaction_type"):
            value = getattr(result, name, None)
            if isinstance(value, torch.Tensor):
                rows.append((name, value))
    if not rows:
        raise RuntimeError("result exposes no canonical tensors")
    field = getattr(result, "field", None)
    if isinstance(field, torch.Tensor):
        rows.append(("field", torch.view_as_real(field)))
    components = getattr(result, "component_power", {})
    if isinstance(components, dict):
        rows.extend((f"component:{name}", components[name]) for name in sorted(components))
    return rows


def _time_workload(
    solve: Callable[[], object], *, semantic_hash: Callable[[object], str] | None = None,
) -> tuple[list[float], list[float], object, list[str]]:
    with torch.inference_mode():
        warmup_result = solve()
    torch.cuda.synchronize()
    del warmup_result
    torch.cuda.reset_peak_memory_stats()
    cuda_ms: list[float] = []
    wall_ms: list[float] = []
    hashes: list[str] = []
    final_result: object | None = None
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        with torch.inference_mode():
            result = solve()
        end.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000.0)
        cuda_ms.append(float(start.elapsed_time(end)))
        if semantic_hash is not None:
            hashes.append(semantic_hash(result))
        final_result = result
    if final_result is None:
        raise RuntimeError("fixed steady measurement produced no result")
    return cuda_ms, wall_ms, final_result, hashes


def _non_target_workloads() -> dict[str, Callable[[], object]]:
    point = _wall_scene(point_receiver=True)
    wedge = _wedge_scene()
    point_config = DeterministicConfig(components={"los"})
    path_config = PathConfig(
        components={"los", "reflection"}, max_depth=1
    )
    deterministic_config = DeterministicConfig(
        components={"los", "reflection"}, max_depth=1
    )
    bdpt_config = BDPTConfig(
        samples=1024, seed=13027, max_depth=1, components={"los", "reflection"}
    )
    frequency = FIXTURE_FREQUENCY_HZ
    return {
        "receiver_point_end_to_end": lambda: solve_deterministic(
            point, point_config, reference_frequency_hz=frequency
        ),
        "path_non_diffraction_end_to_end": lambda: solve_path(
            wedge, path_config, reference_frequency_hz=frequency
        ),
        "deterministic_non_diffraction_end_to_end": lambda: solve_deterministic(
            wedge, deterministic_config, reference_frequency_hz=frequency
        ),
        "bdpt_end_to_end": lambda: solve_bdpt(wedge, bdpt_config, reference_frequency_hz=frequency),
    }


def _profile(args: argparse.Namespace) -> None:
    expected = GROUP_SCENARIOS[args.group]
    if args.profile_only != expected:
        raise RuntimeError("profile scenario does not match comparison group")
    solve = _target(args.group, args.munich_scene_xml, args.sionna_source_root)
    solve()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(7):
        solve()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(json.dumps({
        "schema": PROFILE_SCHEMA, "group": args.group,
        "scenario": expected, "warmup": 1, "steady_repeats": 7,
    }, sort_keys=True), flush=True)


def _run(args: argparse.Namespace) -> None:
    target_solve = _target(args.group, args.munich_scene_xml, args.sionna_source_root)
    target_cuda, target_wall, final_result, target_hashes = _time_workload(
        target_solve,
        semantic_hash=lambda result: _hash_tensors(
            [("target", _target_tensor(args.group, result))]
        ),
    )
    target_tensor = _target_tensor(args.group, final_result)
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    timings = [{
        "name": {
            "enumerated_penetration": "enumerated_penetration_end_to_end",
            "montecarlo_penetration": "montecarlo_basic_penetration_end_to_end",
            "diffraction": "deterministic_munich_end_to_end",
        }[args.group],
        "steady_cuda_ms": target_cuda, "steady_wall_ms": target_wall,
    }]
    unaffected: list[dict[str, str]] = []
    non_target_results: dict[str, object] = {}
    for name, solve in _non_target_workloads().items():
        cuda_ms, wall_ms, result, _ = _time_workload(solve)
        timings.append({"name": name, "steady_cuda_ms": cuda_ms, "steady_wall_ms": wall_ms})
        non_target_results[name] = result
    point = non_target_results["receiver_point_end_to_end"]
    path_result = non_target_results["path_non_diffraction_end_to_end"]
    deterministic = non_target_results["deterministic_non_diffraction_end_to_end"]
    unaffected.extend([
        {"name": "receiver_point", "sha256": _hash_tensors(_result_tensors(point))},
        {"name": "path_export", "sha256": _hash_tensors(_result_tensors(path_result))},
        {"name": "path_non_diffraction", "sha256": _hash_tensors(_result_tensors(path_result))},
        {"name": "deterministic_non_diffraction", "sha256": _hash_tensors(_result_tensors(deterministic))},
        {"name": "coherent_total_field", "sha256": _hash_tensors(_result_tensors(deterministic))},
        {"name": "winner", "sha256": _hash_tensors([("valid", path_result.valid)])},
        {"name": "topology", "sha256": _hash_tensors([
            ("interaction_type", path_result.interaction_type),
            ("primitive_id", path_result.primitive_id),
        ])},
        {"name": "metadata", "sha256": _hash_json({
            "point": point.metadata.get("components"),
            "path": path_result.metadata.get("components"),
            "deterministic": deterministic.metadata.get("components"),
        })},
    ])
    info = build_info()
    record = {
        "schema": SCHEMA, "group": args.group, "variant": args.variant,
        "process_index": args.process_index, "pair_order": args.pair_order,
        "measurement_policy": {
            "warmup": 1, "steady_repeats": 7,
            "timed_region_excludes_result_serialization": True,
            "cuda_event_timing": True, "wall_timing": True,
        },
        "build_crosscheck": {
            "channel_commit": info["channel_git_sha"],
            "rayd_commit": info["rayd_commit"],
            "integration_header_sha256": info["rayd_integration_abi_sha256"],
            "build_fingerprint": info["build_fingerprint"],
        },
        "timings": timings,
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "hashes": {
            "format": HASH_FORMAT, "schema_version": HASH_SCHEMA_VERSION,
            "target_shape": list(target_tensor.shape),
            "target_dtype": str(target_tensor.dtype).removeprefix("torch."),
            "target": target_hashes[-1],
            "full_result": _hash_tensors(_result_tensors(final_result)),
            "repeat_target": target_hashes, "unaffected": unaffected,
        },
    }
    print(json.dumps(record, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=tuple(GROUP_SCENARIOS), required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--process-index", type=int, choices=range(5), required=True)
    parser.add_argument("--pair-order", choices=("AB", "BA"), required=True)
    parser.add_argument("--warmup", type=int, choices=(1,), required=True)
    parser.add_argument("--steady-repeats", type=int, choices=(7,), required=True)
    parser.add_argument("--munich-scene-xml", type=Path, required=True)
    parser.add_argument("--sionna-source-root", type=Path, required=True)
    parser.add_argument("--profile-only", choices=tuple(GROUP_SCENARIOS.values()))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("canonical Phase 12 worker requires CUDA")
    if args.profile_only is not None:
        _profile(args)
    else:
        _run(args)


if __name__ == "__main__":
    main()