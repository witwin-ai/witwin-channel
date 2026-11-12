# Copyright Xingyu Chen.
# Benchmarks diagnostics.

"""Benchmarks diagnostics."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

from phase13_phase12_worker import (
    FIXTURE_FREQUENCY_HZ,
    MUNICH_FREQUENCY_HZ,
    _munich_scene,
    _wall_scene,
    pair_field_sum,
)


SCHEMA = {
    "name": "witwin.channel.phase13-phase12-diagnostic-worker",
    "version": 1,
}

HASH_FORMAT = "semantic-dtype-shape-little-endian-contiguous"
HASH_SCHEMA_VERSION = 1


def _to_host(value: torch.Tensor) -> np.ndarray:
    array = value.detach().contiguous().cpu().numpy()
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and np.little_endian is False
    ):
        array = array.byteswap().newbyteorder("<")
    return array


def _semantic_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = arrays[name]
        header = json.dumps(
            {"name": name, "dtype": str(array.dtype), "shape": list(array.shape)},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
        digest.update(struct.pack("<I", len(header)))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _peak_host_bytes() -> int:
    """Return process peak resident bytes without adding a sampled monitor."""
    if hasattr(ctypes, "windll"):
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            raise RuntimeError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if peak > (1 << 30) else peak * 1024


def _bind(scene: object, frequency_hz: float) -> tuple[object, object]:
    from witwin.channel.scene import compile as compile_scene
    from witwin.channel.scene.endpoints import bind_solver_scene

    compiled = compile_scene(scene, reference_frequency_hz=frequency_hz)
    return bind_solver_scene(compiled), compiled


def _enumerated() -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    from witwin.channel.interactions.transmission import _transmission_topology
    from witwin.channel.runtime import create_capacity_failure_state
    from witwin.channel.scene.compiler import (
        receiver_positions,
        transmitter_positions,
    )

    scene, compiled = _bind(_wall_scene(), FIXTURE_FREQUENCY_HZ)
    device = torch.device("cuda")
    tx, _ = transmitter_positions(scene, device=device)
    rx = receiver_positions(scene, device=device, reference=tx)
    block, _launch_count, execution = _transmission_topology(
        scene, compiled, tx, rx, max_depth=2, ad_mode="none",
        failure_state=create_capacity_failure_state(tx),
    )
    if execution is None:
        raise RuntimeError("canonical wall diagnostic produced no transmission execution record")
    names = (
        "valid", "tx_id", "rx_id", "depth", "component_id", "primitive_id",
        "edge_id", "path_length_m", "interaction_position", "interaction_normal",
        "material_id", "primitive_sequence", "material_sequence",
        "interaction_positions", "interaction_normals",
    )
    # The owner returns the valid rows already compacted in stable order.
    arrays = {name: block[name] for name in names}
    arrays.update(
        {
            "candidate_count": execution.device_candidate_count,
            "guardrail_count": execution.device_guardrail_count,
        }
    )
    return arrays, {"mode": "high_level_owner"}


def _montecarlo() -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    from witwin.channel.interactions.transmission import straight_transmission_chains
    from witwin.channel.runtime import create_capacity_failure_state
    from witwin.channel.scene.compiler import (
        receiver_grid_points,
        transmitter_positions,
    )

    scene, compiled = _bind(_wall_scene(), FIXTURE_FREQUENCY_HZ)
    device = torch.device("cuda")
    tx, _ = transmitter_positions(scene, device=device)
    rx = receiver_grid_points(scene.receivers[0], reference=tx)
    origins = tx[0].reshape(1, 3).repeat(int(rx.shape[0]), 1)
    chains = straight_transmission_chains(
        compiled.rayd,
        origins,
        rx,
        vertices=None,
        max_depth=2,
        scene_diagonal=compiled.montecarlo_penetration_scene_diagonal_m,
        failure_state=create_capacity_failure_state(tx),
    )
    arrays = {
        name: getattr(chains, name)
        for name in (
            "distance", "global_primitive_id", "num_hits", "overflow", "reached_target", "t",
            "valid",
        )
    }
    return arrays, {"mode": "high_level_owner"}


def _diffraction(
    variant: str, munich_scene_xml: Path, sionna_source_root: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    from witwin.channel.interactions.diffraction import _diffraction_topology_order1
    from witwin.channel.scene.compiler import (
        receiver_positions,
        transmitter_positions,
    )

    if variant != "baseline":
        # The diffraction exporter now has one compact layout; the pair-major source-lane
        # layout with a device path count and a query-owned failure state is not available.
        raise RuntimeError(
            "the source-lane diffraction candidate cannot be captured from this tree: "
            "its exporter has only the compact layout"
        )
    scene, compiled = _bind(
        _munich_scene(munich_scene_xml, sionna_source_root), MUNICH_FREQUENCY_HZ
    )
    device = torch.device("cuda")
    tx, tx_power = transmitter_positions(scene, device=device)
    rx = receiver_positions(scene, device=device, reference=tx)
    if int(tx.shape[0]) != 1:
        raise RuntimeError("canonical Munich diagnostic requires exactly one transmitter")
    block, _ = _diffraction_topology_order1(
        scene, compiled, tx, tx_power, rx, frequency_hz=MUNICH_FREQUENCY_HZ
    )
    field = pair_field_sum(
        block["tx_id"], block["rx_id"], block["field_xyz"], tx_count=int(tx.shape[0]),
        rx_count=int(rx.shape[0]),
    )
    target = torch.view_as_real(field).reshape(-1, 6).contiguous()
    return {"target": target}, {"mode": "old_compact_atomic_target"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=("enumerated_penetration", "montecarlo_penetration", "diffraction"),
        required=True,
    )
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--process-index", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--munich-scene-xml", type=Path, required=True)
    parser.add_argument("--sionna-source-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("diagnostic output already exists")
    torch.cuda.reset_peak_memory_stats()
    if args.group == "enumerated_penetration":
        tensors, metadata = _enumerated()
    elif args.group == "montecarlo_penetration":
        tensors, metadata = _montecarlo()
    else:
        tensors, metadata = _diffraction(
            args.variant, args.munich_scene_xml, args.sionna_source_root
        )
    torch.cuda.synchronize()
    peak_device = int(torch.cuda.max_memory_allocated())
    arrays = {name: _to_host(value) for name, value in tensors.items()}
    host_bytes = sum(int(value.nbytes) for value in arrays.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez(stream, **arrays)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "group": args.group,
                "variant": args.variant,
                "process_index": args.process_index,
                "array_names": sorted(arrays),
                "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
                "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
                "hash_format": HASH_FORMAT,
                "hash_schema_version": HASH_SCHEMA_VERSION,
                "semantic_sha256": _semantic_hash(arrays),
                "peak_device_bytes": peak_device,
                "host_array_bytes": host_bytes,
                "peak_host_bytes": _peak_host_bytes(),
                "metadata": metadata,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()