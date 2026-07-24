"""Non-timed direct-facade correctness capture for Phase 12 evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

from phase13_phase12_worker import _munich_scene, _wall_scene


SCHEMA = {
    "name": "witwin.channel.phase13-phase12-diagnostic-worker",
    "version": 1,
}

HASH_FORMAT = "semantic-dtype-shape-little-endian-contiguous"
HASH_SCHEMA_VERSION = 1


def _diffraction_state_capacity() -> int:
    path = Path(__file__).with_name("phase13_phase12_diagnostic_contract.json")
    contract = json.loads(path.read_text(encoding="utf-8"))
    capacity = contract["groups"]["diffraction"]["diffraction_state_capacity"]
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise RuntimeError("diagnostic diffraction_state_capacity is not positive")
    return capacity


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


def _enumerated() -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    from witwin.channel.propagation.enumerated.transmission import (
        _transmission_topology,
    )
    from witwin.channel.scene.tensors import (
        receiver_positions,
        transmitter_positions,
    )

    scene = _wall_scene()
    compiled = scene.compile()
    device = torch.device("cuda")
    tx, _ = transmitter_positions(scene, device=device)
    rx = receiver_positions(scene, device=device, reference=tx)
    block, _launch_count, candidate_count, guardrail_count = _transmission_topology(
        scene, compiled, tx, rx, max_depth=2
    )
    names = (
        "valid", "tx_id", "rx_id", "depth", "component_id", "primitive_id",
        "edge_id", "path_length_m", "interaction_position", "interaction_normal",
        "material_id", "primitive_sequence", "material_sequence",
        "interaction_positions", "interaction_normals",
    )
    # Diagnostics run outside the timed production path and may compact the
    # public fixed-capacity block for a semantic A/B comparison with the old
    # K-row owner. Production never performs this device-selected compaction.
    valid = block["valid"]
    arrays = {name: block[name][valid] for name in names}
    reference = valid
    candidate_count_value = (
        candidate_count
        if isinstance(candidate_count, torch.Tensor)
        else torch.full(
            (1,), candidate_count, device=reference.device, dtype=torch.int32
        )
    )
    guardrail_count_value = (
        guardrail_count
        if isinstance(guardrail_count, torch.Tensor)
        else torch.full(
            (1,), guardrail_count, device=reference.device, dtype=torch.int32
        )
    )
    arrays.update(
        {
            "candidate_count": candidate_count_value,
            "guardrail_count": guardrail_count_value,
        }
    )
    return arrays, {"mode": "high_level_owner"}


def _montecarlo() -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    from witwin.channel.materials.encoding import face_material_field_bundle
    from witwin.channel.montecarlo.events.transmission import (
        layer_csr_view,
        scene_diagonal_m,
        straight_transmission_chains,
    )
    from witwin.channel.scene.tensors import (
        receiver_grid_points,
        transmitter_polarizations,
        transmitter_positions,
    )

    scene = _wall_scene()
    compiled = scene.compile()
    device = torch.device("cuda")
    tx, _ = transmitter_positions(scene, device=device)
    rx = receiver_grid_points(scene.receivers[0], reference=tx)
    bundle = face_material_field_bundle(scene, device=device)
    origins = tx[0].reshape(1, 3).repeat(int(rx.shape[0]), 1)
    polarization = transmitter_polarizations(scene, device=device)[0]
    arrays = straight_transmission_chains(
        compiled.rayd,
        origins,
        rx,
        face_material_id=bundle["material_id"],
        layer_csr=layer_csr_view(bundle),
        polarization=polarization,
        frequency_hz=float(scene.frequency),
        max_depth=2,
        scene_diagonal=scene_diagonal_m(scene),
    )
    return arrays, {"mode": "high_level_owner"}


def _diffraction(
    variant: str, munich_scene_xml: Path, sionna_source_root: Path
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    from witwin.channel.propagation.enumerated.diffraction import (
        _deterministic_diffraction_states,
        _diffraction_topology_order1,
    )
    from witwin.channel.propagation.geometry.diffraction import (
        DiffractionOrder1Query,
        plan_tx_visible_diffraction_states,
        query_diffraction_order1,
    )
    from witwin.channel.scene.tensors import (
        LIGHT_SPEED_M_PER_S,
        receiver_positions,
        transmitter_positions,
    )
    from witwin.channel.propagation.topology.discovery.diffraction import (
        prepare_diffraction_order1_plan,
    )

    scene = _munich_scene(munich_scene_xml, sionna_source_root)
    compiled = scene.compile()
    device = torch.device("cuda")
    tx, tx_power = transmitter_positions(scene, device=device)
    rx = receiver_positions(scene, device=device, reference=tx)
    if int(tx.shape[0]) != 1:
        raise RuntimeError("canonical Munich diagnostic requires exactly one transmitter")
    _, _, field = _diffraction_topology_order1(
        scene, compiled, tx, tx_power, rx, frequency_hz=float(scene.frequency)
    )
    target = torch.view_as_real(field).reshape(-1, 6).contiguous()
    if variant == "baseline":
        return {"target": target}, {"mode": "old_compact_atomic_target"}

    from witwin.channel.field_state import transmitter_polarizations as field_polarizations
    from witwin.channel.materials.encoding import face_material_tensors
    from witwin.channel.propagation.geometry.diffraction import DiffractionPathLayout
    from witwin.channel.runtime.capacity import create_capacity_failure_state

    face_eps_r, face_sigma_e, face_mu_r, material_gain, material_valid = (
        face_material_tensors(compiled, device=device)
    )
    plan = prepare_diffraction_order1_plan(
        metadata=scene.metadata,
        tx_count=int(tx.shape[0]),
        rx_count=int(rx.shape[0]),
    )
    states = _deterministic_diffraction_states(
        compiled.rayd, tx[0], tx_power, 0,
        preserve_imported_edges=plan.preserve_imported_edges,
    )
    visible = plan_tx_visible_diffraction_states(compiled.rayd, states, tx[0])
    failure_state = create_capacity_failure_state(tx)
    state_capacity = _diffraction_state_capacity()
    out = query_diffraction_order1(
        DiffractionOrder1Query(
            handle=compiled.rayd.require_resource(),
            tx_position=tx[0].reshape(1, 3).contiguous(),
            tx_polarization=field_polarizations(scene, device=device)[0]
            .reshape(1, 3).contiguous(),
            rx_positions=rx.contiguous(),
            active=visible.active,
            states=visible,
            material_eta_r=face_eps_r,
            material_sigma=face_sigma_e,
            material_mu_r=face_mu_r,
            material_gain=material_gain,
            material_valid=material_valid,
            state_count=int(visible.edge_index.shape[0]),
            capacity=state_capacity,
            wavelength=LIGHT_SPEED_M_PER_S / float(scene.frequency),
            layout=DiffractionPathLayout.SOURCE_LANE,
            failure_state=failure_state,
        )
    )
    if out.failure_state is not failure_state:
        raise RuntimeError("source-lane query replaced the solve-owned failure state")
    if (
        out.failure_state.bits is not failure_state.bits
        or out.failure_state.bits.data_ptr() != failure_state.bits.data_ptr()
    ):
        raise RuntimeError("source-lane query replaced failure-state storage")
    if (
        out.num_paths.device.type != "cuda"
        or out.num_paths.dtype != torch.int32
        or out.num_paths.shape != (1,)
        or not out.num_paths.is_contiguous()
    ):
        raise RuntimeError("source-lane num_paths is not contiguous CUDA int32[1]")
    lane_count = int(rx.shape[0]) * state_capacity
    if (
        out.valid.device.type != "cuda"
        or out.valid.dtype != torch.bool
        or out.valid.numel() != lane_count
        or not out.valid.is_contiguous()
    ):
        raise RuntimeError("source-lane valid storage differs from pair-major capacity")
    for name in ("x_re", "x_im", "y_re", "y_im", "z_re", "z_im"):
        value = getattr(out, name)
        if (
            value.device.type != "cuda"
            or value.dtype != torch.float32
            or value.numel() != lane_count
            or not value.is_contiguous()
        ):
            raise RuntimeError(f"source-lane {name} storage differs from capacity layout")
    arrays = {
        "target": target,
        "valid": out.valid,
        "x_re": out.x_re,
        "x_im": out.x_im,
        "y_re": out.y_re,
        "y_im": out.y_im,
        "z_re": out.z_re,
        "z_im": out.z_im,
        "num_paths": out.num_paths,
        "failure": out.failure_state.bits,
    }
    return arrays, {
        "mode": "source_lane_pair_reduce",
        "pair_count": int(tx.shape[0]) * int(rx.shape[0]),
        "state_capacity": state_capacity,
        "row_order": "pair-major-state-fast",
        "failure_state_storage_alias_verified": True,
    }


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
