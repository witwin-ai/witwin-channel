"""Independent non-timed correctness diagnostics and deterministic replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from .artifacts import ArtifactStore, RAW_ARG_PREFIX, safe_relative_path
from .contracts import (
    COMPARISON_GROUPS,
    EvidenceError,
    ROOT,
    RunnerConfig,
    controlled_environment,
    exact_keys,
    process_schedule,
    read_json,
    strict_object,
)
from .workers import diagnostic_argv, run_captured


DIAGNOSTIC_CONTRACT_REPO_PATH = Path(
    "benchmarks/phase13_phase12_diagnostic_contract.json"
)
DIAGNOSTIC_SCHEMA = {
    "name": "witwin.channel.phase13-phase12-diagnostic-worker",
    "version": 1,
}
HASH_FORMAT = "semantic-dtype-shape-little-endian-contiguous"
HASH_SCHEMA_VERSION = 1


def load_diagnostic_contract() -> dict[str, object]:
    contract = read_json(ROOT / DIAGNOSTIC_CONTRACT_REPO_PATH)
    exact_keys(contract, {"schema", "execution", "groups"}, label="diagnostic contract")
    if contract["schema"] != {
        "name": "witwin.channel.phase13-phase12-diagnostic-contract",
        "version": 1,
    }:
        raise EvidenceError("diagnostic contract identity is not accepted")
    execution = exact_keys(
        contract["execution"],
        {
            "after_timed_and_nsys", "separate_process_per_variant_pair_member",
            "timed_gate_includes_diagnostic_memory", "record_device_host_and_artifact_bytes",
        },
        label="diagnostic contract execution",
    )
    if execution != {
        "after_timed_and_nsys": True,
        "separate_process_per_variant_pair_member": True,
        "timed_gate_includes_diagnostic_memory": False,
        "record_device_host_and_artifact_bytes": True,
    }:
        raise EvidenceError("diagnostic execution policy is not canonical")
    groups = contract["groups"]
    if not isinstance(groups, dict) or set(groups) != set(COMPARISON_GROUPS):
        raise EvidenceError("diagnostic contract group set is not canonical")
    expected_callables = {
        "enumerated_penetration": (
            "witwin.channel.propagation.enumerated.transmission._transmission_topology"
        ),
        "montecarlo_penetration": (
            "witwin.channel.montecarlo.events.transmission.straight_transmission_chains"
        ),
        "diffraction": (
            "witwin.channel.propagation.enumerated.diffraction._diffraction_topology_order1"
        ),
    }
    for group, callable_name in expected_callables.items():
        row = groups[group]
        if not isinstance(row, dict) or row.get("callable") != callable_name:
            raise EvidenceError(f"{group} diagnostic callable is not canonical")
        variants = row.get("variants")
        if not isinstance(variants, dict) or set(variants) != {"baseline", "candidate"}:
            raise EvidenceError(f"{group} diagnostic variant contract is malformed")
        for variant, available in (
            ("baseline", False),
            ("candidate", group == "diffraction"),
        ):
            variant_row = variants[variant]
            if (
                not isinstance(variant_row, dict)
                or set(variant_row) != {"mode", "source_lane_available"}
                or variant_row["source_lane_available"] is not available
            ):
                raise EvidenceError(f"{group} {variant} diagnostic availability differs")
    return contract


def _parse_stdout(stdout: bytes) -> dict[str, object]:
    records: list[dict[str, object]] = []
    try:
        lines = stdout.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise EvidenceError("diagnostic stdout is not UTF-8") from exc
    for line in lines:
        try:
            value = json.loads(
                line,
                object_pairs_hook=strict_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    EvidenceError(f"non-finite diagnostic JSON token: {token}")
                ),
            )
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == DIAGNOSTIC_SCHEMA:
            records.append(value)
    if len(records) != 1:
        raise EvidenceError(
            f"diagnostic worker must emit exactly one record; observed {len(records)}"
        )
    return records[0]


def _validate_subprocess_capture(
    capture: object, *, group: str, variant: str, process_index: int,
    artifact_path: str,
) -> dict[str, object]:
    row = exact_keys(
        capture,
        {
            "argv", "cwd", "returncode", "timed_out", "started_time_ns",
            "completed_time_ns", "stdout_artifact", "stderr_artifact",
        },
        label="diagnostic subprocess capture",
    )
    argv = row["argv"]
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise EvidenceError("diagnostic argv is malformed")
    expected_options = {
        "--group": group,
        "--variant": variant,
        "--process-index": str(process_index),
        "--output": RAW_ARG_PREFIX + artifact_path,
    }
    if (
        len(argv) != 19
        or argv[1] != "-I"
        or Path(argv[2]).name != "phase13_phase12_bootstrap.py"
        or argv[3] != "--site-packages"
        or argv[5] != "--script"
        or Path(argv[6]).name != "phase13_phase12_diagnostics.py"
        or argv[7:15:2] != list(expected_options)
        or argv[8:16:2] != list(expected_options.values())
        or argv[15] != "--munich-scene-xml"
        or argv[17] != "--sionna-source-root"
        or row["returncode"] != 0
        or row["timed_out"] is not False
        or int(row["completed_time_ns"]) < int(row["started_time_ns"])
    ):
        raise EvidenceError("diagnostic subprocess capture differs from the canonical invocation")
    return row


def _expected_arrays(contract: Mapping[str, object], group: str, variant: str) -> list[str]:
    groups = contract["groups"]
    assert isinstance(groups, dict)
    row = groups[group]
    assert isinstance(row, dict)
    key = f"{variant}_arrays" if group == "diffraction" else "arrays"
    arrays = row.get(key)
    if not isinstance(arrays, list) or not all(isinstance(name, str) for name in arrays):
        raise EvidenceError(f"{group} {variant} diagnostic array contract is malformed")
    if arrays != sorted(arrays) or len(arrays) != len(set(arrays)):
        raise EvidenceError("diagnostic array names must be unique and sorted")
    return arrays


def _validate_record(
    record: object, contract: Mapping[str, object], *, group: str, variant: str,
    process_index: int,
) -> dict[str, object]:
    row = exact_keys(
        record,
        {
            "schema", "group", "variant", "process_index", "array_names",
            "array_shapes", "array_dtypes", "hash_format", "hash_schema_version",
            "semantic_sha256", "peak_device_bytes", "host_array_bytes",
            "peak_host_bytes", "metadata",
        },
        label="diagnostic worker record",
    )
    if (
        row["schema"] != DIAGNOSTIC_SCHEMA
        or row["group"] != group
        or row["variant"] != variant
        or row["process_index"] != process_index
        or row["hash_format"] != HASH_FORMAT
        or row["hash_schema_version"] != HASH_SCHEMA_VERSION
    ):
        raise EvidenceError("diagnostic worker identity/policy differs")
    names = _expected_arrays(contract, group, variant)
    if row["array_names"] != names:
        raise EvidenceError("diagnostic worker array set differs from the contract")
    for field in ("array_shapes", "array_dtypes"):
        value = row[field]
        if not isinstance(value, dict) or set(value) != set(names):
            raise EvidenceError(f"diagnostic {field} key set differs")
    if (
        not isinstance(row["semantic_sha256"], str)
        or len(row["semantic_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in row["semantic_sha256"])
    ):
        raise EvidenceError("diagnostic semantic hash is malformed")
    for field in ("peak_device_bytes", "host_array_bytes", "peak_host_bytes"):
        if isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < 0:
            raise EvidenceError(f"diagnostic {field} must be a non-negative integer")
    metadata = row["metadata"]
    if not isinstance(metadata, dict):
        raise EvidenceError("diagnostic metadata must be an object")
    groups = contract["groups"]
    assert isinstance(groups, dict)
    group_contract = groups[group]
    assert isinstance(group_contract, dict)
    variants = group_contract["variants"]
    assert isinstance(variants, dict)
    variant_contract = variants[variant]
    assert isinstance(variant_contract, dict)
    expected_metadata = {"mode": variant_contract["mode"]}
    if group == "diffraction" and variant == "candidate":
        keys = group_contract["candidate_metadata"]
        assert isinstance(keys, list)
        for key in keys:
            if key == "row_order":
                expected_metadata[str(key)] = group_contract["row_order"]
            elif key == "failure_state_storage_alias_verified":
                expected_metadata[str(key)] = True
            elif key == "state_capacity":
                expected_metadata[str(key)] = group_contract["diffraction_state_capacity"]
            else:
                expected_metadata[str(key)] = metadata.get(str(key))
    if set(metadata) != set(expected_metadata) or any(
        metadata.get(key) != value for key, value in expected_metadata.items()
    ):
        raise EvidenceError("diagnostic metadata differs from the central contract")
    if group == "diffraction" and variant == "candidate":
        for key in ("pair_count", "state_capacity"):
            if isinstance(metadata[key], bool) or not isinstance(metadata[key], int) or metadata[key] <= 0:
                raise EvidenceError(f"diffraction diagnostic {key} must be positive")
    return row


def _semantic_hash(arrays: Mapping[str, np.ndarray]) -> str:
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


def _load_arrays(
    store: ArtifactStore, row: Mapping[str, object]
) -> dict[str, np.ndarray]:
    reference = row["arrays_artifact"]
    store.verify_reference(reference, label="diagnostic NPZ before replay")
    assert isinstance(reference, dict)
    relative = safe_relative_path(reference["path"])
    path = store.root.joinpath(*relative.parts)
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = sorted(archive.files)
            arrays = {name: np.ascontiguousarray(archive[name]) for name in names}
    except (OSError, ValueError) as exc:
        raise EvidenceError("cannot parse retained diagnostic NPZ") from exc
    store.verify_reference(reference, label="diagnostic NPZ after replay")
    record = row["record"]
    assert isinstance(record, dict)
    if names != record["array_names"]:
        raise EvidenceError("diagnostic NPZ array set differs from stdout")
    shapes = {name: list(array.shape) for name, array in arrays.items()}
    dtypes = {name: str(array.dtype) for name, array in arrays.items()}
    if shapes != record["array_shapes"] or dtypes != record["array_dtypes"]:
        raise EvidenceError("diagnostic NPZ shape/dtype differs from stdout")
    if sum(int(array.nbytes) for array in arrays.values()) != record["host_array_bytes"]:
        raise EvidenceError("diagnostic host-array byte count does not replay")
    if _semantic_hash(arrays) != record["semantic_sha256"]:
        raise EvidenceError("diagnostic semantic hash differs from retained arrays")
    return arrays


def _array_hash(name: str, array: np.ndarray) -> str:
    return _semantic_hash({name: array})


def _error_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise EvidenceError("diagnostic comparison arrays are incompatible or non-finite")
    actual64 = actual.astype(np.float64, copy=False)
    reference64 = reference.astype(np.float64, copy=False)
    difference = np.abs(actual64 - reference64)
    denominator = np.maximum(
        np.maximum(np.abs(reference64), np.abs(actual64)),
        np.finfo(np.float64).tiny,
    )
    actual32 = actual.astype(np.float32, copy=False)
    reference32 = reference.astype(np.float32, copy=False)
    actual_bits = actual32.view(np.uint32).copy()
    reference_bits = reference32.view(np.uint32).copy()
    actual_bits[actual32 == 0.0] = 0
    reference_bits[reference32 == 0.0] = 0
    actual_ordered = np.where(
        actual_bits & np.uint32(0x80000000), ~actual_bits,
        actual_bits | np.uint32(0x80000000),
    ).astype(np.int64)
    reference_ordered = np.where(
        reference_bits & np.uint32(0x80000000), ~reference_bits,
        reference_bits | np.uint32(0x80000000),
    ).astype(np.int64)
    return {
        "max_abs_error": float(difference.max(initial=0.0)),
        "max_rel_error": float((difference / denominator).max(initial=0.0)),
        "max_ulp_error": int(np.abs(actual_ordered - reference_ordered).max(initial=0)),
    }


def _diffraction_oracle(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> dict[str, object]:
    pair_count = int(metadata["pair_count"])
    state_capacity = int(metadata["state_capacity"])
    valid = arrays["valid"].reshape(pair_count, state_capacity)
    if valid.dtype != np.bool_:
        raise EvidenceError("source-lane valid must be bool")
    if arrays["failure"].dtype != np.int32 or arrays["failure"].shape != (1,):
        raise EvidenceError("source-lane failure state must be contiguous int32[1]")
    if int(arrays["failure"][0]) != 0:
        raise EvidenceError("source-lane diagnostic observed capacity failure")
    num_paths = arrays["num_paths"]
    if num_paths.dtype != np.int32 or num_paths.shape != (1,):
        raise EvidenceError("source-lane num_paths must be contiguous int32[1]")
    if int(num_paths[0]) != int(np.count_nonzero(valid)):
        raise EvidenceError("source-lane num_paths differs from valid cardinality")
    component_names = ("x_re", "x_im", "y_re", "y_im", "z_re", "z_im")
    oracle = np.empty((pair_count, 6), dtype=np.float64)
    for component_index, name in enumerate(component_names):
        component = arrays[name]
        if component.dtype != np.float32 or component.size != pair_count * state_capacity:
            raise EvidenceError("source-lane component storage differs from pair/state layout")
        values = np.where(valid, component.reshape(pair_count, state_capacity), 0.0).astype(
            np.float64, copy=False
        )
        oracle[:, component_index] = np.add.accumulate(values, axis=1)[:, -1]
    target = arrays["target"]
    if target.dtype != np.float32 or target.shape != (pair_count, 6):
        raise EvidenceError("diffraction reducer target must be float32[pair_count,6]")
    power = np.zeros((pair_count,), dtype=np.float64)
    for component_index in range(6):
        component = oracle[:, component_index]
        power = np.add(power, np.multiply(component, component))
    return {
        "target_vs_float64": _error_metrics(target, oracle),
        "oracle_target_sha256": _array_hash("oracle_target_float64", oracle),
        "oracle_component_power_sha256": _array_hash("oracle_component_power_float64", power),
        "valid_count": int(np.count_nonzero(valid)),
    }


def _analyze(
    rows: Sequence[Mapping[str, object]], gate: Mapping[str, object], *, group: str,
    contract: Mapping[str, object], store: ArtifactStore,
) -> dict[str, object]:
    schedule = process_schedule(gate, group)
    expected = [
        (int(pair["process_index"]), str(pair["order"]), str(variant))
        for pair in schedule for variant in pair["variants"]
    ]
    observed = [
        (int(row["process_index"]), str(row["order"]), str(row["variant"]))
        for row in rows
    ]
    if observed != expected:
        raise EvidenceError("diagnostic process order differs from the canonical A/B schedule")
    semantic = {
        variant: sorted(
            {str(row["record"]["semantic_sha256"]) for row in rows if row["variant"] == variant}  # type: ignore[index]
        )
        for variant in ("baseline", "candidate")
    }
    candidate_stable = len(semantic["candidate"]) == 1
    if not candidate_stable:
        raise EvidenceError("candidate diagnostic arrays are not bitwise stable across processes")
    policy = gate["comparison_groups"][group]  # type: ignore[index]
    assert isinstance(policy, dict)
    correctness = policy["correctness"]
    budget = policy["resource_budgets"]
    if not isinstance(correctness, dict) or not isinstance(budget, dict):
        raise EvidenceError("diagnostic correctness/resource gates are not frozen")
    if semantic["candidate"][0] != correctness["diagnostic_candidate_semantic_sha256"]:
        raise EvidenceError("candidate diagnostic hash differs from the frozen gate")
    if semantic["baseline"] != correctness["diagnostic_baseline_semantic_sha256_values"]:
        raise EvidenceError("baseline diagnostic hashes differ from frozen raw facts")
    exact_ab = group != "diffraction"
    ab_metrics: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    candidate_target_hashes: set[str] = set()
    for pair in schedule:
        index = int(pair["process_index"])
        baseline_row = next(
            row for row in rows
            if int(row["process_index"]) == index and row["variant"] == "baseline"
        )
        candidate_row = next(
            row for row in rows
            if int(row["process_index"]) == index and row["variant"] == "candidate"
        )
        baseline = _load_arrays(store, baseline_row)
        candidate = _load_arrays(store, candidate_row)
        if "target" in candidate:
            candidate_target_hashes.add(_array_hash("target", candidate["target"]))
        if exact_ab:
            if set(baseline) != set(candidate) or any(
                not np.array_equal(baseline[name], candidate[name]) for name in baseline
            ):
                raise EvidenceError(f"{group} direct-facade A/B arrays differ")
        else:
            ab_metrics.append(
                {"process_index": index, **_error_metrics(candidate["target"], baseline["target"])}
            )
            metadata = candidate_row["record"]["metadata"]  # type: ignore[index]
            assert isinstance(metadata, dict)
            oracle_rows.append({"process_index": index, **_diffraction_oracle(candidate, metadata)})
    if group == "diffraction":
        contract_groups = contract["groups"]
        assert isinstance(contract_groups, dict)
        diffraction_contract = contract_groups["diffraction"]
        assert isinstance(diffraction_contract, dict)
        if diffraction_contract["diffraction_state_capacity"] != gate["munich"]["diffraction_state_capacity"]:  # type: ignore[index]
            raise EvidenceError("diagnostic and gate diffraction capacities differ")
        limits = correctness["diagnostic_float64_oracle_limits"]
        if not isinstance(limits, dict) or set(limits) != {
            "max_abs_error", "max_rel_error", "max_ulp_error"
        }:
            raise EvidenceError("diffraction float64 oracle limits are not exact")
        for row in oracle_rows:
            metrics = row["target_vs_float64"]
            assert isinstance(metrics, dict)
            if any(float(metrics[name]) > float(limits[name]) for name in limits):
                raise EvidenceError("diffraction reducer differs from the float64 oracle budget")
        if len({str(row["oracle_target_sha256"]) for row in oracle_rows}) != 1:
            raise EvidenceError("diffraction float64 oracle is not stable across processes")
    if candidate_target_hashes and candidate_target_hashes != {
        str(correctness["candidate_target_sha256"])
    }:
        raise EvidenceError("diagnostic candidate target differs from the timed frozen target")
    peak_device = max(int(row["record"]["peak_device_bytes"]) for row in rows)  # type: ignore[index]
    peak_host = max(int(row["record"]["peak_host_bytes"]) for row in rows)  # type: ignore[index]
    artifact_bytes = max(int(row["arrays_artifact"]["bytes"]) for row in rows)  # type: ignore[index]
    resource_checks = {
        "diagnostic_peak_device_bytes_max": peak_device
        <= int(budget["diagnostic_peak_device_bytes_max"]),
        "diagnostic_peak_host_bytes_max": peak_host
        <= int(budget["diagnostic_peak_host_bytes_max"]),
        "diagnostic_artifact_bytes_max": artifact_bytes
        <= int(budget["diagnostic_artifact_bytes_max"]),
    }
    if not all(resource_checks.values()):
        raise EvidenceError("diagnostic resource gate failed")
    return {
        "candidate_semantic_sha256": semantic["candidate"][0],
        "timed_target_crosscheck": {
            "applicable": bool(candidate_target_hashes),
            "sha256": next(iter(candidate_target_hashes)) if candidate_target_hashes else "",
            "passed": True,
        },
        "baseline_semantic_sha256_values": semantic["baseline"],
        "candidate_bitwise_stable": candidate_stable,
        "exact_ab_required": exact_ab,
        "exact_ab_passed": True,
        "ab_target_error_metrics": ab_metrics,
        "float64_oracle": oracle_rows,
        "resources": {
            "peak_device_bytes_max": peak_device,
            "peak_host_bytes_max": peak_host,
            "artifact_bytes_max": artifact_bytes,
            "checks": resource_checks,
            "passed": True,
        },
        "passed": True,
    }


def collect_diagnostics(
    config: RunnerConfig, gate: Mapping[str, object], *, group: str,
    timeout_seconds: int, store: ArtifactStore,
) -> dict[str, object]:
    contract = load_diagnostic_contract()
    rows: list[dict[str, object]] = []
    for pair in process_schedule(gate, group):
        index = int(pair["process_index"])
        for raw_variant in pair["variants"]:
            variant_name = str(raw_variant)
            variant = config.variant(group, variant_name)
            relative = f"diagnostics/{group}-{index:02d}-{variant_name}.npz"
            output = store.root.joinpath(*safe_relative_path(relative).parts)
            if output.exists():
                raise EvidenceError("diagnostic output path already exists")
            output.parent.mkdir(parents=True, exist_ok=True)
            captured = run_captured(
                diagnostic_argv(
                    variant, group=group, name=variant_name, process_index=index,
                    output=output,
                    munich_scene_xml=config.datasets.munich_scene_xml,
                    sionna_source_root=config.datasets.sionna_source_root,
                ),
                cwd=variant.checkout,
                environment=controlled_environment(config),
                timeout_seconds=timeout_seconds,
                store=store,
                stem=f"diagnostic-{group}-{index:02d}-{variant_name}",
            )
            record = _validate_record(
                _parse_stdout(captured.pop("stdout_bytes")), contract,
                group=group, variant=variant_name, process_index=index,
            )
            captured.pop("stderr_bytes")
            artifact = store.inspect(
                relative, label="diagnostic NPZ",
                minimum_mtime_ns=int(captured["started_time_ns"]),
            )
            _validate_subprocess_capture(
                captured,
                group=group,
                variant=variant_name,
                process_index=index,
                artifact_path=relative,
            )
            rows.append(
                {
                    "group": group,
                    "process_index": index,
                    "order": pair["order"],
                    "variant": variant_name,
                    "record": record,
                    "arrays_artifact": artifact,
                    "capture": captured,
                }
            )
    analysis = _analyze(rows, gate, group=group, contract=contract, store=store)
    return {"captures": rows, "analysis": analysis, "passed": True}


def replay_diagnostics(
    section: object, gate: Mapping[str, object], *, group: str, store: ArtifactStore,
) -> dict[str, object]:
    row = exact_keys(section, {"captures", "analysis", "passed"}, label="diagnostics section")
    captures = row["captures"]
    if not isinstance(captures, list):
        raise EvidenceError("diagnostic captures must be an array")
    contract = load_diagnostic_contract()
    for capture in captures:
        item = exact_keys(
            capture,
            {"group", "process_index", "order", "variant", "record", "arrays_artifact", "capture"},
            label="diagnostic capture",
        )
        group_name = str(item["group"])
        variant = str(item["variant"])
        process_index = int(item["process_index"])
        if group_name != group:
            raise EvidenceError("diagnostic capture group differs")
        raw_capture = item["capture"]
        arrays_artifact = item["arrays_artifact"]
        if not isinstance(arrays_artifact, dict) or not isinstance(arrays_artifact.get("path"), str):
            raise EvidenceError("diagnostic arrays artifact is malformed")
        raw_capture = _validate_subprocess_capture(
            raw_capture,
            group=group,
            variant=variant,
            process_index=process_index,
            artifact_path=arrays_artifact["path"],
        )
        parsed = _validate_record(
            _parse_stdout(
                store.read_verified(
                    raw_capture["stdout_artifact"], label="diagnostic stdout"
                )
            ),
            contract, group=group, variant=variant, process_index=process_index,
        )
        if parsed != item["record"]:
            raise EvidenceError("diagnostic record differs from retained stdout")
    analysis = _analyze(captures, gate, group=group, contract=contract, store=store)
    if analysis != row["analysis"] or row["passed"] is not True:
        raise EvidenceError("diagnostic analysis differs from raw replay")
    return dict(row)


__all__ = [
    "DIAGNOSTIC_CONTRACT_REPO_PATH", "collect_diagnostics",
    "load_diagnostic_contract", "replay_diagnostics",
]
