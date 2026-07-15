"""Collect reduced, exact runtime and performance refactor baselines.

The parent process starts at least two independent child processes for every
solver/scenario pair.  Each child delegates timing and CUDA-memory accounting
to :mod:`benchmarks.harness`; result serialization happens after measurement
and therefore does not contaminate the timing samples.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_NAME = "witwin.channel_native.refactor-runtime-baseline"
SCHEMA_VERSION = 1
SOLVERS = ("path", "deterministic", "montecarlo-basic", "montecarlo-bdpt")
REDUCED_SCENARIOS = ("empty-los", "single-reflection")
MIN_PROCESSES = 2
MIN_WARMUP = 1
MIN_REPEATS = 7
_IDENTITY_NAMES = frozenset(
    {
        "component_id",
        "depth",
        "edge_id",
        "grid_linear_id",
        "interaction_type",
        "light_depth",
        "material_id",
        "material_sequence",
        "num_paths",
        "primitive_id",
        "primitive_sequence",
        "rx_id",
        "sensor_depth",
        "topology",
        "tx_id",
        "valid",
    }
)
_LAUNCH_FIELDS = (
    "primitive",
    "launch_count",
    "forward_launch_count",
    "backward_launch_count",
    "jvp_launch_count",
    "fused_stages",
    "tape_bytes",
    "intermediate_bytes",
    "accumulation_strategy",
    "scheduling_strategy",
    "ad_status",
    "peak_memory_bytes",
)
_VOLATILE_METADATA_FIELDS = frozenset(
    {
        "backward_time_ms",
        "forward_time_ms",
        "jvp_time_ms",
        "peak_memory_bytes",
        "solve_time_ms",
    }
)


class RuntimeBaselineError(RuntimeError):
    """Raised when a runtime baseline would be incomplete or ambiguous."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_sha(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        raise RuntimeBaselineError(
            f"git rev-parse failed: {(completed.stderr or completed.stdout).strip()}"
        )
    return completed.stdout.strip()


def validate_measurement_policy(processes: int, warmup: int, repeats: int) -> None:
    if processes < MIN_PROCESSES:
        raise RuntimeBaselineError(
            f"performance baselines require at least {MIN_PROCESSES} independent processes"
        )
    if warmup < MIN_WARMUP:
        raise RuntimeBaselineError(
            f"performance baselines require at least {MIN_WARMUP} warmup"
        )
    if repeats < MIN_REPEATS:
        raise RuntimeBaselineError(
            f"performance baselines require at least {MIN_REPEATS} steady repeats"
        )


def _float_value(value: float) -> object:
    if math.isfinite(value):
        return value
    return {"kind": "nonfinite-float", "hex": value.hex()}


class Snapshotter:
    """Serialize values exactly while assigning deterministic alias groups."""

    def __init__(self) -> None:
        self._storage_groups: dict[tuple[object, ...], str] = {}
        self.tensor_records: list[dict[str, object]] = []

    def _tensor(self, value: Any, path: str) -> dict[str, object]:
        import torch

        detached = value.detach().resolve_conj().resolve_neg()
        logical = detached.contiguous().cpu()
        # PyTorch may call a tensor contiguous when a size-1 dimension has an
        # arbitrary stride. Flattening and cloning forces a physical stride-1
        # staging buffer before the element-size-changing byte view.
        byte_staging = torch.empty(
            (logical.numel(),), dtype=logical.dtype, device="cpu"
        )
        byte_staging.copy_(logical.reshape(-1))
        raw = byte_staging.view(torch.uint8).numpy().tobytes(order="C")
        storage = value.untyped_storage()
        pointer = int(storage.data_ptr())
        storage_identity = (
            str(value.device),
            pointer if pointer else int(storage._cdata),
            int(storage.nbytes()),
        )
        alias_group = self._storage_groups.setdefault(
            storage_identity, f"storage-{len(self._storage_groups)}"
        )
        manifest = {
            "kind": "tensor",
            "sha256": _sha256(raw),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "layout": str(value.layout),
            "stride": list(value.stride()),
            "storage_offset": int(value.storage_offset()),
            "storage_nbytes": int(storage.nbytes()),
            "contiguous": bool(value.is_contiguous()),
            "requires_grad": bool(value.requires_grad),
            "alias_group": alias_group,
            "numel": int(value.numel()),
            "element_size": int(value.element_size()),
            "hash_semantics": "detached-logical-contiguous-raw-bytes",
        }
        self.tensor_records.append({"path": path, "tensor": manifest})
        return manifest

    def capture(self, value: object, path: str = "$") -> object:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            return self._tensor(value, path)
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return _float_value(value)
        if isinstance(value, complex):
            return {
                "kind": "complex",
                "real": _float_value(value.real),
                "imag": _float_value(value.imag),
            }
        if isinstance(value, enum.Enum):
            return {
                "kind": "enum",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "name": value.name,
                "value": self.capture(value.value, f"{path}.value"),
            }
        if isinstance(value, Path):
            return {"kind": "path", "name": value.name}
        if isinstance(value, bytes):
            return {"kind": "bytes", "size": len(value), "sha256": _sha256(value)}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            captured_fields = []
            for field in dataclasses.fields(value):
                if field.name.startswith("_"):
                    continue
                captured_fields.append(
                    {
                        "name": field.name,
                        "value": self.capture(
                            getattr(value, field.name), f"{path}.{field.name}"
                        ),
                    }
                )
            return {
                "kind": "dataclass",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": captured_fields,
            }
        if isinstance(value, Mapping):
            items = []
            for key, child in value.items():
                captured_key = self.capture(key, f"{path}.<key>")
                child_path = f"{path}[{key!r}]"
                items.append(
                    {
                        "key": captured_key,
                        "value": self.capture(child, child_path),
                    }
                )
            items.sort(key=lambda item: _canonical_bytes(item["key"]))
            return {"kind": "mapping", "items": items}
        if isinstance(value, (tuple, list)):
            return {
                "kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [
                    self.capture(child, f"{path}[{index}]")
                    for index, child in enumerate(value)
                ],
            }
        if isinstance(value, (set, frozenset)):
            items = [self.capture(child, f"{path}.<set-item>") for child in value]
            items.sort(key=_canonical_bytes)
            return {"kind": type(value).__name__, "items": items}
        raise RuntimeBaselineError(
            f"unsupported value at {path}: {type(value).__module__}.{type(value).__qualname__}"
        )


def value_manifest(value: object) -> dict[str, object]:
    snapshotter = Snapshotter()
    snapshot = snapshotter.capture(value)
    return {
        "snapshot": snapshot,
        "fingerprint": _sha256(_canonical_bytes(snapshot)),
        "tensors": snapshotter.tensor_records,
    }


def _strip_volatile_metadata_snapshot(
    snapshot: object, path: str = "$.metadata"
) -> tuple[object, list[dict[str, object]]]:
    volatile = []
    if isinstance(snapshot, dict):
        if snapshot.get("kind") == "mapping":
            stable_items = []
            for item in snapshot["items"]:
                key = item["key"]
                child_path = f"{path}[{key!r}]"
                if isinstance(key, str) and key in _VOLATILE_METADATA_FIELDS:
                    volatile.append({"path": child_path, "value": item["value"]})
                    continue
                stable_child, child_volatile = _strip_volatile_metadata_snapshot(
                    item["value"], child_path
                )
                stable_items.append({"key": key, "value": stable_child})
                volatile.extend(child_volatile)
            return {**snapshot, "items": stable_items}, volatile
        if snapshot.get("kind") in {"list", "tuple", "set", "frozenset"}:
            stable_items = []
            for index, item in enumerate(snapshot["items"]):
                stable_child, child_volatile = _strip_volatile_metadata_snapshot(
                    item, f"{path}[{index}]"
                )
                stable_items.append(stable_child)
                volatile.extend(child_volatile)
            return {**snapshot, "items": stable_items}, volatile
        if snapshot.get("kind") == "dataclass":
            stable_fields = []
            for field in snapshot["fields"]:
                stable_child, child_volatile = _strip_volatile_metadata_snapshot(
                    field["value"], f"{path}.{field['name']}"
                )
                stable_fields.append({"name": field["name"], "value": stable_child})
                volatile.extend(child_volatile)
            return {**snapshot, "fields": stable_fields}, volatile
    return snapshot, volatile


def _semantic_result_snapshot(
    snapshot: object,
) -> tuple[object, object, list[dict[str, object]]]:
    if not isinstance(snapshot, dict) or snapshot.get("kind") != "dataclass":
        raise RuntimeBaselineError("public solver Result must be a dataclass")
    stable_fields = []
    stable_metadata: object = None
    volatile = []
    found_metadata = False
    for field in snapshot["fields"]:
        if field["name"] != "metadata":
            stable_fields.append(field)
            continue
        found_metadata = True
        stable_metadata, volatile = _strip_volatile_metadata_snapshot(field["value"])
        stable_fields.append({"name": "metadata", "value": stable_metadata})
    if not found_metadata:
        raise RuntimeBaselineError("public solver Result has no metadata field")
    return {**snapshot, "fields": stable_fields}, stable_metadata, volatile


def result_manifest(result: object) -> dict[str, object]:
    captured = value_manifest(result)
    identity = [
        row
        for row in captured["tensors"]
        if str(row["path"]).rsplit(".", 1)[-1].split("[", 1)[0] in _IDENTITY_NAMES
    ]
    semantic_snapshot, semantic_metadata, volatile_metadata = _semantic_result_snapshot(
        captured["snapshot"]
    )
    full_metadata_manifest = value_manifest(getattr(result, "metadata", None))
    semantic_result_fingerprint = _sha256(_canonical_bytes(semantic_snapshot))
    semantic_metadata_fingerprint = _sha256(_canonical_bytes(semantic_metadata))
    return {
        "type": f"{type(result).__module__}.{type(result).__qualname__}",
        "result_fingerprint": semantic_result_fingerprint,
        "semantic_result_fingerprint": semantic_result_fingerprint,
        "full_result_fingerprint": captured["fingerprint"],
        "snapshot": captured["snapshot"],
        "semantic_snapshot": semantic_snapshot,
        "tensor_count": len(captured["tensors"]),
        "tensors": captured["tensors"],
        "path_identity": identity,
        "metadata_fingerprint": semantic_metadata_fingerprint,
        "semantic_metadata_fingerprint": semantic_metadata_fingerprint,
        "full_metadata_fingerprint": full_metadata_manifest["fingerprint"],
        "metadata": full_metadata_manifest["snapshot"],
        "semantic_metadata": semantic_metadata,
        "volatile_metadata": volatile_metadata,
        "volatile_metadata_allowlist": sorted(_VOLATILE_METADATA_FIELDS),
        "fingerprint_semantics": (
            "All tensor fields, path identity, alias contracts, and metadata are "
            "exact except the explicitly listed timing/high-water fields."
        ),
    }


def launch_ledger(result: object) -> dict[str, object]:
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    kernel = metadata.get("kernel")
    kernel = kernel if isinstance(kernel, dict) else {}
    aggregate = {field: kernel[field] for field in _LAUNCH_FIELDS if field in kernel}
    if "launch_count" not in aggregate and "launch_count" in metadata:
        aggregate["launch_count"] = metadata["launch_count"]
    return {
        "availability": "aggregate-only" if aggregate else "unavailable",
        "source": "result.metadata.kernel",
        "aggregate": aggregate,
        "per_launch": [],
        "unavailable_fields": [
            "per_launch_name",
            "grid",
            "block",
            "stream",
            "synchronization_points",
        ],
        "note": (
            "The current public runtime exposes aggregate launch/tape accounting, "
            "not a per-launch grid/block/stream ledger. Missing fields are explicit "
            "and are never represented as zero."
        ),
    }


def _plain_config(config: object) -> dict[str, object]:
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise RuntimeBaselineError("solver config must be a dataclass instance")
    values = {}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if isinstance(value, (set, frozenset)):
            value = sorted(value)
        elif isinstance(value, tuple):
            value = list(value)
        values[field.name] = value
    return values


def _load_case(solver: str, scenario: str) -> tuple[object, object, Any]:
    if solver not in SOLVERS:
        raise RuntimeBaselineError(f"unknown solver: {solver}")
    if scenario not in REDUCED_SCENARIOS:
        raise RuntimeBaselineError(f"unknown reduced scenario: {scenario}")

    from tests.support.scenes import (
        empty_space_los_scene,
        same_side_wall_reflection_scene,
    )

    if scenario == "empty-los":
        scene = empty_space_los_scene()
        components = {"los"}
        max_depth = 0
        samples = 128
        seed = 7
    else:
        scene = same_side_wall_reflection_scene()
        components = {"reflection"}
        max_depth = 1
        samples = 2048
        seed = 5

    if solver == "path":
        from witwin.channel_native.path import Config, solve

        config = Config(max_depth=max_depth, components=components)
    elif solver == "deterministic":
        from witwin.channel_native.deterministic import Config, solve

        config = Config(
            max_depth=max_depth,
            components=components,
            export_paths=True,
            diagnostics=True,
        )
    elif solver == "montecarlo-basic":
        from witwin.channel_native.montecarlo.basic import Config, solve

        config = Config(
            samples=samples,
            max_depth=max_depth,
            seed=seed,
            components=components,
            diagnostics=True,
        )
    else:
        from witwin.channel_native.montecarlo.bdpt import Config, solve

        config = Config(
            samples=samples,
            max_depth=max_depth,
            seed=seed,
            components=components,
            diagnostics=True,
            export_paths=True,
            max_exported_paths=samples,
            receiver_strategy=(
                "point_sphere" if scenario == "single-reflection" else "grid_area"
            ),
        )
    return scene, config, solve


def _child_environment(torch: Any) -> dict[str, object]:
    from witwin.channel_native import build_info

    device = torch.cuda.current_device()
    return {
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(part) for part in sys.version_info[:3]),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "device": {
            "index": int(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
        },
        "native_build": dict(build_info()),
    }


def run_child(
    *,
    repo: Path,
    solver: str,
    scenario: str,
    process_index: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    validate_measurement_policy(MIN_PROCESSES, warmup, repeats)
    sys.path[:0] = [str(repo / "src"), str(repo)]
    from tests.support.native_ext import inject_native_paths

    if not inject_native_paths():
        raise RuntimeBaselineError("compiled _channel_native extension was not found")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeBaselineError("runtime baseline requires CUDA")
    from benchmarks.harness import benchmark_operation

    scene, config, solve = _load_case(solver, scenario)
    scene_input = value_manifest(scene)

    def operation() -> object:
        return solve(scene, config)

    result, measurement = benchmark_operation(
        operation,
        warmup=warmup,
        repeats=repeats,
    )
    captured_result = result_manifest(result)
    ledger = launch_ledger(result)
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "git_sha": _git_sha(repo),
        "solver": solver,
        "scenario": scenario,
        "process_index": process_index,
        "config": _plain_config(config),
        "scene_input_fingerprint": scene_input["fingerprint"],
        "scene_input": scene_input["snapshot"],
        "result": captured_result,
        "launch_ledger": ledger,
        "performance": measurement.as_dict(),
        "environment": _child_environment(torch),
        "capture_note": (
            "Tensor D2H copies and hashing occur only after benchmark timing and are "
            "excluded from all wall/CUDA samples."
        ),
    }


def _parse_child_stdout(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "result" in value:
            return value
    raise RuntimeBaselineError("runtime child did not emit a JSON result")


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def aggregate_case(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if len(rows) < MIN_PROCESSES:
        raise RuntimeBaselineError("case aggregation requires independent process rows")
    fingerprints = {
        str(row["result"]["result_fingerprint"])  # type: ignore[index]
        for row in rows
    }
    metadata_fingerprints = {
        str(row["result"]["metadata_fingerprint"])  # type: ignore[index]
        for row in rows
    }
    scene_fingerprints = {str(row["scene_input_fingerprint"]) for row in rows}
    if len(fingerprints) != 1:
        raise RuntimeBaselineError("result hash differs across independent processes")
    if len(metadata_fingerprints) != 1:
        raise RuntimeBaselineError("metadata differs across independent processes")
    if len(scene_fingerprints) != 1:
        raise RuntimeBaselineError("scene input differs across independent processes")

    steady = [
        sample
        for row in rows
        for sample in row["performance"]["steady"]  # type: ignore[index]
    ]
    wall = [float(sample["wall_ms"]) for sample in steady]
    cuda = [
        float(sample["cuda_event_ms"])
        for sample in steady
        if sample["cuda_event_ms"] is not None
    ]
    memories = [row["performance"]["memory"] for row in rows]  # type: ignore[index]
    return {
        "solver": rows[0]["solver"],
        "scenario": rows[0]["scenario"],
        "config": rows[0]["config"],
        "scene_input_fingerprint": next(iter(scene_fingerprints)),
        "result_fingerprint": next(iter(fingerprints)),
        "metadata_fingerprint": next(iter(metadata_fingerprints)),
        "exact_across_processes": True,
        "processes": list(rows),
        "performance_distribution": {
            "process_count": len(rows),
            "steady_sample_count": len(wall),
            "wall_median_ms": float(statistics.median(wall)),
            "wall_p95_ms": _percentile(wall, 0.95),
            "cuda_median_ms": float(statistics.median(cuda)) if cuda else None,
            "cuda_p95_ms": _percentile(cuda, 0.95) if cuda else None,
            "peak_allocated_bytes_max": max(
                int(memory["peak_allocated_bytes"]) for memory in memories
            ),
            "peak_reserved_bytes_max": max(
                int(memory["peak_reserved_bytes"]) for memory in memories
            ),
            "peak_temporary_allocated_bytes_max": max(
                int(memory["peak_temporary_allocated_bytes"]) for memory in memories
            ),
        },
    }


def _child_command(
    script: Path,
    *,
    solver: str,
    scenario: str,
    process_index: int,
    warmup: int,
    repeats: int,
) -> list[str]:
    return [
        sys.executable,
        str(script),
        "--child",
        "--solver",
        solver,
        "--scenario",
        scenario,
        "--process-index",
        str(process_index),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]


def collect_reduced(
    repo: Path,
    *,
    solvers: Sequence[str],
    scenarios: Sequence[str],
    processes: int,
    warmup: int,
    repeats: int,
    timeout_seconds: int,
) -> dict[str, object]:
    validate_measurement_policy(processes, warmup, repeats)
    unknown_solvers = sorted(set(solvers) - set(SOLVERS))
    unknown_scenarios = sorted(set(scenarios) - set(REDUCED_SCENARIOS))
    if unknown_solvers or unknown_scenarios:
        raise RuntimeBaselineError(
            f"unknown solvers/scenarios: {unknown_solvers + unknown_scenarios}"
        )
    script = Path(__file__).resolve()
    cases = []
    for solver in solvers:
        for scenario in scenarios:
            rows = []
            for process_index in range(processes):
                command = _child_command(
                    script,
                    solver=solver,
                    scenario=scenario,
                    process_index=process_index,
                    warmup=warmup,
                    repeats=repeats,
                )
                env = os.environ.copy()
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                completed = subprocess.run(
                    command,
                    cwd=repo,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                )
                if completed.returncode:
                    detail = (completed.stderr or completed.stdout).strip()
                    raise RuntimeBaselineError(
                        f"child failed for {solver}/{scenario}/process-{process_index}: {detail}"
                    )
                rows.append(_parse_child_stdout(completed.stdout))
            cases.append(aggregate_case(rows))
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "git_sha": _git_sha(repo),
        "profile": "reduced",
        "measurement_policy": {
            "independent_processes": processes,
            "warmup_per_process": warmup,
            "steady_repeats_per_process": repeats,
            "minimums_enforced": {
                "independent_processes": MIN_PROCESSES,
                "warmup_per_process": MIN_WARMUP,
                "steady_repeats_per_process": MIN_REPEATS,
            },
        },
        "coverage": {
            "solvers": list(solvers),
            "scenarios": list(scenarios),
            "ad_modes": ["none"],
            "fixed_seeds": {"empty-los": 7, "single-reflection": 5},
            "status": "reduced-not-full-g0-matrix",
        },
        "cases": cases,
    }


def _project_runtime_report(report: dict[str, object]) -> dict[str, dict[str, object]]:
    common = {
        "schema": report["schema"],
        "git_sha": report["git_sha"],
        "profile": report["profile"],
        "measurement_policy": report["measurement_policy"],
        "coverage": report["coverage"],
    }
    solver_cases = []
    launch_cases = []
    performance_cases = []
    for case in report["cases"]:
        case_common = {
            "solver": case["solver"],
            "scenario": case["scenario"],
            "config": case["config"],
            "scene_input_fingerprint": case["scene_input_fingerprint"],
            "result_fingerprint": case["result_fingerprint"],
            "metadata_fingerprint": case["metadata_fingerprint"],
            "exact_across_processes": case["exact_across_processes"],
        }
        solver_cases.append(
            {
                **case_common,
                "processes": [
                    {
                        "process_index": row["process_index"],
                        "scene_input": row["scene_input"],
                        "result": row["result"],
                    }
                    for row in case["processes"]
                ],
            }
        )
        launch_cases.append(
            {
                **case_common,
                "processes": [
                    {
                        "process_index": row["process_index"],
                        "launch_ledger": row["launch_ledger"],
                    }
                    for row in case["processes"]
                ],
            }
        )
        performance_cases.append(
            {
                "solver": case["solver"],
                "scenario": case["scenario"],
                "config": case["config"],
                "distribution": case["performance_distribution"],
                "processes": [
                    {
                        "process_index": row["process_index"],
                        "environment": row["environment"],
                        "performance": row["performance"],
                    }
                    for row in case["processes"]
                ],
            }
        )
    return {
        "solver-results.json": {
            **common,
            "kind": "solver-results",
            "cases": solver_cases,
        },
        "launch-ledger.json": {
            **common,
            "kind": "launch-ledger",
            "cases": launch_cases,
        },
        "performance.json": {
            **common,
            "kind": "performance",
            "cases": performance_cases,
        },
    }


def write_immutable_report(report: dict[str, object], output_root: Path) -> Path:
    import shutil

    sha = str(report["git_sha"])
    output_root = output_root.resolve()
    directory = output_root / sha
    if directory.exists():
        raise RuntimeBaselineError(
            f"immutable runtime baseline already exists: {directory}"
        )
    staging = output_root / f".{sha}.{os.getpid()}.tmp"
    projections = _project_runtime_report(report)
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        artifact_index = {}
        for filename, payload in projections.items():
            content = _canonical_bytes(payload)
            (staging / filename).write_bytes(content)
            artifact_index[filename] = {
                "kind": payload["kind"],
                "sha256": _sha256(content),
            }
        index = {
            "schema": report["schema"],
            "git_sha": report["git_sha"],
            "profile": report["profile"],
            "measurement_policy": report["measurement_policy"],
            "coverage": report["coverage"],
            "artifacts": artifact_index,
        }
        (staging / "reduced.json").write_bytes(_canonical_bytes(index))
        staging.rename(directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return directory / "reduced.json"


def _comma_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--solver", choices=SOLVERS, help=argparse.SUPPRESS)
    parser.add_argument("--scenario", choices=REDUCED_SCENARIOS, help=argparse.SUPPRESS)
    parser.add_argument("--process-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--solvers", type=_comma_values, default=SOLVERS)
    parser.add_argument("--scenarios", type=_comma_values, default=REDUCED_SCENARIOS)
    parser.add_argument("--processes", type=int, default=MIN_PROCESSES)
    parser.add_argument("--warmup", type=int, default=MIN_WARMUP)
    parser.add_argument("--repeats", type=int, default=MIN_REPEATS)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="defaults to REPO/artifacts/refactor_runtime_baseline",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = args.repo.resolve()
    try:
        if args.child:
            if not args.solver or not args.scenario:
                raise RuntimeBaselineError(
                    "child mode requires --solver and --scenario"
                )
            report = run_child(
                repo=repo,
                solver=args.solver,
                scenario=args.scenario,
                process_index=args.process_index,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
            return 0
        report = collect_reduced(
            repo,
            solvers=args.solvers,
            scenarios=args.scenarios,
            processes=args.processes,
            warmup=args.warmup,
            repeats=args.repeats,
            timeout_seconds=args.timeout_seconds,
        )
        output_root = (
            args.output_root
            if args.output_root is not None
            else repo / "artifacts" / "refactor_runtime_baseline"
        )
        destination = write_immutable_report(report, output_root)
    except (RuntimeBaselineError, OSError, subprocess.SubprocessError) as error:
        print(f"runtime baseline failed: {error}", file=sys.stderr)
        return 2
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
