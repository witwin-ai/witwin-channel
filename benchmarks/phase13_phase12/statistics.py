# Copyright Xingyu Chen.
# Strict worker validation, statistics, correctness, and resource gates.

"""Strict worker validation, statistics, correctness, and resource gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import random
import statistics

from .contracts import (
    EvidenceError,
    VARIANTS,
    WORKER_SCHEMA_NAME,
    WORKER_SCHEMA_VERSION,
    exact_keys,
    finite_number,
    finite_samples,
    require_sha,
    require_sha256,
    schema_identity,
    validate_exact_schedule,
)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise EvidenceError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * quantile
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_identity(value: object, *, label: str) -> dict[str, object]:
    identity = exact_keys(
        value,
        {
            "channel_commit", "rayd_commit", "rayd_lock_sha256",
            "integration_header_sha256", "build_fingerprint", "build_type",
            "extension_load_source", "extension_path", "extension_sha256",
            "python_executable_sha256", "runtime_dependencies",
            "cuda_architectures", "device_index", "device_uuid", "gpu_name",
            "driver_version", "python_version",
            "torch_version", "cuda_version", "compiler", "cuda_compiler_version",
        },
        label=label,
    )
    for name in ("channel_commit", "rayd_commit"):
        require_sha(identity[name], label=f"{label}.{name}")
    for name in (
        "rayd_lock_sha256", "integration_header_sha256", "build_fingerprint",
        "extension_sha256", "python_executable_sha256",
    ):
        require_sha256(identity[name], label=f"{label}.{name}")
    if identity["build_type"] != "Release":
        raise EvidenceError(f"{label}.build_type must be Release")
    if identity["extension_load_source"] != "packaged":
        raise EvidenceError(f"{label}.extension_load_source must be packaged")
    if identity["cuda_architectures"] != ["120-real"]:
        raise EvidenceError(f"{label}.cuda_architectures must be the real SM120 evidence build")
    if type(identity["device_index"]) is not int or int(identity["device_index"]) < 0:
        raise EvidenceError(f"{label}.device_index must be a non-negative integer")
    for name in (
        "extension_path", "device_uuid", "gpu_name", "driver_version", "python_version",
        "torch_version", "cuda_version", "compiler", "cuda_compiler_version",
    ):
        if not isinstance(identity[name], str) or not identity[name]:
            raise EvidenceError(f"{label}.{name} must be a non-empty string")
    dependencies = identity["runtime_dependencies"]
    if not isinstance(dependencies, list) or not dependencies:
        raise EvidenceError(f"{label}.runtime_dependencies must be non-empty")
    dependency_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(dependencies):
        item = exact_keys(
            row, {"path", "name", "sha256", "bytes"},
            label=f"{label}.runtime_dependencies[{index}]",
        )
        require_sha256(item["sha256"], label=f"{label}.runtime_dependencies[{index}]")
        if not isinstance(item["path"], str) or not isinstance(item["name"], str):
            raise EvidenceError("runtime dependency path/name must be strings")
        if type(item["bytes"]) is not int or int(item["bytes"]) <= 0:
            raise EvidenceError("runtime dependency byte size must be positive")
        key = (str(item["name"]).casefold(), str(item["sha256"]))
        if key in dependency_keys:
            raise EvidenceError("runtime dependency identity is duplicated")
        dependency_keys.add(key)
    return identity


def _validate_hash_rows(
    value: object, *, label: str, expected_names: set[str],
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise EvidenceError(f"{label} must be an array")
    result: list[dict[str, object]] = []
    names: set[str] = set()
    for index, raw in enumerate(value):
        row = exact_keys(raw, {"name", "sha256"}, label=f"{label}[{index}]")
        name = row["name"]
        if not isinstance(name, str) or not name or name in names:
            raise EvidenceError(f"{label} names must be non-empty and unique")
        require_sha256(row["sha256"], label=f"{label}[{index}].sha256")
        names.add(name)
        result.append(row)
    if names != expected_names:
        raise EvidenceError(
            f"{label} names differ; missing={sorted(expected_names - names)}, "
            f"extra={sorted(names - expected_names)}"
        )
    return result


def validate_worker_record(
    raw: object, *, expected_group: str, expected_variant: str, process_index: int, order: str,
    gate: Mapping[str, object],
) -> dict[str, object]:
    record = exact_keys(
        raw,
        {
            "schema", "group", "variant", "process_index", "pair_order",
            "measurement_policy", "identity", "identity_capture", "timings",
            "memory", "hashes", "build_crosscheck", "capture",
        },
        label=f"worker[{expected_variant}:{process_index}]",
    )
    schema_identity(
        record["schema"], name=WORKER_SCHEMA_NAME, version=WORKER_SCHEMA_VERSION,
        label="worker.schema",
    )
    if (
        record["group"] != expected_group
        or record["variant"] != expected_variant
        or record["process_index"] != process_index
        or record["pair_order"] != order
    ):
        raise EvidenceError("worker process identity disagrees with the runner")
    if record["measurement_policy"] != {
        "warmup": 1, "steady_repeats": 7,
        "timed_region_excludes_result_serialization": True,
        "cuda_event_timing": True, "wall_timing": True,
    }:
        raise EvidenceError("worker measurement policy differs from the frozen policy")
    identity = _validate_identity(record["identity"], label="worker.identity")
    if not isinstance(record["identity_capture"], dict):
        raise EvidenceError("runner-owned identity capture must be an object")
    crosscheck = exact_keys(
        record["build_crosscheck"],
        {"channel_commit", "rayd_commit", "integration_header_sha256", "build_fingerprint"},
        label="worker.build_crosscheck",
    )
    expected_crosscheck = {
        name: identity[name] for name in crosscheck
    }
    if crosscheck != expected_crosscheck:
        raise EvidenceError("worker build cross-check differs from parent identity probe")
    if not isinstance(record["capture"], dict):
        raise EvidenceError("worker capture must be runner-owned")

    group_policy = gate["comparison_groups"][expected_group]  # type: ignore[index]
    assert isinstance(group_policy, dict)
    expected_metrics = {
        str(group_policy["target_end_to_end_metric"]),
        *[str(name) for name in gate["non_target_metrics"]],  # type: ignore[union-attr]
    }
    timings = record["timings"]
    if not isinstance(timings, list):
        raise EvidenceError("worker.timings must be an array")
    normalized_timings: list[dict[str, object]] = []
    timing_names: set[str] = set()
    for index, raw_timing in enumerate(timings):
        timing = exact_keys(
            raw_timing, {"name", "steady_cuda_ms", "steady_wall_ms"},
            label=f"worker.timings[{index}]",
        )
        name = timing["name"]
        if not isinstance(name, str) or name in timing_names:
            raise EvidenceError("worker timing names must be unique strings")
        timing_names.add(name)
        normalized_timings.append(
            {
                "name": name,
                "steady_cuda_ms": finite_samples(timing["steady_cuda_ms"], label=f"{name}.cuda"),
                "steady_wall_ms": finite_samples(timing["steady_wall_ms"], label=f"{name}.wall"),
            }
        )
    if timing_names != expected_metrics:
        raise EvidenceError("worker timing metric set differs from the canonical gate")

    memory = exact_keys(
        record["memory"],
        {"peak_allocated_bytes", "peak_reserved_bytes"},
        label="worker.memory",
    )
    for name, value in memory.items():
        if type(value) is not int or value < 0:
            raise EvidenceError(f"worker.memory.{name} must be a non-negative integer")

    correctness_policy = group_policy["correctness"]
    if not isinstance(correctness_policy, dict):
        raise EvidenceError("group correctness facts are not frozen")
    hashes = exact_keys(
        record["hashes"],
        {
            "format", "schema_version", "target_shape", "target_dtype", "target",
            "full_result", "repeat_target", "unaffected",
        },
        label="worker.hashes",
    )
    tensor_contract = gate["frozen_inputs"]["tensor_hash"]  # type: ignore[index]
    assert isinstance(tensor_contract, dict)
    if hashes["format"] != tensor_contract["format"] or hashes.get(
        "schema_version"
    ) != tensor_contract["schema_version"]:
        raise EvidenceError("worker tensor hash encoding differs from the canonical format")
    if hashes["target_shape"] != correctness_policy["target_shape"]:
        raise EvidenceError("worker target tensor shape differs from frozen facts")
    if hashes["target_dtype"] != correctness_policy["target_dtype"]:
        raise EvidenceError("worker target tensor dtype differs from frozen facts")
    require_sha256(hashes["target"], label="target result hash")
    require_sha256(hashes["full_result"], label="full result hash")
    repeats = hashes["repeat_target"]
    if not isinstance(repeats, list) or len(repeats) != 7:
        raise EvidenceError("repeat_target must contain seven hashes")
    for index, value in enumerate(repeats):
        require_sha256(value, label=f"repeat_target[{index}]")
    unaffected = _validate_hash_rows(
        hashes["unaffected"], label="worker.hashes.unaffected",
        expected_names=set(gate["required_unaffected_hashes"]),  # type: ignore[arg-type]
    )
    return {
        **record, "identity": identity, "timings": normalized_timings,
        "memory": memory,
        "hashes": {**hashes, "unaffected": unaffected},
    }


def validate_process_identity(
    pairs: Sequence[Mapping[str, object]], implementation: Mapping[str, object],
    gate: Mapping[str, object],
) -> dict[str, object]:
    group = str(pairs[0]["group"])
    identities: dict[str, list[dict[str, object]]] = {name: [] for name in VARIANTS}
    for pair in pairs:
        for variant in VARIANTS:
            record = pair[variant]
            assert isinstance(record, dict)
            identity = record["identity"]
            assert isinstance(identity, dict)
            identities[variant].append(identity)
    canonical: dict[str, dict[str, object]] = {}
    for variant, rows in identities.items():
        encoded = {repr(sorted(row.items())) for row in rows}
        if len(encoded) != 1:
            raise EvidenceError(f"{variant} build/tool identity changed between processes")
        canonical[variant] = rows[0]
    for variant in VARIANTS:
        group_history = implementation["groups"][group]  # type: ignore[index]
        assert isinstance(group_history, dict)
        if canonical[variant]["channel_commit"] != group_history[f"{variant}_commit"]:
            raise EvidenceError(f"{variant} worker build differs from checkout HEAD")
        if canonical[variant]["rayd_commit"] != implementation["rayd_commit"]:
            raise EvidenceError(f"{variant} worker RayD commit differs from lock")
        if canonical[variant]["rayd_lock_sha256"] != implementation["rayd_lock_sha256"]:
            raise EvidenceError(f"{variant} worker RayD lock differs from checkout")
        if canonical[variant]["integration_header_sha256"] != implementation["integration_header_sha256"]:
            raise EvidenceError(f"{variant} worker header differs from checkout")
    equal_fields = (
        "rayd_commit", "rayd_lock_sha256", "integration_header_sha256", "build_type",
        "extension_load_source", "cuda_architectures", "device_index", "device_uuid",
        "gpu_name", "driver_version",
        "python_version", "torch_version", "cuda_version", "compiler",
        "cuda_compiler_version",
    )
    mismatched = [name for name in equal_fields if canonical["baseline"][name] != canonical["candidate"][name]]
    baseline_dependencies = {
        (str(row["name"]).casefold(), str(row["sha256"]), int(row["bytes"]))
        for row in canonical["baseline"]["runtime_dependencies"]
    }
    candidate_dependencies = {
        (str(row["name"]).casefold(), str(row["sha256"]), int(row["bytes"]))
        for row in canonical["candidate"]["runtime_dependencies"]
    }
    if baseline_dependencies != candidate_dependencies:
        mismatched.append("runtime_dependencies")
    canonical_dependencies = sorted(
        (
            {
                "path": str(row["path"]),
                "name": str(row["name"]),
                "sha256": str(row["sha256"]),
                "bytes": int(row["bytes"]),
            }
            for row in canonical["candidate"]["runtime_dependencies"]
        ),
        key=lambda row: (row["path"].casefold(), row["name"].casefold()),
    )
    dependency_digest = hashlib.sha256(
        json.dumps(
            canonical_dependencies,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    frozen = gate.get("frozen_inputs")
    if not isinstance(frozen, dict) or dependency_digest != frozen.get(
        "runtime_dependency_sha256"
    ):
        mismatched.append("frozen_runtime_dependency_sha256")
    if canonical["baseline"]["python_executable_sha256"] != canonical["candidate"]["python_executable_sha256"]:
        mismatched.append("python_executable_sha256")
    if mismatched:
        raise EvidenceError("A/B RayD/toolchain differs: " + ", ".join(mismatched))
    if canonical["baseline"]["build_fingerprint"] == canonical["candidate"]["build_fingerprint"]:
        raise EvidenceError("distinct A/B builds unexpectedly share one fingerprint")
    if canonical["baseline"]["extension_sha256"] == canonical["candidate"]["extension_sha256"]:
        raise EvidenceError("baseline and candidate unexpectedly loaded the same extension bytes")
    return {
        "group": group,
        "history": group_history,
        "rayd_commit": implementation["rayd_commit"],
        "rayd_lock_sha256": implementation["rayd_lock_sha256"],
        "integration_header_sha256": implementation["integration_header_sha256"],
        "baseline_build_fingerprint": canonical["baseline"]["build_fingerprint"],
        "candidate_build_fingerprint": canonical["candidate"]["build_fingerprint"],
        "baseline_extension_sha256": canonical["baseline"]["extension_sha256"],
        "candidate_extension_sha256": canonical["candidate"]["extension_sha256"],
        "runtime_dependencies": canonical_dependencies,
        "runtime_dependency_sha256": dependency_digest,
        "same_rayd_pin_and_toolchain_verified": True,
    }


def paired_bootstrap(
    values: Sequence[float], *, resamples: int, confidence: float = 0.95, seed: int = 13029030,
) -> dict[str, object]:
    if len(values) != 5 or resamples != 100000 or confidence != 0.95:
        raise EvidenceError("paired bootstrap requires five pairs, 100000 resamples, 95% confidence")
    finite = [finite_number(value, label="paired improvement") for value in values]
    generator = random.Random(seed)
    draws = [
        statistics.mean(finite[generator.randrange(5)] for _ in range(5))
        for _ in range(resamples)
    ]
    return {
        "method": "paired-mean-of-process-medians-percent-improvement",
        "seed": seed, "resamples": resamples, "confidence": confidence,
        "estimate_percent": statistics.mean(finite),
        "lower_percent": percentile(draws, 0.025),
        "upper_percent": percentile(draws, 0.975),
    }


def _timing(record: Mapping[str, object], name: str) -> Mapping[str, object]:
    rows = record["timings"]
    if not isinstance(rows, list):
        raise EvidenceError("worker timing rows are malformed")
    profile_rows = record.get("profile_timings", [])
    if not isinstance(profile_rows, list):
        raise EvidenceError("profile timing rows are malformed")
    matches = [
        row
        for row in [*rows, *profile_rows]
        if isinstance(row, dict) and row.get("name") == name
    ]
    if len(matches) != 1:
        raise EvidenceError(f"worker lacks unique timing metric {name}")
    return matches[0]


def compare_timings(
    pairs: Sequence[Mapping[str, object]], gate: Mapping[str, object], *, bootstrap_resamples: int,
) -> list[dict[str, object]]:
    validate_exact_schedule(pairs, gate)
    group = str(pairs[0]["group"])
    group_policy = gate["comparison_groups"][group]  # type: ignore[index]
    assert isinstance(group_policy, dict)
    regression = gate["non_target_regression_policy"]
    assert isinstance(regression, dict)
    policies: list[dict[str, object]] = [
        {
            "name": group_policy["target_stage_metric"],
            "category": "target_stage", "clock": "cuda",
            "minimum_improvement_percent": group_policy["minimum_stage_improvement_percent"],
        },
        {
            "name": group_policy["target_end_to_end_metric"],
            "category": "target_end_to_end", "clock": "wall",
            "minimum_improvement_percent": group_policy["minimum_end_to_end_improvement_percent"],
        },
        *[
            {
                "name": name, "category": "non_target", "clock": "wall",
                **regression,
            }
            for name in gate["non_target_metrics"]  # type: ignore[union-attr]
        ],
    ]
    comparisons: list[dict[str, object]] = []
    for policy in policies:
        assert isinstance(policy, dict)
        name, clock = str(policy["name"]), str(policy["clock"])
        key = f"steady_{clock}_ms"
        baseline_all: list[float] = []
        candidate_all: list[float] = []
        paired: list[float] = []
        process_rows: list[dict[str, object]] = []
        for pair in pairs:
            baseline = list(_timing(pair["baseline"], name)[key])  # type: ignore[arg-type,index]
            candidate = list(_timing(pair["candidate"], name)[key])  # type: ignore[arg-type,index]
            baseline_median = statistics.median(baseline)
            candidate_median = statistics.median(candidate)
            improvement = 100.0 * (baseline_median - candidate_median) / baseline_median
            paired.append(improvement)
            baseline_all.extend(baseline)
            candidate_all.extend(candidate)
            process_rows.append(
                {
                    "process_index": pair["process_index"], "order": pair["order"],
                    "baseline_median_ms": baseline_median,
                    "candidate_median_ms": candidate_median,
                    "improvement_percent": improvement,
                }
            )
        baseline_median = statistics.median(baseline_all)
        candidate_median = statistics.median(candidate_all)
        baseline_p95, candidate_p95 = percentile(baseline_all, 0.95), percentile(candidate_all, 0.95)
        improvement = 100.0 * (baseline_median - candidate_median) / baseline_median
        median_regression = -improvement
        p95_regression = 100.0 * (candidate_p95 - baseline_p95) / baseline_p95
        bootstrap = paired_bootstrap(paired, resamples=bootstrap_resamples)
        if policy["category"] in {"target_stage", "target_end_to_end"}:
            checks = {
                "minimum_improvement": improvement >= float(policy["minimum_improvement_percent"]),
                "bootstrap_lower_bound_positive": float(bootstrap["lower_percent"]) > 0.0,
            }
        else:
            checks = {
                "median_regression": median_regression <= float(policy["maximum_median_regression_percent"]),
                "p95_regression": p95_regression <= float(policy["maximum_p95_regression_percent"]),
            }
        comparisons.append(
            {
                "name": name, "category": policy["category"], "clock": clock,
                "baseline_median_ms": baseline_median, "candidate_median_ms": candidate_median,
                "baseline_p95_ms": baseline_p95, "candidate_p95_ms": candidate_p95,
                "improvement_percent": improvement,
                "median_regression_percent": median_regression,
                "p95_regression_percent": p95_regression,
                "process_pairs": process_rows, "paired_bootstrap": bootstrap,
                "checks": checks, "passed": all(checks.values()),
            }
        )
    return comparisons


def validate_correctness(
    pairs: Sequence[Mapping[str, object]], gate: Mapping[str, object],
) -> dict[str, object]:
    validate_exact_schedule(pairs, gate)
    target_hashes: set[str] = set()
    full_hashes: set[str] = set()
    baseline_hashes: set[str] = set()
    unaffected: dict[str, set[str]] = {}
    for pair in pairs:
        for variant in VARIANTS:
            record = pair[variant]
            assert isinstance(record, dict)
            hashes = record["hashes"]
            assert isinstance(hashes, dict)
            target = str(hashes["target"])
            repeats = set(str(value) for value in hashes["repeat_target"])  # type: ignore[union-attr]
            if variant == "candidate":
                target_hashes.add(target)
                full_hashes.add(str(hashes["full_result"]))
                if repeats != {target}:
                    raise EvidenceError("candidate target is not bitwise stable across repeats")
            else:
                baseline_hashes.add(target)
            for row in hashes["unaffected"]:  # type: ignore[union-attr]
                unaffected.setdefault(str(row["name"]), set()).add(str(row["sha256"]))
    if len(target_hashes) != 1 or len(full_hashes) != 1:
        raise EvidenceError("candidate target/full result is not stable across processes")
    if any(len(values) != 1 for values in unaffected.values()):
        raise EvidenceError("an unaffected result changed across A/B processes")
    group = str(pairs[0]["group"])
    correctness = gate["comparison_groups"][group]["correctness"]  # type: ignore[index]
    assert isinstance(correctness, dict)
    target = next(iter(target_hashes))
    full = next(iter(full_hashes))
    if target != correctness["candidate_target_sha256"] or full != correctness["candidate_full_result_sha256"]:
        raise EvidenceError("candidate result differs from the accepted deterministic hashes")
    if sorted(baseline_hashes) != correctness["baseline_target_sha256_values"]:
        raise EvidenceError("baseline result differs from frozen raw facts")
    return {
        "candidate_target_sha256": target,
        "candidate_full_result_sha256": full,
        "baseline_target_sha256_values": sorted(baseline_hashes),
        "unaffected_hashes": [
            {"name": name, "sha256": next(iter(values))}
            for name, values in sorted(unaffected.items())
        ],
        "candidate_bitwise_stable": True,
        "unaffected_exact": True,
    }


def summarize_resources(
    pairs: Sequence[Mapping[str, object]], gate: Mapping[str, object],
) -> dict[str, object]:
    validate_exact_schedule(pairs, gate)
    memories = [pair["candidate"]["memory"] for pair in pairs]  # type: ignore[index]
    group = str(pairs[0]["group"])
    budget = gate["comparison_groups"][group]["resource_budgets"]  # type: ignore[index]
    assert isinstance(budget, dict)
    observed = {
        "peak_allocated_bytes_max": max(int(row["peak_allocated_bytes"]) for row in memories),
        "peak_reserved_bytes_max": max(int(row["peak_reserved_bytes"]) for row in memories),
    }
    checks = {
        name: value <= int(budget[name]) for name, value in observed.items()
    }
    return {**observed, "checks": checks, "passed": all(checks.values())}


__all__ = [
    "compare_timings", "paired_bootstrap", "percentile", "summarize_resources",
    "validate_correctness", "validate_process_identity", "validate_worker_record",
]