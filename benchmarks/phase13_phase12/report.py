# Copyright Xingyu Chen.
# Three-group measured report construction and raw-evidence replay.

"""Three-group measured report construction and raw-evidence replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path

from .artifacts import ArtifactStore
from .builds import (
    compiler_resource_checks,
    prepare_fresh_channel_builds,
    validate_channel_build_records,
)
from .contracts import (
    CANONICAL_GATE_REPO_PATH,
    COMPARISON_GROUPS,
    DEFAULT_GATE,
    EvidenceError,
    ROOT,
    RunnerConfig,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    exact_keys,
    load_gate,
    process_schedule,
    reject_developer_override_environment,
    require_measured_policy_ready,
)
from .diagnostics import collect_diagnostics, replay_diagnostics
from .profilers import (
    attach_profile_timings,
    parse_ncu_csv,
    parse_nsys_sqlite,
    load_profile_contract,
    run_ncu_candidate,
    run_nsys_matrix,
    validate_ncu_blocker,
    validate_nsys_timelines,
)
from .release import run_release_evidence, validate_release_report
from .statistics import (
    compare_timings,
    summarize_resources,
    validate_correctness,
    validate_process_identity,
    validate_worker_record,
)
from .workers import (
    collect_process_pairs,
    parse_identity_stdout,
    parse_worker_stdout,
    sha256_path,
    validate_retained_identity,
    validate_retained_worker_capture,
    verify_checkouts,
    verify_evidence_commit_binding,
)


def _gate_policy() -> dict[str, object]:
    return {"path": CANONICAL_GATE_REPO_PATH, "sha256": sha256_path(DEFAULT_GATE)}


def build_dry_run(
    config: RunnerConfig, gate: Mapping[str, object], *, gate_path: Path
) -> dict[str, object]:
    reject_developer_override_environment()
    implementation = verify_checkouts(config, require_clean_rayd=False)
    planned: dict[str, list[dict[str, object]]] = {}
    for group in COMPARISON_GROUPS:
        rows: list[dict[str, object]] = []
        for pair in process_schedule(gate, group):
            invocations = []
            for raw_name in pair["variants"]:
                name = str(raw_name)
                variant = config.variant(group, name)
                invocations.append(
                    {
                        "variant": name,
                        "cwd": str(variant.checkout),
                        "python_executable": str(variant.python_executable),
                        "canonical_worker": "benchmarks/phase13_phase12_worker.py",
                    }
                )
            rows.append({"group": group, **pair, "invocations": invocations})
        planned[group] = rows
    blockers: list[str] = []
    try:
        require_measured_policy_ready(gate)
    except EvidenceError as exc:
        blockers.append(str(exc))
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "status": "dry_run",
        "release_claim": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_policy": _gate_policy(),
        "implementation": implementation,
        "measurement_policy": gate["measurement_policy"],
        "munich": gate["munich"],
        "planned_groups": planned,
        "blockers": blockers,
        "gate": {"status": "not_evaluated", "passed": None, "checks": []},
    }


def _normalize_pairs(
    pairs: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
    *,
    group: str,
) -> list[dict[str, object]]:
    schedule = process_schedule(gate, group)
    if len(pairs) != len(schedule):
        raise EvidenceError(f"{group} worker pair count differs from the fixed schedule")
    normalized: list[dict[str, object]] = []
    for pair, expected in zip(pairs, schedule, strict=True):
        index = int(expected["process_index"])
        order = str(expected["order"])
        row: dict[str, object] = {
            "group": group,
            "process_index": index,
            "order": order,
        }
        for variant in ("baseline", "candidate"):
            source = pair[variant]
            if not isinstance(source, dict):
                raise EvidenceError("worker pair variant must be an object")
            parsed = validate_worker_record(
                source,
                expected_group=group,
                expected_variant=variant,
                process_index=index,
                order=order,
                gate=gate,
            )
            parsed["capture"] = source["capture"]
            parsed["identity_capture"] = source["identity_capture"]
            row[variant] = parsed
        normalized.append(row)
    return normalized


def _group_checks(
    group: str,
    *,
    comparisons: Sequence[Mapping[str, object]],
    correctness: Mapping[str, object],
    resources: Mapping[str, object],
    nsys: Mapping[str, object],
    diagnostics: Mapping[str, object],
    compiler_resources: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = [
        {"name": f"{group}:timing:{item['name']}", "passed": item["passed"] is True}
        for item in comparisons
    ]
    rows.extend(
        [
            {
                "name": f"{group}:correctness:target",
                "passed": correctness["candidate_bitwise_stable"] is True,
            },
            {
                "name": f"{group}:correctness:unaffected",
                "passed": correctness["unaffected_exact"] is True,
            },
            {"name": f"{group}:resources", "passed": resources["passed"] is True},
            {
                "name": f"{group}:diagnostics",
                "passed": diagnostics["passed"] is True,
            },
            {
                "name": f"{group}:compiler_resources",
                "passed": compiler_resources["passed"] is True,
            },
            {"name": f"{group}:nsys", "passed": nsys["passed"] is True},
        ]
    )
    return rows


def build_measured_report(
    config: RunnerConfig,
    gate: Mapping[str, object],
    *,
    gate_path: Path,
    timeout_seconds: int,
    bootstrap_resamples: int,
) -> dict[str, object]:
    reject_developer_override_environment()
    canonical = load_gate(gate_path, measured=True)
    if gate != canonical:
        raise EvidenceError("caller gate object differs from the canonical measured gate")
    require_measured_policy_ready(gate)
    if bootstrap_resamples != 100000:
        raise EvidenceError("formal evidence requires exactly 100000 bootstrap resamples")
    store = ArtifactStore.create(config.raw_artifact_parent)
    initial = verify_checkouts(
        config, gate, store=store, require_clean_rayd=False
    )
    runtime_config, channel_builds = prepare_fresh_channel_builds(
        config, initial, store=store, timeout_seconds=timeout_seconds
    )
    built_identity = verify_checkouts(runtime_config, gate)
    if built_identity["groups"] != initial["groups"]:
        raise EvidenceError("runner-built Channel source commits differ from verified inputs")
    channel_builds["compiler_resource_checks"] = compiler_resource_checks(
        channel_builds, gate
    )
    group_reports: dict[str, dict[str, object]] = {}
    group_identities: dict[str, dict[str, object]] = {}
    gate_checks: list[dict[str, object]] = []
    for group in COMPARISON_GROUPS:
        pairs = _normalize_pairs(
            collect_process_pairs(
                runtime_config, gate, group=group, timeout_seconds=timeout_seconds, store=store
            ),
            gate,
            group=group,
        )
        group_identity = validate_process_identity(pairs, initial, gate)
        nsys = run_nsys_matrix(
            runtime_config,
            gate,
            group=group,
            timeout_seconds=timeout_seconds,
            store=store,
        )
        diagnostics = collect_diagnostics(
            runtime_config,
            gate,
            group=group,
            timeout_seconds=timeout_seconds,
            store=store,
        )
        pairs = attach_profile_timings(
            pairs, nsys["captures"], gate, group=group  # type: ignore[arg-type]
        )
        comparisons = compare_timings(
            pairs, gate, bootstrap_resamples=bootstrap_resamples
        )
        correctness = validate_correctness(pairs, gate)
        resources = summarize_resources(pairs, gate)
        checks = _group_checks(
            group,
            comparisons=comparisons,
            correctness=correctness,
            resources=resources,
            nsys=nsys,
            diagnostics=diagnostics,
            compiler_resources=channel_builds["compiler_resource_checks"][group],
        )
        if not all(row["passed"] for row in checks):
            raise EvidenceError(f"{group} acceptance gate failed")
        group_identities[group] = group_identity
        group_reports[group] = {
            "process_pairs": pairs,
            "comparisons": comparisons,
            "correctness": correctness,
            "resources": resources,
            "nsight_systems": nsys,
            "diagnostics": diagnostics,
            "checks": checks,
        }
        gate_checks.extend(checks)
    dependency_digests = {
        str(identity["runtime_dependency_sha256"])
        for identity in group_identities.values()
    }
    if dependency_digests != {str(gate["frozen_inputs"]["runtime_dependency_sha256"])}:  # type: ignore[index]
        raise EvidenceError("runtime dependency identity changed across comparison groups")
    final_identity = group_identities["diffraction"]
    implementation = {
        **initial,
        "group_identities": group_identities,
        "final_build_fingerprint": final_identity["candidate_build_fingerprint"],
        "final_extension_sha256": final_identity["candidate_extension_sha256"],
    }
    ncu = run_ncu_candidate(runtime_config, timeout_seconds=timeout_seconds, store=store)
    release = run_release_evidence(
        runtime_config,
        gate,
        implementation=implementation,
        timeout_seconds=timeout_seconds,
        store=store,
    )
    final = verify_checkouts(runtime_config, gate)
    if final["groups"] != initial["groups"] or final["rayd_commit"] != initial["rayd_commit"]:
        raise EvidenceError("checkout identity changed during Phase 12 capture")
    implementation["post_capture_reverified"] = True
    gate_checks.extend(
        [
            {"name": "profiler:nsight_compute", "passed": ncu["status"] in {"captured", "blocked"}},
            {"name": "release:all", "passed": True},
        ]
    )
    report = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "status": "measured",
        "release_claim": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_policy": _gate_policy(),
        "implementation": implementation,
        "channel_builds": channel_builds,
        "measurement_policy": gate["measurement_policy"],
        "munich": gate["munich"],
        "groups": group_reports,
        "nsight_compute": ncu,
        "release": release,
        "raw_artifacts": {"bundle_id": store.root.name, "inventory": store.inventory()},
        "gate": {"status": "passed", "passed": True, "checks": gate_checks},
    }
    return report


def _walk_artifact_references(value: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if isinstance(value, dict):
        if set(value) == {"path", "sha256", "bytes"}:
            rows.append(value)
        else:
            for item in value.values():
                rows.extend(_walk_artifact_references(item))
    elif isinstance(value, list):
        for item in value:
            rows.extend(_walk_artifact_references(item))
    return rows


def _artifact_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    return str(row["path"]), str(row["sha256"]), int(row["bytes"])


def _replay_worker_pairs(
    reported_pairs: object,
    gate: Mapping[str, object],
    store: ArtifactStore,
    *,
    group: str,
) -> list[dict[str, object]]:
    if not isinstance(reported_pairs, list):
        raise EvidenceError(f"{group} process_pairs must be an array")
    schedule = process_schedule(gate, group)
    if len(reported_pairs) != len(schedule):
        raise EvidenceError(f"{group} process pair count is not canonical")
    replayed: list[dict[str, object]] = []
    for expected, pair in zip(schedule, reported_pairs, strict=True):
        if not isinstance(pair, dict):
            raise EvidenceError("measured process pair must be an object")
        row: dict[str, object] = {
            "group": group,
            "process_index": expected["process_index"],
            "order": expected["order"],
        }
        for variant in ("baseline", "candidate"):
            reported = pair[variant]
            if not isinstance(reported, dict):
                raise EvidenceError("reported worker must be an object")
            capture = reported.get("capture")
            identity_capture = reported.get("identity_capture")
            identity_summary = reported.get("identity")
            if not isinstance(capture, dict) or not isinstance(identity_capture, dict):
                raise EvidenceError("worker/identity capture is malformed")
            validate_retained_worker_capture(
                capture,
                identity_capture=identity_capture,
                expected_group=group,
                expected_variant=variant,
                process_index=int(expected["process_index"]),
                order=str(expected["order"]),
            )
            worker = parse_worker_stdout(
                store.read_verified(
                    capture["stdout_artifact"], label=f"{group} {variant} stdout"
                )
            )
            identity = validate_retained_identity(
                parse_identity_stdout(
                    store.read_verified(
                        identity_capture["stdout_artifact"],
                        label=f"{group} {variant} identity stdout",
                    )
                ),
                capture=identity_capture,
                identity=identity_summary,
                store=store,
            )
            worker["identity"] = identity
            worker["identity_capture"] = identity_capture
            worker["capture"] = capture
            parsed = validate_worker_record(
                worker,
                expected_group=group,
                expected_variant=variant,
                process_index=int(expected["process_index"]),
                order=str(expected["order"]),
                gate=gate,
            )
            parsed["capture"] = capture
            parsed["identity_capture"] = identity_capture
            reported_base = dict(reported)
            reported_base.pop("profile_timings", None)
            if parsed != reported_base:
                raise EvidenceError("reported worker row differs from retained raw stdout")
            row[variant] = parsed
        replayed.append(row)
    return replayed


def _replay_nsys(
    section: Mapping[str, object],
    gate: Mapping[str, object],
    store: ArtifactStore,
    *,
    group: str,
) -> dict[str, object]:
    captures = section.get("captures")
    if not isinstance(captures, list):
        raise EvidenceError(f"{group} Nsight captures are malformed")
    for capture in captures:
        if not isinstance(capture, dict) or not isinstance(capture.get("sqlite"), dict):
            raise EvidenceError("Nsight capture lacks retained SQLite")
        sqlite_ref = capture["sqlite"]
        store.verify_reference(sqlite_ref, label="Nsight SQLite before replay")
        sqlite_path = store.root / str(sqlite_ref["path"])
        if parse_nsys_sqlite(sqlite_path) != capture.get("timeline"):
            raise EvidenceError("Nsight timeline differs from retained SQLite")
        store.verify_reference(sqlite_ref, label="Nsight SQLite after replay")
    checks = validate_nsys_timelines(captures, gate, group=group)
    if checks != section.get("checks") or section.get("passed") is not all(
        row["passed"] for row in checks
    ):
        raise EvidenceError(f"{group} Nsight checks do not replay")
    return dict(section)


def _replay_ncu(section: object, store: ArtifactStore) -> dict[str, object]:
    if (
        not isinstance(section, dict)
        or set(section) != {"status", "tool", "targets"}
        or not isinstance(section.get("targets"), list)
    ):
        raise EvidenceError("Nsight Compute section is malformed")
    targets = section["targets"]
    if len(targets) != 3:
        raise EvidenceError("Nsight Compute must contain three fixed candidate targets")
    manifest_groups = load_profile_contract()["groups"]
    expected = {
        group: (
            str(manifest_groups[group]["scenario"]),
            list(
                manifest_groups[group]["variants"]["candidate"]["ncu_kernel_families"]  # type: ignore[index]
            ),
        )
        for group in COMPARISON_GROUPS
    }
    seen: set[str] = set()
    for row in targets:
        if not isinstance(row, dict) or not isinstance(row.get("capture"), dict):
            raise EvidenceError("Nsight Compute target row is malformed")
        group = str(row.get("group"))
        if group in seen or group not in expected:
            raise EvidenceError("Nsight Compute target group set is not canonical")
        seen.add(group)
        scenario, kernels = expected[group]
        if row.get("scenario") != scenario or row.get("required_kernels") != kernels:
            raise EvidenceError("Nsight Compute target scenario/kernel contract differs")
        capture = row["capture"]
        if row.get("status") == "blocked":
            validate_ncu_blocker(
                store.read_verified(capture["stdout_artifact"], label="NCU stdout", allow_empty=True),
                store.read_verified(capture["stderr_artifact"], label="NCU stderr", allow_empty=True),
            )
            if row.get("report") is not None or row.get("csv") is not None or row.get("metrics") is not None:
                raise EvidenceError("blocked NCU target claims output artifacts")
        elif row.get("status") == "captured":
            csv_ref = row.get("csv")
            report_ref = row.get("report")
            if not isinstance(csv_ref, dict) or not isinstance(report_ref, dict):
                raise EvidenceError("successful NCU target lacks retained outputs")
            store.verify_reference(report_ref, label="NCU report")
            store.verify_reference(csv_ref, label="NCU CSV before replay")
            replayed = parse_ncu_csv(
                store.root / str(csv_ref["path"]), required_kernels=set(kernels)
            )
            store.verify_reference(csv_ref, label="NCU CSV after replay")
            if replayed != row.get("metrics"):
                raise EvidenceError("NCU summary differs from retained CSV")
        else:
            raise EvidenceError("Nsight Compute target status is not accepted")
    if seen != set(expected) or {row.get("status") for row in targets} != {section.get("status")}:
        raise EvidenceError("Nsight Compute target status/group set is inconsistent")
    if section.get("status") not in {"captured", "blocked"}:
        raise EvidenceError("Nsight Compute status is not accepted")
    return dict(section)


def replay_measured_report(
    report: Mapping[str, object], *, raw_root: Path, repository: Path = ROOT
) -> dict[str, object]:
    gate = load_gate(DEFAULT_GATE, measured=True)
    require_measured_policy_ready(gate)
    exact_keys(
        report,
        {
            "schema", "status", "release_claim", "generated_at_utc", "gate_policy",
            "implementation", "measurement_policy", "munich", "groups",
            "channel_builds", "nsight_compute", "release", "raw_artifacts", "gate",
        },
        label="measured report",
    )
    if report["schema"] != {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}:
        raise EvidenceError("measured report schema identity is not accepted")
    if report["status"] != "measured" or report["release_claim"] is not True:
        raise EvidenceError("report does not make a measured release claim")
    if report["gate_policy"] != _gate_policy():
        raise EvidenceError("report is not bound to the frozen gate")
    raw = report["raw_artifacts"]
    if not isinstance(raw, dict) or set(raw) != {"bundle_id", "inventory"}:
        raise EvidenceError("raw artifact envelope is malformed")
    store = ArtifactStore.open_existing(raw_root)
    if raw_root.name != raw["bundle_id"]:
        raise EvidenceError("raw artifact bundle id differs")
    store.verify_inventory(raw["inventory"])
    without_inventory = dict(report)
    without_inventory["raw_artifacts"] = {"bundle_id": raw["bundle_id"]}
    references = _walk_artifact_references(without_inventory)
    inventory = raw["inventory"]
    assert isinstance(inventory, list)
    reference_keys = {_artifact_key(row) for row in references}
    inventory_keys = {_artifact_key(row) for row in inventory if isinstance(row, dict)}
    if len(inventory_keys) != len(inventory) or reference_keys != inventory_keys:
        raise EvidenceError("raw inventory and report artifact-reference set must match exactly")
    for reference in references:
        store.verify_reference(reference, label="reported artifact", allow_empty=True)

    groups = report["groups"]
    if not isinstance(groups, dict) or set(groups) != set(COMPARISON_GROUPS):
        raise EvidenceError("measured report comparison group set is not canonical")
    implementation = report["implementation"]
    if not isinstance(implementation, dict):
        raise EvidenceError("implementation identity is malformed")
    validate_channel_build_records(
        report["channel_builds"], implementation=implementation, store=store
    )
    replayed_compiler_resources = compiler_resource_checks(report["channel_builds"], gate)
    if replayed_compiler_resources != report["channel_builds"]["compiler_resource_checks"]:  # type: ignore[index]
        raise EvidenceError("compiler resource gates do not replay")
    retained_inputs = implementation.get("retained_inputs")
    gate_reference = retained_inputs.get("gate") if isinstance(retained_inputs, dict) else None
    if not isinstance(gate_reference, dict):
        raise EvidenceError("implementation lacks retained frozen gate bytes")
    retained_gate = store.verify_reference(gate_reference, label="frozen gate")
    if (
        retained_gate["sha256"] != implementation.get("gate_source_sha256")
        or retained_gate["sha256"] != report["gate_policy"]["sha256"]  # type: ignore[index]
    ):
        raise EvidenceError("retained gate, checkout gate, and gate policy differ")
    profile_reference = retained_inputs.get("profile_contract")  # type: ignore[union-attr]
    if not isinstance(profile_reference, dict):
        raise EvidenceError("implementation lacks retained profile contract")
    retained_profile = store.verify_reference(profile_reference, label="profile contract")
    if (
        retained_profile["sha256"] != implementation.get("profile_contract_sha256")
        or retained_profile["sha256"] != gate["frozen_inputs"]["profile_contract_sha256"]  # type: ignore[index]
    ):
        raise EvidenceError("retained profile contract differs from frozen identity")
    diagnostic_reference = retained_inputs.get("diagnostic_contract")  # type: ignore[union-attr]
    if not isinstance(diagnostic_reference, dict):
        raise EvidenceError("implementation lacks retained diagnostic contract")
    retained_diagnostic = store.verify_reference(
        diagnostic_reference, label="diagnostic contract"
    )
    if (
        retained_diagnostic["sha256"] != implementation.get("diagnostic_contract_sha256")
        or retained_diagnostic["sha256"]
        != gate["frozen_inputs"]["diagnostic_contract_sha256"]  # type: ignore[index]
    ):
        raise EvidenceError("retained diagnostic contract differs from frozen identity")
    recomputed_checks: list[dict[str, object]] = []
    for group in COMPARISON_GROUPS:
        section = groups[group]
        if not isinstance(section, dict):
            raise EvidenceError(f"{group} report is malformed")
        exact_keys(
            section,
            {
                "process_pairs", "comparisons", "correctness", "resources",
                "nsight_systems", "diagnostics", "checks",
            },
            label=f"report.groups.{group}",
        )
        pairs = _replay_worker_pairs(section["process_pairs"], gate, store, group=group)
        nsys = _replay_nsys(section["nsight_systems"], gate, store, group=group)  # type: ignore[arg-type]
        diagnostics = replay_diagnostics(
            section["diagnostics"], gate, group=group, store=store
        )
        pairs = attach_profile_timings(
            pairs, nsys["captures"], gate, group=group  # type: ignore[arg-type]
        )
        if pairs != section["process_pairs"]:
            raise EvidenceError(f"{group} profile timings differ from retained SQLite")
        identity = validate_process_identity(pairs, implementation, gate)
        if identity != implementation["group_identities"][group]:  # type: ignore[index]
            raise EvidenceError(f"{group} process identity summary does not replay")
        comparisons = compare_timings(pairs, gate, bootstrap_resamples=100000)
        correctness = validate_correctness(pairs, gate)
        resources = summarize_resources(pairs, gate)
        if (
            comparisons != section["comparisons"]
            or correctness != section["correctness"]
            or resources != section["resources"]
        ):
            raise EvidenceError(f"{group} derived evidence differs from raw replay")
        checks = _group_checks(
            group,
            comparisons=comparisons,
            correctness=correctness,
            resources=resources,
            nsys=nsys,
            diagnostics=diagnostics,
            compiler_resources=replayed_compiler_resources[group],
        )
        if checks != section["checks"]:
            raise EvidenceError(f"{group} gate checks differ from replay")
        recomputed_checks.extend(checks)
    ncu = _replay_ncu(report["nsight_compute"], store)
    release = report["release"]
    if not isinstance(release, dict):
        raise EvidenceError("release section is malformed")
    validate_release_report(release, implementation=implementation, gate=gate, store=store)
    recomputed_checks.extend(
        [
            {"name": "profiler:nsight_compute", "passed": ncu["status"] in {"captured", "blocked"}},
            {"name": "release:all", "passed": True},
        ]
    )
    gate_section = report["gate"]
    if not isinstance(gate_section, dict) or gate_section != {
        "status": "passed", "passed": True, "checks": recomputed_checks
    }:
        raise EvidenceError("top-level gate differs from complete semantic replay")
    retained = implementation.get("retained_inputs")
    if not isinstance(retained, dict):
        raise EvidenceError("implementation lacks retained tool identities")
    executables = retained.get("executables")
    tools = executables.get("tools") if isinstance(executables, dict) else None
    git_reference = tools.get("git") if isinstance(tools, dict) else None
    if not isinstance(git_reference, dict):
        raise EvidenceError("implementation lacks retained Git executable")
    store.verify_reference(git_reference, label="Git executable")
    identities = implementation.get("tool_executable_identity")
    git_identity = identities.get("git") if isinstance(identities, dict) else None
    if (
        not isinstance(git_identity, dict)
        or git_identity.get("sha256") != git_reference.get("sha256")
        or not isinstance(git_identity.get("path"), str)
        or sha256_path(Path(git_identity["path"])) != git_reference.get("sha256")
    ):
        raise EvidenceError("current Git executable differs from retained measured bytes")
    verify_evidence_commit_binding(
        repository,
        implementation,
        git_executable=Path(git_identity["path"]),
        frozen_runner_hashes=gate["frozen_inputs"]["runner_blob_sha256"],  # type: ignore[index]
    )
    return dict(report)


def write_report(report: Mapping[str, object], output: Path) -> None:
    if output.exists():
        raise EvidenceError(f"refusing to overwrite evidence report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    except OSError as exc:
        raise EvidenceError(f"cannot write evidence report {output}: {exc}") from exc


__all__ = [
    "build_dry_run", "build_measured_report", "replay_measured_report", "write_report"
]