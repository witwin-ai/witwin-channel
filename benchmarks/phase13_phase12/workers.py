"""Canonical worker execution and source/build identity verification."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .artifacts import ArtifactStore, hash_external_stable, read_external_stable
from .contracts import (
    COMPARISON_GROUPS,
    EvidenceError,
    IDENTITY_SCHEMA_NAME,
    IDENTITY_SCHEMA_VERSION,
    RunnerConfig,
    VariantConfig,
    WORKER_SCHEMA_NAME,
    controlled_environment,
    process_schedule,
    sanitized_subprocess_environment,
    strict_object,
)


WORKER_REPO_PATH = Path("benchmarks/phase13_phase12_worker.py")
DIAGNOSTIC_REPO_PATH = Path("benchmarks/phase13_phase12_diagnostics.py")
IDENTITY_PROBE_REPO_PATH = Path("benchmarks/phase13_phase12_identity_probe.py")
BOOTSTRAP_REPO_PATH = Path("benchmarks/phase13_phase12_bootstrap.py")
RAYD_LOCK_REPO_PATH = Path("dependencies/rayd.lock.json")
DIFFRACTION_ROUTE_REPO_PATH = Path(
    "src/witwin/channel/propagation/enumerated/diffraction.py"
)
NATIVE_MANIFEST_REPO_PATH = Path("ci/native-binding-manifest.json")
RUNNER_REPO_PATHS = (
    Path("benchmarks/phase13_phase12/__init__.py"),
    Path("benchmarks/phase13_phase12/artifacts.py"),
    Path("benchmarks/phase13_phase12/builds.py"),
    Path("benchmarks/phase13_phase12/contracts.py"),
    Path("benchmarks/phase13_phase12/diagnostics.py"),
    Path("benchmarks/phase13_phase12/profilers.py"),
    Path("benchmarks/phase13_phase12/release.py"),
    Path("benchmarks/phase13_phase12/report.py"),
    Path("benchmarks/phase13_phase12/statistics.py"),
    Path("benchmarks/phase13_phase12/workers.py"),
    WORKER_REPO_PATH,
    DIAGNOSTIC_REPO_PATH,
    IDENTITY_PROBE_REPO_PATH,
    BOOTSTRAP_REPO_PATH,
    Path("benchmarks/phase13_phase12_evidence.py"),
    Path("benchmarks/schemas/phase13-phase12-evidence.schema.json"),
    Path("ci/check_phase13_phase12_evidence.py"),
    Path("ci/wheel_smoke.py"),
    Path("tools/phase13_phase12_evidence.py"),
)


def _clean_git_environment(git_executable: Path) -> dict[str, str]:
    """Return a minimal environment that cannot redirect repository identity."""
    return sanitized_subprocess_environment(
        runtime_search_paths=(git_executable.resolve().parent,)
    )


def _git_argv(
    git_executable: Path, checkout: Path, *arguments: str
) -> list[str]:
    return [
        str(git_executable),
        "-c",
        f"safe.directory={checkout.resolve()}",
        *arguments,
    ]


def sha256_path(path: Path) -> str:
    return hash_external_stable(path, label="hashed file", allow_empty=False)[0]


def executable_identity(path: Path, *, label: str) -> dict[str, object]:
    resolved = path.resolve()
    digest, size = hash_external_stable(
        resolved, label=f"{label} executable", allow_empty=False
    )
    return {
        "path": str(resolved),
        "name": resolved.name,
        "sha256": digest,
        "bytes": size,
    }


def _git(
    git_executable: Path, checkout: Path, *arguments: str, binary: bool = False
) -> str | bytes:
    try:
        completed = subprocess.run(
            _git_argv(git_executable, checkout, *arguments),
            cwd=checkout, capture_output=True,
            text=not binary, encoding=None if binary else "utf-8",
            errors=None if binary else "replace", check=False, timeout=60,
            env=_clean_git_environment(git_executable),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvidenceError(f"git {' '.join(arguments)} failed in {checkout}: {exc}") from exc
    if completed.returncode:
        stderr = completed.stderr
        detail = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
        raise EvidenceError(f"git {' '.join(arguments)} failed in {checkout}: {detail.strip()}")
    if binary:
        return bytes(completed.stdout)
    return str(completed.stdout).strip()


def _verify_repo_file_at_head(
    git_executable: Path, checkout: Path, relative: Path
) -> str:
    working = checkout / relative
    if not working.is_file():
        raise EvidenceError(f"canonical repository file is missing: {working}")
    blob = _git(git_executable, checkout, "show", f"HEAD:{relative.as_posix()}", binary=True)
    assert isinstance(blob, bytes)
    actual = read_external_stable(
        working, label=f"repository file {relative}", allow_empty=False
    )[0]
    if blob != actual:
        raise EvidenceError(f"canonical repository file differs from HEAD: {working}")
    return hashlib.sha256(actual).hexdigest()


def _checkout_state(
    git_executable: Path, checkout: Path, *, label: str, require_clean: bool = True
) -> dict[str, object]:
    if not checkout.is_dir():
        raise EvidenceError(f"{label} checkout does not exist: {checkout}")
    status = _git(git_executable, checkout, "status", "--porcelain=v1", "--untracked-files=all")
    if require_clean and status:
        raise EvidenceError(f"{label} checkout must be clean")
    return {
        "head": _git(git_executable, checkout, "rev-parse", "HEAD"),
        "toplevel": _git(git_executable, checkout, "rev-parse", "--show-toplevel"),
        "clean": not bool(status),
    }


def _lock_identity(checkout: Path) -> dict[str, object]:
    path = checkout / RAYD_LOCK_REPO_PATH
    payload, digest, _ = read_external_stable(path, label="RayD lock")
    try:
        lock = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EvidenceError(f"RayD lock contains non-finite JSON: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"RayD lock is not strict JSON: {path}: {exc}") from exc
    if set(lock) != {"schema_version", "repository_url", "commit", "integration_abi"}:
        raise EvidenceError(f"RayD lock has an unexpected schema: {path}")
    integration = lock["integration_abi"]
    if not isinstance(integration, dict) or set(integration) != {"kind", "path", "sha256"}:
        raise EvidenceError(f"RayD integration ABI lock is malformed: {path}")
    return {
        "lock_sha256": digest,
        "repository_url": lock["repository_url"],
        "rayd_commit": lock["commit"],
        "integration_header_path": integration["path"],
        "integration_header_sha256": integration["sha256"],
    }


def verify_checkouts(
    config: RunnerConfig,
    gate: Mapping[str, object] | None = None,
    *,
    store: ArtifactStore | None = None,
    require_clean_rayd: bool = True,
) -> dict[str, object]:
    states: dict[str, dict[str, object]] = {}
    locks: dict[str, dict[str, object]] = {}
    parents: dict[str, str] = {}
    for group in COMPARISON_GROUPS:
        for name in ("baseline", "candidate"):
            key = f"{group}:{name}"
            variant = config.variant(group, name)
            states[key] = _checkout_state(config.tools.git, variant.checkout, label=key)
            locks[key] = _lock_identity(variant.checkout)
        candidate = config.variant(group, "candidate")
        parents[group] = str(_git(config.tools.git, candidate.checkout, "rev-parse", "HEAD^"))
        if parents[group] != states[f"{group}:baseline"]["head"]:
            raise EvidenceError(f"{group} candidate must be its baseline's direct child")
    if states["montecarlo_penetration:baseline"]["head"] != states[
        "enumerated_penetration:candidate"
    ]["head"]:
        raise EvidenceError("Monte Carlo baseline must equal enumerated candidate")
    diffraction_parent = str(
        _git(
            config.tools.git,
            config.variant("diffraction", "baseline").checkout,
            "rev-parse", "HEAD^",
        )
    )
    if diffraction_parent != states["montecarlo_penetration:candidate"]["head"]:
        raise EvidenceError(
            "diffraction dormant baseline must directly follow the Monte Carlo switch"
        )
    rayd_state = _checkout_state(
        config.tools.git, config.rayd_checkout, label="RayD",
        require_clean=require_clean_rayd,
    )
    canonical_lock = next(iter(locks.values()))
    if any(lock != canonical_lock for lock in locks.values()):
        raise EvidenceError("all comparison commits must use one exact RayD lock/header")
    if rayd_state["head"] != canonical_lock["rayd_commit"]:
        raise EvidenceError("RayD checkout HEAD differs from the Channel dependency lock")
    header_relative = Path(str(canonical_lock["integration_header_path"]))
    header_blob = _git(
        config.tools.git, config.rayd_checkout,
        "show", f"HEAD:{header_relative.as_posix()}", binary=True,
    )
    assert isinstance(header_blob, bytes)
    if hashlib.sha256(header_blob).hexdigest() != canonical_lock["integration_header_sha256"]:
        raise EvidenceError("RayD integration header bytes differ from the Channel lock")
    rayd_cmake_blob = _git(
        config.tools.git, config.rayd_checkout,
        "show", "HEAD:backends/torch/CMakeLists.txt", binary=True,
    )
    rayd_build_script_blob = _git(
        config.tools.git, config.rayd_checkout,
        "show", "HEAD:scripts/build_local.ps1", binary=True,
    )
    assert isinstance(rayd_cmake_blob, bytes) and isinstance(rayd_build_script_blob, bytes)
    rayd_cmake_source_sha256 = hashlib.sha256(rayd_cmake_blob).hexdigest()
    rayd_build_script_sha256 = hashlib.sha256(rayd_build_script_blob).hexdigest()
    scene_sha256 = sha256_path(config.datasets.munich_scene_xml)
    if gate is not None:
        frozen_inputs = gate.get("frozen_inputs")
        if not isinstance(frozen_inputs, dict) or scene_sha256 != frozen_inputs.get(
            "munich_scene_xml_sha256"
        ):
            raise EvidenceError("Munich scene XML differs from the canonical evidence input")
    runner_source_hashes: dict[str, dict[str, str]] = {}
    for relative in RUNNER_REPO_PATHS:
        values = {
            key: _verify_repo_file_at_head(
                config.tools.git, config.variant(*key.split(":")), relative
            )
            for key in states
        }
        if len(set(values.values())) != 1:
            raise EvidenceError(
                f"comparison commits do not use identical runner bytes: {relative}"
            )
        runner_source_hashes[relative.as_posix()] = values
    gate_relative = Path("benchmarks/gates/phase13_phase12.json")
    gate_hashes = {
        key: _verify_repo_file_at_head(
            config.tools.git, config.variant(*key.split(":")), gate_relative
        )
        for key in states
    }
    if len(set(gate_hashes.values())) != 1:
        raise EvidenceError("comparison commits do not use identical frozen gate bytes")
    gate_source_sha256 = next(iter(gate_hashes.values()))
    profile_contract_relative = Path("benchmarks/phase13_phase12_profile_contract.json")
    profile_contract_hashes = {
        key: _verify_repo_file_at_head(
            config.tools.git, config.variant(*key.split(":")), profile_contract_relative
        )
        for key in states
    }
    if len(set(profile_contract_hashes.values())) != 1:
        raise EvidenceError("comparison commits do not use identical profile contract bytes")
    profile_contract_sha256 = next(iter(profile_contract_hashes.values()))
    if gate is not None and gate["frozen_inputs"].get("profile_contract_sha256") != profile_contract_sha256:  # type: ignore[union-attr]
        raise EvidenceError("profile contract bytes differ from the frozen gate")
    diagnostic_contract_relative = Path(
        "benchmarks/phase13_phase12_diagnostic_contract.json"
    )
    diagnostic_contract_hashes = {
        key: _verify_repo_file_at_head(
            config.tools.git, config.variant(*key.split(":")), diagnostic_contract_relative
        )
        for key in states
    }
    if len(set(diagnostic_contract_hashes.values())) != 1:
        raise EvidenceError("comparison commits do not use identical diagnostic contract bytes")
    diagnostic_contract_sha256 = next(iter(diagnostic_contract_hashes.values()))
    if gate is not None and gate["frozen_inputs"].get(
        "diagnostic_contract_sha256"
    ) != diagnostic_contract_sha256:  # type: ignore[union-attr]
        raise EvidenceError("diagnostic contract bytes differ from the frozen gate")
    python_hashes = {
        key: sha256_path(config.variant(*key.split(":")).python_executable)
        for key in states
    }
    if len(set(python_hashes.values())) != 1:
        raise EvidenceError("comparison groups do not use identical Python executable bytes")
    tool_identities = {
        name: executable_identity(getattr(config.tools, name), label=name)
        for name in (
            "nsys", "ncu", "conda", "ctest", "cmake", "ninja", "cuobjdump",
            "dumpbin", "powershell", "cmd", "vcvars64", "git",
            "nvcc", "cl", "link", "nvidia_smi",
        )
    }
    tool_hashes = {name: row["sha256"] for name, row in tool_identities.items()}
    if gate is not None:
        frozen = gate["frozen_inputs"]
        assert isinstance(frozen, dict)
        expected_runner = frozen.get("runner_blob_sha256")
        actual_runner = {
            name: next(iter(values.values()))
            for name, values in runner_source_hashes.items()
        }
        if expected_runner != actual_runner:
            raise EvidenceError("runner HEAD blobs differ from frozen gate inputs")
        if frozen.get("python_executable_sha256") != next(iter(python_hashes.values())):
            raise EvidenceError("Python executable differs from frozen gate input")
        if frozen.get("tool_executable_sha256") != tool_hashes:
            raise EvidenceError("tool executable bytes differ from frozen gate inputs")
    route_audit = {
        group: verify_route_transition(
            group,
            config.variant(group, "baseline").checkout,
            config.variant(group, "candidate").checkout,
            git_executable=config.tools.git,
        )
        for group in COMPARISON_GROUPS
    }
    retained: dict[str, object] = {}
    if store is not None:
        for relative in RUNNER_REPO_PATHS:
            checkout = config.variant("enumerated_penetration", "baseline").checkout
            blob = _git(
                config.tools.git, checkout, "show", f"HEAD:{relative.as_posix()}",
                binary=True,
            )
            assert isinstance(blob, bytes)
            retained[f"blob:{relative.as_posix()}"] = store.write_bytes(
                f"inputs/head-blobs/{relative.as_posix()}", blob, allow_empty=False
            )
        retained["rayd_lock"] = store.retain_external(
            config.variant("enumerated_penetration", "baseline").checkout / RAYD_LOCK_REPO_PATH,
            "inputs/rayd.lock.json", label="RayD lock", allow_empty=False,
        )
        retained["integration_header"] = store.write_bytes(
            "inputs/rayd-integration.h", header_blob, allow_empty=False,
        )
        retained["munich_scene"] = store.retain_external(
            config.datasets.munich_scene_xml, "inputs/munich.xml",
            label="Munich scene XML", allow_empty=False,
        )
        retained["gate"] = store.retain_external(
            config.variant("enumerated_penetration", "baseline").checkout / gate_relative,
            "inputs/phase13_phase12_gate.json",
            label="frozen gate",
            allow_empty=False,
        )
        retained["profile_contract"] = store.retain_external(
            config.variant("enumerated_penetration", "baseline").checkout
            / profile_contract_relative,
            "inputs/phase13_phase12_profile_contract.json",
            label="profile contract",
            allow_empty=False,
        )
        retained["diagnostic_contract"] = store.retain_external(
            config.variant("enumerated_penetration", "baseline").checkout
            / diagnostic_contract_relative,
            "inputs/phase13_phase12_diagnostic_contract.json",
            label="diagnostic contract", allow_empty=False,
        )
        retained["executables"] = {
            "python": {
                key: store.retain_external(
                    config.variant(*key.split(":")).python_executable,
                    f"inputs/executables/python-{key.replace(':', '-')}.exe",
                    label=f"{key} Python executable", allow_empty=False,
                )
                for key in states
            },
            "tools": {
                name: store.retain_external(
                    getattr(config.tools, name),
                    f"inputs/executables/{name}-{getattr(config.tools, name).name}",
                    label=name, allow_empty=False,
                )
                for name in (
                    "nsys", "ncu", "conda", "ctest", "cmake", "ninja", "cuobjdump",
                    "dumpbin", "powershell", "cmd", "vcvars64", "git",
                    "nvcc", "cl", "link", "nvidia_smi",
                )
            },
        }
        route_blobs: dict[str, object] = {}
        for group, audit in route_audit.items():
            assert isinstance(audit, dict)
            for relative in audit["changed_tracked_files"]:
                if not (
                    str(relative).startswith(("src/", "native/", "ci/", "cmake/"))
                    or relative == "CMakeLists.txt"
                ):
                    continue
                for name in ("baseline", "candidate"):
                    checkout = config.variant(group, name).checkout
                    listed = str(
                        _git(
                            config.tools.git, checkout, "ls-tree", "-r", "--name-only",
                            "HEAD", "--", str(relative),
                        )
                    ).splitlines()
                    if str(relative) not in listed:
                        continue
                    blob = _git(
                        config.tools.git, checkout, "show", f"HEAD:{relative}", binary=True
                    )
                    assert isinstance(blob, bytes)
                    key = f"{group}:{name}:{relative}"
                    route_blobs[key] = store.write_bytes(
                        f"inputs/routes/{group}/{name}/{relative}", blob,
                        allow_empty=False,
                    )
        retained["route_blobs"] = route_blobs
    return {
        "groups": {
            group: {
                "baseline_commit": states[f"{group}:baseline"]["head"],
                "candidate_commit": states[f"{group}:candidate"]["head"],
                "candidate_parent": parents[group],
            }
            for group in COMPARISON_GROUPS
        },
        "rayd_commit": canonical_lock["rayd_commit"],
        "rayd_repository_url": canonical_lock["repository_url"],
        "rayd_lock_sha256": canonical_lock["lock_sha256"],
        "integration_header_sha256": canonical_lock["integration_header_sha256"],
        "rayd_cmake_source_sha256": rayd_cmake_source_sha256,
        "rayd_build_script_sha256": rayd_build_script_sha256,
        "runner_source_sha256": {
            name: next(iter(values.values())) for name, values in runner_source_hashes.items()
        },
        "gate_source_sha256": gate_source_sha256,
        "profile_contract_sha256": profile_contract_sha256,
        "diagnostic_contract_sha256": diagnostic_contract_sha256,
        "python_executable_sha256": next(iter(python_hashes.values())),
        "tool_executable_sha256": tool_hashes,
        "tool_executable_identity": tool_identities,
        "munich_scene_xml_sha256": scene_sha256,
        "route_audit": route_audit,
        "retained_inputs": retained,
        "direct_parent_verified": True,
        "clean_channel_checkouts_verified": True,
        "clean_rayd_checkout_verified": bool(rayd_state["clean"]),
    }


_PRODUCTION_ROUTE_PATHS = ("src", "native", "ci", "cmake", "CMakeLists.txt")
_ROUTE_POLICY = {
    "enumerated_penetration": {
        "old": (
            "TransmissionClosestHitQuery",
            "query_transmission_closest_hit",
            "iter_transmission_active_rows",
        ),
        "new": "segment_penetration_forward",
        "owner_fragment": "propagation/enumerated",
    },
    "montecarlo_penetration": {
        "old": ("incident_te_tm_fractions",),
        "stable": ("straight_transmission_chains",),
        "new": "MonteCarloTargetInset",
        "owner_fragment": "montecarlo/events/transmission.py",
    },
    "diffraction": {
        "old": ("deterministic_diffraction_order1_compact",),
        "new": "deterministic_diffraction_pair_reduce",
        "owner_fragment": "propagation/enumerated/diffraction.py",
    },
}


def _tracked_matches(git_executable: Path, checkout: Path, pattern: str) -> list[str]:
    completed = subprocess.run(
        _git_argv(
            git_executable, checkout,
            "grep", "-n", "-F", "-e", pattern, "HEAD", "--", *_PRODUCTION_ROUTE_PATHS,
        ),
        cwd=checkout, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False, timeout=60,
        env=_clean_git_environment(git_executable),
    )
    if completed.returncode not in (0, 1):
        raise EvidenceError(f"tracked route scan failed for {pattern}: {completed.stderr}")
    return sorted(
        line.removeprefix("HEAD:") for line in completed.stdout.splitlines() if line
    )


def _tracked_matches_revision(
    git_executable: Path, repository: Path, revision: str, pattern: str
) -> list[str]:
    completed = subprocess.run(
        _git_argv(
            git_executable, repository,
            "grep", "-n", "-F", "-e", pattern,
            revision, "--", *_PRODUCTION_ROUTE_PATHS,
        ),
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
        env=_clean_git_environment(git_executable),
    )
    if completed.returncode not in (0, 1):
        raise EvidenceError(f"tracked route replay failed for {pattern}: {completed.stderr}")
    prefix = revision + ":"
    return sorted(
        line.removeprefix(prefix)
        for line in completed.stdout.splitlines()
        if line
    )


def _route_transition_at_revisions(
    group: str,
    baseline: str,
    candidate: str,
    *,
    repository: Path,
    git_executable: Path,
) -> dict[str, object]:
    policy = _ROUTE_POLICY[group]
    old_rows: dict[str, dict[str, object]] = {}
    checks: dict[str, bool] = {}
    for symbol in policy["old"]:
        baseline_matches = _tracked_matches_revision(
            git_executable, repository, baseline, str(symbol)
        )
        candidate_matches = _tracked_matches_revision(
            git_executable, repository, candidate, str(symbol)
        )
        old_rows[str(symbol)] = {"baseline": baseline_matches, "candidate": candidate_matches}
        checks[f"baseline_old:{symbol}"] = bool(baseline_matches)
        checks[f"candidate_deleted:{symbol}"] = not candidate_matches
    stable_rows: dict[str, dict[str, object]] = {}
    for symbol in policy.get("stable", ()):
        baseline_matches = _tracked_matches_revision(
            git_executable, repository, baseline, str(symbol)
        )
        candidate_matches = _tracked_matches_revision(
            git_executable, repository, candidate, str(symbol)
        )
        stable_rows[str(symbol)] = {
            "baseline": baseline_matches,
            "candidate": candidate_matches,
        }
        checks[f"baseline_stable:{symbol}"] = bool(baseline_matches)
        checks[f"candidate_stable:{symbol}"] = bool(candidate_matches)
    baseline_new = _tracked_matches_revision(
        git_executable, repository, baseline, str(policy["new"])
    )
    candidate_new = _tracked_matches_revision(
        git_executable, repository, candidate, str(policy["new"])
    )
    owner = str(policy["owner_fragment"])
    baseline_callers = [line for line in baseline_new if owner in line]
    candidate_callers = [line for line in candidate_new if owner in line]
    checks["candidate_new_direct_caller"] = bool(candidate_callers)
    checks["direct_caller_activated_by_switch"] = len(candidate_callers) > len(baseline_callers)
    changed = str(
        _git(git_executable, repository, "diff", "--name-only", baseline, candidate)
    ).splitlines()
    checks["production_route_changed"] = any(
        item.startswith(("src/", "native/", "ci/", "cmake/")) or item == "CMakeLists.txt"
        for item in changed
    )
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise EvidenceError(f"{group} route replay failed: {failed}")
    return {
        "group": group,
        "old_symbol_matches": old_rows,
        "stable_symbol_matches": stable_rows,
        "new_symbol": str(policy["new"]),
        "baseline_new_matches": baseline_new,
        "candidate_new_matches": candidate_new,
        "changed_tracked_files": sorted(changed),
        "checks": checks,
        "passed": True,
    }


def verify_route_transition(
    group: str, baseline: Path, candidate: Path, *, git_executable: Path
) -> dict[str, object]:
    policy = _ROUTE_POLICY[group]
    old_rows: dict[str, dict[str, object]] = {}
    checks: dict[str, bool] = {}
    for symbol in policy["old"]:
        baseline_matches = _tracked_matches(git_executable, baseline, str(symbol))
        candidate_matches = _tracked_matches(git_executable, candidate, str(symbol))
        old_rows[str(symbol)] = {
            "baseline": baseline_matches,
            "candidate": candidate_matches,
        }
        checks[f"baseline_old:{symbol}"] = bool(baseline_matches)
        checks[f"candidate_deleted:{symbol}"] = not candidate_matches
    stable_rows: dict[str, dict[str, object]] = {}
    for symbol in policy.get("stable", ()):
        baseline_matches = _tracked_matches(git_executable, baseline, str(symbol))
        candidate_matches = _tracked_matches(git_executable, candidate, str(symbol))
        stable_rows[str(symbol)] = {
            "baseline": baseline_matches,
            "candidate": candidate_matches,
        }
        checks[f"baseline_stable:{symbol}"] = bool(baseline_matches)
        checks[f"candidate_stable:{symbol}"] = bool(candidate_matches)
    baseline_new = _tracked_matches(git_executable, baseline, str(policy["new"]))
    candidate_new = _tracked_matches(git_executable, candidate, str(policy["new"]))
    owner = str(policy["owner_fragment"])
    baseline_callers = [line for line in baseline_new if owner in line]
    candidate_callers = [line for line in candidate_new if owner in line]
    checks["candidate_new_direct_caller"] = bool(candidate_callers)
    checks["direct_caller_activated_by_switch"] = len(candidate_callers) > len(baseline_callers)
    changed = str(
        _git(
            git_executable,
            candidate,
            "diff", "--name-only",
            str(_git(git_executable, baseline, "rev-parse", "HEAD")), "HEAD",
        )
    ).splitlines()
    checks["production_route_changed"] = any(
        item.startswith(("src/", "native/", "ci/", "cmake/")) or item == "CMakeLists.txt"
        for item in changed
    )
    if not all(checks.values()):
        raise EvidenceError(
            f"{group} tracked production route transition failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )
    return {
        "old_symbol_matches": old_rows,
        "stable_symbol_matches": stable_rows,
        "new_symbol": str(policy["new"]),
        "baseline_new_matches": baseline_new,
        "candidate_new_matches": candidate_new,
        "changed_tracked_files": sorted(changed),
        "checks": checks,
        "passed": True,
    }


def _canonical_runner_hash_map(
    raw: object, *, label: str
) -> dict[str, str]:
    expected_paths = {path.as_posix() for path in RUNNER_REPO_PATHS}
    if not isinstance(raw, dict) or set(raw) != expected_paths:
        raise EvidenceError(
            f"{label} must contain the complete canonical runner source hash map"
        )
    result: dict[str, str] = {}
    for relative, digest in raw.items():
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise EvidenceError(f"{label} contains a malformed SHA-256 identity")
        result[relative] = digest
    return result


def _bind_runner_hash_maps(
    reported: object, frozen: object
) -> dict[str, str]:
    reported_hashes = _canonical_runner_hash_map(
        reported, label="reported runner blob identity"
    )
    frozen_hashes = _canonical_runner_hash_map(
        frozen, label="frozen runner blob identity"
    )
    if reported_hashes != frozen_hashes:
        raise EvidenceError("reported runner blobs differ from the complete frozen hash map")
    return reported_hashes


def verify_evidence_commit_binding(
    repository: Path,
    implementation: Mapping[str, object],
    *,
    git_executable: Path,
    frozen_runner_hashes: object,
) -> dict[str, object]:
    groups = implementation.get("groups")
    runner_hashes = _bind_runner_hash_maps(
        implementation.get("runner_source_sha256"),
        frozen_runner_hashes,
    )
    if not isinstance(groups, dict) or set(groups) != set(COMPARISON_GROUPS):
        raise EvidenceError("reported implementation history is malformed")
    commits: list[str] = []
    for group in COMPARISON_GROUPS:
        history = groups[group]
        if not isinstance(history, dict):
            raise EvidenceError(f"reported history is malformed: {group}")
        baseline = str(history["baseline_commit"])
        candidate = str(history["candidate_commit"])
        parent = str(_git(git_executable, repository, "rev-parse", f"{candidate}^"))
        if parent != baseline or history.get("candidate_parent") != baseline:
            raise EvidenceError(f"{group} candidate is not the reported direct child")
        commits.extend((baseline, candidate))
    if groups["montecarlo_penetration"]["baseline_commit"] != groups["enumerated_penetration"]["candidate_commit"]:  # type: ignore[index]
        raise EvidenceError("Monte Carlo baseline does not equal enumerated candidate")
    diffraction_baseline = str(groups["diffraction"]["baseline_commit"])  # type: ignore[index]
    if str(_git(git_executable, repository, "rev-parse", f"{diffraction_baseline}^")) != groups["montecarlo_penetration"]["candidate_commit"]:  # type: ignore[index]
        raise EvidenceError("diffraction dormant baseline is not directly after Monte Carlo switch")
    unique_commits = list(dict.fromkeys(commits))
    for relative, expected_hash in runner_hashes.items():
        for commit in unique_commits:
            blob = _git(
                git_executable,
                repository,
                "show",
                f"{commit}:{relative}",
                binary=True,
            )
            assert isinstance(blob, bytes)
            if hashlib.sha256(blob).hexdigest() != expected_hash:
                raise EvidenceError(
                    f"runner blob changed within the measured history: {relative}"
                )
    gate_hash = implementation.get("gate_source_sha256")
    if not isinstance(gate_hash, str):
        raise EvidenceError("reported gate source identity is malformed")
    for commit in unique_commits:
        gate_blob = _git(
            git_executable,
            repository,
            "show",
            f"{commit}:benchmarks/gates/phase13_phase12.json",
            binary=True,
        )
        assert isinstance(gate_blob, bytes)
        if hashlib.sha256(gate_blob).hexdigest() != gate_hash:
            raise EvidenceError("frozen gate bytes changed within the measured history")
    profile_hash = implementation.get("profile_contract_sha256")
    if not isinstance(profile_hash, str):
        raise EvidenceError("reported profile contract identity is malformed")
    for commit in unique_commits:
        profile_blob = _git(
            git_executable,
            repository,
            "show",
            f"{commit}:benchmarks/phase13_phase12_profile_contract.json",
            binary=True,
        )
        assert isinstance(profile_blob, bytes)
        if hashlib.sha256(profile_blob).hexdigest() != profile_hash:
            raise EvidenceError("profile contract bytes changed within measured history")
    diagnostic_hash = implementation.get("diagnostic_contract_sha256")
    if not isinstance(diagnostic_hash, str):
        raise EvidenceError("reported diagnostic contract identity is malformed")
    for commit in unique_commits:
        diagnostic_blob = _git(
            git_executable,
            repository,
            "show",
            f"{commit}:benchmarks/phase13_phase12_diagnostic_contract.json",
            binary=True,
        )
        assert isinstance(diagnostic_blob, bytes)
        if hashlib.sha256(diagnostic_blob).hexdigest() != diagnostic_hash:
            raise EvidenceError("diagnostic contract bytes changed within measured history")
    route_audit = implementation.get("route_audit")
    if not isinstance(route_audit, dict) or set(route_audit) != set(COMPARISON_GROUPS):
        raise EvidenceError("reported route audit is malformed")
    for group in COMPARISON_GROUPS:
        history = groups[group]
        assert isinstance(history, dict)
        replayed_route = _route_transition_at_revisions(
            group,
            str(history["baseline_commit"]),
            str(history["candidate_commit"]),
            repository=repository,
            git_executable=git_executable,
        )
        if replayed_route != route_audit[group]:
            raise EvidenceError(f"{group} route audit differs from Git replay")
    current = str(_git(git_executable, repository, "rev-parse", "HEAD"))
    parent = str(_git(git_executable, repository, "rev-parse", "HEAD^"))
    final_candidate = str(groups["diffraction"]["candidate_commit"])  # type: ignore[index]
    if parent != final_candidate:
        raise EvidenceError(
            "committed evidence must be the direct child of the measured live-switch commit"
        )
    changed = str(
        _git(git_executable, repository, "diff", "--name-only", "HEAD^", "HEAD")
    ).splitlines()
    allowed_prefixes = ("benchmarks/evidence/", "docs/dev/audit/")
    if not changed or any(not item.startswith(allowed_prefixes) for item in changed):
        raise EvidenceError("evidence commit contains non-evidence implementation changes")
    return {
        "evidence_commit": current,
        "implementation_parent": parent,
        "history_replayed": True,
        "runner_blobs_replayed": True,
    }


def _parse_worker_stdout(stdout: bytes) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for raw_line in stdout.decode("utf-8", errors="strict").splitlines():
        try:
            value = json.loads(
                raw_line,
                object_pairs_hook=strict_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    EvidenceError(f"non-finite worker JSON token: {token}")
                ),
            )
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            schema = value.get("schema")
            if isinstance(schema, dict) and schema.get("name") == WORKER_SCHEMA_NAME:
                records.append(value)
    if len(records) != 1:
        raise EvidenceError(
            f"worker must emit exactly one Phase 12 JSON record; observed {len(records)}"
        )
    return records[0]


def parse_worker_stdout(stdout: bytes) -> dict[str, object]:
    """Parse exactly one strict worker record from retained stdout bytes."""
    return _parse_worker_stdout(stdout)


def parse_identity_stdout(stdout: bytes) -> dict[str, object]:
    """Parse the one runner-owned identity-probe record."""
    records: list[dict[str, object]] = []
    for raw_line in stdout.decode("utf-8", errors="strict").splitlines():
        try:
            value = json.loads(
                raw_line,
                object_pairs_hook=strict_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    EvidenceError(f"non-finite identity JSON token: {token}")
                ),
            )
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == {
            "name": IDENTITY_SCHEMA_NAME,
            "version": IDENTITY_SCHEMA_VERSION,
        }:
            records.append(value)
    if len(records) != 1:
        raise EvidenceError(
            f"identity probe must emit exactly one JSON record; observed {len(records)}"
        )
    return records[0]


def identity_probe_argv(variant: VariantConfig) -> list[str]:
    if variant.runner_site_packages is None or variant.runner_extension is None:
        raise EvidenceError("identity probe requires a runner-built Channel installation")
    return [
        str(variant.python_executable),
        "-I",
        str(variant.checkout / BOOTSTRAP_REPO_PATH),
        "--site-packages",
        str(variant.runner_site_packages),
        "--script",
        str(variant.checkout / IDENTITY_PROBE_REPO_PATH),
    ]


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_captured(
    argv: list[str], *, cwd: Path, environment: Mapping[str, str], timeout_seconds: int,
    store: ArtifactStore, stem: str, expected_returncode: int | None = 0,
) -> dict[str, object]:
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    started_time_ns = time.time_ns()
    process = subprocess.Popen(
        argv, cwd=cwd, env=dict(environment), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        _terminate_process_tree(process)
        try:
            remaining_stdout, remaining_stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                remaining_stdout, remaining_stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired as final:
                remaining_stdout = final.output or exc.output or b""
                remaining_stderr = final.stderr or exc.stderr or b""
        # CPython may return the complete buffered stream from the second
        # communicate call.  Prefer that stream so retained evidence cannot
        # duplicate the timeout prefix.
        stdout = bytes(remaining_stdout or exc.output or b"")
        stderr = bytes(remaining_stderr or exc.stderr or b"")
    stdout_artifact = store.write_bytes(f"captures/{stem}.stdout.log", bytes(stdout))
    stderr_artifact = store.write_bytes(f"captures/{stem}.stderr.log", bytes(stderr))
    capture = {
        "argv": store.normalize_argv(argv),
        "cwd": str(cwd.resolve()),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "started_time_ns": started_time_ns,
        "completed_time_ns": time.time_ns(),
        "stdout_artifact": stdout_artifact,
        "stderr_artifact": stderr_artifact,
    }
    if timed_out:
        raise EvidenceError(f"subprocess timed out; raw capture retained as {stem}")
    if expected_returncode is not None and process.returncode != expected_returncode:
        raise EvidenceError(
            f"subprocess {stem} returned {process.returncode}, expected {expected_returncode}"
        )
    return {**capture, "stdout_bytes": stdout, "stderr_bytes": stderr}


def probe_variant_identity(
    config: RunnerConfig,
    *,
    group: str,
    name: str,
    process_index: int,
    store: ArtifactStore,
    timeout_seconds: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Probe build/runtime identity separately from numerical measurement."""
    variant = config.variant(group, name)
    captured = run_captured(
        identity_probe_argv(variant),
        cwd=variant.checkout,
        environment=controlled_environment(config),
        timeout_seconds=timeout_seconds,
        store=store,
        stem=f"identity-{group}-{process_index:02d}-{name}",
    )
    record = parse_identity_stdout(captured.pop("stdout_bytes"))
    captured.pop("stderr_bytes")
    if set(record) != {
        "schema", "build_info", "runtime", "extension_path",
        "python_executable", "loaded_dependencies",
    }:
        raise EvidenceError("identity probe record has unexpected fields")
    extension_path = Path(str(record["extension_path"])).resolve()
    if variant.runner_extension is None or extension_path != variant.runner_extension.resolve():
        raise EvidenceError("identity probe did not load the runner-built extension")
    build = record["build_info"]
    runtime = record["runtime"]
    if not isinstance(build, dict) or not isinstance(runtime, dict):
        raise EvidenceError("identity probe build/runtime payload is malformed")
    lock = _lock_identity(variant.checkout)
    extension_sha256 = sha256_path(extension_path)
    retained_relative = f"identity/{group}-{name}-{extension_sha256}-{extension_path.name}"
    retained_path = store.root / retained_relative
    if retained_path.exists():
        extension_artifact = store.inspect(
            retained_relative, label=f"{name} packaged extension"
        )
    else:
        extension_artifact = store.retain_external(
            extension_path,
            retained_relative,
            label=f"{name} packaged extension",
            allow_empty=False,
        )
    captured["extension_artifact"] = extension_artifact
    python_path = Path(str(record["python_executable"])).resolve()
    if python_path != variant.python_executable.resolve():
        raise EvidenceError("identity probe used an unconfigured Python executable")
    python_sha256 = sha256_path(python_path)
    python_relative = f"identity/python-{python_sha256}-{python_path.name}"
    if (store.root / python_relative).exists():
        python_artifact = store.inspect(python_relative, label="Python executable")
    else:
        python_artifact = store.retain_external(
            python_path, python_relative, label="Python executable", allow_empty=False
        )
    captured["python_artifact"] = python_artifact
    dependencies = record["loaded_dependencies"]
    if not isinstance(dependencies, list) or not dependencies:
        raise EvidenceError("identity probe did not resolve runtime dependencies")
    dependency_rows: list[dict[str, object]] = []
    dependency_artifacts: list[dict[str, object]] = []
    for raw_path in dependencies:
        path = Path(str(raw_path)).resolve()
        digest = sha256_path(path)
        relative = f"identity/dependencies/{digest}-{path.name}"
        if (store.root / relative).exists():
            artifact = store.inspect(relative, label="runtime dependency")
        else:
            artifact = store.retain_external(
                path, relative, label="runtime dependency", allow_empty=False
            )
        dependency_artifacts.append(artifact)
        dependency_rows.append(
            {"path": str(path), "name": path.name, "sha256": digest,
             "bytes": artifact["bytes"]}
        )
    captured["dependency_artifacts"] = dependency_artifacts
    identity = {
        "channel_commit": build.get("channel_native_git_sha"),
        "rayd_commit": build.get("rayd_commit"),
        "rayd_lock_sha256": lock["lock_sha256"],
        "integration_header_sha256": build.get("rayd_integration_abi_sha256"),
        "build_fingerprint": build.get("build_fingerprint"),
        "build_type": build.get("build_type"),
        "extension_load_source": "packaged",
        "extension_path": str(extension_path),
        "extension_sha256": extension_sha256,
        "python_executable_sha256": python_sha256,
        "runtime_dependencies": sorted(dependency_rows, key=lambda row: str(row["path"])),
        "cuda_architectures": build.get("cuda_architectures"),
        "device_index": runtime.get("device_index"),
        "device_uuid": runtime.get("device_uuid"),
        "gpu_name": runtime.get("gpu_name"),
        "driver_version": runtime.get("driver_version"),
        "python_version": runtime.get("python_version"),
        "torch_version": build.get("torch_version"),
        "cuda_version": build.get("cuda_version"),
        "compiler": build.get("compiler"),
        "cuda_compiler_version": build.get("cuda_compiler_version"),
    }
    return identity, captured


def validate_retained_identity(
    record: object,
    *,
    capture: object,
    identity: object,
    store: ArtifactStore,
) -> dict[str, object]:
    """Recompute reported identity from probe stdout and retained extension."""
    if not isinstance(record, dict) or set(record) != {
        "schema", "build_info", "runtime", "extension_path",
        "python_executable", "loaded_dependencies",
    }:
        raise EvidenceError("retained identity probe record is malformed")
    if not isinstance(capture, dict) or not isinstance(identity, dict):
        raise EvidenceError("retained identity capture/summary is malformed")
    argv = capture.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) != 7
        or argv[1] != "-I"
        or Path(str(argv[2])).name != BOOTSTRAP_REPO_PATH.name
        or argv[3] != "--site-packages"
        or argv[5] != "--script"
        or Path(str(argv[6])).name != IDENTITY_PROBE_REPO_PATH.name
    ):
        raise EvidenceError("identity probe argv is not canonical")
    build = record["build_info"]
    runtime = record["runtime"]
    if not isinstance(build, dict) or not isinstance(runtime, dict):
        raise EvidenceError("retained identity build/runtime payload is malformed")
    artifact = capture.get("extension_artifact")
    if not isinstance(artifact, dict):
        raise EvidenceError("identity capture lacks retained extension bytes")
    store.verify_reference(artifact, label="identity packaged extension", allow_empty=False)
    python_artifact = capture.get("python_artifact")
    dependency_artifacts = capture.get("dependency_artifacts")
    if not isinstance(python_artifact, dict) or not isinstance(dependency_artifacts, list):
        raise EvidenceError("identity capture lacks Python/dependency artifacts")
    store.verify_reference(python_artifact, label="identity Python", allow_empty=False)
    for row in dependency_artifacts:
        store.verify_reference(row, label="runtime dependency", allow_empty=False)
    raw_dependencies = record["loaded_dependencies"]
    if not isinstance(raw_dependencies, list) or len(raw_dependencies) != len(dependency_artifacts):
        raise EvidenceError("runtime dependency artifact count differs from probe")
    dependency_rows = sorted(
        [
            {
                "path": str(Path(str(path)).resolve()),
                "name": Path(str(path)).name,
                "sha256": artifact_row["sha256"],
                "bytes": artifact_row["bytes"],
            }
            for path, artifact_row in zip(raw_dependencies, dependency_artifacts, strict=True)
        ],
        key=lambda row: str(row["path"]),
    )
    expected = {
        "channel_commit": build.get("channel_native_git_sha"),
        "rayd_commit": build.get("rayd_commit"),
        "integration_header_sha256": build.get("rayd_integration_abi_sha256"),
        "build_fingerprint": build.get("build_fingerprint"),
        "build_type": build.get("build_type"),
        "extension_load_source": "packaged",
        "extension_path": str(Path(str(record["extension_path"])).resolve()),
        "extension_sha256": artifact.get("sha256"),
        "python_executable_sha256": python_artifact.get("sha256"),
        "runtime_dependencies": dependency_rows,
        "cuda_architectures": build.get("cuda_architectures"),
        "device_index": runtime.get("device_index"),
        "device_uuid": runtime.get("device_uuid"),
        "gpu_name": runtime.get("gpu_name"),
        "driver_version": runtime.get("driver_version"),
        "python_version": runtime.get("python_version"),
        "torch_version": build.get("torch_version"),
        "cuda_version": build.get("cuda_version"),
        "compiler": build.get("compiler"),
        "cuda_compiler_version": build.get("cuda_compiler_version"),
    }
    for name, value in expected.items():
        if identity.get(name) != value:
            raise EvidenceError(f"identity field differs from retained probe: {name}")
    if not isinstance(identity.get("rayd_lock_sha256"), str):
        raise EvidenceError("identity lacks parent-computed RayD lock SHA")
    return dict(identity)


def worker_argv(
    variant: VariantConfig, *, group: str, name: str, process_index: int, order: str,
    munich_scene_xml: Path | None = None, sionna_source_root: Path | None = None,
) -> list[str]:
    if variant.runner_site_packages is None or variant.runner_extension is None:
        raise EvidenceError("worker requires a runner-built Channel installation")
    script = variant.checkout / WORKER_REPO_PATH
    argv = [
        str(variant.python_executable), "-I", str(variant.checkout / BOOTSTRAP_REPO_PATH),
        "--site-packages", str(variant.runner_site_packages), "--script", str(script),
        "--group", group, "--variant", name, "--process-index", str(process_index),
        "--pair-order", order, "--warmup", "1", "--steady-repeats", "7",
    ]
    if munich_scene_xml is not None and sionna_source_root is not None:
        argv.extend(
            [
                "--munich-scene-xml", str(munich_scene_xml),
                "--sionna-source-root", str(sionna_source_root),
            ]
        )
    return argv


def validate_retained_worker_capture(
    capture: object,
    *,
    identity_capture: object,
    expected_group: str,
    expected_variant: str,
    process_index: int,
    order: str,
) -> dict[str, object]:
    """Bind a retained timed-worker capture to its canonical invocation."""
    capture_keys = {
        "argv", "cwd", "returncode", "timed_out", "started_time_ns",
        "completed_time_ns", "stdout_artifact", "stderr_artifact",
    }
    if not isinstance(capture, dict) or set(capture) != capture_keys:
        raise EvidenceError("retained timed worker capture schema is not canonical")
    if type(capture["returncode"]) is not int or capture["returncode"] != 0:
        raise EvidenceError("retained timed worker did not return success")
    if capture["timed_out"] is not False:
        raise EvidenceError("retained timed worker must not be timed out")
    started = capture["started_time_ns"]
    completed = capture["completed_time_ns"]
    if (
        type(started) is not int
        or type(completed) is not int
        or started <= 0
        or completed < started
    ):
        raise EvidenceError("retained timed worker capture timestamps are malformed")
    argv = capture["argv"]
    cwd_value = capture["cwd"]
    if (
        not isinstance(argv, list)
        or len(argv) != 23
        or any(not isinstance(item, str) for item in argv)
        or not isinstance(cwd_value, str)
    ):
        raise EvidenceError("retained timed worker argv/cwd is malformed")
    checkout = Path(cwd_value)
    if not checkout.is_absolute() or str(checkout.resolve()) != cwd_value:
        raise EvidenceError("retained timed worker cwd is not canonical")
    if not isinstance(identity_capture, dict):
        raise EvidenceError("retained identity capture is malformed")
    identity_argv = identity_capture.get("argv")
    if (
        not isinstance(identity_argv, list)
        or len(identity_argv) != 7
        or any(not isinstance(item, str) for item in identity_argv)
        or identity_capture.get("cwd") != cwd_value
    ):
        raise EvidenceError("timed worker and identity capture invocation differ")
    variant = VariantConfig(
        checkout=checkout,
        python_executable=Path(identity_argv[0]),
        runner_site_packages=Path(identity_argv[4]),
        runner_extension=checkout / "_capture_validation_only.pyd",
    )
    expected_argv = worker_argv(
        variant,
        group=expected_group,
        name=expected_variant,
        process_index=process_index,
        order=order,
        munich_scene_xml=Path(argv[20]),
        sionna_source_root=Path(argv[22]),
    )
    if argv != expected_argv:
        raise EvidenceError("retained timed worker argv is not canonical")
    return dict(capture)


def diagnostic_argv(
    variant: VariantConfig, *, group: str, name: str, process_index: int,
    output: Path, munich_scene_xml: Path, sionna_source_root: Path,
) -> list[str]:
    if variant.runner_site_packages is None or variant.runner_extension is None:
        raise EvidenceError("diagnostic worker requires a runner-built Channel installation")
    return [
        str(variant.python_executable), "-I",
        str(variant.checkout / BOOTSTRAP_REPO_PATH),
        "--site-packages", str(variant.runner_site_packages),
        "--script", str(variant.checkout / DIAGNOSTIC_REPO_PATH),
        "--group", group, "--variant", name,
        "--process-index", str(process_index), "--output", str(output),
        "--munich-scene-xml", str(munich_scene_xml),
        "--sionna-source-root", str(sionna_source_root),
    ]


def collect_process_pairs(
    config: RunnerConfig, gate: Mapping[str, object], *, group: str,
    timeout_seconds: int,
    store: ArtifactStore,
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for scheduled in process_schedule(gate, group):
        index = int(scheduled["process_index"])
        order = str(scheduled["order"])
        variants: dict[str, dict[str, object]] = {}
        for raw_name in scheduled["variants"]:
            name = str(raw_name)
            variant = config.variant(group, name)
            environment = controlled_environment(config)
            identity, identity_capture = probe_variant_identity(
                config,
                group=group,
                name=name,
                process_index=index,
                store=store,
                timeout_seconds=timeout_seconds,
            )
            captured = run_captured(
                worker_argv(
                    variant, group=group, name=name, process_index=index, order=order,
                    munich_scene_xml=config.datasets.munich_scene_xml,
                    sionna_source_root=config.datasets.sionna_source_root,
                ),
                cwd=variant.checkout, environment=environment,
                timeout_seconds=timeout_seconds, store=store,
                stem=f"worker-{group}-{index:02d}-{name}",
            )
            record = _parse_worker_stdout(captured.pop("stdout_bytes"))
            captured.pop("stderr_bytes")
            if "identity" in record or "identity_capture" in record:
                raise EvidenceError("measurement worker must not self-report extension identity")
            record["identity"] = identity
            record["identity_capture"] = identity_capture
            record["capture"] = captured
            variants[name] = record
        pairs.append(
            {
                "group": group, "process_index": index, "order": order,
                "baseline": variants["baseline"], "candidate": variants["candidate"],
            }
        )
    return pairs


__all__ = [
    "DIAGNOSTIC_REPO_PATH", "IDENTITY_PROBE_REPO_PATH", "WORKER_REPO_PATH",
    "collect_process_pairs", "diagnostic_argv", "executable_identity",
    "identity_probe_argv", "parse_identity_stdout",
    "parse_worker_stdout", "probe_variant_identity", "run_captured", "sha256_path",
    "validate_retained_identity", "validate_retained_worker_capture", "verify_checkouts",
    "verify_evidence_commit_binding", "verify_route_transition", "worker_argv",
]
