"""Canonical facade for the split Phase 12 evidence implementation."""

from __future__ import annotations

from .phase13_phase12.artifacts import ArtifactStore, safe_relative_path
from .phase13_phase12.contracts import (
    DEFAULT_GATE,
    DEFAULT_SCHEMA,
    EvidenceError,
    RunnerConfig,
    VariantConfig,
    load_config,
    load_gate,
    process_schedule,
    read_json,
    reject_developer_override_environment,
    require_measured_policy_ready,
    sanitized_subprocess_environment,
    validate_config_paths,
)
from .phase13_phase12.diagnostics import (
    collect_diagnostics,
    load_diagnostic_contract,
    replay_diagnostics,
)
from .phase13_phase12.profilers import (
    parse_ncu_csv,
    parse_nsys_sqlite,
    run_ncu_candidate,
    run_nsys_matrix,
    validate_ncu_blocker,
    validate_nsys_timelines,
)
from .phase13_phase12.release import (
    run_release_evidence,
    validate_release_report,
)
from .phase13_phase12.report import (
    build_dry_run,
    build_measured_report,
    replay_measured_report,
    write_report,
)
from .phase13_phase12.statistics import (
    compare_timings,
    paired_bootstrap,
    summarize_resources,
    validate_correctness,
    validate_process_identity,
    validate_worker_record,
)
from .phase13_phase12.workers import (
    collect_process_pairs,
    parse_worker_stdout,
    run_captured,
    sha256_path,
    verify_checkouts,
    verify_evidence_commit_binding,
    verify_route_transition,
    worker_argv,
)


sha256_file = sha256_path


__all__ = [
    "ArtifactStore",
    "DEFAULT_GATE",
    "DEFAULT_SCHEMA",
    "EvidenceError",
    "RunnerConfig",
    "VariantConfig",
    "build_dry_run",
    "build_measured_report",
    "collect_diagnostics",
    "collect_process_pairs",
    "compare_timings",
    "load_config",
    "load_diagnostic_contract",
    "load_gate",
    "paired_bootstrap",
    "parse_ncu_csv",
    "parse_nsys_sqlite",
    "parse_worker_stdout",
    "process_schedule",
    "read_json",
    "reject_developer_override_environment",
    "replay_measured_report",
    "replay_diagnostics",
    "require_measured_policy_ready",
    "run_captured",
    "run_ncu_candidate",
    "run_nsys_matrix",
    "run_release_evidence",
    "safe_relative_path",
    "sanitized_subprocess_environment",
    "sha256_file",
    "summarize_resources",
    "validate_config_paths",
    "validate_correctness",
    "validate_ncu_blocker",
    "validate_nsys_timelines",
    "validate_process_identity",
    "validate_release_report",
    "validate_worker_record",
    "verify_checkouts",
    "verify_evidence_commit_binding",
    "verify_route_transition",
    "worker_argv",
    "write_report",
]
