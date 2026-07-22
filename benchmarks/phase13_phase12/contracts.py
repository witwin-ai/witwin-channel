"""Strict, versioned historical Phase 12 ADR-030 runner contracts.

Measured execution remains fail-closed. ADR-032 superseded this production
candidate after the capacity route regressed Munich E2E latency, peak memory,
and throughput. Dry-run planning remains available to audit the historical
contract; it cannot activate ADR-030 or manufacture missing evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import stat


SCHEMA_NAME = "witwin.channel.phase13-phase12-evidence"
SCHEMA_VERSION = 2
WORKER_SCHEMA_NAME = "witwin.channel.phase13-phase12-worker"
WORKER_SCHEMA_VERSION = 2
IDENTITY_SCHEMA_NAME = "witwin.channel.phase13-phase12-identity-probe"
IDENTITY_SCHEMA_VERSION = 3
SUPPORT_SCHEMA_NAME = "witwin.channel.phase13-phase12-support"
SUPPORT_SCHEMA_VERSION = 2
RUNNER_CONFIG_NAME = "witwin.channel.phase13-phase12-runner-config"
RUNNER_CONFIG_VERSION = 3
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE = ROOT / "benchmarks/gates/phase13_phase12.json"
DEFAULT_SCHEMA = ROOT / "benchmarks/schemas/phase13-phase12-evidence.schema.json"
CANONICAL_GATE_REPO_PATH = "benchmarks/gates/phase13_phase12.json"
SHA_LENGTH = 40
SHA256_LENGTH = 64
VARIANTS = ("baseline", "candidate")
COMPARISON_GROUPS = (
    "enumerated_penetration",
    "montecarlo_penetration",
    "diffraction",
)
DEVELOPER_OVERRIDE_ENV = (
    "WITWIN_CHANNEL_NATIVE_DEVELOPER_OVERRIDE",
    "WITWIN_CHANNEL_NATIVE_EXTENSION_PATH",
    "WITWIN_CHANNEL_NATIVE_EXPECTED_FINGERPRINT",
)
PYTHON_INJECTION_ENV = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
)

# This historical runner cannot become a production switch by filling its
# frozen inputs. ADR-032 keeps the compact route authoritative.
ADDENDUM_ACCEPTED = False
ADR030_PRODUCTION_CANDIDATE_SUPERSEDED = True


class EvidenceError(RuntimeError):
    """Raised when evidence is incomplete, mutable, or contradictory."""


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise EvidenceError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_file_bytes(path: Path) -> bytes:
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise EvidenceError(f"JSON source is not a regular file: {path}")
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise EvidenceError(f"cannot read file {path}: {exc}") from exc
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(payload) != after.st_size
    ):
        raise EvidenceError(f"file changed while being read: {path}")
    return payload


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            _read_file_bytes(path).decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EvidenceError(f"non-finite JSON token: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON root must be an object: {path}")
    return value


def exact_keys(
    value: object,
    expected: set[str] | frozenset[str],
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be an object")
    actual = set(value)
    if actual != set(expected):
        raise EvidenceError(
            f"{label} keys differ; missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )
    return value


def schema_identity(
    value: object, *, name: str, version: int, label: str
) -> dict[str, object]:
    row = exact_keys(value, {"name", "version"}, label=label)
    if row != {"name": name, "version": version}:
        raise EvidenceError(f"{label} identity is not accepted")
    return row


def is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def require_sha(value: object, *, label: str) -> str:
    if not is_sha(value):
        raise EvidenceError(f"{label} must be a lowercase 40-character Git SHA")
    return str(value)


def require_sha256(value: object, *, label: str) -> str:
    if not is_sha256(value):
        raise EvidenceError(f"{label} must be a lowercase SHA-256")
    return str(value)


def finite_number(
    value: object, *, label: str, positive: bool = False, non_negative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceError(f"{label} must be finite")
    if positive and number <= 0.0:
        raise EvidenceError(f"{label} must be positive")
    if non_negative and number < 0.0:
        raise EvidenceError(f"{label} must be non-negative")
    return number


def finite_samples(value: object, *, label: str, count: int = 7) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise EvidenceError(f"{label} must contain exactly {count} samples")
    return [finite_number(item, label=label, positive=True) for item in value]


def reject_developer_override_environment(
    environment: Mapping[str, str] | None = None,
) -> None:
    values = os.environ if environment is None else environment
    present = [name for name in DEVELOPER_OVERRIDE_ENV if name in values]
    if present:
        raise EvidenceError(
            "Phase 12 requires the packaged extension; remove: " + ", ".join(present)
        )


def sanitized_subprocess_environment(
    environment: Mapping[str, str] | None = None,
    *,
    runtime_search_paths: Sequence[Path] = (),
) -> dict[str, str]:
    source = os.environ if environment is None else environment
    allowed = (
        "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR",
        "OS", "LANG", "LC_ALL", "NUMBER_OF_PROCESSORS", "PROCESSOR_IDENTIFIER",
    )
    result = {name: source[name] for name in allowed if name in source}
    if not runtime_search_paths:
        raise EvidenceError("controlled subprocess environment requires fixed runtime paths")
    result["PATH"] = os.pathsep.join(str(path.resolve()) for path in runtime_search_paths)
    result.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
    )
    return result


def controlled_environment(config: "RunnerConfig") -> dict[str, str]:
    return sanitized_subprocess_environment(
        runtime_search_paths=config.runtime_search_paths
    )


@dataclass(frozen=True, slots=True)
class VariantConfig:
    checkout: Path
    python_executable: Path
    runner_site_packages: Path | None = None
    runner_extension: Path | None = None


@dataclass(frozen=True, slots=True)
class ToolConfig:
    nsys: Path
    ncu: Path
    conda: Path
    ctest: Path
    cmake: Path
    ninja: Path
    cuobjdump: Path
    dumpbin: Path
    powershell: Path
    cmd: Path
    vcvars64: Path
    git: Path
    nvcc: Path
    cl: Path
    link: Path
    nvidia_smi: Path


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    munich_scene_xml: Path
    sionna_source_root: Path


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    baseline: VariantConfig
    candidate: VariantConfig


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    comparisons: Mapping[str, ComparisonConfig]
    rayd_checkout: Path
    raw_artifact_parent: Path
    build_parent: Path
    output: Path
    tools: ToolConfig
    datasets: DatasetConfig
    runtime_search_paths: tuple[Path, ...]
    runner_build_environment: Mapping[str, str] | None = None

    def variant(self, group: str, name: str) -> VariantConfig:
        if group not in COMPARISON_GROUPS or name not in VARIANTS:
            raise EvidenceError(f"unknown comparison variant: {group}/{name}")
        return getattr(self.comparisons[group], name)


def _absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise EvidenceError(f"{label} must be absolute")
    return path.resolve()


def load_config(path: Path) -> RunnerConfig:
    raw = read_json(path)
    schema_identity(
        raw.get("schema"), name=RUNNER_CONFIG_NAME, version=RUNNER_CONFIG_VERSION,
        label="config.schema"
    )
    exact_keys(
        raw,
        {
            "schema", "comparisons", "rayd_checkout", "raw_artifact_parent",
            "build_parent", "output", "tools", "datasets", "runtime_search_paths",
        },
        label="config",
    )

    def variant(name: str) -> VariantConfig:
        row = exact_keys(
            raw[name], {"checkout", "python_executable"},
            label=f"config.{name}"
        )
        return VariantConfig(
            checkout=_absolute_path(row["checkout"], label=f"config.{name}.checkout"),
            python_executable=_absolute_path(
                row["python_executable"], label=f"config.{name}.python_executable"
            ),
        )

    comparison_rows = exact_keys(
        raw["comparisons"], set(COMPARISON_GROUPS), label="config.comparisons"
    )
    comparisons: dict[str, ComparisonConfig] = {}
    for group in COMPARISON_GROUPS:
        row = exact_keys(
            comparison_rows[group], {"baseline", "candidate"},
            label=f"config.comparisons.{group}",
        )
        # Reuse the strict variant parser without trusting a shared path alias.
        raw[group + "_baseline"] = row["baseline"]
        raw[group + "_candidate"] = row["candidate"]
        comparisons[group] = ComparisonConfig(
            baseline=variant(group + "_baseline"),
            candidate=variant(group + "_candidate"),
        )
    tools = exact_keys(
        raw["tools"],
        {
            "nsys", "ncu", "conda", "ctest", "cmake", "ninja", "cuobjdump",
            "dumpbin", "powershell",
            "cmd", "vcvars64", "git",
            "nvcc", "cl", "link", "nvidia_smi",
        },
        label="config.tools",
    )
    datasets = exact_keys(
        raw["datasets"], {"munich_scene_xml", "sionna_source_root"},
        label="config.datasets",
    )
    config = RunnerConfig(
        comparisons=comparisons,
        rayd_checkout=_absolute_path(raw["rayd_checkout"], label="config.rayd_checkout"),
        raw_artifact_parent=_absolute_path(
            raw["raw_artifact_parent"], label="config.raw_artifact_parent"
        ),
        build_parent=_absolute_path(raw["build_parent"], label="config.build_parent"),
        output=_absolute_path(raw["output"], label="config.output"),
        tools=ToolConfig(
            nsys=_absolute_path(tools["nsys"], label="config.tools.nsys"),
            ncu=_absolute_path(tools["ncu"], label="config.tools.ncu"),
            conda=_absolute_path(tools["conda"], label="config.tools.conda"),
            ctest=_absolute_path(tools["ctest"], label="config.tools.ctest"),
            cmake=_absolute_path(tools["cmake"], label="config.tools.cmake"),
            ninja=_absolute_path(tools["ninja"], label="config.tools.ninja"),
            cuobjdump=_absolute_path(tools["cuobjdump"], label="config.tools.cuobjdump"),
            dumpbin=_absolute_path(tools["dumpbin"], label="config.tools.dumpbin"),
            powershell=_absolute_path(tools["powershell"], label="config.tools.powershell"),
            cmd=_absolute_path(tools["cmd"], label="config.tools.cmd"),
            vcvars64=_absolute_path(tools["vcvars64"], label="config.tools.vcvars64"),
            git=_absolute_path(tools["git"], label="config.tools.git"),
            nvcc=_absolute_path(tools["nvcc"], label="config.tools.nvcc"),
            cl=_absolute_path(tools["cl"], label="config.tools.cl"),
            link=_absolute_path(tools["link"], label="config.tools.link"),
            nvidia_smi=_absolute_path(tools["nvidia_smi"], label="config.tools.nvidia_smi"),
        ),
        datasets=DatasetConfig(
            munich_scene_xml=_absolute_path(
                datasets["munich_scene_xml"], label="config.datasets.munich_scene_xml"
            ),
            sionna_source_root=_absolute_path(
                datasets["sionna_source_root"], label="config.datasets.sionna_source_root"
            ),
        ),
        runtime_search_paths=tuple(
            _absolute_path(value, label=f"config.runtime_search_paths[{index}]")
            for index, value in enumerate(raw["runtime_search_paths"])
        ) if isinstance(raw["runtime_search_paths"], list) else (),
    )
    if not isinstance(raw["runtime_search_paths"], list) or not config.runtime_search_paths:
        raise EvidenceError("config.runtime_search_paths must be a non-empty array")
    validate_config_paths(config)
    return config


def validate_config_paths(config: RunnerConfig) -> None:
    protected = tuple(
        config.variant(group, name).checkout
        for group in COMPARISON_GROUPS
        for name in VARIANTS
    ) + (config.rayd_checkout,)
    parent = config.raw_artifact_parent.resolve()
    build_parent = config.build_parent.resolve()
    output = config.output.resolve()
    for checkout in protected:
        if parent == checkout or parent.is_relative_to(checkout):
            raise EvidenceError("raw_artifact_parent must be outside every checkout")
        if (
            build_parent == checkout
            or build_parent.is_relative_to(checkout)
            or checkout.is_relative_to(build_parent)
        ):
            raise EvidenceError("build_parent must have no ancestry overlap with a checkout")
        if output == checkout or output.is_relative_to(checkout):
            raise EvidenceError("output must be outside every checkout")
    if output == parent or not output.is_relative_to(parent):
        raise EvidenceError("output must be a file beneath raw_artifact_parent")
    if (
        build_parent == parent
        or build_parent.is_relative_to(parent)
        or parent.is_relative_to(build_parent)
    ):
        raise EvidenceError("build_parent and raw_artifact_parent must be disjoint")
    python_paths = {
        config.variant(group, name).python_executable.resolve()
        for group in COMPARISON_GROUPS
        for name in VARIANTS
    }
    if len(python_paths) != 1:
        raise EvidenceError("all Channel variants must use one exact witwin2 Python")
    python_path = next(iter(python_paths))
    folded_parts = tuple(part.casefold() for part in python_path.parts)
    if (
        python_path.name.casefold() != "python.exe"
        or not any(
            folded_parts[index:index + 2] == ("envs", "witwin2")
            for index in range(len(folded_parts) - 1)
        )
    ):
        raise EvidenceError("formal Channel evidence Python must be the witwin2 environment")


def canonical_gate_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != DEFAULT_GATE.resolve():
        raise EvidenceError(
            f"measured Phase 12 evidence requires {CANONICAL_GATE_REPO_PATH}"
        )
    return resolved


def _validate_gate_structure(gate: Mapping[str, object]) -> None:
    exact_keys(
        gate,
        {
            "schema", "history_policy", "measurement_policy", "frozen_inputs",
            "munich", "non_target_metrics", "non_target_regression_policy",
            "comparison_groups", "required_unaffected_hashes",
            "required_release_checks", "wheel_cuda_architectures",
        },
        label="gate",
    )
    schema_identity(
        gate["schema"], name="witwin.channel.phase13-phase12-gate",
        version=3, label="gate.schema",
    )
    groups = exact_keys(
        gate["comparison_groups"], set(COMPARISON_GROUPS),
        label="gate.comparison_groups",
    )
    for group in COMPARISON_GROUPS:
        exact_keys(
            groups[group],
            {
                "target_stage_metric", "target_end_to_end_metric",
                "minimum_stage_improvement_percent",
                "minimum_end_to_end_improvement_percent", "correctness",
                "range_multiplicity_per_solve", "marker_multiplicity_per_solve",
                "resource_budgets",
            },
            label=f"gate.comparison_groups.{group}",
        )


def load_gate(path: Path = DEFAULT_GATE, *, measured: bool = False) -> dict[str, object]:
    if measured:
        canonical_gate_path(path)
    gate = read_json(path)
    _validate_gate_structure(gate)
    return gate


def require_measured_policy_ready(gate: Mapping[str, object]) -> None:
    history = gate["history_policy"]
    assert isinstance(history, dict)
    blockers: list[str] = []
    if ADR030_PRODUCTION_CANDIDATE_SUPERSEDED:
        blockers.append("ADR-030 production candidate superseded by ADR-032")
    elif not ADDENDUM_ACCEPTED or history.get("accepted") is not True:
        blockers.append("accepted ADR-030 dormant-pin/direct-switch addendum")
    frozen_inputs = gate["frozen_inputs"]
    if not isinstance(frozen_inputs, dict):
        blockers.append("frozen_inputs")
    else:
        blockers.extend(
            f"frozen_inputs.{name}" for name, value in frozen_inputs.items()
            if value is None
        )
    groups = gate["comparison_groups"]
    assert isinstance(groups, dict)
    for group, policy in groups.items():
        if not isinstance(policy, dict):
            blockers.append(f"comparison_groups.{group}")
            continue
        for name in ("correctness", "resource_budgets"):
            if policy.get(name) is None:
                blockers.append(f"comparison_groups.{group}.{name}")
    def collect_nulls(value: object, prefix: str) -> None:
        if value is None:
            blockers.append(prefix)
        elif isinstance(value, dict):
            for name, item in value.items():
                collect_nulls(item, f"{prefix}.{name}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect_nulls(item, f"{prefix}[{index}]")

    collect_nulls(gate, "gate")
    blockers = list(dict.fromkeys(blockers))
    if blockers:
        raise EvidenceError(
            "measured Phase 12 evidence is blocked until these accepted facts are frozen: "
            + ", ".join(blockers)
        )


def process_schedule(
    gate: Mapping[str, object], group: str
) -> list[dict[str, object]]:
    if group not in COMPARISON_GROUPS:
        raise EvidenceError(f"unknown comparison group: {group}")
    policy = gate["measurement_policy"]
    assert isinstance(policy, dict)
    orders = policy["pair_order"]
    assert isinstance(orders, list)
    return [
        {
            "process_index": index,
            "order": order,
            "variants": ["baseline", "candidate"]
            if order == "AB"
            else ["candidate", "baseline"],
        }
        for index, order in enumerate(orders)
    ]


def validate_exact_schedule(pairs: Sequence[Mapping[str, object]], gate: Mapping[str, object]) -> None:
    if not pairs:
        raise EvidenceError("comparison group has no process pairs")
    group = pairs[0].get("group")
    if not isinstance(group, str):
        raise EvidenceError("comparison pairs lack group identity")
    expected = process_schedule(gate, group)
    if len(pairs) != len(expected):
        raise EvidenceError("Phase 12 requires exactly five process pairs")
    observed = [
        {"group": row.get("group"), "process_index": row.get("process_index"), "order": row.get("order")}
        for row in pairs
    ]
    wanted = [
        {"group": group, "process_index": row["process_index"], "order": row["order"]}
        for row in expected
    ]
    if observed != wanted:
        raise EvidenceError("process pair indices/order differ from the frozen schedule")


__all__ = [
    "ADDENDUM_ACCEPTED", "ADR030_PRODUCTION_CANDIDATE_SUPERSEDED",
    "CANONICAL_GATE_REPO_PATH", "COMPARISON_GROUPS", "DEFAULT_GATE",
    "DEFAULT_SCHEMA", "DEVELOPER_OVERRIDE_ENV", "EvidenceError", "ROOT",
    "ComparisonConfig", "DatasetConfig", "RunnerConfig", "SCHEMA_NAME", "SCHEMA_VERSION", "SUPPORT_SCHEMA_NAME",
    "SUPPORT_SCHEMA_VERSION", "ToolConfig", "VARIANTS", "VariantConfig",
    "WORKER_SCHEMA_NAME", "WORKER_SCHEMA_VERSION", "canonical_gate_path",
    "controlled_environment",
    "exact_keys", "finite_number", "finite_samples", "is_sha", "is_sha256",
    "load_config", "load_gate", "process_schedule", "read_json",
    "reject_developer_override_environment", "require_measured_policy_ready",
    "require_sha", "require_sha256", "sanitized_subprocess_environment",
    "schema_identity", "strict_object", "validate_config_paths",
    "validate_exact_schedule",
]
