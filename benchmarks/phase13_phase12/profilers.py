"""Runner-owned Nsight Systems/Compute capture and semantic validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import math
import os
from pathlib import Path
import sqlite3
import stat

from .artifacts import ArtifactStore, reject_reparse_chain
from .contracts import (
    COMPARISON_GROUPS,
    EvidenceError,
    ROOT,
    RunnerConfig,
    controlled_environment,
    exact_keys,
    process_schedule,
    read_json,
)
from .workers import executable_identity, run_captured, worker_argv


PROFILE_CONTRACT_REPO_PATH = Path("benchmarks/phase13_phase12_profile_contract.json")


def load_profile_contract() -> dict[str, object]:
    contract = read_json(ROOT / PROFILE_CONTRACT_REPO_PATH)
    exact_keys(contract, {"schema", "steady_repeats", "groups"}, label="profile contract")
    if contract["schema"] != {
        "name": "witwin.channel_native.phase13-phase12-profile-contract",
        "version": 1,
    } or contract["steady_repeats"] != 7:
        raise EvidenceError("profile contract identity/repeat count is not accepted")
    groups = exact_keys(contract["groups"], set(COMPARISON_GROUPS), label="profile contract groups")
    for group in COMPARISON_GROUPS:
        row = exact_keys(
            groups[group],
            {
                "scenario", "target_timing_range", "solver_entrypoint",
                "ncu_kernel_family_match", "variants",
            },
            label=f"profile contract {group}",
        )
        if row["ncu_kernel_family_match"] != "case_sensitive_substring":
            raise EvidenceError("profile contract NCU family matching policy differs")
        variants = exact_keys(row["variants"], {"baseline", "candidate"}, label=f"profile variants {group}")
        for name in ("baseline", "candidate"):
            variant = exact_keys(
                variants[name],
                {
                    "required_ranges", "forbidden_ranges", "required_markers",
                    "known_range_multiplicity_per_solve", "ncu_kernel_families",
                },
                label=f"profile variant {group}/{name}",
            )
            required = variant["required_ranges"]
            forbidden = variant["forbidden_ranges"]
            markers = variant["required_markers"]
            known = variant["known_range_multiplicity_per_solve"]
            ncu_families = variant["ncu_kernel_families"]
            if not all(isinstance(value, list) for value in (required, forbidden, markers)) or not isinstance(known, dict):
                raise EvidenceError("profile contract variant lists/multiplicity are malformed")
            if (
                not isinstance(ncu_families, list)
                or not all(isinstance(value, str) and value for value in ncu_families)
                or ncu_families != sorted(set(ncu_families))
                or (name == "baseline" and ncu_families)
                or (name == "candidate" and not ncu_families)
            ):
                raise EvidenceError("profile contract NCU kernel families are malformed")
            if set(required) & set(forbidden) or set(required) & set(markers):
                raise EvidenceError("profile contract required/forbidden/marker names overlap")
            if not set(known).issubset(required) or any(
                type(value) is not int or value <= 0 for value in known.values()
            ):
                raise EvidenceError("profile contract known multiplicity is malformed")
    return contract


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    if not table.replace("_", "").isalnum():
        raise EvidenceError(f"unsafe SQLite table name: {table}")
    return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]


def _enum_map(connection: sqlite3.Connection, table: str) -> dict[int, str]:
    columns = _columns(connection, table)
    id_column = next((name for name in ("id", "value", "cbid") if name in columns), None)
    name_column = next((name for name in ("name", "label", "valueName") if name in columns), None)
    if id_column is None or name_column is None:
        raise EvidenceError(f"Nsight enum table {table} lacks id/name columns")
    return {
        int(row[0]): str(row[1])
        for row in connection.execute(
            f'SELECT "{id_column}", "{name_column}" FROM "{table}"'
        ).fetchall()
    }


def _api_names(
    connection: sqlite3.Connection, activity_table: str, enum_table: str
) -> list[str]:
    columns = _columns(connection, activity_table)
    if "name" in columns:
        return [str(row[0]) for row in connection.execute(f'SELECT "name" FROM "{activity_table}"')]
    if "cbid" not in columns:
        raise EvidenceError(f"Nsight activity table {activity_table} lacks name/cbid")
    mapping = _enum_map(connection, enum_table)
    names: list[str] = []
    for row in connection.execute(f'SELECT "cbid" FROM "{activity_table}"'):
        key = int(row[0])
        if key not in mapping:
            raise EvidenceError(f"Nsight API cbid {key} is absent from {enum_table}")
        names.append(mapping[key])
    return names


def _column(
    columns: Sequence[str], candidates: Sequence[str], *, label: str
) -> str:
    value = next((name for name in candidates if name in columns), None)
    if value is None:
        raise EvidenceError(f"Nsight {label} lacks one of {tuple(candidates)}")
    return value


def _activity_rows(
    connection: sqlite3.Connection, activity_table: str, enum_table: str
) -> list[dict[str, object]]:
    """Resolve CUDA API rows without trusting exporter-specific column order."""
    columns = _columns(connection, activity_table)
    start_column = _column(columns, ("start",), label=f"{activity_table} start")
    end_column = _column(columns, ("end",), label=f"{activity_table} end")
    correlation_column = _column(
        columns, ("correlationId", "correlation_id"),
        label=f"{activity_table} correlation",
    )
    name_column = "name" if "name" in columns else None
    cbid_column = "cbid" if "cbid" in columns else None
    if name_column is None and cbid_column is None:
        raise EvidenceError(f"Nsight activity table {activity_table} lacks name/cbid")
    selected_name = name_column or cbid_column
    assert selected_name is not None
    enum = {} if name_column is not None else _enum_map(connection, enum_table)
    rows: list[dict[str, object]] = []
    for raw_name, raw_start, raw_end, raw_correlation in connection.execute(
        f'SELECT "{selected_name}", "{start_column}", "{end_column}", '
        f'"{correlation_column}" FROM "{activity_table}" ORDER BY "{start_column}"'
    ):
        if raw_start is None or raw_end is None:
            raise EvidenceError(f"Nsight {activity_table} contains an incomplete API row")
        start = int(raw_start)
        end = int(raw_end)
        if end < start:
            raise EvidenceError(f"Nsight {activity_table} contains a negative API interval")
        if name_column is not None:
            name = str(raw_name)
        else:
            key = int(raw_name)
            if key not in enum:
                raise EvidenceError(f"Nsight API cbid {key} is absent from {enum_table}")
            name = enum[key]
        rows.append(
            {
                "name": name,
                "start": start,
                "end": end,
                "correlation_id": (
                    None if raw_correlation is None else int(raw_correlation)
                ),
            }
        )
    return rows


def _gpu_activity_rows(
    connection: sqlite3.Connection, table: str, *, include_copy_kind: bool
) -> list[dict[str, object]]:
    columns = _columns(connection, table)
    start_column = _column(columns, ("start",), label=f"{table} start")
    end_column = _column(columns, ("end",), label=f"{table} end")
    correlation_column = _column(
        columns, ("correlationId", "correlation_id"), label=f"{table} correlation"
    )
    stream_column = _column(
        columns, ("streamId", "stream_id"), label=f"{table} stream"
    )
    device_column = _column(
        columns, ("deviceId", "device_id"), label=f"{table} device"
    )
    selected = [start_column, end_column, correlation_column, stream_column, device_column]
    if include_copy_kind:
        selected.append(_column(columns, ("copyKind", "copy_kind"), label=f"{table} kind"))
        selected.append(_column(columns, ("bytes",), label=f"{table} bytes"))
    query = ", ".join(f'"{name}"' for name in selected)
    rows: list[dict[str, object]] = []
    for raw in connection.execute(
        f'SELECT {query} FROM "{table}" ORDER BY "{start_column}"'
    ):
        start, end = int(raw[0]), int(raw[1])
        if end <= start:
            raise EvidenceError(f"Nsight {table} contains a non-positive GPU interval")
        if raw[2] is None:
            raise EvidenceError(f"Nsight {table} contains an uncorrelated GPU activity")
        stream_id = int(raw[3])
        device_id = int(raw[4])
        if stream_id < 0 or device_id < 0:
            raise EvidenceError(f"Nsight {table} contains a negative device/stream id")
        row: dict[str, object] = {
            "start": start,
            "end": end,
            "correlation_id": int(raw[2]),
            "stream_id": stream_id,
            "device_id": device_id,
        }
        if include_copy_kind:
            row["copy_kind"] = int(raw[5])
            row["bytes"] = int(raw[6])
            if int(row["bytes"]) < 0:
                raise EvidenceError(f"Nsight {table} contains negative copy bytes")
        rows.append(row)
    return rows


def _strictly_nested(
    first: tuple[int, int], second: tuple[int, int]
) -> bool:
    if first == second:
        return False
    return (
        first[0] <= second[0] and second[1] <= first[1]
    ) or (
        second[0] <= first[0] and first[1] <= second[1]
    )


def _ranges_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return max(first[0], second[0]) < min(first[1], second[1])


def _require_single_target_placement(
    rows: Sequence[Mapping[str, object]], *, name: str
) -> tuple[int, int]:
    if len(rows) != 7:
        raise EvidenceError(f"Nsight target range {name} does not have seven samples")
    placements = {
        (int(row["device_id"]), int(row["stream_id"]))
        for row in rows
    }
    if len(placements) != 1:
        raise EvidenceError(
            f"Nsight target range {name} samples cross CUDA devices/streams"
        )
    return next(iter(placements))


def _nvtx_names(connection: sqlite3.Connection) -> list[str]:
    columns = _columns(connection, "NVTX_EVENTS")
    if "text" in columns:
        return [
            str(row[0])
            for row in connection.execute('SELECT "text" FROM "NVTX_EVENTS"')
            if row[0] is not None
        ]
    text_id = next((name for name in ("textId", "text_id") if name in columns), None)
    if text_id is None or "StringIds" not in _tables(connection):
        raise EvidenceError("Nsight NVTX tables lack resolvable text")
    string_columns = _columns(connection, "StringIds")
    value_column = next((name for name in ("value", "str", "text") if name in string_columns), None)
    if "id" not in string_columns or value_column is None:
        raise EvidenceError("Nsight StringIds table lacks id/value")
    strings = {
        int(row[0]): str(row[1])
        for row in connection.execute(f'SELECT "id", "{value_column}" FROM "StringIds"')
    }
    result: list[str] = []
    for row in connection.execute(f'SELECT "{text_id}" FROM "NVTX_EVENTS"'):
        if row[0] is None:
            continue
        key = int(row[0])
        if key not in strings:
            raise EvidenceError(f"Nsight NVTX text id {key} is unresolved")
        result.append(strings[key])
    return result


def _nvtx_range_rows(connection: sqlite3.Connection) -> list[tuple[str, int, int]]:
    columns = _columns(connection, "NVTX_EVENTS")
    if "start" not in columns or "end" not in columns:
        raise EvidenceError("Nsight NVTX table lacks start/end timestamps")
    names = _nvtx_names(connection)
    # Resolve names and timestamp rows in the same SQLite order used by
    # _nvtx_names.  NULL-text rows are excluded by both queries.
    if "text" in columns:
        rows = connection.execute(
            'SELECT "text", "start", "end" FROM "NVTX_EVENTS" '
            'WHERE "text" IS NOT NULL AND "end" IS NOT NULL ORDER BY "start"'
        ).fetchall()
        return [
            (str(name), int(start), int(end))
            for name, start, end in rows
            if int(end) > int(start)
        ]
    text_id = next((name for name in ("textId", "text_id") if name in columns), None)
    if text_id is None:
        raise EvidenceError("Nsight NVTX table lacks a text identifier")
    string_columns = _columns(connection, "StringIds")
    value_column = next(
        (name for name in ("value", "str", "text") if name in string_columns), None
    )
    if value_column is None:
        raise EvidenceError("Nsight StringIds table lacks text values")
    rows = connection.execute(
        f'SELECT s."{value_column}", e."start", e."end" '
        f'FROM "NVTX_EVENTS" e JOIN "StringIds" s ON s."id" = e."{text_id}" '
        'WHERE e."end" IS NOT NULL ORDER BY e."start"'
    ).fetchall()
    resolved = [
        (str(name), int(start), int(end))
        for name, start, end in rows
        if int(end) > int(start)
    ]
    if not resolved and names:
        raise EvidenceError("Nsight NVTX names exist but no complete ranges were resolved")
    return resolved


def parse_nsys_sqlite(path: Path) -> dict[str, object]:
    reject_reparse_chain(path)
    try:
        guard = path.open("rb")
        before = os.fstat(guard.fileno())
    except OSError as exc:
        raise EvidenceError(f"cannot guard Nsight Systems SQLite: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
        guard.close()
        raise EvidenceError(f"Nsight Systems SQLite is missing or empty: {path}")
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error as exc:
        guard.close()
        raise EvidenceError(f"cannot open Nsight Systems SQLite: {exc}") from exc
    try:
        tables = _tables(connection)
        required = {
            "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_RUNTIME", "ENUM_CUDA_RUNTIME_API",
            "ENUM_CUDA_MEMCPY_OPER", "NVTX_EVENTS",
        }
        missing = required - tables
        if missing:
            raise EvidenceError(f"Nsight Systems SQLite lacks required tables: {sorted(missing)}")
        kernel_rows = _gpu_activity_rows(
            connection, "CUPTI_ACTIVITY_KIND_KERNEL", include_copy_kind=False
        )
        kernel_count = len(kernel_rows)
        copy_map = _enum_map(connection, "ENUM_CUDA_MEMCPY_OPER")
        memcpy_rows = _gpu_activity_rows(
            connection, "CUPTI_ACTIVITY_KIND_MEMCPY", include_copy_kind=True
        )
        memcpy_names: list[str] = []
        for row in memcpy_rows:
            key = int(row["copy_kind"])
            if key not in copy_map:
                raise EvidenceError(f"Nsight memcpy kind {key} is unresolved")
            memcpy_names.append(copy_map[key])
        runtime_rows = _activity_rows(
            connection, "CUPTI_ACTIVITY_KIND_RUNTIME", "ENUM_CUDA_RUNTIME_API"
        )
        driver_rows: list[dict[str, object]] = []
        if "CUPTI_ACTIVITY_KIND_DRIVER" in tables:
            if "ENUM_CUDA_DRIVER_API" not in tables:
                raise EvidenceError("Nsight driver activity lacks its enum table")
            driver_rows = _activity_rows(
                connection, "CUPTI_ACTIVITY_KIND_DRIVER", "ENUM_CUDA_DRIVER_API"
            )
        nvtx_names = _nvtx_names(connection)
        nvtx_ranges = _nvtx_range_rows(connection)
    except sqlite3.Error as exc:
        raise EvidenceError(f"cannot query Nsight Systems SQLite: {exc}") from exc
    finally:
        connection.close()
        after = os.fstat(guard.fileno())
        path_after = path.stat(follow_symlinks=False)
        guard.close()
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity_fields) or any(
        getattr(before, name) != getattr(path_after, name) for name in identity_fields
    ):
        raise EvidenceError("Nsight Systems SQLite changed or was replaced during replay")

    def count_fragments(values: Sequence[str], fragments: Sequence[str]) -> int:
        return sum(
            any(fragment.casefold() in value.casefold() for fragment in fragments)
            for value in values
        )

    phase_names = sorted({name for name in nvtx_names if name.startswith("witwin.channel_native:")})
    if not phase_names:
        raise EvidenceError("Nsight capture contains no Phase 12 NVTX ranges")
    api_rows = runtime_rows + driver_rows
    api_names = [str(row["name"]) for row in api_rows]
    launch_api_by_correlation: dict[int, list[dict[str, object]]] = {}
    memcpy_api_by_correlation: dict[int, list[dict[str, object]]] = {}
    for row in api_rows:
        correlation = row["correlation_id"]
        folded = str(row["name"]).casefold()
        if correlation is None:
            continue
        if "launch" in folded:
            launch_api_by_correlation.setdefault(int(correlation), []).append(row)
        if "memcpy" in folded:
            memcpy_api_by_correlation.setdefault(int(correlation), []).append(row)
    kernels_by_correlation: dict[int, list[dict[str, object]]] = {}
    for row in kernel_rows:
        kernels_by_correlation.setdefault(int(row["correlation_id"]), []).append(row)
    copies_by_correlation: dict[int, list[dict[str, object]]] = {}
    for row in memcpy_rows:
        copies_by_correlation.setdefault(int(row["correlation_id"]), []).append(row)
    for correlation, rows in kernels_by_correlation.items():
        if len(rows) != 1 or len(launch_api_by_correlation.get(correlation, ())) != 1:
            raise EvidenceError(
                f"Nsight kernel correlation {correlation} is missing or ambiguous"
            )
    for correlation, rows in copies_by_correlation.items():
        if len(rows) != 1 or len(memcpy_api_by_correlation.get(correlation, ())) != 1:
            raise EvidenceError(
                f"Nsight memcpy correlation {correlation} is missing or ambiguous"
            )

    phase_spans = [
        (index, name, start, end)
        for index, (name, start, end) in enumerate(nvtx_ranges)
        if name.startswith("witwin.channel_native:")
    ]
    spans_by_name: dict[str, list[tuple[int, int]]] = {}
    for _, name, start, end in phase_spans:
        spans = spans_by_name.setdefault(name, [])
        if any(_ranges_overlap((start, end), other) for other in spans):
            raise EvidenceError(f"Nsight range {name} has overlapping instances")
        spans.append((start, end))

    correlation_families: dict[int, str] = {}

    def register_correlation(correlation: int, family: str) -> None:
        previous = correlation_families.setdefault(correlation, family)
        if previous != family:
            raise EvidenceError(
                f"Nsight correlation {correlation} is reused across {previous}/{family} families"
            )

    for correlation in launch_api_by_correlation:
        register_correlation(correlation, "kernel")
    for correlation in kernels_by_correlation:
        register_correlation(correlation, "kernel")
    for correlation in memcpy_api_by_correlation:
        register_correlation(correlation, "memcpy")
    for correlation in copies_by_correlation:
        register_correlation(correlation, "memcpy")

    correlation_owners: dict[int, list[tuple[str, int, int, int]]] = {}
    range_samples: dict[str, list[dict[str, object]]] = {}
    for span_index, name, start, end in phase_spans:
        launch_rows = [
            row for row in api_rows
            if "launch" in str(row["name"]).casefold()
            and row["correlation_id"] is not None
            and start <= int(row["start"])
            and int(row["end"]) <= end
        ]
        kernel_correlations = [int(row["correlation_id"]) for row in launch_rows]
        if len(kernel_correlations) != len(set(kernel_correlations)):
            raise EvidenceError("Nsight range contains duplicate launch correlations")
        missing_kernel = [
            value for value in kernel_correlations if value not in kernels_by_correlation
        ]
        if missing_kernel:
            raise EvidenceError(
                f"Nsight range {name} has launch APIs without GPU kernels: {missing_kernel}"
            )
        matched_kernels = [kernels_by_correlation[value][0] for value in kernel_correlations]
        copy_api_rows = [
            row for row in api_rows
            if "memcpy" in str(row["name"]).casefold()
            and row["correlation_id"] is not None
            and start <= int(row["start"])
            and int(row["end"]) <= end
        ]
        memcpy_correlations = [int(row["correlation_id"]) for row in copy_api_rows]
        if len(memcpy_correlations) != len(set(memcpy_correlations)):
            raise EvidenceError("Nsight range contains duplicate memcpy correlations")
        missing_memcpy = [
            value for value in memcpy_correlations if value not in copies_by_correlation
        ]
        if missing_memcpy:
            raise EvidenceError(
                f"Nsight range {name} has memcpy APIs without GPU activity: {missing_memcpy}"
            )
        matched_copies = [copies_by_correlation[value][0] for value in memcpy_correlations]
        for correlations in (kernel_correlations, memcpy_correlations):
            for correlation in correlations:
                correlation_owners.setdefault(correlation, []).append(
                    (name, span_index, start, end)
                )
        kernel_stream_keys = {
            (int(row["device_id"]), int(row["stream_id"]))
            for row in matched_kernels
        }
        if len(kernel_stream_keys) > 1:
            raise EvidenceError(
                f"Nsight range {name} has ambiguous multi-stream kernel activity"
            )
        device_ids = {int(row["device_id"]) for row in [*matched_kernels, *matched_copies]}
        if len(device_ids) > 1:
            raise EvidenceError(f"Nsight range {name} crosses CUDA devices")
        copy_stream_ids = sorted({int(row["stream_id"]) for row in matched_copies})
        if matched_kernels:
            all_gpu_rows = [*matched_kernels, *matched_copies]
            gpu_start = min(int(row["start"]) for row in all_gpu_rows)
            gpu_end = max(int(row["end"]) for row in all_gpu_rows)
            duration_ms = (gpu_end - gpu_start) / 1_000_000.0
            active_ms = sum(
                int(row["end"]) - int(row["start"]) for row in matched_kernels
            ) / 1_000_000.0
            copy_active_ms = sum(
                int(row["end"]) - int(row["start"]) for row in matched_copies
            ) / 1_000_000.0
            device_id = next(iter(device_ids))
            _, stream_id = next(iter(kernel_stream_keys))
        else:
            duration_ms = 0.0
            active_ms = 0.0
            copy_active_ms = 0.0
            device_id = None
            stream_id = None
        copy_names = [copy_map[int(row["copy_kind"])] for row in matched_copies]
        parents = sorted(
            {
                parent_name
                for parent_index, parent_name, parent_start, parent_end in phase_spans
                if parent_index != span_index
                and parent_start <= start and end <= parent_end
                and (parent_start, parent_end) != (start, end)
            }
        )
        range_samples.setdefault(name, []).append(
            {
                "duration_ms": duration_ms,
                "kernel_active_ms": active_ms,
                "copy_active_ms": copy_active_ms,
                "copy_bytes": sum(int(row["bytes"]) for row in matched_copies),
                "copy_stream_ids": copy_stream_ids,
                "copy_stream_mismatch_count": sum(
                    value != stream_id for value in copy_stream_ids
                ),
                "kernel_launch_count": len(matched_kernels),
                "h2d_copy_count": count_fragments(
                    copy_names, ("htod", "hosttodevice", "host to device")
                ),
                "d2h_copy_count": count_fragments(
                    copy_names, ("dtoh", "devicetohost", "device to host")
                ),
                "device_copy_count": len(copy_names) - count_fragments(
                    copy_names,
                    (
                        "htod", "hosttodevice", "host to device",
                        "dtoh", "devicetohost", "device to host",
                    ),
                ),
                "kernel_correlation_ids": sorted(kernel_correlations),
                "memcpy_correlation_ids": sorted(memcpy_correlations),
                "device_id": device_id,
                "stream_id": stream_id,
                "parent_range_names": parents,
            }
        )
    for correlation, owners in correlation_owners.items():
        family = correlation_families[correlation]
        for index, first in enumerate(owners):
            for second in owners[index + 1:]:
                if not _strictly_nested((first[2], first[3]), (second[2], second[3])):
                    raise EvidenceError(
                        f"Nsight {family} correlation {correlation} crosses sibling ranges"
                    )
    return {
        "cuda_kernel_launch_count": kernel_count,
        "memcpy_count": len(memcpy_names),
        "memcpy_bytes": sum(int(row["bytes"]) for row in memcpy_rows),
        "h2d_copy_count": count_fragments(memcpy_names, ("htod", "hosttodevice", "host to device")),
        "d2h_copy_count": count_fragments(memcpy_names, ("dtoh", "devicetohost", "device to host")),
        "synchronization_api_count": count_fragments(api_names, ("synchronize",)),
        "stream_wait_api_count": count_fragments(api_names, ("streamwaitevent", "waitexternal")),
        "runtime_api_count": len(runtime_rows),
        "driver_api_count": len(driver_rows),
        "nvtx_counts": {name: nvtx_names.count(name) for name in phase_names},
        "nvtx_ranges": range_samples,
    }


def _nsys_commands(
    config: RunnerConfig, *, group: str, variant_name: str, scenario: str,
    process_index: int, order: str, store: ArtifactStore,
) -> tuple[list[str], list[str], Path, Path]:
    variant = config.variant(group, variant_name)
    stem = f"profiles/nsys-{group}-{process_index:02d}-{order}-{scenario}-{variant_name}"
    report = store.root / f"{stem}.nsys-rep"
    sqlite_path = store.root / f"{stem}.sqlite"
    base_worker = worker_argv(
        variant, group=group, name=variant_name, process_index=process_index, order=order,
        munich_scene_xml=config.datasets.munich_scene_xml,
        sionna_source_root=config.datasets.sionna_source_root,
    )
    profile = [
        str(config.tools.nsys), "profile", "--trace=cuda,nvtx,osrt",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--force-overwrite=false", "--output", str(report),
        *base_worker, "--profile-only", scenario,
    ]
    export = [
        str(config.tools.nsys), "export", "--type=sqlite",
        "--force-overwrite=false", "--output", str(sqlite_path), str(report),
    ]
    return profile, export, report, sqlite_path


def _capture_reference(
    store: ArtifactStore,
    path: Path,
    *,
    label: str,
    minimum_mtime_ns: int,
) -> dict[str, object]:
    return store.inspect(
        store.relative_for_created_file(path),
        label=label,
        minimum_mtime_ns=minimum_mtime_ns,
    )


def run_nsys_matrix(
    config: RunnerConfig, gate: Mapping[str, object], *, group: str,
    timeout_seconds: int,
    store: ArtifactStore,
) -> dict[str, object]:
    tool = executable_identity(config.tools.nsys, label="Nsight Systems")
    captures: list[dict[str, object]] = []
    policy = gate["comparison_groups"][group]  # type: ignore[index]
    assert isinstance(policy, dict)
    manifest_group = load_profile_contract()["groups"][group]  # type: ignore[index]
    assert isinstance(manifest_group, dict)
    scenario = str(manifest_group["scenario"])
    for scheduled in process_schedule(gate, group):
        process_index = int(scheduled["process_index"])
        order = str(scheduled["order"])
        for raw_variant in scheduled["variants"]:
                variant_name = str(raw_variant)
                profile, export, report, sqlite_path = _nsys_commands(
                    config, group=group, variant_name=variant_name, scenario=scenario,
                    process_index=process_index, order=order, store=store,
                )
                environment = controlled_environment(config)
                profile_capture = run_captured(
                    profile, cwd=config.variant(group, variant_name).checkout,
                    environment=environment, timeout_seconds=timeout_seconds,
                    store=store,
                    stem=f"nsys-profile-{group}-{process_index:02d}-{scenario}-{variant_name}",
                )
                profile_capture.pop("stdout_bytes")
                profile_capture.pop("stderr_bytes")
                if not report.is_file():
                    raise EvidenceError("Nsight Systems profile did not create its bound report")
                export_capture = run_captured(
                    export, cwd=config.variant(group, variant_name).checkout,
                    environment=environment, timeout_seconds=timeout_seconds,
                    store=store,
                    stem=f"nsys-export-{group}-{process_index:02d}-{scenario}-{variant_name}",
                )
                export_capture.pop("stdout_bytes")
                export_capture.pop("stderr_bytes")
                report_reference = _capture_reference(
                    store,
                    report,
                    label="Nsight report",
                    minimum_mtime_ns=int(profile_capture["started_time_ns"]),
                )
                sqlite_reference = _capture_reference(
                    store,
                    sqlite_path,
                    label="Nsight SQLite",
                    minimum_mtime_ns=int(export_capture["started_time_ns"]),
                )
                store.verify_reference(sqlite_reference, label="Nsight SQLite before parse")
                timeline = parse_nsys_sqlite(sqlite_path)
                store.verify_reference(sqlite_reference, label="Nsight SQLite after parse")
                manifest_variant = manifest_group["variants"][variant_name]  # type: ignore[index]
                assert isinstance(manifest_variant, dict)
                required = tuple(manifest_variant["required_ranges"])  # type: ignore[arg-type]
                forbidden = set(manifest_variant["forbidden_ranges"])  # type: ignore[arg-type]
                multiplicity = policy["range_multiplicity_per_solve"][variant_name]  # type: ignore[index]
                if not isinstance(multiplicity, dict):
                    raise EvidenceError("range multiplicity is not frozen")
                ranges = timeline["nvtx_ranges"]
                assert isinstance(ranges, dict)
                if set(ranges) != set(required):
                    raise EvidenceError(
                        f"{scenario} NVTX range set differs from the fixed manifest"
                    )
                if forbidden & set(timeline["nvtx_counts"]):  # type: ignore[arg-type]
                    raise EvidenceError(f"{scenario} emitted a forbidden variant range")
                if any(
                    len(ranges[name]) != 7 * int(multiplicity[name]) for name in required
                ):
                    raise EvidenceError(
                        f"{scenario} NVTX range count differs from frozen multiplicity"
                    )
                captures.append(
                    {
                        "name": scenario,
                        "group": group,
                        "variant": variant_name,
                        "process_index": process_index,
                        "order": order,
                        "profile": profile_capture,
                        "sqlite_export": export_capture,
                        "report": report_reference,
                        "sqlite": sqlite_reference,
                        "timeline": timeline,
                    }
                )
    checks = validate_nsys_timelines(captures, gate, group=group)
    return {"tool": tool, "captures": captures, "checks": checks, "passed": all(row["passed"] for row in checks)}


def validate_nsys_timelines(
    captures: Sequence[Mapping[str, object]], gate: Mapping[str, object], *, group: str
) -> list[dict[str, object]]:
    if len(captures) != 10:
        raise EvidenceError("Nsight capture count is not the fixed five-pair matrix")
    observed_keys = {
        (
            int(capture.get("process_index", -1)),
            str(capture.get("order", "")),
            str(capture.get("name", "")),
            str(capture.get("variant", "")),
        )
        for capture in captures
    }
    expected_keys = {
        (int(row["process_index"]), str(row["order"]), scenario, str(variant))
        for row in process_schedule(gate, group)
        for scenario in (load_profile_contract()["groups"][group]["scenario"],)  # type: ignore[index]
        for variant in row["variants"]
    }
    if observed_keys != expected_keys:
        raise EvidenceError("Nsight captures do not match the canonical A/B schedule")
    group_budget = gate["comparison_groups"][group]["resource_budgets"]  # type: ignore[index]
    if not isinstance(group_budget, dict):
        raise EvidenceError("group resource budgets are not frozen")
    if set(group_budget) != {
        "nsys_candidate_counters", "kernel_launch_counts", "range_copy_counts",
        "peak_allocated_bytes_max", "peak_reserved_bytes_max",
        "diagnostic_peak_device_bytes_max", "diagnostic_peak_host_bytes_max",
        "diagnostic_artifact_bytes_max", "compiler_resources",
    }:
        raise EvidenceError("group resource budget fields are not exact")
    expected = group_budget["nsys_candidate_counters"]
    launch_budgets = group_budget["kernel_launch_counts"]
    copy_budgets = group_budget["range_copy_counts"]
    if not all(isinstance(value, dict) for value in (expected, launch_budgets, copy_budgets)):
        raise EvidenceError("Nsight range budgets are not frozen")
    policy = gate["comparison_groups"][group]  # type: ignore[index]
    assert isinstance(policy, dict)
    manifest_group = load_profile_contract()["groups"][group]  # type: ignore[index]
    assert isinstance(manifest_group, dict)
    scenario = str(manifest_group["scenario"])
    range_multiplicity = policy["range_multiplicity_per_solve"]
    marker_multiplicity = policy["marker_multiplicity_per_solve"]
    if not all(isinstance(value, dict) for value in (range_multiplicity, marker_multiplicity)):
        raise EvidenceError("variant-specific NVTX policy is malformed")
    for variant in ("baseline", "candidate"):
        manifest_variant = manifest_group["variants"][variant]  # type: ignore[index]
        assert isinstance(manifest_variant, dict)
        manifest_required = set(manifest_variant["required_ranges"])  # type: ignore[arg-type]
        manifest_markers = set(manifest_variant["required_markers"])  # type: ignore[arg-type]
        known = manifest_variant["known_range_multiplicity_per_solve"]
        assert isinstance(known, dict)
        if any(range_multiplicity[variant].get(name) != value for name, value in known.items()):
            raise EvidenceError("gate multiplicity differs from shared known facts")
        if set(range_multiplicity[variant]) != manifest_required:
            raise EvidenceError("range multiplicity key set differs from required ranges")
        if set(marker_multiplicity[variant]) != manifest_markers:
            raise EvidenceError("marker multiplicity key set differs from required markers")
        values = list(range_multiplicity[variant].values()) + list(marker_multiplicity[variant].values())
        if any(type(value) is not int or value <= 0 for value in values):
            raise EvidenceError("NVTX multiplicity must be a frozen positive integer")
    required_launch_names = {
        variant: set(manifest_group["variants"][variant]["required_ranges"])  # type: ignore[index]
        for variant in ("baseline", "candidate")
    }
    marker_names = set().union(
        *(
            set(manifest_group["variants"][variant]["required_markers"])  # type: ignore[index]
            for variant in ("baseline", "candidate")
        )
    )
    marker_counter_names = {
        name: name.removeprefix("witwin.channel_native:") for name in marker_names
    }
    if len(set(marker_counter_names.values())) != len(marker_counter_names):
        raise EvidenceError("profile marker counter names are not unique")
    if set(launch_budgets) != {"baseline", "candidate"} or any(
        set(launch_budgets[variant]) != required_launch_names[variant]
        for variant in ("baseline", "candidate")
    ):
        raise EvidenceError("kernel launch budget set differs from the range manifest")
    if set(copy_budgets) != {"baseline", "candidate"} or any(
        not isinstance(copy_budgets[variant], dict)
        or set(copy_budgets[variant]) != required_launch_names[variant]
        for variant in ("baseline", "candidate")
    ):
        raise EvidenceError("range copy budget set differs from the range manifest")
    for variant in ("baseline", "candidate"):
        for name in required_launch_names[variant]:
            row = copy_budgets[variant][name]
            if (
                not isinstance(row, dict)
                or set(row) != {
                    "h2d", "d2h", "device", "bytes_max", "streams_max",
                    "active_ms_max", "mismatch_max",
                }
                or any(
                    type(row[name]) is not int or int(row[name]) < 0
                    for name in (
                        "h2d", "d2h", "device", "bytes_max", "streams_max",
                        "mismatch_max",
                    )
                )
                or not isinstance(row["active_ms_max"], (int, float))
                or isinstance(row["active_ms_max"], bool)
                or not math.isfinite(float(row["active_ms_max"]))
                or float(row["active_ms_max"]) < 0.0
            ):
                raise EvidenceError("range copy budget is malformed")
    observations: dict[tuple[int, str], dict[str, object]] = {}
    launch_observations: dict[tuple[int, str], dict[str, list[int]]] = {}
    placement_observations: dict[tuple[int, str], tuple[int, int]] = {}
    for capture in captures:
        timeline = capture["timeline"]
        if not isinstance(timeline, dict):
            raise EvidenceError("Nsight capture timeline is malformed")
        ranges = timeline.get("nvtx_ranges")
        variant = str(capture["variant"])
        manifest_variant = manifest_group["variants"][variant]  # type: ignore[index]
        assert isinstance(manifest_variant, dict)
        required = tuple(manifest_variant["required_ranges"])  # type: ignore[arg-type]
        markers = tuple(manifest_variant["required_markers"])  # type: ignore[arg-type]
        forbidden = set(manifest_variant["forbidden_ranges"])  # type: ignore[arg-type]
        if (
            not isinstance(ranges, dict)
            or set(ranges) != set(required)
        ):
            raise EvidenceError("Nsight required-range manifest is not satisfied")
        range_checks: dict[str, bool] = {}
        launch_observations[(int(capture["process_index"]), str(capture["variant"]))] = {}
        for name in required:
            rows = ranges[name]
            expected_rows = 7 * int(range_multiplicity[variant][name])
            if not isinstance(rows, list) or len(rows) != expected_rows:
                raise EvidenceError(
                    f"Nsight range {name} count differs from frozen multiplicity"
                )
            for row in rows:
                if (
                    not isinstance(row, dict)
                    or set(row) != {
                        "duration_ms", "kernel_active_ms", "copy_active_ms",
                        "copy_bytes", "copy_stream_ids", "copy_stream_mismatch_count",
                        "kernel_launch_count",
                        "h2d_copy_count", "d2h_copy_count", "device_copy_count",
                        "kernel_correlation_ids", "memcpy_correlation_ids",
                        "device_id", "stream_id", "parent_range_names",
                    }
                    or not isinstance(row["duration_ms"], (int, float))
                    or isinstance(row["duration_ms"], bool)
                    or not math.isfinite(float(row["duration_ms"]))
                    or float(row["duration_ms"]) <= 0.0
                    or type(row["kernel_launch_count"]) is not int
                    or int(row["kernel_launch_count"]) <= 0
                    or not isinstance(row["kernel_active_ms"], (int, float))
                    or isinstance(row["kernel_active_ms"], bool)
                    or not math.isfinite(float(row["kernel_active_ms"]))
                    or float(row["kernel_active_ms"]) <= 0.0
                    or float(row["kernel_active_ms"]) > float(row["duration_ms"])
                    or not isinstance(row["copy_active_ms"], (int, float))
                    or isinstance(row["copy_active_ms"], bool)
                    or not math.isfinite(float(row["copy_active_ms"]))
                    or float(row["copy_active_ms"]) < 0.0
                    or type(row["copy_bytes"]) is not int
                    or int(row["copy_bytes"]) < 0
                    or not isinstance(row["copy_stream_ids"], list)
                    or row["copy_stream_ids"] != sorted(set(row["copy_stream_ids"]))
                    or not all(type(value) is int for value in row["copy_stream_ids"])
                    or any(int(value) < 0 for value in row["copy_stream_ids"])
                    or type(row["copy_stream_mismatch_count"]) is not int
                    or int(row["copy_stream_mismatch_count"]) < 0
                    or any(
                        type(row[field]) is not int or int(row[field]) < 0
                        for field in (
                            "h2d_copy_count", "d2h_copy_count", "device_copy_count"
                        )
                    )
                    or not isinstance(row["kernel_correlation_ids"], list)
                    or len(row["kernel_correlation_ids"]) != int(row["kernel_launch_count"])
                    or len(set(row["kernel_correlation_ids"])) != len(row["kernel_correlation_ids"])
                    or not all(type(value) is int for value in row["kernel_correlation_ids"])
                    or not isinstance(row["memcpy_correlation_ids"], list)
                    or len(set(row["memcpy_correlation_ids"])) != len(row["memcpy_correlation_ids"])
                    or not all(type(value) is int for value in row["memcpy_correlation_ids"])
                    or type(row["device_id"]) is not int
                    or int(row["device_id"]) < 0
                    or type(row["stream_id"]) is not int
                    or int(row["stream_id"]) < 0
                    or not isinstance(row["parent_range_names"], list)
                    or not all(isinstance(value, str) for value in row["parent_range_names"])
                ):
                    raise EvidenceError(f"Nsight range {name} sample is malformed")
            counts = {int(row["kernel_launch_count"]) for row in rows}
            copies = {
                (
                    int(row["h2d_copy_count"]), int(row["d2h_copy_count"]),
                    int(row["device_copy_count"]),
                )
                for row in rows
            }
            copy_budget = copy_budgets[variant][name]
            range_checks[name] = (
                counts == {launch_budgets[variant][name]}
                and copies == {
                    (
                        int(copy_budget["h2d"]), int(copy_budget["d2h"]),
                        int(copy_budget["device"]),
                    )
                }
                and max(int(row["copy_bytes"]) for row in rows)
                <= int(copy_budget["bytes_max"])
                and max(len(row["copy_stream_ids"]) for row in rows)
                <= int(copy_budget["streams_max"])
                and max(float(row["copy_active_ms"]) for row in rows)
                <= float(copy_budget["active_ms_max"])
                and max(int(row["copy_stream_mismatch_count"]) for row in rows)
                <= int(copy_budget["mismatch_max"])
            )
            launch_observations[
                (int(capture["process_index"]), str(capture["variant"]))
            ][name] = [int(row["kernel_launch_count"]) for row in rows]
        target_range = str(manifest_group["target_timing_range"])
        placement_observations[
            (int(capture["process_index"]), str(capture["variant"]))
        ] = _require_single_target_placement(
            ranges[target_range], name=target_range
        )
        if len(required) > 1:
            for name in required:
                if name == target_range:
                    continue
                if any(target_range not in row["parent_range_names"] for row in ranges[name]):
                    raise EvidenceError("Nsight stage range is not nested in the target range")
        observed = {
            "cuda_kernel_launch_count": timeline["cuda_kernel_launch_count"],
            "h2d_copy_count": timeline["h2d_copy_count"],
            "d2h_copy_count": timeline["d2h_copy_count"],
            "synchronization_api_count": timeline["synchronization_api_count"],
            "stream_wait_api_count": timeline["stream_wait_api_count"],
        }
        nvtx = timeline["nvtx_counts"]
        assert isinstance(nvtx, dict)
        if forbidden & set(nvtx):
            raise EvidenceError("Nsight emitted a forbidden variant range")
        expected_nvtx_names = set(required) | set(markers)
        if set(nvtx) != expected_nvtx_names:
            raise EvidenceError("Nsight range/auxiliary marker set is not exact")
        for name in required:
            if int(nvtx[name]) != 7 * int(range_multiplicity[variant][name]):
                raise EvidenceError("Nsight range name count differs from its span count")
        for name in markers:
            if int(nvtx[name]) != 7 * int(marker_multiplicity[variant][name]):
                raise EvidenceError("Nsight marker count differs from frozen multiplicity")
        observed.update(
            {counter: int(nvtx.get(name, 0)) for name, counter in marker_counter_names.items()}
        )
        all_required_samples = [sample for name in required for sample in ranges[name]]
        observed.update(
            {
                "range_copy_stream_count_max": max(
                    len(sample["copy_stream_ids"]) for sample in all_required_samples
                ),
                "range_copy_bytes_max": max(
                    int(sample["copy_bytes"]) for sample in all_required_samples
                ),
                "range_copy_active_ms_max": max(
                    float(sample["copy_active_ms"]) for sample in all_required_samples
                ),
                "range_copy_stream_mismatch_max": max(
                    int(sample["copy_stream_mismatch_count"])
                    for sample in all_required_samples
                ),
            }
        )
        observations[(int(capture["process_index"]), str(capture["variant"]))] = observed
        if not all(range_checks.values()):
            raise EvidenceError("range launch count differs from exact gate")
    checks: list[dict[str, object]] = []
    paired_fields = (
        "cuda_kernel_launch_count", "h2d_copy_count", "d2h_copy_count",
        "synchronization_api_count", "stream_wait_api_count",
        "range_copy_stream_count_max", "range_copy_bytes_max",
        "range_copy_active_ms_max",
        "range_copy_stream_mismatch_max",
        *sorted(marker_counter_names.values()),
    )
    for scheduled in process_schedule(gate, group):
        index = int(scheduled["process_index"])
        baseline = observations[(index, "baseline")]
        candidate = observations[(index, "candidate")]
        paired = {
            name: float(candidate[name]) <= float(baseline[name]) for name in paired_fields
        }
        common_launch_names = required_launch_names["baseline"] & required_launch_names["candidate"]
        launch_paired = {
            name: max(launch_observations[(index, "candidate")][name])
            <= max(launch_observations[(index, "baseline")][name])
            for name in common_launch_names
        }
        baseline_placement = placement_observations[(index, "baseline")]
        candidate_placement = placement_observations[(index, "candidate")]
        same_device = candidate_placement[0] == baseline_placement[0]
        passed = (
            candidate == expected
            and all(paired.values())
            and all(launch_paired.values())
            and same_device
        )
        checks.append(
            {
                "capture": scenario,
                "process_index": index,
                "order": scheduled["order"],
                "baseline": baseline,
                "candidate": candidate,
                "expected_candidate": expected,
                "paired_counter_checks": paired,
                "paired_launch_checks": launch_paired,
                "target_placement": {
                    "baseline": {
                        "device_id": baseline_placement[0],
                        "stream_id": baseline_placement[1],
                    },
                    "candidate": {
                        "device_id": candidate_placement[0],
                        "stream_id": candidate_placement[1],
                    },
                    "same_device": same_device,
                },
                "passed": passed,
            }
        )
    return checks


def attach_profile_timings(
    pairs: Sequence[Mapping[str, object]],
    captures: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
    *,
    group: str,
) -> list[dict[str, object]]:
    """Attach stage samples recomputed only from retained Nsight ranges."""
    normalized: list[dict[str, object]] = []
    by_key = {
        (
            int(capture["process_index"]),
            str(capture["variant"]),
            str(capture["name"]),
        ): capture
        for capture in captures
    }
    manifest_group = load_profile_contract()["groups"][group]  # type: ignore[index]
    assert isinstance(manifest_group, dict)
    for pair in pairs:
        pair_row = dict(pair)
        for variant in ("baseline", "candidate"):
            worker = dict(pair[variant])  # type: ignore[arg-type]
            profile_rows: list[dict[str, object]] = []
            scenario = str(manifest_group["scenario"])
            for scenario in (scenario,):
                capture = by_key.get((int(pair["process_index"]), variant, scenario))
                if capture is None or capture.get("order") != pair["order"]:
                    raise EvidenceError("Nsight stage sample is missing from its A/B pair")
                timeline = capture["timeline"]
                assert isinstance(timeline, dict)
                ranges = timeline["nvtx_ranges"]
                assert isinstance(ranges, dict)
                range_name = str(manifest_group["target_timing_range"])
                samples = ranges[range_name]
                profile_rows.append(
                    {
                        "name": range_name.removeprefix("witwin.channel_native:"),
                        "steady_cuda_ms": [float(row["duration_ms"]) for row in samples],
                        "source": "nsys_nvtx_sqlite",
                    }
                )
            worker["profile_timings"] = profile_rows
            pair_row[variant] = worker
        normalized.append(pair_row)
    return normalized


def parse_ncu_csv(path: Path, *, required_kernels: set[str]) -> dict[str, object]:
    reject_reparse_chain(path)
    classes: set[str] = set()
    kernels: set[str] = set()
    summaries: dict[tuple[str, str, str], dict[str, object]] = {}
    row_count = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
                raise EvidenceError("Nsight Compute CSV is empty or not regular")
            for index, row in enumerate(csv.DictReader(stream)):
                if index >= 1_000_000:
                    raise EvidenceError("Nsight Compute CSV exceeds the fixed row limit")
                folded = {str(key).strip().casefold(): value for key, value in row.items()}
                kernel = folded.get("kernel name") or folded.get("kernel")
                metric = folded.get("metric name") or folded.get("metric")
                value = folded.get("metric value") or folded.get("value")
                unit = folded.get("metric unit") or folded.get("unit") or ""
                if not kernel or not metric or value is None:
                    raise EvidenceError("Nsight Compute CSV lacks kernel/metric/value")
                try:
                    numeric = float(str(value).replace(",", ""))
                except ValueError as exc:
                    raise EvidenceError(f"Nsight Compute metric is not numeric: {value}") from exc
                if not math.isfinite(numeric):
                    raise EvidenceError("Nsight Compute metric is non-finite")
                kernels.add(str(kernel))
                folded_metric = str(metric).casefold()
                for fragment in ("register", "occupancy", "shared", "local", "branch"):
                    if fragment in folded_metric:
                        classes.add(fragment)
                key = str(kernel), str(metric), str(unit)
                summary = summaries.setdefault(
                    key, {"count": 0, "minimum": numeric, "maximum": numeric}
                )
                summary["count"] = int(summary["count"]) + 1
                summary["minimum"] = min(float(summary["minimum"]), numeric)
                summary["maximum"] = max(float(summary["maximum"]), numeric)
                row_count += 1
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise EvidenceError(f"cannot read Nsight Compute CSV: {exc}") from exc
    if any(
        getattr(before, name) != getattr(after, name)
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    ):
        raise EvidenceError("Nsight Compute CSV changed during parsing")
    if classes != {"register", "occupancy", "shared", "local", "branch"}:
        raise EvidenceError("Nsight Compute CSV lacks required resource classes")
    if not all(any(required in kernel for kernel in kernels) for required in required_kernels):
        raise EvidenceError("Nsight Compute CSV lacks a required target kernel family")
    return {
        "metric_row_count": row_count,
        "metric_summaries": [
            {
                "kernel": kernel, "metric": metric, "unit": unit,
                "count": summary["count"], "minimum": summary["minimum"],
                "maximum": summary["maximum"],
            }
            for (kernel, metric, unit), summary in sorted(summaries.items())
        ],
    }


def validate_ncu_blocker(stdout: bytes, stderr: bytes) -> None:
    if b"ERR_NVGPUCTRPERM" not in stdout + b"\n" + stderr:
        raise EvidenceError("Nsight Compute blocker output lacks ERR_NVGPUCTRPERM")


def run_ncu_candidate(
    config: RunnerConfig, *, timeout_seconds: int, store: ArtifactStore,
) -> dict[str, object]:
    contract_groups = load_profile_contract()["groups"]
    rows: list[dict[str, object]] = []
    for group in COMPARISON_GROUPS:
        scenario = str(contract_groups[group]["scenario"])  # type: ignore[index]
        required_kernels = set(
            contract_groups[group]["variants"]["candidate"]["ncu_kernel_families"]  # type: ignore[index]
        )
        variant = config.variant(group, "candidate")
        report = store.root / f"profiles/ncu-{group}.ncu-rep"
        csv_path = store.root / f"profiles/ncu-{group}.csv"
        argv = [
            str(config.tools.ncu), "--target-processes", "all", "--csv",
            "--log-file", str(csv_path), "--export", str(report),
            "--section", "LaunchStats", "--section", "Occupancy",
            "--section", "MemoryWorkloadAnalysis", "--section", "SourceCounters",
            *worker_argv(
                variant, group=group, name="candidate", process_index=0, order="AB",
                munich_scene_xml=config.datasets.munich_scene_xml,
                sionna_source_root=config.datasets.sionna_source_root,
            ),
            "--profile-only", scenario,
        ]
        capture = run_captured(
            argv, cwd=variant.checkout, environment=controlled_environment(config),
            timeout_seconds=timeout_seconds, store=store, stem=f"ncu-{group}",
            expected_returncode=None,
        )
        stdout = capture.pop("stdout_bytes")
        stderr = capture.pop("stderr_bytes")
        assert isinstance(stdout, bytes) and isinstance(stderr, bytes)
        if capture["returncode"] != 0:
            validate_ncu_blocker(stdout, stderr)
            if report.exists() or csv_path.exists():
                raise EvidenceError("blocked Nsight Compute capture left ambiguous output files")
            rows.append(
                {
                    "group": group, "scenario": scenario, "status": "blocked",
                    "capture": capture, "blocker": {"code": "ERR_NVGPUCTRPERM"},
                    "report": None, "csv": None, "metrics": None,
                    "required_kernels": sorted(required_kernels),
                }
            )
            continue
        report_reference = _capture_reference(
            store,
            report,
            label="Nsight Compute report",
            minimum_mtime_ns=int(capture["started_time_ns"]),
        )
        csv_reference = _capture_reference(
            store,
            csv_path,
            label="Nsight Compute CSV",
            minimum_mtime_ns=int(capture["started_time_ns"]),
        )
        store.verify_reference(csv_reference, label="Nsight Compute CSV before parse")
        metrics = parse_ncu_csv(csv_path, required_kernels=required_kernels)
        store.verify_reference(csv_reference, label="Nsight Compute CSV after parse")
        rows.append(
            {
                "group": group, "scenario": scenario, "status": "captured",
                "capture": capture, "report": report_reference, "csv": csv_reference,
                "metrics": metrics, "blocker": None,
                "required_kernels": sorted(required_kernels),
            }
        )
    statuses = {str(row["status"]) for row in rows}
    if len(statuses) != 1:
        raise EvidenceError("Nsight Compute permission/result status differs across fixed targets")
    return {
        "status": next(iter(statuses)),
        "tool": executable_identity(config.tools.ncu, label="Nsight Compute"),
        "targets": rows,
    }


__all__ = [
    "PROFILE_CONTRACT_REPO_PATH", "load_profile_contract",
    "attach_profile_timings", "parse_ncu_csv", "parse_nsys_sqlite",
    "run_ncu_candidate", "run_nsys_matrix", "validate_ncu_blocker",
    "validate_nsys_timelines",
]
