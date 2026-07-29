# Copyright Xingyu Chen.
# Tests nsys parser performance evidence.

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from benchmarks.phase13_phase12.contracts import EvidenceError
from benchmarks.phase13_phase12.profilers import parse_nsys_sqlite


def _database(
    path: Path, *, ranges: tuple[tuple[str, int, int], ...],
    runtime: tuple[tuple[int, int, int | None, int], ...],
    kernels: tuple[tuple[int, int, int | None, int, int], ...],
    copies: tuple[tuple[int, int, int | None, int, int, int, int], ...] = (),
) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE ENUM_CUDA_RUNTIME_API (id INTEGER, name TEXT);
        INSERT INTO ENUM_CUDA_RUNTIME_API VALUES
          (1, 'cudaLaunchKernel'),
          (2, 'cudaMemcpyAsync'),
          (3, 'cudaStreamSynchronize');
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
          start INTEGER, end INTEGER, correlationId INTEGER, cbid INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
          start INTEGER, end INTEGER, correlationId INTEGER,
          streamId INTEGER, deviceId INTEGER
        );
        CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER, name TEXT);
        INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES
          (1, 'HtoD'), (2, 'DtoH'), (3, 'DtoD');
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
          start INTEGER, end INTEGER, correlationId INTEGER,
          streamId INTEGER, deviceId INTEGER, copyKind INTEGER, bytes INTEGER
        );
        CREATE TABLE NVTX_EVENTS (text TEXT, start INTEGER, end INTEGER);
        """
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)", runtime
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)", kernels
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?, ?, ?)", copies
    )
    connection.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)", ranges)
    connection.commit()
    connection.close()
    return path


def test_nsys_uses_launch_correlation_and_gpu_timeline_for_stage_time(tmp_path: Path) -> None:
    database = _database(
        tmp_path / "capture.sqlite",
        ranges=(
            ("witwin.channel:total", 100, 300),
            ("witwin.channel:child", 110, 200),
        ),
        runtime=((120, 130, 41, 1), (140, 145, 42, 2), (146, 150, 43, 2)),
        # GPU work intentionally occurs after both CPU NVTX ranges close.
        kernels=((1_000, 1_100, 41, 7, 0),),
        copies=((900, 950, 42, 7, 0, 1, 4096), (1_150, 1_200, 43, 9, 0, 2, 2048)),
    )

    parsed = parse_nsys_sqlite(database)
    child = parsed["nvtx_ranges"]["witwin.channel:child"][0]

    assert child["duration_ms"] == pytest.approx(300 / 1_000_000)
    assert child["kernel_active_ms"] == pytest.approx(100 / 1_000_000)
    assert child["copy_active_ms"] == pytest.approx(100 / 1_000_000)
    assert child["copy_bytes"] == 6144
    assert child["copy_stream_ids"] == [7, 9]
    assert child["copy_stream_mismatch_count"] == 1
    assert child["kernel_correlation_ids"] == [41]
    assert child["memcpy_correlation_ids"] == [42, 43]
    assert child["h2d_copy_count"] == 1
    assert child["d2h_copy_count"] == 1
    assert child["device_copy_count"] == 0
    assert child["parent_range_names"] == ["witwin.channel:total"]


def test_nsys_rejects_missing_or_duplicate_kernel_correlation(tmp_path: Path) -> None:
    missing = _database(
        tmp_path / "missing.sqlite",
        ranges=(("witwin.channel:stage", 100, 200),),
        runtime=((120, 130, 41, 1),),
        kernels=((1_000, 1_100, 99, 7, 0),),
    )
    with pytest.raises(EvidenceError, match="missing or ambiguous"):
        parse_nsys_sqlite(missing)

    duplicate = _database(
        tmp_path / "duplicate.sqlite",
        ranges=(("witwin.channel:stage", 100, 200),),
        runtime=((120, 130, 41, 1),),
        kernels=((1_000, 1_100, 41, 7, 0), (1_110, 1_200, 41, 7, 0)),
    )
    with pytest.raises(EvidenceError, match="missing or ambiguous"):
        parse_nsys_sqlite(duplicate)


def test_nsys_rejects_memcpy_api_without_gpu_activity(tmp_path: Path) -> None:
    database = _database(
        tmp_path / "missing-copy.sqlite",
        ranges=(("witwin.channel:stage", 100, 200),),
        runtime=((120, 130, 41, 1), (140, 150, 42, 2)),
        kernels=((1_000, 1_100, 41, 7, 0),),
    )

    with pytest.raises(EvidenceError, match="memcpy APIs without GPU activity"):
        parse_nsys_sqlite(database)


def test_nsys_allows_nested_ownership_but_rejects_sibling_contamination(tmp_path: Path) -> None:
    database = _database(
        tmp_path / "sibling.sqlite",
        ranges=(
            ("witwin.channel:first", 100, 190),
            ("witwin.channel:second", 150, 240),
        ),
        runtime=((160, 170, 41, 1),),
        kernels=((1_000, 1_100, 41, 7, 0),),
    )

    with pytest.raises(EvidenceError, match="crosses sibling ranges"):
        parse_nsys_sqlite(database)


def test_nsys_rejects_ambiguous_multi_stream_stage(tmp_path: Path) -> None:
    database = _database(
        tmp_path / "streams.sqlite",
        ranges=(("witwin.channel:stage", 100, 200),),
        runtime=((120, 125, 41, 1), (130, 135, 42, 1)),
        kernels=((1_000, 1_100, 41, 7, 0), (1_110, 1_200, 42, 8, 0)),
    )

    with pytest.raises(EvidenceError, match="multi-stream"):
        parse_nsys_sqlite(database)


def test_nsys_records_copy_streams_but_rejects_cross_device_copy(tmp_path: Path) -> None:
    database = _database(
        tmp_path / "cross-device.sqlite",
        ranges=(("witwin.channel:stage", 100, 200),),
        runtime=((120, 125, 41, 1), (130, 135, 42, 2)),
        kernels=((1_000, 1_100, 41, 7, 0),),
        copies=((900, 950, 42, 8, 1, 1, 1024),),
    )

    with pytest.raises(EvidenceError, match="crosses CUDA devices"):
        parse_nsys_sqlite(database)