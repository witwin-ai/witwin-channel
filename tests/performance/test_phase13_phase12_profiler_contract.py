from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from benchmarks.phase13_phase12.contracts import EvidenceError
from benchmarks.phase13_phase12.profilers import (
    _require_single_target_placement,
    parse_nsys_sqlite,
)


def _database(
    path: Path,
    *,
    ranges: tuple[tuple[str, int, int], ...],
    runtime: tuple[tuple[int, int, int | None, int], ...],
    kernels: tuple[tuple[int, int, int | None, int, int], ...],
    copies: tuple[tuple[int, int, int | None, int, int, int, int], ...] = (),
) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE ENUM_CUDA_RUNTIME_API (id INTEGER, name TEXT);
        INSERT INTO ENUM_CUDA_RUNTIME_API VALUES
          (1, 'cudaLaunchKernel'), (2, 'cudaMemcpyAsync');
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
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?, ?, ?)",
        copies,
    )
    connection.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)", ranges)
    connection.commit()
    connection.close()
    return path


@pytest.mark.parametrize(
    ("stream_id", "device_id"),
    ((-1, 0), (0, -1)),
)
def test_nsys_rejects_negative_gpu_placement_ids(
    tmp_path: Path, stream_id: int, device_id: int
) -> None:
    database = _database(
        tmp_path / f"negative-{stream_id}-{device_id}.sqlite",
        ranges=(("witwin.channel_native:stage", 100, 200),),
        runtime=((120, 130, 41, 1),),
        kernels=((1_000, 1_100, 41, stream_id, device_id),),
    )

    with pytest.raises(EvidenceError, match="negative device/stream id"):
        parse_nsys_sqlite(database)


def test_nsys_rejects_correlation_reuse_across_activity_families(
    tmp_path: Path,
) -> None:
    database = _database(
        tmp_path / "cross-family.sqlite",
        ranges=(("witwin.channel_native:stage", 100, 200),),
        runtime=((120, 125, 41, 1), (130, 135, 41, 2)),
        kernels=((1_000, 1_100, 41, 7, 0),),
        copies=((1_110, 1_120, 41, 7, 0, 3, 16),),
    )

    with pytest.raises(EvidenceError, match="reused across kernel/memcpy families"):
        parse_nsys_sqlite(database)


@pytest.mark.parametrize("second", ((120, 180), (150, 250)))
def test_nsys_rejects_overlapping_instances_of_one_required_range(
    tmp_path: Path, second: tuple[int, int]
) -> None:
    database = _database(
        tmp_path / f"overlap-{second[0]}-{second[1]}.sqlite",
        ranges=(
            ("witwin.channel_native:stage", 100, 200),
            ("witwin.channel_native:stage", *second),
        ),
        runtime=((130, 140, 41, 1),),
        kernels=((1_000, 1_100, 41, 7, 0),),
    )

    with pytest.raises(EvidenceError, match="overlapping instances"):
        parse_nsys_sqlite(database)


@pytest.mark.parametrize(("field", "replacement"), (("device_id", 1), ("stream_id", 8)))
def test_target_samples_must_share_one_device_and_stream(
    field: str, replacement: int
) -> None:
    rows = [{"device_id": 0, "stream_id": 7} for _ in range(7)]
    rows[-1][field] = replacement

    with pytest.raises(EvidenceError, match="cross CUDA devices/streams"):
        _require_single_target_placement(rows, name="witwin.channel_native:stage")


def test_target_samples_accept_one_device_and_stream() -> None:
    rows = [{"device_id": 0, "stream_id": 7} for _ in range(7)]

    assert _require_single_target_placement(
        rows, name="witwin.channel_native:stage"
    ) == (0, 7)
