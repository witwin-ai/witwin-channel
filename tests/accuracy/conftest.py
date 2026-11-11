# Copyright Xingyu Chen.
# Collect each accuracy test's measured error and print them as one table at the end of the run.

from __future__ import annotations

import json
import os
from typing import Any

import pytest

_RECORDS: list[dict[str, Any]] = []
REPORT_ENV = "CHANNEL_ACCURACY_REPORT"


class AccuracyRecorder:
    """Record ``(name, error, tolerance)`` rows; ``check`` asserts after recording."""

    def __init__(self, test_id: str) -> None:
        self.test_id = test_id

    def record(self, name: str, error: float, tolerance: float, **extra: Any) -> bool:
        ok = bool(error <= tolerance)
        _RECORDS.append({"test": self.test_id, "case": name, "error": float(error),
                         "tolerance": float(tolerance), "ok": ok, **extra})
        return ok

    def check(self, name: str, error: float, tolerance: float, **extra: Any) -> None:
        assert self.record(name, error, tolerance, **extra), (
            f"{name}: error {error:.3e} exceeds tolerance {tolerance:.3e} {extra or ''}")


@pytest.fixture
def accuracy(request: pytest.FixtureRequest) -> AccuracyRecorder:
    return AccuracyRecorder(request.node.nodeid.split("::", 1)[-1])


def pytest_terminal_summary(terminalreporter: Any) -> None:
    if not _RECORDS:
        return
    terminalreporter.section("reference/accuracy errors")
    for row in _RECORDS:
        flag = "ok  " if row["ok"] else "FAIL"
        terminalreporter.write_line(
            f"{flag} {row['error']:10.3e} <= {row['tolerance']:9.2e}  {row['test']} :: {row['case']}")
    path = os.environ.get(REPORT_ENV)
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_RECORDS, handle, indent=1)
