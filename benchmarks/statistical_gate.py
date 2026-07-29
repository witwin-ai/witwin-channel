# Copyright Xingyu Chen.
# Benchmarks statistical gate.

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Iterable


_T_CRITICAL_95 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
)
_T_CRITICAL_99 = (
    0.0,
    63.657,
    9.925,
    5.841,
    4.604,
    4.032,
    3.707,
    3.499,
    3.355,
    3.250,
    3.169,
    3.106,
    3.055,
    3.012,
    2.977,
    2.947,
)


def _student_t_critical(table: tuple[float, ...], count: int) -> float:
    if count <= 1:
        return 0.0
    return table[min(count - 1, len(table) - 1)]


@dataclass(frozen=True, slots=True)
class Observation:
    seed: int
    value: float | None
    finite_count: int
    total_count: int
    error: str | None = None


def summarize_observations(
    observations: Iterable[Observation], *, reference: float | None = None,
) -> dict[str, Any]:
    rows = tuple(observations)
    values = [
        float(row.value)
        for row in rows
        if row.error is None
        and row.value is not None
        and math.isfinite(float(row.value))
    ]
    finite_count = sum(int(row.finite_count) for row in rows)
    total_count = sum(int(row.total_count) for row in rows)
    failures = sum(row.error is not None for row in rows)
    mean = float(statistics.mean(values)) if values else None
    sample_variance = (
        float(statistics.variance(values))
        if len(values) > 1
        else 0.0
        if values
        else None
    )
    standard_error = (
        math.sqrt(sample_variance / len(values))
        if sample_variance is not None and values
        else None
    )

    def interval(critical: float) -> dict[str, float | None]:
        if mean is None or standard_error is None:
            return {"lower": None, "upper": None, "half_width": None}
        half_width = critical * standard_error
        return {
            "lower": mean - half_width,
            "upper": mean + half_width,
            "half_width": half_width,
        }

    relative_bias = None
    if reference is not None and mean is not None and reference != 0.0:
        relative_bias = abs(mean - reference) / abs(reference)
    return {
        "observations": [asdict(row) for row in rows],
        "attempt_count": len(rows),
        "success_count": len(values),
        "failure_count": failures,
        "failure_rate": failures / len(rows) if rows else 1.0,
        "finite_ratio": finite_count / total_count if total_count else 0.0,
        "mean": mean,
        "sample_variance": sample_variance,
        "standard_error": standard_error,
        "ci95": interval(_student_t_critical(_T_CRITICAL_95, len(values))),
        "ci99": interval(_student_t_critical(_T_CRITICAL_99, len(values))),
        "reference": reference,
        "absolute_bias": abs(mean - reference)
        if mean is not None and reference is not None
        else None,
        "relative_bias": relative_bias,
    }


def evaluate_thresholds(summary: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "failure_rate": summary["failure_rate"]
        <= float(thresholds["failure_rate_max"]),
        "finite_ratio": summary["finite_ratio"]
        >= float(thresholds["finite_ratio_min"]),
    }
    if "mean_min" in thresholds:
        checks["mean_min"] = summary["mean"] is not None and summary["mean"] >= float(
            thresholds["mean_min"]
        )
    if "relative_bias_max" in thresholds:
        checks["relative_bias"] = summary["relative_bias"] is not None and summary[
            "relative_bias"
        ] <= float(thresholds["relative_bias_max"])
    if thresholds.get("reference_in_ci99", False):
        reference = summary["reference"]
        interval = summary["ci99"]
        checks["reference_in_ci99"] = (
            reference is not None
            and interval["lower"] is not None
            and interval["lower"] <= reference <= interval["upper"]
        )
    if "relative_ci95_half_width_max" in thresholds:
        mean = summary["mean"]
        half_width = summary["ci95"]["half_width"]
        checks["relative_ci95_half_width"] = (
            mean is not None
            and mean != 0.0
            and half_width is not None
            and half_width / abs(mean)
            <= float(thresholds["relative_ci95_half_width_max"])
        )
    return checks


__all__ = ["Observation", "evaluate_thresholds", "summarize_observations"]