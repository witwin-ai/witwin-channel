from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .models import FieldMap


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    complex_scale_real: float
    complex_scale_imag: float
    raw_nmse: float
    calibrated_nmse: float
    magnitude_correlation: float
    magnitude_rmse_db: float


@dataclass(frozen=True, slots=True)
class MagnitudeMetrics:
    amplitude_scale: float
    reference_to_candidate_energy_ratio: float
    calibrated_reference_to_candidate_energy_ratio: float
    calibrated_magnitude_nmse: float
    magnitude_correlation: float
    magnitude_rmse_db: float
    complex_coherence: float


@dataclass(frozen=True, slots=True)
class BoundaryMetrics:
    kind: str
    component: str
    edge_count: int
    deterministic_jump_db_median: float | None
    deterministic_jump_db_p95: float | None
    deterministic_jump_db_max: float | None
    fullwave_jump_db_median: float | None
    fullwave_jump_db_p95: float | None
    fullwave_jump_db_max: float | None
    p95_excess_jump_db: float | None


def resample_regular(source: FieldMap, x: np.ndarray, y: np.ndarray) -> FieldMap:
    target_x = np.asarray(x, dtype=np.float64)
    target_y = np.asarray(y, dtype=np.float64)
    field = _resample_array(source.x, source.y, source.field, target_x, target_y)
    components = {
        name: _resample_array(source.x, source.y, value, target_x, target_y)
        for name, value in source.components.items()
    }
    return FieldMap(
        x=target_x,
        y=target_y,
        field=field,
        components=components,
        metadata=source.metadata,
    )


def _resample_array(
    source_x: np.ndarray,
    source_y: np.ndarray,
    values: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    x_margin = 0.51 * float(np.min(np.diff(source_x)))
    y_margin = 0.51 * float(np.min(np.diff(source_y)))
    if (
        target_x[0] < source_x[0] - x_margin
        or target_x[-1] > source_x[-1] + x_margin
        or target_y[0] < source_y[0] - y_margin
        or target_y[-1] > source_y[-1] + y_margin
    ):
        raise ValueError("target grid extends outside the full-wave reference grid")
    clipped_x = np.clip(target_x, source_x[0], source_x[-1])
    clipped_y = np.clip(target_y, source_y[0], source_y[-1])
    along_x = np.empty((source_y.size, target_x.size), dtype=np.complex128)
    for row in range(source_y.size):
        along_x[row] = _interp_complex(clipped_x, source_x, values[row])
    output = np.empty((target_y.size, target_x.size), dtype=np.complex128)
    for col in range(target_x.size):
        output[:, col] = _interp_complex(clipped_y, source_y, along_x[:, col])
    return output


def _interp_complex(x: np.ndarray, xp: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, values.real) + 1j * np.interp(x, xp, values.imag)


def compare_fields(
    candidate: FieldMap,
    reference: FieldMap,
    *,
    valid_mask: np.ndarray | None = None,
) -> ErrorMetrics:
    aligned = resample_regular(reference, candidate.x, candidate.y)
    mask = _validated_mask(valid_mask, candidate.field.shape)
    cand = candidate.field[mask]
    ref = aligned.field[mask]
    ref_energy = float(np.vdot(ref, ref).real)
    cand_energy = float(np.vdot(cand, cand).real)
    if ref_energy <= 0.0 or cand_energy <= 0.0:
        raise ValueError("candidate and reference fields must have non-zero energy")
    scale = np.vdot(cand, ref) / np.vdot(cand, cand)
    calibrated = scale * cand
    raw_nmse = float(np.vdot(cand - ref, cand - ref).real / ref_energy)
    calibrated_nmse = float(
        np.vdot(calibrated - ref, calibrated - ref).real / ref_energy
    )
    cand_mag = np.abs(calibrated)
    ref_mag = np.abs(ref)
    if np.std(cand_mag) == 0.0 or np.std(ref_mag) == 0.0:
        correlation = 1.0 if np.allclose(cand_mag, ref_mag) else 0.0
    else:
        correlation = float(np.corrcoef(cand_mag, ref_mag)[0, 1])
    floor = max(float(ref_mag.max()) * 1.0e-8, 1.0e-30)
    cand_db = 20.0 * np.log10(np.maximum(cand_mag, floor))
    ref_db = 20.0 * np.log10(np.maximum(ref_mag, floor))
    return ErrorMetrics(
        complex_scale_real=float(scale.real),
        complex_scale_imag=float(scale.imag),
        raw_nmse=raw_nmse,
        calibrated_nmse=calibrated_nmse,
        magnitude_correlation=correlation,
        magnitude_rmse_db=float(np.sqrt(np.mean((cand_db - ref_db) ** 2))),
    )


def compare_magnitudes(
    candidate: FieldMap,
    reference: FieldMap,
    *,
    valid_mask: np.ndarray | None = None,
    amplitude_scale: float | None = None,
) -> MagnitudeMetrics:
    """Compare field envelopes after matching RMS amplitude.

    Use this when the solvers do not export the same complex observable or
    source normalization. The scale is real and positive, so spatial phase
    disagreement cannot suppress the candidate field. By default it matches
    RMS amplitude; pass a scale measured from an empty-scene baseline to keep
    calibration independent of the scatterer under test.
    """
    aligned = resample_regular(reference, candidate.x, candidate.y)
    mask = _validated_mask(valid_mask, candidate.field.shape)
    cand = candidate.field[mask]
    ref = aligned.field[mask]
    cand_energy = float(np.vdot(cand, cand).real)
    ref_energy = float(np.vdot(ref, ref).real)
    if ref_energy <= 0.0 or cand_energy <= 0.0:
        raise ValueError("candidate and reference fields must have non-zero energy")

    energy_ratio = ref_energy / cand_energy
    scale = float(np.sqrt(energy_ratio) if amplitude_scale is None else amplitude_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("amplitude_scale must be finite and positive")
    cand_mag = scale * np.abs(cand)
    ref_mag = np.abs(ref)
    magnitude_nmse = float(np.sum((cand_mag - ref_mag) ** 2) / ref_energy)
    if np.std(cand_mag) == 0.0 or np.std(ref_mag) == 0.0:
        correlation = 1.0 if np.allclose(cand_mag, ref_mag) else 0.0
    else:
        correlation = float(np.corrcoef(cand_mag, ref_mag)[0, 1])
    floor = max(float(ref_mag.max()) * 1.0e-8, 1.0e-30)
    cand_db = 20.0 * np.log10(np.maximum(cand_mag, floor))
    ref_db = 20.0 * np.log10(np.maximum(ref_mag, floor))
    coherence = float(abs(np.vdot(cand, ref)) ** 2 / (cand_energy * ref_energy))
    return MagnitudeMetrics(
        amplitude_scale=scale,
        reference_to_candidate_energy_ratio=energy_ratio,
        calibrated_reference_to_candidate_energy_ratio=energy_ratio / scale**2,
        calibrated_magnitude_nmse=magnitude_nmse,
        magnitude_correlation=correlation,
        magnitude_rmse_db=float(np.sqrt(np.mean((cand_db - ref_db) ** 2))),
        complex_coherence=coherence,
    )


def analyze_boundaries(
    deterministic: FieldMap,
    fullwave: FieldMap,
    *,
    support_floor_db: float = -80.0,
    valid_mask: np.ndarray | None = None,
) -> dict[str, BoundaryMetrics]:
    aligned = resample_regular(fullwave, deterministic.x, deterministic.y)
    mask = _validated_mask(valid_mask, deterministic.field.shape)
    definitions = {"ISB": "los", "RSB": "reflection"}
    return {
        kind: _boundary_metrics(
            kind,
            component,
            deterministic,
            aligned,
            support_floor_db=support_floor_db,
            valid_mask=mask,
        )
        for kind, component in definitions.items()
    }


def _boundary_metrics(
    kind: str,
    component: str,
    deterministic: FieldMap,
    fullwave: FieldMap,
    *,
    support_floor_db: float,
    valid_mask: np.ndarray,
) -> BoundaryMetrics:
    if component not in deterministic.components:
        raise ValueError(f"deterministic reference is missing {component!r}")
    values = np.abs(deterministic.components[component])
    peak = float(values.max())
    active = (
        values > peak * 10.0 ** (support_floor_db / 20.0)
        if peak > 0.0
        else np.zeros_like(values, dtype=bool)
    )
    x_edges = active[:, 1:] != active[:, :-1]
    y_edges = active[1:, :] != active[:-1, :]
    x_edges &= valid_mask[:, 1:] & valid_mask[:, :-1]
    y_edges &= valid_mask[1:, :] & valid_mask[:-1, :]
    det_jumps = _edge_jumps_db(deterministic.field, x_edges, y_edges)
    full_jumps = _edge_jumps_db(fullwave.field, x_edges, y_edges)
    if det_jumps.size == 0:
        return BoundaryMetrics(
            kind, component, 0, None, None, None, None, None, None, None
        )
    det_p95 = float(np.percentile(det_jumps, 95.0))
    full_p95 = float(np.percentile(full_jumps, 95.0))
    return BoundaryMetrics(
        kind=kind,
        component=component,
        edge_count=int(det_jumps.size),
        deterministic_jump_db_median=float(np.median(det_jumps)),
        deterministic_jump_db_p95=det_p95,
        deterministic_jump_db_max=float(det_jumps.max()),
        fullwave_jump_db_median=float(np.median(full_jumps)),
        fullwave_jump_db_p95=full_p95,
        fullwave_jump_db_max=float(full_jumps.max()),
        p95_excess_jump_db=det_p95 - full_p95,
    )


def _edge_jumps_db(
    field: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray
) -> np.ndarray:
    magnitude = np.abs(field)
    floor = max(float(magnitude.max()) * 1.0e-10, 1.0e-30)
    db = 20.0 * np.log10(np.maximum(magnitude, floor))
    x_jump = np.abs(db[:, 1:] - db[:, :-1])[x_edges]
    y_jump = np.abs(db[1:, :] - db[:-1, :])[y_edges]
    return np.concatenate((x_jump, y_jump))


def _validated_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    normalized = np.asarray(mask, dtype=bool)
    if normalized.shape != shape:
        raise ValueError(f"valid_mask must have shape {shape}; got {normalized.shape}")
    if not np.any(normalized):
        raise ValueError("valid_mask must contain at least one valid sample")
    return normalized


def comparison_report(deterministic: FieldMap, fullwave: FieldMap) -> dict[str, Any]:
    _require_same_case(deterministic, fullwave)
    return {
        "schema": {
            "name": "witwin.channel_native.fullwave-comparison",
            "version": 1,
        },
        "case_id": deterministic.metadata.get("case_id"),
        "case_fingerprint": deterministic.metadata.get("case_fingerprint"),
        "field_metrics": asdict(compare_fields(deterministic, fullwave)),
        "magnitude_metrics": asdict(compare_magnitudes(deterministic, fullwave)),
        "boundaries": {
            name: asdict(metrics)
            for name, metrics in analyze_boundaries(deterministic, fullwave).items()
        },
    }


def _require_same_case(first: FieldMap, second: FieldMap) -> None:
    for key in ("case_id", "case_fingerprint", "frequency_hz"):
        left = first.metadata.get(key)
        right = second.metadata.get(key)
        if left is not None and right is not None and left != right:
            raise ValueError(f"reference mismatch for {key}: {left!r} != {right!r}")
