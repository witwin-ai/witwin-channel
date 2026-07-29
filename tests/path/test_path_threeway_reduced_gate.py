# Copyright Xingyu Chen.
# Tests path threeway reduced gate.

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks import bench_path_solver_threeway as benchmark


def _record(
    delay_s: float,
    coefficient: complex,
    *,
    angle_offset: float = 0.0,
    position_offset: float = 0.0,
) -> dict[str, object]:
    return {
        "delay_s": delay_s,
        "coefficient": [coefficient.real, coefficient.imag],
        "angles_rad": {
            "theta_t": 0.5 + angle_offset,
            "phi_t": -0.25 + angle_offset,
            "theta_r": 1.0 + angle_offset,
            "phi_r": 0.75 + angle_offset,
        },
        "interaction_types": [1],
        "interaction_positions_m": [[1.0 + position_offset, 2.0, 3.0]],
    }


def _stats(records: list[dict[str, object]]) -> dict[str, object]:
    tau = np.array([[[record["delay_s"] for record in records]]], dtype=np.float64)
    valid = np.ones_like(tau, dtype=bool)
    labels = np.full(tau.shape, "reflection", dtype=object)
    indexed = {(0, 0, index): record for index, record in enumerate(records)}
    return benchmark._stats_from_labeled_paths(
        tau=tau,
        valid=valid,
        labels=labels,
        num_rx=1,
        num_tx=1,
        records_by_index=indexed,
        frequency_offsets_hz=(-1.0e6, 0.0, 1.0e6),
    )


def test_versioned_schema_and_cold_steady_memory_contract() -> None:
    calls = 0

    def operation() -> list[int]:
        nonlocal calls
        calls += 1
        return [calls]

    result, cold_ms, steady_ms, memory = benchmark._time_repeated(
        operation,
        lambda value: value,
        warmup=1,
        repeats=2,
    )

    assert benchmark.SCHEMA_NAME == "witwin.channel.path_solver_threeway"
    assert benchmark.SCHEMA_VERSION == "1.0.0"
    assert result == [4]
    assert cold_ms >= 0.0
    assert len(steady_ms) == 2
    assert memory["host_traced_peak_bytes"] >= memory["host_traced_current_bytes"]


def test_reduced_exact_gate_covers_complex_angles_geometry_cir_and_cfr() -> None:
    phase_error = 5.0e-4
    magnitude_ratio = 10.0 ** (0.1 / 20.0)
    reference_records = [
        _record(10.0e-9, 1.0 + 0.0j),
        _record(20.0e-9, 0.5 + 0.25j),
    ]
    native_records = [
        _record(
            float(record["delay_s"]) + 1.0e-11,
            complex(*record["coefficient"]) * magnitude_ratio * complex(math.cos(phase_error), math.sin(phase_error)),
            angle_offset=5.0e-4,
            position_offset=5.0e-4,
        )
        for record in reference_records
    ]
    native = {
        "provider": "channel",
        "cases": {"reflection": {"component_stats": _stats(native_records)}},
    }
    reference = {
        "provider": "sionna",
        "cases": {"reflection": {"component_stats": _stats(reference_records)}},
    }

    report = benchmark._component_delay_comparison(
        native,
        reference,
        component="reflection",
        case="reflection",
        tau_tol_s=1.0e-9,
        exact_counts=True,
    )

    assert report["comparison_mode"] == "exact_count"
    assert report["passed"]
    assert report["path_metrics"]["matched_paths"] == 2
    assert report["path_metrics"]["median_magnitude_error_db"] < 0.25
    assert report["signal_views"]["cir"]["finite"]
    assert report["signal_views"]["cfr"]["samples"] == 3


def test_diffraction_coverage_mode_and_confidence_interval() -> None:
    interval = benchmark._confidence_interval([0.96, 0.98, 1.0])

    assert interval["count"] == 3
    assert 0.0 <= interval["lower_95"] <= interval["mean"] <= interval["upper_95"] <= 1.0


def test_native_stats_extracts_padded_signal_and_geometry() -> None:
    path_shape = (1, 1, 1, 1, 2)
    result = SimpleNamespace(
        a=torch.tensor([[[[[[1.0 + 0.0j], [0.5 + 0.25j]]]]]], dtype=torch.complex64),
        tau=torch.tensor([[[[[1.0e-9, 2.0e-9]]]]], dtype=torch.float32),
        valid=torch.ones(path_shape, dtype=torch.bool),
        theta_t=torch.zeros(path_shape),
        phi_t=torch.zeros(path_shape),
        theta_r=torch.zeros(path_shape),
        phi_r=torch.zeros(path_shape),
        interaction_type=torch.tensor([[[[[[0], [1]]]]]], dtype=torch.int32),
        position=torch.zeros((*path_shape, 1, 3), dtype=torch.float32),
    )

    stats = benchmark._native_case_stats(
        result,
        num_rx=1,
        num_tx=1,
        frequency_offsets_hz=(0.0,),
    )

    assert stats["los"]["total"] == 1
    assert stats["reflection"]["total"] == 1
    assert stats["reflection"]["records_by_pair"]["0,0"][0]["coefficient"] == [
        0.5,
        0.25,
    ]


def test_provider_subprocess_preserves_negative_cfr_offsets(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def run(cmd, **kwargs):
        del kwargs
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=1, stdout="", stderr="expected")

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    args = SimpleNamespace(
        scene="san_francisco",
        sionna_source_root="sionna",
        channel_root="channel",
        frequency_hz=3.5e9,
        tx="0,0,180",
        rx="250,0,180",
        samples=256,
        max_num_paths=16,
        diffraction_state_budget=1024,
        inserted_reflection_state_budget=1024,
        warmup=0,
        repeats=1,
        seed=7,
        cfr_offsets_hz="-1000000,0,1000000",
    )

    benchmark._run_provider_subprocess(args, "native")

    assert "--cfr-offsets-hz=-1000000,0,1000000" in captured["cmd"]