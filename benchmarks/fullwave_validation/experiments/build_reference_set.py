# Copyright Xingyu Chen.
# Estimate, submit and store the Tidy3D full-wave reference set (tests/data/fullwave).

"""Estimate, submit and store the Tidy3D full-wave reference set (tests/data/fullwave)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.fullwave_validation import reference_set as rs  # noqa: E402


# A calibration case serves the cases whose dipole axis it shares by symmetry; a case at another
# frequency or grid names its own calibration case.
_CALIBRATION = {"z": "empty_z", "x": "empty_x", "y": "empty_x"}


def _calibration_case(case: rs.ReferenceCase) -> str:
    return case.calibration or _CALIBRATION[case.polarization]


_CALIBRATION_MIN_DISTANCE_M = 0.2


def _tasks(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def estimate(args: argparse.Namespace) -> None:
    from tidy3d import web

    tasks = _tasks(args.tasks)
    for case_id in args.cases or rs.case_ids():
        case = rs.load_reference_case(case_id)
        simulation = rs.build_tidy3d_simulation(case)
        task_id = web.upload(simulation, task_name=f"fw-{case_id}-{case.fingerprint[:8]}",
                             verbose=False)
        tasks[case_id] = {"task_id": task_id, "fingerprint": case.fingerprint,
                          "estimated_flex_credits": web.estimate_cost(task_id, verbose=False),
                          "cells": int(simulation.num_cells),
                          "time_steps": int(simulation.num_time_steps)}
        print(case_id, tasks[case_id], flush=True)
    args.tasks.write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    total = sum(entry["estimated_flex_credits"] for entry in tasks.values())
    print(f"estimated total: {total:.3f} FlexCredits")


def submit(args: argparse.Namespace) -> None:
    from tidy3d import web

    tasks = _tasks(args.tasks)
    total = sum(entry["estimated_flex_credits"] for entry in tasks.values())
    if total > args.max_credits:
        raise SystemExit(f"estimated {total:.3f} FlexCredits exceeds --max-credits {args.max_credits}")
    for case_id, entry in tasks.items():
        web.start(entry["task_id"])
    for case_id, entry in tasks.items():
        web.monitor(entry["task_id"], verbose=False)
        web.download(entry["task_id"], path=str(args.data_dir / f"{case_id}.hdf5"), verbose=False)
        entry["real_flex_credits"] = web.get_info(entry["task_id"]).realFlexUnit
    args.tasks.write_text(json.dumps(tasks, indent=1), encoding="utf-8")


def calibration_constant(case: rs.ReferenceCase, data_dir: Path) -> complex:
    """Complex scale from the Tidy3D phasor (conjugated) to the exact unit-dipole field."""

    import tidy3d as td

    empty = rs.load_reference_case(_calibration_case(case))
    plane = rs.extract_planes(td.SimulationData.from_file(str(data_dir / f"{empty.case_id}.hdf5")),
                              empty)[0]
    points = rs.plane_points(plane["x"], plane["y"], plane["z"])
    distance = np.linalg.norm(points - np.asarray(empty.tx_position), axis=-1)
    exact = rs.dipole_field(empty, points, near_field=True)
    usable = (distance > _CALIBRATION_MIN_DISTANCE_M) & (np.abs(exact) * distance > 0.3)
    ratio = np.conj(plane["field"][usable]) / exact[usable]
    return complex(np.median(ratio.real), np.median(ratio.imag))


def extract(args: argparse.Namespace) -> None:
    import tidy3d as td

    tasks = _tasks(args.tasks)
    for case_id in args.cases or rs.case_ids():
        case = rs.load_reference_case(case_id)
        if case.role == "calibration":
            continue
        scale = calibration_constant(case, args.data_dir)
        data = td.SimulationData.from_file(str(args.data_dir / f"{case_id}.hdf5"))
        planes = rs.extract_planes(data, case)
        for plane in planes:
            plane["field"] = np.conj(plane["field"]) / scale
            plane["regions"] = rs.classify_regions(case, plane["x"], plane["y"], plane["z"])
        task = tasks.get(case_id, {})
        metadata = {
            "solver": "tidy3d", "tidy3d_version": td.__version__,
            "task_id": task.get("task_id"), "flex_credits": task.get("real_flex_credits"),
            "field_component": "E" + case.polarization, "decimation": rs.DECIMATION,
            "normalization": "conj(Tidy3D phasor) / C: the exact unit-dipole field exp(-jkr)/r "
                             "times the transverse dipole projection, near-field terms included",
            "calibration_case": _calibration_case(case),
            "calibration_constant": [scale.real, scale.imag],
            "field_decay": _field_decay(data),
        }
        path = rs.save_reference(case, planes, metadata)
        print(case_id, path, path.stat().st_size, "bytes", flush=True)


def _field_decay(data) -> float | None:
    for line in reversed((data.log or "").splitlines()):
        if "field decay:" in line:
            return float(line.rsplit("field decay:", 1)[1].split()[0])
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("estimate", "submit", "extract"))
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--tasks", type=Path, required=True, help="task/estimate JSON ledger")
    parser.add_argument("--data-dir", type=Path, required=True, help="downloaded SimulationData")
    parser.add_argument("--max-credits", type=float, default=20.0)
    args = parser.parse_args(argv)
    {"estimate": estimate, "submit": submit, "extract": extract}[args.command](args)


if __name__ == "__main__":
    main()
