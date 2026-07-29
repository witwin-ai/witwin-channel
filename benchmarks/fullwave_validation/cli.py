# Copyright Xingyu Chen.
# Benchmarks cli.

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from .backends import (
    load_tidy3d_data,
    save_tidy3d_simulation,
    solve_deterministic,
    submit_tidy3d,
)
from .metrics import comparison_report
from .models import FieldMap
from .scenarios import MATERIALS, SCENARIOS, load_case


def _case_parser(subparsers: argparse._SubParsersAction, name: str, help_text: str):
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--material", choices=MATERIALS, required=True)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare channel deterministic fields with Tidy3D ground truth."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = _case_parser(
        commands, "prepare", "Write a Tidy3D simulation without submitting it."
    )
    prepare.add_argument("--output-dir", type=Path, required=True)

    deterministic = _case_parser(
        commands, "solve-deterministic", "Run channel deterministic."
    )
    deterministic.add_argument("--output", type=Path, required=True)

    tidy3d = _case_parser(
        commands, "solve-tidy3d", "Import or explicitly submit a Tidy3D solve."
    )
    source = tidy3d.add_mutually_exclusive_group(required=True)
    source.add_argument("--simulation-data", type=Path)
    source.add_argument(
        "--submit",
        action="store_true",
        help="Explicitly authorize a cloud task that may consume Tidy3D credits.",
    )
    tidy3d.add_argument("--output", type=Path, required=True)
    tidy3d.add_argument("--task-name")
    tidy3d.add_argument("--tidy3d-data-output", type=Path)

    compare = commands.add_parser(
        "compare", help="Compare two saved FieldMap references."
    )
    compare.add_argument("--deterministic", type=Path, required=True)
    compare.add_argument("--fullwave", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        report = comparison_report(
            FieldMap.load(args.deterministic), FieldMap.load(args.fullwave)
        )
        _write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    spec = load_case(args.scenario, args.material)
    if args.command == "prepare":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        simulation_path = save_tidy3d_simulation(
            spec, args.output_dir / f"{spec.case_id}.tidy3d.hdf5"
        )
        _write_json(
            args.output_dir / f"{spec.case_id}.case.json",
            {"case": asdict(spec), "case_fingerprint": spec.fingerprint},
        )
        print(simulation_path)
        return 0

    if args.command == "solve-deterministic":
        output = solve_deterministic(spec).save(args.output)
        print(output)
        return 0

    if args.simulation_data is not None:
        reference = load_tidy3d_data(args.simulation_data, spec)
    else:
        data_output = args.tidy3d_data_output or args.output.with_suffix(".tidy3d.hdf5")
        reference = submit_tidy3d(
            spec,
            task_name=args.task_name or f"channel-{spec.case_id}",
            data_path=data_output,
        )
    output = reference.save(args.output)
    print(output)
    return 0


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )