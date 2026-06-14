from __future__ import annotations

import argparse
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "support" / "bin"))

from benchmark_munich_deterministic_native_vs_original import main as parity_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--artifact-dir", default=str(_REPO_ROOT / "artifacts" / "deterministic_munich"))
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--original-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--original-enable-rd-diffraction", action="store_true")
    args = parser.parse_args()
    parity_args = [
        "--artifact-dir",
        args.artifact_dir,
        "--grid-size",
        str(int(args.grid_size)),
        "--max-depth",
        str(int(args.max_depth)),
        "--original-timeout-seconds",
        str(float(args.original_timeout_seconds)),
        "--warmup-runs",
        str(int(args.warmup_runs)),
    ]
    if bool(args.original_enable_rd_diffraction):
        parity_args.append("--original-enable-rd-diffraction")
    if args.json:
        parity_args.append("--json")
    parity_main(parity_args)


if __name__ == "__main__":
    main()
