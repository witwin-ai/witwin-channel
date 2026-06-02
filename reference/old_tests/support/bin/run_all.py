"""Batch runner for channel bin figure-generation scripts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ._paths import FIGURES_DIR, OUTPUT_DIR, REPO_ROOT


MODULE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "forward": ("mesh_2d_detailed.png",),
    "grad_position": ("demo.png",),
    "grad_rotation": ("demo_rotation_grad.png",),
    "grad_transmitter": ("demo_tx_grad.png",),
    "grad_diffraction": ("diffraction_grad_vis.png",),
    "grad_reflection": ("reflection_grad_vis.png",),
    "grad_components": ("gradient_map.png",),
    "grad_multipath": (
        "grad_multipath.png",
        "grad_multipath_per_bounce.png",
        "grad_multipath_crosssection.png",
        "grad_multipath_second_order_combo.png",
    ),
    "compare_gradients": ("compare_gradients.png", "compare_gradients.svg"),
    "optimize": ("optimize.png",),
}

DEFAULT_MODULES: tuple[str, ...] = tuple(
    module for module in MODULE_OUTPUTS if module != "optimize"
)
PYTEST_DEFAULT_MODULES: tuple[str, ...] = DEFAULT_MODULES


def _parse_modules(raw_modules: list[str] | None) -> tuple[str, ...]:
    if not raw_modules:
        return DEFAULT_MODULES

    expanded: list[str] = []
    for raw in raw_modules:
        for chunk in str(raw).split(","):
            name = chunk.strip()
            if name:
                expanded.append(name)
    unknown = [name for name in expanded if name not in MODULE_OUTPUTS]
    if unknown:
        raise ValueError(f"Unknown bin modules: {', '.join(unknown)}")
    return tuple(expanded)


def _expected_paths(module: str) -> tuple[Path, ...]:
    return tuple(FIGURES_DIR / filename for filename in MODULE_OUTPUTS[module])


def run_modules(modules: tuple[str, ...]) -> list[dict[str, object]]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("WITWIN_CHANNEL_BIN_SHOW", "0")

    manifest: list[dict[str, object]] = []
    for module in modules:
        expected_paths = _expected_paths(module)
        for output_path in expected_paths:
            if output_path.exists():
                output_path.unlink()

        command = [sys.executable, "-m", f"tests.support.bin.{module}"]
        started_at = time.time()
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
        )
        missing_outputs = [
            str(path)
            for path in expected_paths
            if (not path.exists()) or path.stat().st_size <= 0 or path.stat().st_mtime < started_at - 1e-6
        ]
        record = {
            "module": module,
            "command": command,
            "returncode": result.returncode,
            "outputs": [str(path) for path in expected_paths],
            "stdout": result.stdout,
            "stderr": result.stderr,
            "missing_outputs": missing_outputs,
        }
        manifest.append(record)

        if result.returncode != 0:
            raise RuntimeError(
                f"Bin module {module} failed with return code {result.returncode}.\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        if missing_outputs:
            raise RuntimeError(
                f"Bin module {module} did not produce expected outputs: {missing_outputs}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

    manifest_path = OUTPUT_DIR / "bin_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run channel bin figure-generation scripts.")
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Bin module name to run. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--pytest-defaults",
        action="store_true",
        help="Run the default automated subset used by pytest.",
    )
    parser.add_argument(
        "--include-optimize",
        action="store_true",
        help="Include the long-running optimize module in the default run set.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available bin modules and exit.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for module in MODULE_OUTPUTS:
            print(module)
        return 0

    modules = PYTEST_DEFAULT_MODULES if args.pytest_defaults else _parse_modules(args.modules)
    if args.include_optimize and "optimize" not in modules:
        modules = tuple(modules) + ("optimize",)
    manifest = run_modules(modules)
    print(f"[OK] Generated {sum(len(item['outputs']) for item in manifest)} figures in {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
