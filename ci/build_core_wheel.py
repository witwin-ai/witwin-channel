# Copyright Xingyu Chen.
# Build the locked Core wheel that the Channel wheel smoke installs beside it.

"""Build the locked Core wheel that the Channel wheel smoke installs beside it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR_ENV = "WITWIN_CORE_SOURCE_DIR"
CORE_DISTRIBUTION = "witwin"


class CoreSourceError(RuntimeError):
    """The Core checkout could not be identified."""


def _declares_core(candidate: Path) -> bool:
    pyproject = candidate / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return data.get("project", {}).get("name") == CORE_DISTRIBUTION


def resolve_core_source(environ: dict[str, str]) -> Path:
    """Return the Core checkout, or raise naming every place that was tried."""

    explicit = (environ.get(SOURCE_DIR_ENV) or "").strip()
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not _declares_core(candidate):
            raise CoreSourceError(
                f"{SOURCE_DIR_ENV}={explicit!r} is not a Core checkout: expected "
                f"a directory whose pyproject.toml declares name = "
                f"{CORE_DISTRIBUTION!r}. An explicit source directory is "
                "authoritative and never falls back."
            )
        return candidate

    candidates = [
        (REPO_ROOT.parent / "core").resolve(),
        (REPO_ROOT.parent.parent / "core").resolve(),
    ]
    for candidate in candidates:
        if _declares_core(candidate):
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise CoreSourceError(
        f"no Core checkout found. Set {SOURCE_DIR_ENV} to one, or place it "
        f"where the release workflow does. Tried: {tried}"
    )


def build_core_wheel(source: Path, outdir: Path, *, isolated: bool) -> Path:
    """Build exactly one Core wheel into ``outdir`` and return it."""

    outdir.mkdir(parents=True, exist_ok=True)
    for stale in outdir.glob("*.whl"):
        stale.unlink()
    command = [sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)]
    if not isolated:
        command.append("--no-isolation")
    command.append(str(source))
    print(f"[core-wheel] {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, check=True)
    wheels = sorted(outdir.glob("*.whl"))
    if len(wheels) != 1:
        raise CoreSourceError(
            f"expected exactly one Core wheel in {outdir}, found "
            f"{[wheel.name for wheel in wheels]}"
        )
    return wheels[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help=(
            "reuse the active environment's build backend, matching how the "
            "Channel wheel itself is built locally"
        ),
    )
    arguments = parser.parse_args(argv)

    try:
        source = resolve_core_source(dict(os.environ))
        wheel = build_core_wheel(
            source, arguments.outdir.resolve(), isolated=not arguments.no_isolation
        )
    except CoreSourceError as error:
        print(f"core wheel build failed: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"core wheel build failed: exit code {error.returncode}", file=sys.stderr)
        return error.returncode or 1

    print(f"core wheel OK: {wheel} (source {source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())