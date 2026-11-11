# Copyright Xingyu Chen.
# Derives the release wheel build matrix, CUDA flags and checks from ci/support-matrix.toml.

"""Derive the release wheel build matrix, CUDA flags and checks from ci/support-matrix.toml."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "ci" / "support-matrix.toml"
SMOKE_SASS = (86, 120)
_TORCH_PIN = re.compile(r'"torch==[0-9][0-9A-Za-z.]*"')
_VERSION = re.compile(r'(?m)^version = "([^"]+)"$')


def load(path: Path = MATRIX) -> dict[str, Any]:
    # Imported here: the wheel jobs run the other commands on Python 3.10, which has no tomllib.
    import tomllib

    with path.open("rb") as stream:
        return tomllib.load(stream)


def torch_tag(version: str) -> str:
    """``2.11.0`` -> ``torch211``: the local version label of a release-asset wheel."""

    major, minor, *_ = version.split(".")
    return f"torch{major}{minor}"


def cuda_architectures(sass: list[int], ptx: list[int]) -> str:
    """The CMAKE_CUDA_ARCHITECTURES value for the given cubin and PTX targets."""

    return ";".join([*(f"{sm}-real" for sm in sass), *(f"{sm}-virtual" for sm in ptx)])


def gencode_flags(sass: list[int], ptx: list[int]) -> str:
    """nvcc flags that build each cubin from the lowest virtual architecture of its major family.

    One PTX compilation per major family keeps the build time near one compilation per family;
    the requested PTX targets are embedded from their own virtual architecture.
    """

    families: dict[int, list[int]] = {}
    for sm in sorted(sass):
        families.setdefault(sm // 10, []).append(sm)
    flags = []
    for major, members in families.items():
        codes = [f"sm_{sm}" for sm in members]
        virtual = members[0]
        codes += [f"compute_{sm}" for sm in ptx if sm == virtual]
        code = codes[0] if len(codes) == 1 else "[" + ",".join(codes) + "]"
        flags.append(f"--generate-code=arch=compute_{virtual},code={code}")
    for sm in ptx:
        if sm not in {members[0] for members in families.values()}:
            flags.append(f"--generate-code=arch=compute_{sm},code=compute_{sm}")
    return " ".join(flags)


def profile(matrix: dict[str, Any], *, smoke: bool = False) -> dict[str, str]:
    """The CUDA build profile: architectures, nvcc flags and the expected SASS/PTX lists."""

    sass = list(SMOKE_SASS) if smoke else list(matrix["sass"])
    ptx = list(matrix["ptx"])
    return {
        "cuda_architectures": cuda_architectures(sass, ptx),
        "gencode": gencode_flags(sass, ptx),
        "expected_sass": ",".join(str(sm) for sm in sass),
        "expected_ptx": ",".join(str(sm) for sm in ptx),
        "max_wheel_mib": str(matrix["max_wheel_mib"]),
    }


def build_matrix(matrix: dict[str, Any], *, smoke: bool = False) -> list[dict[str, str]]:
    """One entry per wheel: Python, PyTorch release, publishing channel and local version."""

    rows = []
    for python in matrix["pythons"]:
        for torch in matrix["torch"]:
            if smoke and (python != "3.11" or torch["channel"] != "pypi"):
                continue
            rows.append(
                {
                    "python": python,
                    "python_tag": "cp" + python.replace(".", ""),
                    "torch": torch["version"],
                    "channel": torch["channel"],
                    "local": "" if torch["channel"] == "pypi" else torch_tag(torch["version"]),
                }
            )
    return rows


def expected_wheels(matrix: dict[str, Any], channel: str, version: str) -> list[str]:
    """Filename prefixes and platform suffixes every wheel of one publishing channel must match."""

    names = []
    for row in build_matrix(matrix):
        if row["channel"] != channel:
            continue
        full = version if not row["local"] else f"{version}+{row['local']}"
        for platform in matrix["platforms"]:
            tag = row["python_tag"]
            names.append(f"witwin_channel-{full}-{tag}-{tag}-{platform}.whl")
    return sorted(names)


def select_torch(pyproject: Path, torch: str, local: str) -> None:
    """Pin ``torch==<torch>`` and append ``+<local>`` to the version for a release-asset wheel.

    The edit is a declared build parameter, not a source change, so the file is marked
    assume-unchanged and the build fingerprint keeps reporting a clean checkout.
    """

    text = pyproject.read_text(encoding="utf-8")
    if len(_TORCH_PIN.findall(text)) != 1 or len(_VERSION.findall(text)) != 1:
        raise SystemExit(f"{pyproject} must contain exactly one torch pin and one version")
    text = _TORCH_PIN.sub(f'"torch=={torch}"', text)
    if local:
        text = _VERSION.sub(lambda match: f'version = "{match.group(1)}+{local}"', text)
    pyproject.write_text(text, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(pyproject.parent), "update-index", "--assume-unchanged", pyproject.name],
        check=True,
    )


def check_wheels(matrix: dict[str, Any], directory: Path, channel: str, version: str) -> None:
    """Require exactly the expected wheels of one channel and enforce the size limit."""

    wheels = sorted(directory.glob("*.whl"))
    names = sorted(wheel.name for wheel in wheels)
    expected = expected_wheels(matrix, channel, version)
    if names != expected:
        raise SystemExit(f"{channel} wheels differ: expected {expected}, found {names}")
    check_sizes(wheels, int(matrix["max_wheel_mib"]))


def check_sizes(wheels: list[Path], max_mib: int) -> None:
    """Fail when a wheel is larger than ``max_mib`` MiB."""

    for wheel in wheels:
        size = wheel.stat().st_size
        print(f"{wheel.name}: {size / 1024 / 1024:.1f} MiB")
        if size > max_mib * 1024 * 1024:
            raise SystemExit(
                f"{wheel.name} is {size / 1024 / 1024:.1f} MiB, over the {max_mib} MiB "
                "limit in ci/support-matrix.toml"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    github = commands.add_parser("github-outputs", help="print GITHUB_OUTPUT lines")
    github.add_argument("--smoke", action="store_true")
    select = commands.add_parser("select-torch")
    select.add_argument("--pyproject", type=Path, required=True)
    select.add_argument("--torch", required=True)
    select.add_argument("--local", default="")
    check = commands.add_parser("check-wheels")
    check.add_argument("directory", type=Path)
    check.add_argument("--channel", required=True, choices=["pypi", "release-asset"])
    check.add_argument("--version", required=True)
    size = commands.add_parser("check-size")
    size.add_argument("wheels", type=Path, nargs="+")
    size.add_argument("--max-mib", type=int, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "select-torch":
        select_torch(arguments.pyproject, arguments.torch, arguments.local)
        return 0
    if arguments.command == "check-size":
        check_sizes(list(arguments.wheels), arguments.max_mib)
        return 0
    matrix = load()
    if arguments.command == "github-outputs":
        values = profile(matrix, smoke=arguments.smoke)
        values["build"] = json.dumps(build_matrix(matrix, smoke=arguments.smoke))
        for key, value in values.items():
            print(f"{key}={value}")
    else:
        check_wheels(matrix, arguments.directory, arguments.channel, arguments.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
