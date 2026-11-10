# Copyright Xingyu Chen.
# Generate the native source inventory and check the native translation-unit layout rules.

"""Generate the native source inventory and check the native translation-unit layout rules."""

# ``--write`` regenerates ``ci/native-inventory.json``: for every native source under
# ``native/channel`` it lists the ``__global__`` kernels it defines, the kernels it launches, the
# host synchronization or device-to-host copy calls it makes, whether it can trap on the device, and
# its per-file compile options. Entries are sorted name sets, never line numbers, counts or hashes,
# so a reviewer sees a new launch, sync or copy site as a one-line diff.
#
# The default mode regenerates the inventory in memory, diffs it against the committed file, and
# checks the layout rules directly against the source:
#
# * every native ``.cu`` and ``.cpp`` file is compiled exactly once by the ``_channel`` target;
# * only ``binding/module.cpp`` defines the pybind module and no binding file launches a kernel;
# * the Channel CMake build never enables ``--use_fast_math``;
# * no CUDA source reaches ``<torch/extension.h>`` or names bare ``torch::`` or ``at::kFloat32``;
# * a native header name never carries a ``common`` or ``plumbing`` suffix.

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = Path("ci/native-inventory.json")
NATIVE_PATH = Path("native/channel")
CMAKE_PATH = Path("CMakeLists.txt")
SCHEMA_VERSION = 1
GENERATOR = "python ci/check_native_inventory.py --write"
SOURCE_SUFFIXES = frozenset({".cu", ".cuh", ".cpp", ".h"})
COMPILED_SUFFIXES = frozenset({".cu", ".cpp"})

_GLOBAL_KERNEL = re.compile(r"__global__\s+[\w:<>,\s*&]*?\b(\w+)\s*\(")
_LAUNCH_BOUNDS = re.compile(r"__launch_bounds__\s*\([^)]*\)")
_LAUNCH = re.compile(r"\b(\w+)\s*(?:<[^<>;(){}]*>)?\s*<<<")
_HOST_SYNC = re.compile(
    r"\b(cudaStreamSynchronize|cudaDeviceSynchronize|cudaEventSynchronize|cudaMemcpy\w*)\s*\("
    r"|(\.item\s*<)|(\.cpu\s*\()|(\.to\s*\(\s*(?:at|torch)::kCPU)"
)
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_SOURCE_PROPERTIES = re.compile(
    r"set_source_files_properties\(\s*(?P<files>[^)]*?)\s+PROPERTIES\s+COMPILE_OPTIONS\s+\"(?P<options>[^\"]*)\"\s*\)"
)


def _strip_comments(text: str) -> str:
    return _COMMENT.sub("", text)


def _native_sources(repo: Path) -> list[Path]:
    root = repo / NATIVE_PATH
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in SOURCE_SUFFIXES)


def _compile_options(cmake: str) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    for match in _SOURCE_PROPERTIES.finditer(cmake):
        for name in match.group("files").split():
            options.setdefault(name, []).extend(match.group("options").split(";"))
    return {name: sorted(set(values)) for name, values in options.items()}


def generate_inventory(repo: Path) -> dict[str, object]:
    options = _compile_options((repo / CMAKE_PATH).read_text(encoding="utf-8"))
    files: dict[str, object] = {}
    for path in _native_sources(repo):
        relative = path.relative_to(repo).as_posix()
        text = _strip_comments(path.read_text(encoding="utf-8-sig"))
        syncs = set()
        for match in _HOST_SYNC.finditer(text):
            token = next(group for group in match.groups() if group)
            syncs.add(re.sub(r"\s+", "", token).rstrip("(<"))
        files[relative] = {
            "kernels": sorted(set(_GLOBAL_KERNEL.findall(_LAUNCH_BOUNDS.sub("", text)))),
            "launches": sorted(set(_LAUNCH.findall(text))),
            "host_synchronization": sorted(syncs),
            "device_trap": "__trap(" in text,
            "compile_options": options.get(relative, []),
        }
    return {"schema_version": SCHEMA_VERSION, "generated_by": GENERATOR, "files": files}


def render_inventory(inventory: dict[str, object]) -> str:
    lines = ["{"]
    lines.append(f'  "schema_version": {json.dumps(inventory["schema_version"])},')
    lines.append(f'  "generated_by": {json.dumps(inventory["generated_by"])},')
    lines.append('  "files": {')
    files = inventory["files"]
    assert isinstance(files, dict)
    for position, (name, entry) in enumerate(files.items()):
        lines.append(f"    {json.dumps(name)}: {{")
        keys = list(entry)
        for key_index, key in enumerate(keys):
            value = entry[key]
            comma = "," if key_index < len(keys) - 1 else ""
            if isinstance(value, list) and value:
                lines.append(f"      {json.dumps(key)}: [")
                lines.extend(
                    f"        {json.dumps(item)}{',' if index < len(value) - 1 else ''}"
                    for index, item in enumerate(value)
                )
                lines.append(f"      ]{comma}")
            else:
                lines.append(f"      {json.dumps(key)}: {json.dumps(value)}{comma}")
        lines.append("    }" + ("," if position < len(files) - 1 else ""))
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _reachable_includes(sources: list[Path], native_root: Path) -> set[Path]:
    pending = list(sources)
    reached: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in reached:
            continue
        reached.add(source)
        for include in re.findall(r'^\s*#include\s+"([^"]+)"', source.read_text(encoding="utf-8-sig"), re.M):
            candidate = (source.parent / include).resolve()
            if candidate.is_file() and candidate.is_relative_to(native_root):
                pending.append(candidate)
    return reached


def check_layout(repo: Path) -> list[str]:
    problems: list[str] = []
    cmake = (repo / CMAKE_PATH).read_text(encoding="utf-8")
    if "Python_add_library(" not in cmake:
        return ["CMakeLists.txt has no Python_add_library(_channel ...) target"]
    target_block = cmake.split("Python_add_library(", 1)[1].split("\n)", 1)[0]
    listed = [line.strip() for line in target_block.splitlines()]
    sources = _native_sources(repo)
    for path in sources:
        if path.suffix not in COMPILED_SUFFIXES:
            continue
        relative = path.relative_to(repo).as_posix()
        count = listed.count(relative)
        if count != 1:
            problems.append(f"{relative} is listed {count} times in the _channel target (expected once)")
    for entry in listed:
        if entry.startswith(NATIVE_PATH.as_posix()) and not (repo / entry).is_file():
            problems.append(f"_channel target lists a missing source: {entry}")
    if "--use_fast_math" in cmake:
        problems.append("CMakeLists.txt enables --use_fast_math for Channel sources")

    binding_root = repo / NATIVE_PATH / "binding"
    for path in sorted(binding_root.glob("*.cpp")):
        text = _strip_comments(path.read_text(encoding="utf-8-sig"))
        relative = path.relative_to(repo).as_posix()
        if "<<<" in text:
            problems.append(f"{relative} launches a kernel; bindings only validate and dispatch")
        if "PYBIND11_MODULE" in text and path.name != "module.cpp":
            problems.append(f"{relative} defines a pybind module; only binding/module.cpp may")

    kernels = repo / NATIVE_PATH / "kernels"
    cuda_sources = sorted(path for path in kernels.rglob("*") if path.suffix in {".cu", ".cuh"})
    for path in sorted(_reachable_includes(cuda_sources, (repo / NATIVE_PATH).resolve())):
        if "#include <torch/extension.h>" in path.read_text(encoding="utf-8-sig"):
            problems.append(f"{path.relative_to(repo.resolve()).as_posix()} reaches <torch/extension.h>")
    for path in cuda_sources:
        text = _strip_comments(path.read_text(encoding="utf-8-sig"))
        relative = path.relative_to(repo).as_posix()
        if re.search(r"(?<!rayd::)\btorch::", text):
            problems.append(f"{relative} names bare torch:: (use the ATen spelling)")
        if "at::kFloat32" in text:
            problems.append(f"{relative} names at::kFloat32 (use at::kFloat)")

    for path in sources:
        if path.suffix in {".h", ".cuh"} and re.search(r"common|plumbing", path.stem):
            problems.append(f"{path.relative_to(repo).as_posix()}: header names are owner nouns")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--write", action="store_true", help="regenerate the committed inventory")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    inventory_path = repo / INVENTORY_PATH
    rendered = render_inventory(generate_inventory(repo))
    if args.write:
        inventory_path.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"wrote {INVENTORY_PATH.as_posix()}")
    problems = check_layout(repo)
    committed = inventory_path.read_text(encoding="utf-8") if inventory_path.is_file() else ""
    if committed != rendered:
        sys.stdout.writelines(
            difflib.unified_diff(
                committed.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                fromfile=f"{INVENTORY_PATH.as_posix()} (committed)",
                tofile=f"{INVENTORY_PATH.as_posix()} (generated)",
                n=2,
            )
        )
        problems.append(f"{INVENTORY_PATH.as_posix()} is out of date; run: {GENERATOR}")
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"native inventory: {len(json.loads(rendered)['files'])} sources, layout rules hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
