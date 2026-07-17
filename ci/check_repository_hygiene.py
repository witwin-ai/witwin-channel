"""Reject tracked build artifacts, oversized blobs, and dirty worktrees."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MAX_TRACKED_BYTES = 10 * 1024 * 1024

_FORBIDDEN_DIRECTORY_NAMES = frozenset(
    {
        ".benchmarks",
        ".hypothesis",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "__pycache__",
        "artifacts",
        "build",
        "cmakefiles",
        "dist",
        "htmlcov",
    }
)
_FORBIDDEN_FILE_NAMES = frozenset(
    {
        ".ninja_deps",
        ".ninja_log",
        "build.ninja",
        "cmake_install.cmake",
        "cmakecache.txt",
        "coverage.xml",
    }
)
_FORBIDDEN_FILE_SUFFIXES = (
    ".a",
    ".bak",
    ".dll",
    ".dylib",
    ".egg",
    ".exe",
    ".exp",
    ".idb",
    ".ilk",
    ".lib",
    ".lastbuildstate",
    ".o",
    ".obj",
    ".pdb",
    ".prof",
    ".pyc",
    ".pyd",
    ".pyo",
    ".recipe",
    ".sln",
    ".so",
    ".swo",
    ".swp",
    ".temp",
    ".tlog",
    ".tmp",
    ".vcxproj",
    ".vcxproj.filters",
    ".whl",
)


class GitCommandError(RuntimeError):
    """Raised when repository metadata cannot be inspected."""


@dataclass(frozen=True, slots=True)
class TrackedFile:
    path: str
    object_id: str
    size: int


@dataclass(frozen=True, slots=True)
class Violation:
    kind: str
    path: str
    detail: str


def _git(root: Path, *args: str, input_data: bytes | None = None) -> bytes:
    command = ["git", "-C", str(root), *args]
    completed = subprocess.run(
        command,
        check=False,
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise GitCommandError(f"{' '.join(command)} failed: {error}")
    return completed.stdout


def repository_root(path: Path) -> Path:
    output = _git(path, "rev-parse", "--show-toplevel")
    return Path(output.decode("utf-8", errors="surrogateescape").strip())


def _index_entries(root: Path) -> list[tuple[str, str]]:
    output = _git(root, "ls-files", "--stage", "-z")
    entries: list[tuple[str, str]] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", maxsplit=1)
        _mode, raw_object_id, stage = metadata.split()
        if stage != b"0":
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        entries.append((path, raw_object_id.decode("ascii")))
    return entries


def _object_sizes(root: Path, object_ids: list[str]) -> dict[str, int]:
    unique_ids = list(dict.fromkeys(object_ids))
    if not unique_ids:
        return {}
    input_data = ("\n".join(unique_ids) + "\n").encode("ascii")
    output = _git(
        root,
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input_data=input_data,
    )
    sizes: dict[str, int] = {}
    for line in output.decode("ascii").splitlines():
        object_id, object_type, raw_size = line.split()
        if object_type == "blob":
            sizes[object_id] = int(raw_size)
    return sizes


def tracked_files(root: Path) -> list[TrackedFile]:
    entries = _index_entries(root)
    sizes = _object_sizes(root, [object_id for _path, object_id in entries])
    return [
        TrackedFile(path, object_id, sizes[object_id])
        for path, object_id in entries
        if object_id in sizes
    ]


def forbidden_path_reason(path: str) -> str | None:
    parts = path.replace("\\", "/").split("/")
    folded_parts = [part.casefold() for part in parts]
    for part in folded_parts[:-1]:
        if part in _FORBIDDEN_DIRECTORY_NAMES:
            return f"generated directory {part!r}"
        if part.startswith(("build-", "cmake-build-")):
            return f"generated build directory {part!r}"
        if part.endswith(".egg-info"):
            return f"generated package metadata directory {part!r}"

    name = folded_parts[-1]
    if name in _FORBIDDEN_FILE_NAMES:
        return f"generated file {name!r}"
    if name == ".coverage" or name.startswith(".coverage."):
        return "coverage data file"
    if name.endswith("~") or name.endswith(_FORBIDDEN_FILE_SUFFIXES):
        return f"temporary or compiled artifact {name!r}"
    if ".so." in name and name.rsplit(".so.", maxsplit=1)[1].replace(".", "").isdigit():
        return f"compiled shared library {name!r}"
    return None


def scan_repository(
    root: Path, *, max_tracked_bytes: int = DEFAULT_MAX_TRACKED_BYTES
) -> list[Violation]:
    violations: list[Violation] = []
    for tracked_file in tracked_files(root):
        reason = forbidden_path_reason(tracked_file.path)
        if reason is not None:
            violations.append(Violation("forbidden-path", tracked_file.path, reason))
        if tracked_file.size > max_tracked_bytes:
            violations.append(
                Violation(
                    "oversized-file",
                    tracked_file.path,
                    f"{tracked_file.size} bytes exceeds {max_tracked_bytes} bytes",
                )
            )
    return violations


def worktree_changes(root: Path) -> list[str]:
    output = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    return [
        record.decode("utf-8", errors="surrogateescape")
        for record in output.split(b"\0")
        if record
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository to inspect (defaults to this repository)",
    )
    parser.add_argument(
        "--max-tracked-bytes",
        type=int,
        default=DEFAULT_MAX_TRACKED_BYTES,
        help=f"maximum tracked blob size (default: {DEFAULT_MAX_TRACKED_BYTES})",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="skip the clean-worktree gate while retaining tracked-file checks",
    )
    args = parser.parse_args(argv)
    if args.max_tracked_bytes < 0:
        parser.error("--max-tracked-bytes must be non-negative")

    try:
        root = repository_root(args.root.resolve())
        violations = scan_repository(root, max_tracked_bytes=args.max_tracked_bytes)
        changes = [] if args.allow_dirty else worktree_changes(root)
    except GitCommandError as error:
        print(error, file=sys.stderr)
        return 2

    for violation in violations:
        print(
            f"{violation.kind}: {violation.path}: {violation.detail}",
            file=sys.stderr,
        )
    for change in changes:
        print(f"dirty-worktree: {change}", file=sys.stderr)
    if violations or changes:
        return 1

    print(f"repository hygiene passed for {len(tracked_files(root))} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
