# Copyright Xingyu Chen.
# Checks that every tracked source file starts with a concise purpose header.

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cu", ".cuh"}
COPYRIGHT = "Copyright Xingyu Chen."
FORBIDDEN_HISTORY = re.compile(
    r"\b(?:ADR[- ]?\d+|phase(?:[- _]?\d+[a-z]?|\s+[A-Z]\b)|plan[- ]?\d+)\b",
    re.IGNORECASE,
)
PYTHON_ENCODING = re.compile(r"coding[:=]\s*[-\w.]+")


def _tracked_sources() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return [
        ROOT / item.decode("utf-8")
        for item in output.split(b"\0")
        if item and Path(item.decode("utf-8")).suffix.lower() in SOURCE_SUFFIXES
    ]


def _header_index(path: Path, lines: list[str]) -> int:
    if path.suffix.lower() != ".py":
        return 0
    index = 1 if lines and lines[0].startswith("#!") else 0
    if index < len(lines) and PYTHON_ENCODING.search(lines[index]):
        index += 1
    return index


def _check(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    index = _header_index(path, lines)
    prefix = "#" if path.suffix.lower() == ".py" else "//"
    expected_copyright = f"{prefix} {COPYRIGHT}"
    relative = path.relative_to(ROOT).as_posix()
    errors: list[str] = []

    if index >= len(lines) or lines[index] != expected_copyright:
        errors.append(f"{relative}: missing exact header '{expected_copyright}'")
        return errors
    if index + 1 >= len(lines) or not lines[index + 1].startswith(f"{prefix} "):
        errors.append(f"{relative}: missing one-line file purpose after copyright")
        return errors

    purpose = lines[index + 1][len(prefix) + 1 :].strip()
    if not purpose:
        errors.append(f"{relative}: file purpose is empty")
    if len(purpose) > 100:
        errors.append(f"{relative}: file purpose exceeds 100 characters")
    if not purpose.endswith("."):
        errors.append(f"{relative}: file purpose must end with a period")
    if FORBIDDEN_HISTORY.search(purpose):
        errors.append(f"{relative}: file purpose describes architecture history")
    if purpose.startswith(("-", "*")):
        errors.append(f"{relative}: file purpose must be a plain sentence")
    return errors


def main() -> int:
    errors = [error for path in _tracked_sources() for error in _check(path)]
    if errors:
        print("\n".join(errors))
        return 1
    print("source headers: all tracked source files have concise purpose headers")
    return 0


if __name__ == "__main__":
    sys.exit(main())