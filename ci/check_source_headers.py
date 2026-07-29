# Copyright Xingyu Chen.
# Checks source headers, docstrings, and comments for plain current-purpose prose.

"""Checks source headers, docstrings, and comments for plain current-purpose prose."""

from __future__ import annotations

import ast
import io
from pathlib import Path
import re
import subprocess
import sys
import tokenize


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cu", ".cuh"}
COPYRIGHT = "Copyright Xingyu Chen."
FORBIDDEN_HISTORY = re.compile(
    r"\b(?:ADR[- ]?\d+|phase(?:[- _]?\d+[a-z]?|\s+[A-Z]\b)|plan[- ]?\d+)\b",
    re.IGNORECASE,
)
FORBIDDEN_COMMENT_JARGON = re.compile(
    r"\b(?:ADR(?:[- ]?\d+)?|Plan[- ]?\d+|Phase[- ]?\d+[A-Za-z]?)\b"
    r"|\b(?:plan|contract|implementation notes?)\s+sections?[- ]?\s*\d"
    r"|\b(?:spec|ruling|wave)\s+\d"
    r"|\bsections?[- ]\s*\d"
    r"|\baudit\s+[A-Z]{1,4}-?\d"
    r"|(?<![A-Za-z0-9_])(?:F[1-6][a-z]?|G[0-5][a-z]?|R[3-5][a-z]?)"
    r"(?![A-Za-z0-9_])"
    r"|\b(?:historical|history|cutover|migration|supersed\w*|former(?:ly)?|"
    r"previously|used to|no longer|re[- ]?baselined|re[- ]?keyed|later phases?|accepted phases?|native ownership transfer|"
    r"scene ownership transfer|frozen native ownership snapshot|ownership "
    r"audit|release evidence|consolidated from|split out of|moved from|concept-axis|current implementation)\b",
    re.IGNORECASE,
)
FORBIDDEN_PHASE_LABEL = re.compile(r"\bPhase\s+[A-Z]\b")
PYTHON_ENCODING = re.compile(r"coding[:=]\s*[-\w.]+")
NATIVE_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


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


def _check_prose(relative: str, line: int, kind: str, prose: str) -> list[str]:
    if FORBIDDEN_COMMENT_JARGON.search(prose) or FORBIDDEN_PHASE_LABEL.search(
        prose
    ):
        return [
            f"{relative}:{line}: {kind} describes architecture history "
            "instead of current behavior"
        ]
    return []


def _python_docstrings(tree: ast.AST) -> list[tuple[int, str, str]]:
    result: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if docstring is None:
            continue
        line = node.body[0].lineno
        kind = "module docstring" if isinstance(node, ast.Module) else "docstring"
        result.append((line, kind, docstring))
    return result


def _check_comment_prose(path: Path, text: str) -> list[str]:
    relative = path.relative_to(ROOT).as_posix()
    errors: list[str] = []
    if path.suffix.lower() == ".py":
        tree = ast.parse(text)
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                errors.extend(
                    _check_prose(relative, token.start[0], "comment", token.string)
                )
        for line, kind, docstring in _python_docstrings(tree):
            errors.extend(_check_prose(relative, line, kind, docstring))
            if kind == "module docstring" and (
                "\n" in docstring or len(docstring) > 120
            ):
                errors.append(
                    f"{relative}:{line}: module docstring must be one concise "
                    "sentence of at most 120 characters"
                )
        return errors

    for match in NATIVE_COMMENT.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        errors.extend(_check_prose(relative, line, "comment", match.group()))
    return errors


def _check(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
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
    errors.extend(_check_comment_prose(path, text))
    return errors


def main() -> int:
    errors = [error for path in _tracked_sources() for error in _check(path)]
    if errors:
        print("\n".join(errors))
        return 1
    print("source prose: headers, docstrings, and comments describe current behavior")
    return 0


if __name__ == "__main__":
    sys.exit(main())