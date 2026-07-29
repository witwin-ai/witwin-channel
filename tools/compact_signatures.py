# Copyright Xingyu Chen.
# Formats Python function signatures with compact 100-column parameter packing.

"""Formats Python function signatures with compact 100-column parameter packing."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import subprocess
import sys
import tokenize


ROOT = Path(__file__).resolve().parents[1]
COLUMN_LIMIT = 100


def _offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _absolute(offsets: list[int], position: tuple[int, int]) -> int:
    line, column = position
    return offsets[line - 1] + column


def _signature_edits(text: str) -> list[tuple[int, int, str]]:
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    offsets = _offsets(text)
    edits: list[tuple[int, int, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type != tokenize.NAME or token.string != "def":
            index += 1
            continue

        cursor = index + 1
        while cursor < len(tokens) and tokens[cursor].string != "(":
            cursor += 1
        if cursor == len(tokens):
            break
        opening = tokens[cursor]

        depth = 0
        commas: list[tokenize.TokenInfo] = []
        closing: tokenize.TokenInfo | None = None
        scan = cursor + 1
        while scan < len(tokens):
            current = tokens[scan]
            if current.type == tokenize.OP:
                if current.string in "([{":
                    depth += 1
                elif current.string in ")]}":
                    if current.string == ")" and depth == 0:
                        closing = current
                        break
                    depth -= 1
                elif current.string == "," and depth == 0:
                    commas.append(current)
            scan += 1
        if closing is None:
            index += 1
            continue

        colon_index = scan + 1
        suffix_depth = 0
        colon: tokenize.TokenInfo | None = None
        while colon_index < len(tokens):
            current = tokens[colon_index]
            if current.type == tokenize.OP:
                if current.string in "([{":
                    suffix_depth += 1
                elif current.string in ")]}":
                    suffix_depth -= 1
                elif current.string == ":" and suffix_depth == 0:
                    colon = current
                    break
            if current.type in {tokenize.NEWLINE, tokenize.ENDMARKER}:
                break
            colon_index += 1
        if colon is None:
            index += 1
            continue

        start = _absolute(offsets, (token.start[0], 0))
        end = _absolute(offsets, colon.end)
        opening_end = _absolute(offsets, opening.end)
        closing_start = _absolute(offsets, closing.start)
        prefix = text[start:opening_end]
        suffix = text[closing_start:end]
        if "\n" in prefix or "\r" in prefix or "\n" in suffix or "\r" in suffix:
            index = colon_index + 1
            continue

        boundaries = [opening.end, *(comma.end for comma in commas), closing.start]
        parameters: list[str] = []
        supported = True
        previous = boundaries[0]
        for boundary in boundaries[1:]:
            raw = text[_absolute(offsets, previous) : _absolute(offsets, boundary)]
            raw = raw[:-1] if raw.endswith(",") else raw
            parameter = raw.strip()
            if parameter:
                if "\n" in parameter or "\r" in parameter or "#" in parameter:
                    supported = False
                    break
                parameters.append(parameter)
            previous = boundary
        if not supported:
            index = colon_index + 1
            continue

        replacement = _render(prefix, parameters, suffix)
        if replacement != text[start:end]:
            edits.append((start, end, replacement))
        index = colon_index + 1
    return edits


def _render(prefix: str, parameters: list[str], suffix: str) -> str:
    one_line = f"{prefix}{', '.join(parameters)}{suffix}"
    if len(one_line) <= COLUMN_LIMIT:
        return one_line
    if not parameters:
        return one_line

    indent = prefix[: len(prefix) - len(prefix.lstrip())]
    continuation = f"{indent}    "
    packed: list[str] = []
    current = continuation
    for parameter in parameters:
        item = f"{parameter},"
        candidate = f"{current} {item}" if current != continuation else f"{current}{item}"
        if current != continuation and len(candidate) > COLUMN_LIMIT:
            packed.append(current)
            current = f"{continuation}{item}"
        else:
            current = candidate
    packed.append(current)
    return "\n".join((prefix, *packed, f"{indent}{suffix}"))


def format_text(text: str) -> str:
    """Return source with supported function signatures compacted."""

    for start, end, replacement in reversed(_signature_edits(text)):
        text = f"{text[:start]}{replacement}{text[end:]}"
    return text


def _tracked_python_files(paths: list[str]) -> list[Path]:
    if paths:
        result: set[Path] = set()
        for item in paths:
            path = (ROOT / item).resolve()
            if path.is_dir():
                result.update(path.rglob("*.py"))
            elif path.suffix == ".py":
                result.add(path)
        return sorted(result)

    output = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return [ROOT / item.decode("utf-8") for item in output.split(b"\0") if item]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report files needing changes")
    parser.add_argument("paths", nargs="*", help="files or directories; defaults to tracked Python")
    args = parser.parse_args(argv)

    changed: list[Path] = []
    for path in _tracked_python_files(args.paths):
        original = path.read_text(encoding="utf-8-sig")
        formatted = format_text(original)
        if formatted == original:
            continue
        changed.append(path)
        if not args.check:
            path.write_text(formatted, encoding="utf-8", newline="")

    for path in changed:
        print(path.relative_to(ROOT).as_posix())
    if args.check and changed:
        print(f"{len(changed)} Python file(s) need compact signature formatting")
        return 1
    if not changed:
        print("Python signatures use compact parameter packing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
