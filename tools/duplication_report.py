# Copyright Xingyu Chen.
# Exact-token duplicate-region detector for channel production sources.

"""Exact-token duplicate-region detector for channel production sources.

The report is static-first: it tokenizes Python and C++/CUDA sources without
importing ``witwin.channel`` or loading its native extension. Two
metrics are produced side by side and never mixed:

* duplicate *regions* are maximal token spans of at least ``MIN_TOKENS`` tokens
  that occur two or more times (within or across files of the same corpus).
  Each region carries a stable ``region_id`` derived from its token content and
  the exact source spans of every occurrence. Regions are the unit that the
  classification ledger (``docs/dev/audit/duplication-classification.json``)
  and gate (``ci/check_duplication.py``) reason about.
* duplicate *line coverage* is the union of physical source lines touched by any
  repeated token window. Coverage is computed independently of region
  enumeration so that nested or partially overlapping repeats are counted once
  and never double counted.

The Python tokenizer uses the standard library ``tokenize`` module, drops
comment/indent/newline noise and statement-leading docstring strings, and keeps
every remaining code token verbatim (exact-token metric, no identifier
normalization). The native tokenizer strips comments and string/char contents
and splits the remainder into identifiers, numbers, and operators.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import re
import sys
import tokenize
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 1
MIN_TOKENS = 100
PACKAGE_RELATIVE = "witwin/channel"
NATIVE_RELATIVE = "native/channel"
NATIVE_SUFFIXES = frozenset({".cpp", ".cu", ".cuh", ".h", ".hpp"})
_EXCLUDED_PARTS = frozenset({"tests", "benchmarks", "__pycache__"})
_STRING_PLACEHOLDER = '"<str>"'
_CHAR_PLACEHOLDER = "'<chr>'"
# Polynomial rolling-hash constants (Rabin-Karp). The 61-bit Mersenne modulus
# keeps every intermediate product inside Python's fast small-int path.
_HASH_MOD = (1 << 61) - 1
_HASH_BASE = 1_000_003


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One tokenized production source, ready for duplicate analysis."""

    path: str
    line_count: int
    token_values: tuple[str, ...]
    token_lines: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Occurrence:
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class Region:
    region_id: str
    corpus: str
    token_count: int
    occurrences: tuple[Occurrence, ...]


@dataclass(frozen=True, slots=True)
class CorpusResult:
    corpus: str
    file_count: int
    total_lines: int
    duplicate_lines: int
    regions: tuple[Region, ...]
    per_file_duplicate_lines: dict[str, int]
    per_file_total_lines: dict[str, int]

    @property
    def coverage_percent(self) -> float:
        return _percent(self.duplicate_lines, self.total_lines)


def _percent(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round(100.0 * part / whole, 6)


# --------------------------------------------------------------------------- #
# Python tokenization
# --------------------------------------------------------------------------- #
_PY_TRIVIA = frozenset(
    {
        tokenize.ENCODING,
        tokenize.NL,
        tokenize.COMMENT,
        tokenize.INDENT,
        tokenize.DEDENT,
    }
)
_PY_STATEMENT_END = frozenset({tokenize.NEWLINE, tokenize.ENDMARKER})


def python_code_tokens(source: str) -> list[tuple[str, int]]:
    """Return ``(token_value, line)`` pairs for real Python code tokens.

    Comments, indentation, and newline bookkeeping tokens are dropped. Any
    statement-leading string expression (module, class, and function
    docstrings, and bare string statements) is dropped as well; every other
    token is kept with its exact source text.
    """

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    result: list[tuple[str, int]] = []
    index = 0
    count = len(tokens)
    at_statement_start = True
    while index < count:
        token = tokens[index]
        if token.type in _PY_TRIVIA:
            index += 1
            continue
        if token.type == tokenize.NEWLINE:
            at_statement_start = True
            index += 1
            continue
        if token.type == tokenize.ENDMARKER:
            break
        if token.type == tokenize.STRING and at_statement_start:
            index = _consume_leading_string_run(tokens, index, result)
            at_statement_start = False
            continue
        result.append((token.string, token.start[0]))
        at_statement_start = False
        index += 1
    return result


def _consume_leading_string_run(
    tokens: Sequence[tokenize.TokenInfo],
    index: int,
    result: list[tuple[str, int]],
) -> int:
    """Handle a string that opens a logical line; return the next index.

    A run of adjacent string literals that forms the entire statement is a
    docstring or bare string statement and is dropped. Otherwise the strings
    belong to an expression and are emitted verbatim.
    """

    run: list[tokenize.TokenInfo] = []
    cursor = index
    count = len(tokens)
    while cursor < count and tokens[cursor].type in (
        tokenize.STRING,
        tokenize.NL,
        tokenize.COMMENT,
    ):
        if tokens[cursor].type == tokenize.STRING:
            run.append(tokens[cursor])
        cursor += 1
    following = tokens[cursor] if cursor < count else None
    is_bare_statement = following is None or following.type in _PY_STATEMENT_END
    if not is_bare_statement:
        for string_token in run:
            result.append((string_token.string, string_token.start[0]))
    return cursor


# --------------------------------------------------------------------------- #
# Native (C/CUDA) tokenization
# --------------------------------------------------------------------------- #
_NATIVE_TOKEN = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|'
    r"[A-Za-z_]\w*|(?:\d+\.\d*|\.\d+|\d+)(?:[eEpP][+-]?\d+)?[A-Za-z_]*|"
    r"::|->\*|->|<<=|>>=|<=>|==|!=|<=|>=|&&|\|\||\+\+|--|"
    r"\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>|\.\.\.|[^\s]"
)


def _mask_native_comments(source: str) -> str:
    """Replace ``//`` and ``/* */`` comments with spaces, preserving newlines."""

    output = list(source)
    index = 0
    length = len(source)
    quote: str | None = None
    escaped = False
    while index < length:
        char = source[index]
        following = source[index + 1] if index + 1 < length else ""
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif char == "/" and following == "/":
            end = source.find("\n", index + 2)
            end = length if end < 0 else end
            for position in range(index, end):
                output[position] = " "
            index = end
            continue
        elif char == "/" and following == "*":
            end = source.find("*/", index + 2)
            end = length if end < 0 else end + 2
            for position in range(index, end):
                if output[position] != "\n":
                    output[position] = " "
            index = end
            continue
        index += 1
    return "".join(output)


def native_code_tokens(source: str) -> list[tuple[str, int]]:
    """Return ``(token_value, line)`` pairs for native code tokens.

    Comments are removed and string/char literal contents are collapsed to a
    single placeholder so that only structural duplication is measured.
    """

    masked = _mask_native_comments(source)
    newline_offsets = [match.start() for match in re.finditer("\n", masked)]
    result: list[tuple[str, int]] = []
    for match in _NATIVE_TOKEN.finditer(masked):
        text = match.group(0)
        first = text[0]
        if first == '"':
            value = _STRING_PLACEHOLDER
        elif first == "'":
            value = _CHAR_PLACEHOLDER
        else:
            value = text
        line = bisect.bisect_right(newline_offsets, match.start()) + 1
        result.append((value, line))
    return result


# --------------------------------------------------------------------------- #
# Source collection
# --------------------------------------------------------------------------- #
def _is_excluded(path: Path, root: Path) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(part in _EXCLUDED_PARTS for part in relative_parts)


def make_source(path_label: str, source: str, language: str) -> SourceFile:
    if language == "python":
        pairs = python_code_tokens(source)
    elif language == "native":
        pairs = native_code_tokens(source)
    else:
        raise ValueError(f"unknown language: {language}")
    values = tuple(value for value, _ in pairs)
    lines = tuple(line for _, line in pairs)
    return SourceFile(
        path=path_label,
        line_count=len(source.splitlines()),
        token_values=values,
        token_lines=lines,
    )


def collect_python_sources(repo: Path) -> list[SourceFile]:
    root = repo / PACKAGE_RELATIVE
    sources: list[SourceFile] = []
    for path in sorted(root.rglob("*.py")):
        if not path.is_file() or _is_excluded(path, root):
            continue
        text = path.read_text(encoding="utf-8-sig")
        sources.append(make_source(path.relative_to(repo).as_posix(), text, "python"))
    return sources


def collect_native_sources(repo: Path) -> list[SourceFile]:
    root = repo / NATIVE_RELATIVE
    sources: list[SourceFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in NATIVE_SUFFIXES:
            continue
        if _is_excluded(path, root):
            continue
        text = path.read_text(encoding="utf-8-sig")
        sources.append(make_source(path.relative_to(repo).as_posix(), text, "native"))
    return sources


# --------------------------------------------------------------------------- #
# Duplicate-region detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _Flat:
    values: list[str]
    ids: list[int]
    file_of: list[int]
    line_of: list[int]
    file_paths: list[str]
    file_bounds: list[tuple[int, int]]


def _flatten(sources: Sequence[SourceFile]) -> _Flat:
    values: list[str] = []
    ids: list[int] = []
    file_of: list[int] = []
    line_of: list[int] = []
    file_paths: list[str] = []
    file_bounds: list[tuple[int, int]] = []
    id_map: dict[str, int] = {}
    for file_index, source in enumerate(sources):
        start = len(values)
        file_paths.append(source.path)
        for value, line in zip(source.token_values, source.token_lines):
            token_id = id_map.get(value)
            if token_id is None:
                token_id = len(id_map) + 1
                id_map[value] = token_id
            values.append(value)
            ids.append(token_id)
            file_of.append(file_index)
            line_of.append(line)
        file_bounds.append((start, len(values)))
    return _Flat(values, ids, file_of, line_of, file_paths, file_bounds)


def _seed_groups(flat: _Flat, min_tokens: int) -> list[list[int]]:
    """Return verified position groups that share an identical ``min_tokens``-gram."""

    ids = flat.ids
    buckets: dict[int, list[int]] = defaultdict(list)
    power = pow(_HASH_BASE, min_tokens - 1, _HASH_MOD)
    for start, end in flat.file_bounds:
        span = end - start
        if span < min_tokens:
            continue
        rolling = 0
        for offset in range(min_tokens):
            rolling = (rolling * _HASH_BASE + ids[start + offset]) % _HASH_MOD
        buckets[rolling].append(start)
        for position in range(start + 1, end - min_tokens + 1):
            leaving = ids[position - 1]
            entering = ids[position + min_tokens - 1]
            rolling = (
                (rolling - leaving * power) * _HASH_BASE + entering
            ) % _HASH_MOD
            buckets[rolling].append(position)
    seeds: list[list[int]] = []
    for positions in buckets.values():
        if len(positions) < 2:
            continue
        by_content: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for position in positions:
            key = tuple(ids[position : position + min_tokens])
            by_content[key].append(position)
        for group in by_content.values():
            if len(group) >= 2:
                seeds.append(group)
    return seeds


class _UnionFind:
    """Disjoint-set forest with path compression and union by size."""

    def __init__(self, count: int) -> None:
        self._parent = list(range(count))
        self._size = [1] * count

    def find(self, node: int) -> int:
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self._size[left] < self._size[right]:
            left, right = right, left
        self._parent[right] = left
        self._size[left] += self._size[right]


def _coalesce_blocks(
    flat: _Flat, covered: bytearray, min_tokens: int
) -> tuple[list[tuple[int, int]], list[int]]:
    """Merge covered token runs into contiguous same-file blocks.

    Returns the block spans ``[start, end)`` and a per-token map from token
    index to its owning block index (``-1`` when uncovered).
    """

    blocks: list[tuple[int, int]] = []
    block_of = [-1] * len(covered)
    index = 0
    total = len(covered)
    while index < total:
        if not covered[index]:
            index += 1
            continue
        start = index
        file_index = flat.file_of[index]
        while (
            index < total
            and covered[index]
            and flat.file_of[index] == file_index
        ):
            index += 1
        end = index
        if end - start >= min_tokens:
            block_id = len(blocks)
            blocks.append((start, end))
            for token in range(start, end):
                block_of[token] = block_id
    return blocks, block_of


def _content_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\x00".join(values).encode("utf-8")).hexdigest()


def _region_id(corpus: str, block_content_hashes: Sequence[str]) -> str:
    payload = corpus + "\x00" + "\x00".join(sorted(block_content_hashes))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def analyze_corpus(
    corpus: str, sources: Sequence[SourceFile], min_tokens: int = MIN_TOKENS
) -> CorpusResult:
    flat = _flatten(sources)
    seeds = _seed_groups(flat, min_tokens)

    covered = bytearray(len(flat.ids))
    for positions in seeds:
        for position in positions:
            covered[position : position + min_tokens] = b"\x01" * min_tokens

    blocks, block_of = _coalesce_blocks(flat, covered, min_tokens)
    union = _UnionFind(len(blocks))
    for positions in seeds:
        blocks_here = {block_of[position] for position in positions}
        blocks_here.discard(-1)
        iterator = iter(sorted(blocks_here))
        try:
            first = next(iterator)
        except StopIteration:
            continue
        for other in iterator:
            union.union(first, other)

    components: dict[int, list[int]] = defaultdict(list)
    for block_id in range(len(blocks)):
        components[union.find(block_id)].append(block_id)

    region_objects: list[Region] = []
    for block_ids in components.values():
        if len(block_ids) < 2:
            continue
        occurrences: list[Occurrence] = []
        content_hashes: list[str] = []
        token_count = 0
        for block_id in block_ids:
            start, end = blocks[block_id]
            occurrences.append(
                Occurrence(
                    path=flat.file_paths[flat.file_of[start]],
                    start_line=flat.line_of[start],
                    end_line=flat.line_of[end - 1],
                )
            )
            content_hashes.append(_content_hash(flat.values[start:end]))
            token_count = max(token_count, end - start)
        occurrences.sort(key=lambda occ: (occ.path, occ.start_line, occ.end_line))
        region_objects.append(
            Region(
                region_id=_region_id(corpus, content_hashes),
                corpus=corpus,
                token_count=token_count,
                occurrences=tuple(occurrences),
            )
        )
    region_objects.sort(key=lambda region: (-region.token_count, region.region_id))

    per_file_total = {source.path: source.line_count for source in sources}
    per_file_duplicate: dict[str, set[int]] = defaultdict(set)
    for token_index, marked in enumerate(covered):
        if marked:
            per_file_duplicate[flat.file_paths[flat.file_of[token_index]]].add(
                flat.line_of[token_index]
            )
    per_file_duplicate_lines = {
        path: len(lines) for path, lines in per_file_duplicate.items()
    }
    duplicate_lines = sum(per_file_duplicate_lines.values())
    total_lines = sum(per_file_total.values())
    return CorpusResult(
        corpus=corpus,
        file_count=len(sources),
        total_lines=total_lines,
        duplicate_lines=duplicate_lines,
        regions=tuple(region_objects),
        per_file_duplicate_lines=per_file_duplicate_lines,
        per_file_total_lines=per_file_total,
    )


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def _corpus_summary(result: CorpusResult) -> dict[str, object]:
    return {
        "files": result.file_count,
        "total_lines": result.total_lines,
        "duplicate_lines": result.duplicate_lines,
        "coverage_percent": result.coverage_percent,
        "region_count": len(result.regions),
    }


def _region_payload(region: Region) -> dict[str, object]:
    return {
        "region_id": region.region_id,
        "corpus": region.corpus,
        "token_count": region.token_count,
        "occurrence_count": len(region.occurrences),
        "occurrences": [
            {
                "file": occ.path,
                "start_line": occ.start_line,
                "end_line": occ.end_line,
            }
            for occ in region.occurrences
        ],
    }


def build_report(repo: Path, min_tokens: int = MIN_TOKENS) -> dict[str, object]:
    python_result = analyze_corpus(
        "python", collect_python_sources(repo), min_tokens
    )
    native_result = analyze_corpus(
        "native", collect_native_sources(repo), min_tokens
    )
    return report_from_results(python_result, native_result, min_tokens)


def report_from_results(
    python_result: CorpusResult,
    native_result: CorpusResult,
    min_tokens: int = MIN_TOKENS,
) -> dict[str, object]:
    results = (python_result, native_result)
    combined_duplicate = sum(result.duplicate_lines for result in results)
    combined_total = sum(result.total_lines for result in results)
    regions = [
        _region_payload(region)
        for result in results
        for region in result.regions
    ]
    regions.sort(
        key=lambda item: (
            str(item["corpus"]),
            -int(item["token_count"]),
            str(item["region_id"]),
        )
    )
    per_file: list[dict[str, object]] = []
    for result in results:
        for path in sorted(result.per_file_total_lines):
            total = result.per_file_total_lines[path]
            duplicate = result.per_file_duplicate_lines.get(path, 0)
            per_file.append(
                {
                    "file": path,
                    "corpus": result.corpus,
                    "total_lines": total,
                    "duplicate_lines": duplicate,
                    "coverage_percent": _percent(duplicate, total),
                }
            )
    per_file.sort(key=lambda item: str(item["file"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "metric": "exact-token-duplicate-regions",
        "min_tokens": min_tokens,
        "generated_by": "tools/duplication_report.py",
        "corpora": {
            "python": _corpus_summary(python_result),
            "native": _corpus_summary(native_result),
        },
        "combined": {
            "files": python_result.file_count + native_result.file_count,
            "total_lines": combined_total,
            "duplicate_lines": combined_duplicate,
            "coverage_percent": _percent(combined_duplicate, combined_total),
            "region_count": len(python_result.regions)
            + len(native_result.regions),
        },
        "per_file": per_file,
        "regions": regions,
    }


def region_ids(report: dict[str, object]) -> set[str]:
    return {str(region["region_id"]) for region in report["regions"]}  # type: ignore[index]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of tools/)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=MIN_TOKENS,
        help="minimum duplicate region length in tokens",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the JSON report to this path instead of stdout",
    )
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    report = build_report(repo, args.min_tokens)
    if args.output is not None:
        _write_json(args.output, report)
        print(str(args.output))
    else:
        sys.stdout.write(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())