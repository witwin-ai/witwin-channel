# Copyright Xingyu Chen.
# Tests duplication report.

from __future__ import annotations

from ci import check_duplication
from tools import duplication_report


def _make_python_function(name: str, tail: str) -> str:
    body = "\n".join(f"    v{i} = w{i} + x{i} * y{i}" for i in range(30))
    return f"def {name}():\n{body}\n    return {tail}\n"


def test_shared_block_detected_once_with_correct_line_spans() -> None:
    # Two files share an identical 30-line (210 token) body wrapped in unique
    # signatures and return statements, so the duplicate block is bounded.
    source_a = _make_python_function("alpha", "alpha_end")
    source_b = _make_python_function("beta", "beta_end")
    sources = [
        duplication_report.make_source("a.py", source_a, "python"),
        duplication_report.make_source("b.py", source_b, "python"),
    ]

    result = duplication_report.analyze_corpus("python", sources)

    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.token_count >= 210
    assert len(region.occurrences) == 2
    assert {occ.path for occ in region.occurrences} == {"a.py", "b.py"}
    for occ in region.occurrences:
        # Exact-token matching pulls in the shared structural tokens '(', ')',
        # ':' on the signature line and the shared 'return' keyword, so the
        # bounded duplicate region spans line 1 through line 32 in both files.
        assert occ.start_line == 1
        assert occ.end_line == 32


def test_internal_repeat_below_threshold_is_not_reported() -> None:
    short = "\n".join(f"    v{i} = w{i} + x{i}" for i in range(5))
    source_a = f"def alpha():\n{short}\n    return alpha_end\n"
    source_b = f"def beta():\n{short}\n    return beta_end\n"
    sources = [
        duplication_report.make_source("a.py", source_a, "python"),
        duplication_report.make_source("b.py", source_b, "python"),
    ]

    result = duplication_report.analyze_corpus("python", sources)

    assert result.regions == ()
    assert result.duplicate_lines == 0


def test_python_tokenizer_drops_docstrings_but_keeps_code() -> None:
    tokens = duplication_report.python_code_tokens(
        'def f():\n    """doc string only."""\n    return value + 1\n'
    )
    values = [value for value, _ in tokens]
    assert "doc string only." not in "".join(values)
    assert values == ["def", "f", "(", ")", ":", "return", "value", "+", "1"]


def test_native_tokenizer_strips_comments_and_string_contents() -> None:
    tokens = duplication_report.native_code_tokens(
        'int f() { // secret\n  const char* s = "literal text"; /* block */ return 0; }\n'
    )
    values = [value for value, _ in tokens]
    assert "secret" not in "".join(values)
    assert "literal text" not in "".join(values)
    assert '"<str>"' in values


def _report(coverage: float, region_ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "metric": "exact-token-duplicate-regions",
        "min_tokens": 100,
        "combined": {
            "coverage_percent": coverage,
            "duplicate_lines": 0,
            "total_lines": 100,
            "region_count": len(region_ids),
            "files": 2,
        },
        "corpora": {},
        "per_file": [],
        "regions": [
            {"region_id": region_id, "corpus": "python", "token_count": 120}
            for region_id in region_ids
        ],
    }


def _ledger(coverage: float, region_ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "baseline": {"coverage_percent": coverage},
        "regions": {
            region_id: {
                "category": "other",
                "owner": "example",
                "reason": "example duplication",
            }
            for region_id in region_ids
        },
    }


def test_gate_passes_when_ledger_matches_report() -> None:
    report = _report(5.0, ["aaaa", "bbbb"])
    ledger = _ledger(6.0, ["aaaa", "bbbb"])
    assert check_duplication.evaluate(report, ledger) == []


def test_gate_flags_unclassified_region() -> None:
    report = _report(5.0, ["aaaa", "cccc"])
    ledger = _ledger(6.0, ["aaaa"])
    violations = check_duplication.evaluate(report, ledger)
    assert any("unclassified region: cccc" in item for item in violations)


def test_gate_flags_stale_ledger_entry() -> None:
    report = _report(5.0, ["aaaa"])
    ledger = _ledger(6.0, ["aaaa", "dddd"])
    violations = check_duplication.evaluate(report, ledger)
    assert any("stale ledger entry: dddd" in item for item in violations)


def test_gate_flags_coverage_regression() -> None:
    report = _report(7.5, ["aaaa"])
    ledger = _ledger(5.0, ["aaaa"])
    violations = check_duplication.evaluate(report, ledger)
    assert any("coverage regression" in item for item in violations)


def test_gate_allows_coverage_at_baseline() -> None:
    report = _report(5.0, ["aaaa"])
    ledger = _ledger(5.0, ["aaaa"])
    assert check_duplication.evaluate(report, ledger) == []


def test_gate_requires_lockstep_tests_for_numeric_exemption() -> None:
    report = _report(5.0, ["aaaa"])
    ledger = _ledger(6.0, ["aaaa"])
    ledger["regions"]["aaaa"]["category"] = "numeric_sensitive_exempt"
    violations = check_duplication.evaluate(report, ledger)
    assert any("missing lockstep_tests" in item for item in violations)