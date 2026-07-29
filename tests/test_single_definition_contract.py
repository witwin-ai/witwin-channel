# Copyright Xingyu Chen.
# The single-definition gate passes here, and FAILS on a planted duplicate.

"""The single-definition gate passes here, and FAILS on a planted duplicate."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from ci import check_single_definition as single


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "witwin" / "channel"

AVAILABILITY = "component availability status"
DEPTH = "component max-depth rule"
SYMBOL = "native symbol lookup"


@pytest.fixture(scope="module")
def real_facts() -> list[single.FunctionFacts]:
    return single.package_facts(PACKAGE_ROOT)


def concept(name: str) -> single.Concept:
    (found,) = [entry for entry in single.CONCEPTS if entry.name == name]
    return found


def planted(source: str, module: str) -> list[single.FunctionFacts]:
    return single.function_facts(source, module, "<planted>")


# A fourth copy of the depth rule, written from scratch the way the real one in
# `deterministic/pipeline.py` was: different name, different parameters,
# per-key assignment instead of a dict literal.
DEPTH_COPY = '''
def _interaction_budget(active, depth):
    budget = {}
    budget["los"] = 0 if "los" in active else -1
    budget["reflection"] = depth if "reflection" in active else -1
    budget["transmission"] = depth if "transmission" in active else -1
    budget["diffraction"] = 1 if "diffraction" in active else -1
    budget["scattering"] = 1 if "scattering" in active else -1
    return budget
'''

AVAILABILITY_COPY = '''
def _describe_enabled(wanted, *, can_reflect, can_diffract):
    report = {
        "los": "enabled" if "los" in wanted else "not_requested",
        "reflection": "not_requested",
        "diffraction": "not_requested",
        "transmission": "enabled" if "transmission" in wanted else "not_requested",
        "scattering": "enabled" if "scattering" in wanted else "not_requested",
    }
    if "reflection" in wanted:
        if not can_reflect:
            raise RuntimeError("reflection unavailable")
        report["reflection"] = "enabled"
    if "diffraction" in wanted:
        if not can_diffract:
            raise RuntimeError("diffraction unavailable")
        report["diffraction"] = "enabled"
    return report
'''

SYMBOL_COPY = '''
def _fetch_kernel(op):
    module = native_extension()
    if module is None or not hasattr(module, op):
        raise RuntimeError(f"_channel.{op} CUDA kernel is required")
    return getattr(module, op)
'''


# --- The tree is clean today ----------------------------------------------


def test_the_real_package_has_one_definition_per_concept():
    assert single.find_violations(PACKAGE_ROOT) == []


def test_each_protected_concept_resolves_to_its_declared_owner(real_facts):
    for entry in single.CONCEPTS:
        owned = [
            site
            for site in single.definition_sites(entry, real_facts)
            if site.module == entry.owner
            and (site.module, site.qualname) not in entry.recorded_duplicates
        ]
        assert len(owned) == entry.owner_definitions, entry.name


def test_the_registry_protects_the_three_named_concepts():
    assert {entry.name for entry in single.CONCEPTS} == {
        AVAILABILITY,
        DEPTH,
        SYMBOL,
    }
    assert concept(AVAILABILITY).owner == "witwin.channel.components"
    assert concept(DEPTH).owner == "witwin.channel.components"
    assert concept(SYMBOL).owner == "witwin.channel.runtime"


def test_a_signature_is_not_satisfied_by_ordinary_code(real_facts):
    """1300+ real function scopes, and only the owners match."""

    assert len(real_facts) > 1000
    matched = {
        (site.module, site.qualname)
        for entry in (concept(AVAILABILITY), concept(DEPTH))
        for site in single.definition_sites(entry, real_facts)
    }
    assert matched == {
        ("witwin.channel.components", "component_availability_status"),
        ("witwin.channel.components", "component_max_depth"),
    }


# --- A planted duplicate is rejected --------------------------------------


@pytest.mark.parametrize(
    ("name", "source", "module", "qualname"),
    [
        (
            DEPTH,
            DEPTH_COPY,
            "witwin.channel.deterministic.pipeline",
            "_interaction_budget",
        ),
        (
            AVAILABILITY,
            AVAILABILITY_COPY,
            "witwin.channel.path.metadata",
            "_describe_enabled",
        ),
        (
            SYMBOL,
            SYMBOL_COPY,
            "witwin.channel.deterministic.pipeline",
            "_fetch_kernel",
        ),
    ],
)
def test_a_renamed_near_copy_is_rejected(
    real_facts, name: str, source: str, module: str, qualname: str
):
    entry = concept(name)
    violations = single.concept_violations(
        entry, real_facts + planted(source, module)
    )

    assert [violation.kind for violation in violations] == ["second-definition"]
    assert f"{module}.{qualname}" in violations[0].detail
    assert entry.owner in violations[0].detail


def test_detection_ignores_every_identifier():
    """The planted copies share no name with the owner they duplicate."""

    for source, owner_name in (
        (DEPTH_COPY, "component_max_depth"),
        (AVAILABILITY_COPY, "component_availability_status"),
        (SYMBOL_COPY, "required_symbol"),
    ):
        assert owner_name not in source
    for shared in ("components", "chain_depth", "single_bounce_depth", "status"):
        assert shared not in DEPTH_COPY


def test_a_second_copy_inside_the_owner_module_is_rejected(real_facts):
    """One site package-wide, not one per module."""

    entry = concept(DEPTH)
    violations = single.concept_violations(
        entry, real_facts + planted(DEPTH_COPY, entry.owner)
    )

    assert [violation.kind for violation in violations] == [
        "owner-definition-count"
    ]
    assert "2 time(s), expected 1" in violations[0].detail


def test_a_module_level_copy_is_rejected(real_facts):
    """A rule written as a top-level dict is still a definition site."""

    source = (
        'BUDGET = {\n'
        '    "los": 0,\n'
        '    "reflection": -1,\n'
        '    "diffraction": -1,\n'
        '    "transmission": -1,\n'
        '    "scattering": -1,\n'
        '}\n'
    )
    violations = single.concept_violations(
        concept(DEPTH),
        real_facts + planted(source, "witwin.channel.path.metadata"),
    )

    assert [violation.kind for violation in violations] == ["second-definition"]
    assert "<module>" in violations[0].detail


# --- The recorded-duplicate ledger only shrinks ---------------------------


def test_the_recorded_duplicate_ledger_is_empty():
    assert all(entry.recorded_duplicates == frozenset() for entry in single.CONCEPTS)
    assert all(not entry.debt for entry in single.CONCEPTS)


def test_every_recorded_duplicate_still_exists(real_facts):
    """Everything that is not the one canonical definition is on the ledger."""

    entry = concept(SYMBOL)
    sites = single.definition_sites(entry, real_facts)
    canonical = {
        (site.module, site.qualname)
        for site in sites
        if site.module == entry.owner
        and (site.module, site.qualname) not in entry.recorded_duplicates
    }
    found = {(site.module, site.qualname) for site in sites} - canonical

    assert len(canonical) == entry.owner_definitions
    assert found == entry.recorded_duplicates


def test_a_cleaned_up_duplicate_must_leave_the_ledger(real_facts):
    """The ratchet: a stale entry fails until the list shrinks."""

    entry = concept(SYMBOL)
    stale = single.Concept(
        name=entry.name,
        owner=entry.owner,
        reason=entry.reason,
        signature=entry.signature,
        recorded_duplicates=entry.recorded_duplicates
        | {("witwin.channel.gone", "already_fixed")},
    )
    violations = single.concept_violations(stale, real_facts)

    assert [violation.kind for violation in violations] == [
        "stale-recorded-duplicate"
    ]
    assert "witwin.channel.gone.already_fixed" in violations[0].detail


def test_an_unrecorded_probe_fails_with_an_empty_ledger(real_facts):
    violations = single.concept_violations(
        concept(SYMBOL),
        real_facts + planted(SYMBOL_COPY, "witwin.channel.scene.compiler"),
    )

    assert [violation.kind for violation in violations] == ["second-definition"]


# --- Fact extraction ------------------------------------------------------


def test_a_nested_function_is_its_own_scope():
    facts = {
        fact.qualname: fact
        for fact in planted(
            'def outer():\n'
            '    def inner():\n'
            '        raise ValueError("boom")\n'
            '    return inner\n',
            "witwin.channel.x",
        )
    }

    assert set(facts) == {"<module>", "outer", "outer.inner"}
    assert facts["outer.inner"].raises
    assert not facts["outer"].raises


def test_negative_and_boolean_constants_are_read_as_written():
    (fact,) = [
        fact
        for fact in planted("def f():\n    return (-1, 0, True)\n", "witwin.channel.x")
        if fact.qualname == "f"
    ]

    assert -1.0 in fact.numbers
    assert 0.0 in fact.numbers
    assert 1.0 not in fact.numbers


# --- CLI ------------------------------------------------------------------


def test_cli_passes_with_repository_defaults(capsys):
    assert single.main(["--repository-root", str(REPOSITORY_ROOT)]) == 0
    assert "single-definition check passed" in capsys.readouterr().out


def test_cli_fails_on_a_duplicate_planted_in_a_mirror(tmp_path: Path, capsys):
    package_root = tmp_path / "witwin" / "channel"
    shutil.copytree(
        PACKAGE_ROOT,
        package_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyd", "*.so"),
    )
    target = package_root / "deterministic.py"
    target.write_text(
        target.read_text(encoding="utf-8") + DEPTH_COPY, encoding="utf-8"
    )

    assert single.main(["--package-root", str(package_root)]) == 1
    output = capsys.readouterr().out
    assert "single-definition check failed" in output
    assert "component max-depth rule" in output
    assert "_interaction_budget" in output